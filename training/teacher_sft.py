"""Teacher SFT 训练脚本。支持 Teacher_R(纯文本) 和 Teacher_P(视频)。
Teacher_R: 输入含完整trajectory(含result)，但只对推理段计算loss。
Teacher_P: 输入视频+感知问题，标准CE loss。
"""
from __future__ import annotations
import argparse, json, math, os, re, sys, time, warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch, torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path: sys.path.insert(0, str(_CODE_ROOT))
from training.latent_sft_helpers import load_system_prompt
from utils.qwen3vl_patch import apply_qwen3vl_patches
apply_qwen3vl_patches()

def is_dist(): return dist.is_available() and dist.is_initialized()
def get_rank(): return dist.get_rank() if is_dist() else 0
def is_main_process(): return get_rank() == 0
def rank0_print(*a, **kw):
    if is_main_process(): print(*a, **kw, flush=True)
def setup_distributed():
    if "RANK" in os.environ:
        if not dist.is_initialized(): dist.init_process_group(backend="nccl")
        lr = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(lr); return lr
    if torch.cuda.is_available(): torch.cuda.set_device(0)
    return 0

class TeacherSFTDataset(Dataset):
    def __init__(self, path, max_samples=0):
        self.items = [json.loads(l) for l in open(path) if l.strip()]
        if max_samples and 0 < max_samples < len(self.items):
            import random; random.seed(42); random.shuffle(self.items)
            self.items = self.items[:max_samples]
        rank0_print(f"[Dataset] {len(self.items)} samples: {path}")
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

class TeacherRCollator:
    def __init__(self, processor, max_length=8192):
        self.processor, self.tokenizer, self.max_length = processor, processor.tokenizer, max_length
    def __call__(self, batch):
        if not batch: return None
        item = batch[0]
        q, traj = item["question"], item["trajectory"]
        no_loss = item.get("no_loss_spans", [])
        sp = load_system_prompt("teacher_r")
        msgs = []
        if sp: msgs.append({"role":"system","content":[{"type":"text","text":sp}]})
        msgs.append({"role":"user","content":[{"type":"text","text":q}]})
        msgs.append({"role":"assistant","content":[{"type":"text","text":traj}]})
        full = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc = self.tokenizer(full, return_tensors="pt", truncation=True, max_length=self.max_length, return_offsets_mapping=True)
        ids, attn, offsets = enc["input_ids"], enc["attention_mask"], enc["offset_mapping"][0]
        labels = ids.clone()
        # prefix mask
        pfx = self.processor.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        plen = self.tokenizer(pfx, return_tensors="pt", truncation=True, max_length=self.max_length)["input_ids"].shape[1]
        labels[0, :plen] = -100
        # result span mask
        if no_loss:
            ts = full.find(traj)
            if ts >= 0:
                for s in no_loss:
                    cs, ce = ts + s["start"], ts + s["end"]
                    for ti in range(ids.shape[1]):
                        a, b = offsets[ti].tolist()
                        if a==0 and b==0: continue
                        if a >= cs and b <= ce: labels[0, ti] = -100
        return {"input_ids": ids, "attention_mask": attn, "labels": labels}

class TeacherPCollator:
    def __init__(self, processor, max_length=32768, max_frames=64, fps=1.0):
        self.processor, self.tokenizer = processor, processor.tokenizer
        self.max_length, self.max_frames, self.fps = max_length, max_frames, fps
    def __call__(self, batch):
        if not batch: return None
        item = batch[0]
        vp, pq, rt = item["video"], item["perception_question"], item["result_text"]
        if not os.path.exists(vp): return None
        sp = load_system_prompt("teacher_p")
        msgs = []
        if sp: msgs.append({"role":"system","content":[{"type":"text","text":sp}]})
        msgs.append({"role":"user","content":[{"type":"video","video":vp,"max_pixels":360*420,"fps":self.fps},{"type":"text","text":pq}]})
        msgs.append({"role":"assistant","content":[{"type":"text","text":rt}]})
        # 用 apply_chat_template(tokenize=True) 一步完成视频解码+tokenize，无需 qwen_vl_utils
        try:
            inp = self.processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_tensors="pt"
            )
        except Exception as e:
            rank0_print(f"[WARN] {vp}: {e}"); return None
        ids, labels = inp["input_ids"], inp["input_ids"].clone()
        # 计算 prefix 长度（不含 assistant 回复）
        try:
            pfx_inp = self.processor.apply_chat_template(
                msgs[:-1], tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
            plen = pfx_inp["input_ids"].shape[1]
        except:
            pfx = self.processor.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            plen = len(self.tokenizer.encode(pfx))
        labels[0, :plen] = -100
        r = {"input_ids": ids, "attention_mask": inp["attention_mask"], "labels": labels}
        # 显式传递模型需要的视觉字段
        for k in ("pixel_values_videos", "video_grid_thw", "mm_token_type_ids",
                  "pixel_values", "image_grid_thw", "second_per_grid_ts"):
            if k in inp:
                r[k] = inp[k]
        return r

def train(args):
    lr_ = setup_distributed(); dev = torch.device(f"cuda:{lr_}"); dt = torch.bfloat16
    rank0_print(f"[Teacher SFT] role={args.role} data={args.train_jsonl}")
    from transformers import AutoProcessor
    try: from transformers import Qwen3VLForConditionalGeneration as MC
    except: from transformers import AutoModelForVision2Seq as MC
    proc = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = MC.from_pretrained(args.model_path, torch_dtype=dt, trust_remote_code=True).to(dev)
    model.train()
    if args.gradient_checkpointing: model.gradient_checkpointing_enable()
    ds = TeacherSFTDataset(args.train_jsonl, args.max_samples)
    col = TeacherRCollator(proc, args.max_length) if args.role=="teacher_r" else TeacherPCollator(proc, args.max_length, args.max_frames, args.fps)
    smp = DistributedSampler(ds, shuffle=True) if is_dist() else None
    dl = DataLoader(ds, batch_size=1, sampler=smp, collate_fn=col, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ts = len(dl)*args.epochs//args.grad_accum; wu = max(1, int(ts*0.03))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s/wu if s<wu else 0.5*(1+math.cos(math.pi*(s-wu)/max(1,ts-wu))))
    if is_main_process() and args.wandb_project:
        try: import wandb; wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        except: pass
    gs = 0; os.makedirs(args.output_dir, exist_ok=True)
    for ep in range(args.epochs):
        if smp: smp.set_epoch(ep)
        rank0_print(f"\n[Epoch {ep+1}/{args.epochs}]"); opt.zero_grad(); al = 0.0
        for st, b in enumerate(tqdm(dl, disable=not is_main_process())):
            if b is None: continue
            # 将所有tensor移到GPU，处理可能的list of tensors情况
            bd = {}
            for k, v in b.items():
                if isinstance(v, torch.Tensor):
                    bd[k] = v.to(dev)
                elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    bd[k] = [t.to(dev) for t in v]
                else:
                    bd[k] = v
            try:
                loss = model(**bd).loss / args.grad_accum; loss.backward(); al += loss.item()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    warnings.warn(f"[OOM] step={st}"); torch.cuda.empty_cache(); opt.zero_grad(); al=0; continue
                raise
            if (st+1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); gs += 1
                if is_main_process() and gs % args.logging_steps == 0:
                    rank0_print(f"  step={gs} loss={al:.4f} lr={sch.get_last_lr()[0]:.2e}")
                    try: import wandb; wandb.log({"train/loss": al, "train/lr": sch.get_last_lr()[0]}, step=gs)
                    except: pass
                al = 0.0
                if args.save_steps and gs % args.save_steps == 0 and is_main_process():
                    d=os.path.join(args.output_dir,f"checkpoint-{gs}"); rank0_print(f"  [Save] {d}"); model.save_pretrained(d); proc.save_pretrained(d)
    if is_main_process():
        d=os.path.join(args.output_dir,"final"); rank0_print(f"\n[Save] {d}"); model.save_pretrained(d); proc.save_pretrained(d)
    # 释放训练显存，避免 barrier 时 OOM
    del model, opt, sch; torch.cuda.empty_cache()
    if is_dist(): dist.barrier()
    rank0_print("[Done] Teacher SFT finished.")

def inference_check(args):
    dev = torch.device("cuda:0"); dt = torch.bfloat16
    ckpt = args.inference_ckpt or os.path.join(args.output_dir, "final")
    if not os.path.exists(ckpt): print(f"[ERROR] no ckpt: {ckpt}"); return
    print(f"\n[Inference Check] role={args.role} ckpt={ckpt}")
    from transformers import AutoProcessor
    try: from transformers import Qwen3VLForConditionalGeneration as MC
    except: from transformers import AutoModelForVision2Seq as MC
    proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    model = MC.from_pretrained(ckpt, torch_dtype=dt, trust_remote_code=True).to(dev).eval()
    ds = TeacherSFTDataset(args.train_jsonl, max_samples=3)
    for i, item in enumerate(ds.items[:3]):
        print(f"\n--- Sample {i+1} ---")
        q = item.get("question", item.get("perception_question",""))
        print(f"  Q: {q}")
        sp = load_system_prompt(args.role)
        msgs = [{"role":"system","content":[{"type":"text","text":sp}]}] if sp else []
        if args.role == "teacher_r":
            msgs.append({"role":"user","content":[{"type":"text","text":q}]})
            t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc.tokenizer(t, return_tensors="pt").to(dev)
            with torch.no_grad(): out = model.generate(**inp, max_new_tokens=512, do_sample=False)
            gen = proc.tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=False)
        else:
            vp = item.get("video","")
            if not os.path.exists(vp): print("  [SKIP] no video"); continue
            msgs.append({"role":"user","content":[{"type":"video","video":vp,"max_pixels":360*420,"fps":1.0},{"type":"text","text":item.get("perception_question","")}]})
            try:
                inp = proc.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt"
                )
                inp = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            except Exception as e: print(f"  [SKIP] {e}"); continue
            with torch.no_grad(): out = model.generate(**inp, max_new_tokens=256, do_sample=False)
            gen = proc.tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=False)
        print(f"  Gen: {gen[:500]}")
        gt = item.get("trajectory", item.get("result_text",""))[:200]
        print(f"  GT:  {gt}...")
    print("\n[Inference Check Done]")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True, choices=["teacher_r","teacher_p"])
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_path", default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--wandb_project", default="")
    p.add_argument("--wandb_run_name", default="")
    p.add_argument("--inference_only", action="store_true")
    p.add_argument("--inference_ckpt", default=None)
    args = p.parse_args()
    if not args.model_path:
        from configs.paths import QWEN3_VL_4B_PATH; args.model_path = str(QWEN3_VL_4B_PATH)
    if args.inference_only: inference_check(args)
    else:
        train(args)
        if is_main_process(): inference_check(args)

if __name__ == "__main__": main()