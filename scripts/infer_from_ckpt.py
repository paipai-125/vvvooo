#!/usr/bin/env python
"""
从 SFT ckpt 加载模型并对训练集前几条样本做推理 —— 用于调试 ckpt 是否能输出期望格式。

支持 3 种 role：
  --role student   : Coconut 式潜空间训练后的学生模型；推理时直接用学生 LM head
                     generate 即可（输出会包含 <think>...</think><answer>...</answer>）。
                     额外可加 --use_init_lm_head_for_think 用初始 LM head 单独
                     重新解码 think 区域，看潜空间是否能解码出更合理内容。
  --role teacher_r : 推理教师；标准 generate（无视频，纯文本）
  --role teacher_p : 感知教师；标准 generate（带视频）

用法：
  python -m scripts.infer_from_ckpt --role student   --ckpt <dir> [--num_samples 3]
  python -m scripts.infer_from_ckpt --role teacher_r --ckpt <dir>
  python -m scripts.infer_from_ckpt --role teacher_p --ckpt <dir>

ckpt 目录可以是：
  1. DeepSpeed ZeRO 保存的 checkpoint-N 目录
     (内含 zero_pp_rank_*_mp_rank_*_model_states.pt 等分片)
  2. HF 标准目录 (含 config.json + safetensors)
脚本会自动判断格式。
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from configs import paths as cfg_paths  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("从 SFT ckpt 加载并推理")
    p.add_argument("--role", required=True,
                   choices=["student", "teacher_r", "teacher_p"])
    p.add_argument("--ckpt", required=True,
                   help="ckpt 目录: 既可是 DeepSpeed ZeRO checkpoint-N 目录, "
                        "也可是 HF 标准目录")
    p.add_argument("--base_model", default=str(cfg_paths.QWEN3_VL_4B_PATH),
                   help="基础模型路径 (用于初始化结构 + 拿 processor + 兜底 base 权重)")
    p.add_argument("--train_jsonl", default=None,
                   help="训练 jsonl, 默认按 role 自动推断")
    p.add_argument("--num_samples", type=int, default=3,
                   help="从 jsonl 取几条来推理")
    p.add_argument("--shuffle", action="store_true", default=True,
                   help="是否随机抽样（默认 True，避免取到 jsonl 头部同一类任务）")
    p.add_argument("--no_shuffle", dest="shuffle", action="store_false",
                   help="关闭随机抽样，按文件顺序取前 N 条")
    p.add_argument("--seed", type=int, default=42,
                   help="随机抽样种子")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--use_init_lm_head_for_think", action="store_true", default=False,
                   help="(student 专用) 在 generate 完后, 用 base 模型初始 LM head "
                        "重新解码 <think> 区域 hidden states 看潜空间内容")
    return p.parse_args()


def auto_train_jsonl(role: str) -> str:
    sft_dir = cfg_paths.OUTPUT_ROOT / "stage1_sft"
    if role == "student":
        return str(sft_dir / "stage1_sft_template_all.jsonl")
    elif role == "teacher_r":
        return str(sft_dir / "stage1_sft_teacher_r.jsonl")
    elif role == "teacher_p":
        return str(sft_dir / "stage1_sft_teacher_p.jsonl")
    raise ValueError(role)


def is_deepspeed_zero_dir(ckpt_dir: Path) -> bool:
    """判断是不是 DeepSpeed ZeRO 风格 ckpt 目录。"""
    if not ckpt_dir.is_dir():
        return False
    # ZeRO ckpt 特征: 有 zero_pp_rank_*_mp_rank_*_model_states.pt 或 latest 文件
    for f in ckpt_dir.iterdir():
        name = f.name
        if name == "latest" or name.startswith("zero_pp_rank_") or "mp_rank_" in name:
            return True
    return False


def load_model_from_ckpt(args, dtype: torch.dtype, device: torch.device):
    """统一的模型加载入口, 自动判断 HF 目录还是 DeepSpeed ZeRO ckpt 目录。"""
    # 0) Qwen3-VL processor 的 monkey-patch
    from utils.qwen3vl_patch import apply_qwen3vl_patches
    apply_qwen3vl_patches()

    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as _Cls
    except ImportError:
        from transformers import AutoModelForVision2Seq as _Cls

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = args.fps
        if hasattr(processor.video_processor, "max_frames"):
            processor.video_processor.max_frames = args.max_frames
        if hasattr(processor.video_processor, "num_frames"):
            try:
                processor.video_processor.num_frames = None
            except Exception:
                pass

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt 路径不存在: {ckpt_path}")

    # 处理 ckpt 是 父目录 (output_dir) 而非 checkpoint-N 的情况:
    # 自动找最新的 checkpoint-N 子目录
    if ckpt_path.is_dir() and not is_deepspeed_zero_dir(ckpt_path) \
            and not (ckpt_path / "config.json").exists():
        sub_ckpts = sorted(
            [p for p in ckpt_path.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        if sub_ckpts:
            ckpt_path = sub_ckpts[-1]
            print(f"[Infer] 自动选择最新 ckpt: {ckpt_path}", flush=True)

    print(f"[Infer] 加载 base 模型结构从: {args.base_model}", flush=True)
    model = _Cls.from_pretrained(args.base_model, torch_dtype=dtype, trust_remote_code=True)

    if is_deepspeed_zero_dir(ckpt_path):
        # DeepSpeed ZeRO ckpt: 用 zero_to_fp32 合并分片
        print(f"[Infer] 检测到 DeepSpeed ZeRO ckpt, 合并分片: {ckpt_path}", flush=True)
        try:
            from deepspeed.utils.zero_to_fp32 import (
                load_state_dict_from_zero_checkpoint,
                get_fp32_state_dict_from_zero_checkpoint,
            )
        except ImportError:
            raise RuntimeError("无法导入 deepspeed.utils.zero_to_fp32, 请确认 deepspeed 已安装")

        # ZeRO ckpt 的 path 是 ckpt 的 *父*目录 (即 output_dir), tag 是 checkpoint-N
        # 但 load_state_dict_from_zero_checkpoint 接受 ckpt_dir 直接是 checkpoint-N 目录
        # 我们这里两种调用方式都试一下
        try:
            # 方式 A: ckpt_path 直接是 checkpoint-N
            sd = get_fp32_state_dict_from_zero_checkpoint(str(ckpt_path))
        except Exception:
            # 方式 B: ckpt_path 是 output_dir, tag 在 latest 文件里
            sd = get_fp32_state_dict_from_zero_checkpoint(str(ckpt_path.parent), tag=ckpt_path.name)

        # 训练时学生用了 LatentSpaceForward 包装, 保存的 state_dict 里 key 会带
        # "student_model." 前缀;  教师没有包装。这里两个 case 都处理。
        new_sd = {}
        skipped = []
        for k, v in sd.items():
            if k.startswith("student_model."):
                new_sd[k[len("student_model."):]] = v
            elif k.startswith("decoder_lm_head."):
                # 这是冻结的 decoder LM head 副本, 推理时不需要重新加载（base 模型已自带）
                skipped.append(k)
            else:
                new_sd[k] = v
        print(f"[Infer] state_dict: 共 {len(sd)} 项, "
              f"加载 {len(new_sd)} 项, 跳过 decoder_lm_head {len(skipped)} 项", flush=True)
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"[Infer] missing keys ({len(missing)}): "
                  f"{missing[:3]}{' ...' if len(missing) > 3 else ''}", flush=True)
        if unexpected:
            print(f"[Infer] unexpected keys ({len(unexpected)}): "
                  f"{unexpected[:3]}{' ...' if len(unexpected) > 3 else ''}", flush=True)
    else:
        # HF 标准目录, 直接重新 from_pretrained 覆盖
        if (ckpt_path / "config.json").exists():
            print(f"[Infer] 检测到 HF 标准目录, 重新加载: {ckpt_path}", flush=True)
            model = _Cls.from_pretrained(str(ckpt_path), torch_dtype=dtype, trust_remote_code=True)
        else:
            raise RuntimeError(f"ckpt 既不是 ZeRO 也不是 HF 目录: {ckpt_path}")

    model = model.to(device).eval()
    return processor, model


def build_messages(role: str, sample: Dict[str, Any], for_generation: bool = True):
    """根据 role 构造 messages (推理时只放 user 部分, 由 generate 续写)。"""
    if role == "student":
        video_path = sample["video"]
        question = sample["question"]
        gt = sample.get("trajectory", "")
        user_content = [
            {"type": "video", "video": video_path},
            {"type": "text", "text": question},
        ]
    elif role == "teacher_r":
        question = sample["question"]
        gt = sample.get("trajectory", "")
        user_content = [{"type": "text", "text": question}]
    elif role == "teacher_p":
        video_path = sample["video"]
        question = sample.get("perception_question") or sample.get("question")
        gt = sample.get("result_text", "")
        user_content = [
            {"type": "video", "video": video_path},
            {"type": "text", "text": question},
        ]
    else:
        raise ValueError(role)
    msgs = [{"role": "user", "content": user_content}]
    return msgs, gt


def split_think_answer(text: str) -> Dict[str, str]:
    """从模型输出中切出 <think>...</think> 和 <answer>...</answer> 区域。"""
    out = {"think": "", "answer": "", "raw": text}
    m_think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    m_answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m_think:
        out["think"] = m_think.group(1).strip()
    if m_answer:
        out["answer"] = m_answer.group(1).strip()
    if not m_think and not m_answer:
        # 没找到标签, 整段都给 answer
        out["answer"] = text.strip()
    return out


def decode_think_with_init_lm_head(
    model, processor, prompt_inputs, generated_ids: torch.Tensor,
    base_model_path: str, dtype: torch.dtype, device: torch.device,
):
    """[student 专用] 用 base 模型的初始 LM head 重新解码 generated 序列在 think 区域的 hidden states.

    步骤：
      1) 拼接 prompt + generated 序列做一次完整 forward, 拿到所有位置的 hidden states
      2) 找 <think>...</think> 范围的 token 位置
      3) 用 base 初始 LM head 在这些位置 argmax 解码
      4) 与原始 generate 出的 token 做对比
    """
    print("[Infer][student] 用 base 初始 LM head 解码 think 区域 ...", flush=True)
    try:
        from transformers import Qwen3VLForConditionalGeneration as _Cls
    except ImportError:
        from transformers import AutoModelForVision2Seq as _Cls
    base_model = _Cls.from_pretrained(base_model_path, torch_dtype=dtype, trust_remote_code=True)
    init_lm_head = base_model.lm_head if hasattr(base_model, "lm_head") \
        else base_model.model.lm_head
    init_lm_head = init_lm_head.to(device).eval()
    del base_model

    # 拼接 prompt + generated 做完整 forward
    full_ids = generated_ids  # (1, L_full); generate 已经返回 prompt+output 拼好的
    fwd_kwargs = {
        "input_ids": full_ids.to(device),
        "output_hidden_states": True,
        "return_dict": True,
    }
    # 视觉特征也要透传
    for k in ["pixel_values_videos", "video_grid_thw", "mm_token_type_ids", "attention_mask"]:
        if k in prompt_inputs:
            v = prompt_inputs[k]
            if torch.is_tensor(v):
                # attention_mask / mm_token_type_ids 需要扩展到 full_ids 长度
                if k in ("attention_mask", "mm_token_type_ids"):
                    pad_len = full_ids.shape[1] - v.shape[1]
                    if pad_len > 0:
                        pad_val = 1 if k == "attention_mask" else 0
                        v = torch.cat([
                            v, torch.full((v.shape[0], pad_len), pad_val,
                                          dtype=v.dtype, device=v.device)
                        ], dim=1)
                fwd_kwargs[k] = v.to(device)

    with torch.no_grad():
        out = model(**fwd_kwargs)
    last_hidden = out.hidden_states[-1]  # (1, L, D)

    # 找 <think> / </think> token 边界
    tok = processor.tokenizer
    ids = full_ids[0].tolist()
    think_open = tok.encode("<think>", add_special_tokens=False)
    think_close = tok.encode("</think>", add_special_tokens=False)

    def find_subseq(seq, sub):
        n, m = len(seq), len(sub)
        for i in range(n - m + 1):
            if seq[i:i + m] == sub:
                return i
        return None

    s = find_subseq(ids, think_open)
    e = find_subseq(ids, think_close)
    if s is None or e is None or e <= s:
        print("[Infer][student] 未找到完整 <think>...</think> 区间, 跳过潜空间解码", flush=True)
        return None

    span_start = s + len(think_open)
    span_end = e
    span_hidden = last_hidden[0, span_start:span_end].to(dtype)  # (T, D)
    logits = init_lm_head(span_hidden)  # (T, V)
    decoded = logits.argmax(dim=-1).tolist()
    decoded_text = tok.decode(decoded, skip_special_tokens=False)

    # 同时给出原始 generate 出的 think 内容做对比
    raw_think_text = tok.decode(ids[span_start:span_end], skip_special_tokens=False)
    return {"raw_think": raw_think_text, "decoded_with_init_head": decoded_text}


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 GPU")
    device = torch.device("cuda:0")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    train_jsonl = args.train_jsonl or auto_train_jsonl(args.role)
    if not os.path.exists(train_jsonl):
        raise FileNotFoundError(f"训练 jsonl 不存在: {train_jsonl}")

    # 1) 加载模型
    processor, model = load_model_from_ckpt(args, dtype, device)

    # 2) 取 num_samples 条样本（默认随机抽样，避免取到 jsonl 头部同一类任务）
    import random as _random
    all_lines = []
    with open(train_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                all_lines.append(line)
    if args.shuffle:
        rng = _random.Random(args.seed)
        rng.shuffle(all_lines)
        print(f"[Infer] 数据已按 seed={args.seed} 随机打乱（共 {len(all_lines)} 条）", flush=True)
    else:
        print(f"[Infer] 不打乱, 按文件顺序取前 {args.num_samples} 条", flush=True)
    samples = [json.loads(l) for l in all_lines[:args.num_samples]]

    print(f"\n[Infer] role={args.role}  num_samples={len(samples)}\n")

    # 3) 逐条推理
    for idx, sample in enumerate(samples):
        msgs, gt = build_messages(args.role, sample, for_generation=True)

        # 提取 observe type 方便快速识别任务类型
        gt_str = str(gt) if gt else ""
        m_type = re.search(r'type="([^"]+)"', gt_str)
        observe_type = m_type.group(1) if m_type else "<no observe>"

        print("=" * 80)
        print(f"[Sample {idx + 1}/{len(samples)}]  task_type={observe_type}")
        if args.role != "teacher_r":
            print(f"  video:    {sample.get('video', '<n/a>')}")
        print(f"  question: {(sample.get('question') or sample.get('perception_question') or '')[:200]}")

        prompt_inputs = processor.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        moved = {}
        for k, v in prompt_inputs.items():
            if not torch.is_tensor(v):
                moved[k] = v
                continue
            if v.dtype.is_floating_point:
                moved[k] = v.to(device=device, dtype=dtype)
            else:
                moved[k] = v.to(device=device)
        prompt_inputs = moved

        with torch.no_grad():
            generated_ids = model.generate(
                **prompt_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        # 截掉 prompt 部分
        prompt_len = prompt_inputs["input_ids"].shape[1]
        gen_only = generated_ids[:, prompt_len:]
        output_text = processor.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(f"  GT (前 200 字):  {str(gt)[:200]}")
        print(f"  ---- 模型输出 (full) ----")
        print(output_text)

        # 切 think / answer
        parsed = split_think_answer(output_text)
        if parsed["think"] or parsed["answer"]:
            print(f"  ---- 解析 ----")
            print(f"  <think>:  {parsed['think'][:400]}")
            print(f"  <answer>: {parsed['answer'][:400]}")

        # 学生 + 用初始 LM head 解码 think 区域
        if args.role == "student" and args.use_init_lm_head_for_think:
            extra = decode_think_with_init_lm_head(
                model, processor, prompt_inputs, generated_ids,
                args.base_model, dtype, device,
            )
            if extra is not None:
                print(f"  ---- think 区潜空间用初始 LM head 解码 ----")
                print(f"  raw_think:                {extra['raw_think'][:400]}")
                print(f"  decoded_with_init_head:   {extra['decoded_with_init_head'][:400]}")

    print("\n[Infer] 全部完成 ✅")


if __name__ == "__main__":
    main()

