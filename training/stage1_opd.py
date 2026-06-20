"""Stage1-OPD 训练脚本（四模型协同蒸馏 · On-Policy）。

核心流程（GKD 风格 On-Policy Distillation）：
  1. Student 潜空间 forward → [h_1, ..., h_K]
  2. Translator 自回归解码每个 h_i → decoded_text_i（on-policy 轨迹）
  3. 解析 observe 指令 → 调用 pipeline 工具处理视频
  4. Teacher_R/P teacher-forcing 在学生轨迹上 → teacher_logits
  5. Translator teacher-forcing(h_i, decoded_text_i) → student_logits
  6. KL(student_logits, teacher_logits) → 梯度回传到 Student

四卡一组：
  GPU 0: Student（可训练）
  GPU 1: Translator（冻结）
  GPU 2: Teacher_R（冻结，纯文本）
  GPU 3: Teacher_P（冻结，视频）

8卡 = 2组并行，等效 batch_size × 2。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from configs import paths as cfg_paths
from training.latent_forward import LatentForwardEngine, LatentExitHead, THINK_START_ID, THINK_END_ID
from training.losses import compute_aux_loss, compute_answer_loss, compute_think_end_loss
from training.teacher_forward import TeacherRForward, TeacherPForward, compute_kl_loss
from utils.parser import parse_observe, ObserveQuery
from pipelines import run_pipeline, PIPELINES


# =============================================================================
# 常量
# =============================================================================

MAX_SEG_LEN = 256       # 单段 GT 上限 token 数
MAX_LATENT_STEPS = 32   # K 上限
MAX_DECODE_LEN = 256    # Translator 自回归解码最大长度


# =============================================================================
# 分布式工具
# =============================================================================

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return local_rank
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return 0

def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs, flush=True)


# =============================================================================
# 参数
# =============================================================================

@dataclass
class OPDArgs:
    train_jsonl: str = ""
    output_dir: str = ""
    student_ckpt: str = ""          # Student SFT checkpoint 路径
    teacher_r_ckpt: str = ""        # Teacher_R SFT checkpoint 路径
    teacher_p_ckpt: str = ""        # Teacher_P SFT checkpoint 路径
    model_path: str = ""            # 基础模型路径（用于 Translator 结构）
    num_train_epochs: int = 2
    gradient_accumulation_steps: int = 4
    student_lr: float = 5e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_length: int = 32768
    max_frames: int = 64
    fps: float = 1.0
    video_max_pixels: int = 0
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    bf16: bool = True
    max_seg_len: int = MAX_SEG_LEN
    max_latent_steps: int = MAX_LATENT_STEPS
    max_decode_len: int = MAX_DECODE_LEN
    # Loss 权重
    lambda_kl: float = 1.0
    lambda_ans: float = 1.0
    lambda_aux: float = 0.1
    lambda_think_end: float = 1.0
    # KL 温度
    temperature: float = 2.0
    # WandB
    wandb_project: str = ""
    wandb_run_name: str = ""
    wandb_mode: str = "online"
    # 其他
    num_workers: int = 2
    max_samples: int = 0
    teacher_r_max_length: int = 8192
    teacher_p_max_length: int = 32768


def parse_args() -> OPDArgs:
    p = argparse.ArgumentParser("Stage1-OPD")
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--student_ckpt", required=True)
    p.add_argument("--teacher_r_ckpt", required=True)
    p.add_argument("--teacher_p_ckpt", required=True)
    p.add_argument("--model_path", default="")
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--student_lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_length", type=int, default=32768)
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--video_max_pixels", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--max_seg_len", type=int, default=MAX_SEG_LEN)
    p.add_argument("--max_latent_steps", type=int, default=MAX_LATENT_STEPS)
    p.add_argument("--max_decode_len", type=int, default=MAX_DECODE_LEN)
    p.add_argument("--lambda_kl", type=float, default=1.0)
    p.add_argument("--lambda_ans", type=float, default=1.0)
    p.add_argument("--lambda_aux", type=float, default=0.1)
    p.add_argument("--lambda_think_end", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--wandb_project", default="")
    p.add_argument("--wandb_run_name", default="")
    p.add_argument("--wandb_mode", default="online")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--teacher_r_max_length", type=int, default=8192)
    p.add_argument("--teacher_p_max_length", type=int, default=32768)
    a = p.parse_args()
    return OPDArgs(**vars(a))


# =============================================================================
# Dataset（复用 v5 的数据格式）
# =============================================================================

class OPDDataset(Dataset):
    """读取 JSONL 数据，提供 video_path, question, trajectory。"""

    def __init__(self, jsonl_path: str, max_samples: int = 0, shuffle_seed: int = 42):
        self.jsonl_path = str(jsonl_path)
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(self.jsonl_path)

        self.offsets: List[int] = []
        with open(self.jsonl_path, "rb") as f:
            offset = 0
            for line in f:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)
        if not self.offsets:
            raise RuntimeError(f"empty jsonl: {self.jsonl_path}")

        total = len(self.offsets)
        if max_samples and 0 < max_samples < total:
            import random as _r
            rng = _r.Random(int(shuffle_seed))
            rng.shuffle(self.offsets)
            self.offsets = self.offsets[:int(max_samples)]
            rank0_print(f"[Dataset] subsample {total} -> {len(self.offsets)} (seed={shuffle_seed})")
        rank0_print(f"[Dataset] {len(self.offsets)} samples: {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def _read(self, idx: int) -> Dict[str, Any]:
        with open(self.jsonl_path, "rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
        return json.loads(line)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self._read(idx)
        v, q, t = s.get("video"), s.get("question"), s.get("trajectory")
        if not v or not q or not t:
            raise ValueError(f"sample missing fields idx={idx}")
        if not os.path.exists(v):
            raise FileNotFoundError(v)
        return {"video_path": v, "question": q, "trajectory": t}


# =============================================================================
# Collator（复用 v5 的 Phase1 构造逻辑）
# =============================================================================

class OPDCollator:
    """构造 Phase1 输入 + GT segments + answer_ids。"""

    def __init__(self, processor, max_length: int = 32768,
                 max_seg_len: int = MAX_SEG_LEN, max_latent_steps: int = MAX_LATENT_STEPS):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.max_seg_len = max_seg_len
        self.max_latent_steps = max_latent_steps

    def __call__(self, batch):
        if not batch:
            return None
        item = batch[0]
        try:
            return self._process_one(item)
        except Exception as e:
            rank0_print(f"[Collator WARN] {e}")
            return None

    def _process_one(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        video_path = item["video_path"]
        question = item["question"]
        trajectory = item["trajectory"]

        # 提取 <think>...</think> 内容和 answer 部分
        think_start = trajectory.find("<think>")
        think_end = trajectory.find("</think>")
        if think_start < 0 or think_end < 0:
            return None
        think_content = trajectory[think_start + len("<think>"):think_end]
        answer_part = trajectory[think_end:]  # 含 </think>

        # 切分 GT segments
        from training.stage1_sft_v5 import split_trajectory_into_segments
        segments = split_trajectory_into_segments(think_content)
        if not segments:
            return None

        K = len(segments)
        if K > self.max_latent_steps:
            K = self.max_latent_steps
            segments = segments[:K]

        # Tokenize GT segments
        gt_segment_ids = []
        gt_segment_texts = []
        gt_segment_types = []
        for seg_text, seg_type in segments:
            ids = self.tokenizer.encode(seg_text, add_special_tokens=False)
            if len(ids) > self.max_seg_len:
                return None  # 段太长，丢弃
            gt_segment_ids.append(ids)
            gt_segment_texts.append(seg_text)
            gt_segment_types.append(seg_type)

        # 构造 Phase 1 输入
        from training.latent_sft_helpers import load_system_prompt
        sys_prompt = load_system_prompt("student")
        msgs = []
        if sys_prompt:
            msgs.append({"role": "system", "content": [{"type": "text", "text": sys_prompt}]})
        msgs.append({"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": question},
        ]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "<think>"}]})

        inputs = self.processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt"
        )

        phase1_input_ids = inputs["input_ids"].squeeze(0)
        phase1_attention_mask = inputs["attention_mask"].squeeze(0) if "attention_mask" in inputs \
            else torch.ones_like(phase1_input_ids)

        # Answer ids
        answer_ids = self.tokenizer.encode(answer_part, add_special_tokens=False)
        answer_ids = torch.tensor(answer_ids, dtype=torch.long)

        result = {
            "phase1_input_ids": phase1_input_ids,
            "phase1_attention_mask": phase1_attention_mask,
            "gt_segment_ids": gt_segment_ids,
            "gt_segment_texts": gt_segment_texts,
            "gt_segment_types": gt_segment_types,
            "answer_ids": answer_ids,
            "num_latent_steps": K,
            "video_path": video_path,
            "question": question,
        }

        # 视觉特征（对齐 v5 的 shape 处理）
        # pixel_values_videos: [1, T*H*W, C] → [T*H*W, C]（去掉 batch 维）
        # video_grid_thw:      [N, 3]（processor 输出本身无 batch 维，直接存）
        # mm_token_type_ids:   [1, L] → [L]（去掉 batch 维）
        if "pixel_values_videos" in inputs:
            result["pixel_values_videos"] = inputs["pixel_values_videos"][0]
        if "video_grid_thw" in inputs:
            result["video_grid_thw"] = inputs["video_grid_thw"]
        if "mm_token_type_ids" in inputs:
            result["mm_token_type_ids"] = inputs["mm_token_type_ids"][0]

        return result


# =============================================================================
# Translator 自回归解码（On-Policy 核心）
# =============================================================================

def translator_decode(
    translator,
    hidden: torch.Tensor,
    tokenizer,
    max_len: int = MAX_DECODE_LEN,
) -> str:
    """用 Translator 自回归解码一个 latent hidden 为文字（on-policy）。

    Args:
        translator: Translator 模型
        hidden: [1, H] latent hidden state（有梯度，但解码过程 no_grad）
        tokenizer: tokenizer
        max_len: 最大生成长度

    Returns:
        decoded_text: 解码出的文字
    """
    device = hidden.device
    embed_tokens = translator._get_embed_tokens()

    # 第一步：latent hidden 当 input embedding
    input_embeds = hidden.unsqueeze(1)  # [1, 1, H]

    with torch.no_grad():
        outputs = translator.model(
            inputs_embeds=input_embeds,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_id = logits.argmax(dim=-1).item()

        generated_ids = []
        eos_ids = {151645, 151643, 0}  # <|im_end|>, <|endoftext|>, pad

        if next_id in eos_ids:
            return ""

        generated_ids.append(next_id)

        # 后续步骤：KV cache 逐 token 自回归
        attention_mask = torch.ones(1, 2, dtype=torch.long, device=device)
        for step in range(max_len - 1):
            new_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            new_embed = embed_tokens(new_token)

            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, dtype=torch.long, device=device)
            ], dim=1)

            outputs = translator.model(
                inputs_embeds=new_embed,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_id = logits.argmax(dim=-1).item()

            if next_id in eos_ids:
                break
            generated_ids.append(next_id)

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text


# =============================================================================
# Translator teacher-forcing（获取 student logits，有梯度）
# =============================================================================

def translator_teacher_forcing(
    translator,
    hidden: torch.Tensor,
    segment_token_ids: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Translator teacher-forcing forward，返回 logits（有梯度回传到 hidden）。

    Args:
        translator: Translator 模型（冻结，但 hidden 有梯度）
        hidden: [1, H] latent hidden state（有梯度）
        segment_token_ids: 段文本的 token ids
        device: Translator 所在设备

    Returns:
        logits: [1, seg_len, vocab_size]（梯度通过 hidden 回传到 Student）
    """
    seg_ids = torch.tensor([segment_token_ids], dtype=torch.long, device=device)
    embed_tokens = translator._get_embed_tokens()

    # 构造 input embeddings: [hidden] + [embed(seg_ids[:-1])]
    latent_embed = hidden.unsqueeze(1)  # [1, 1, H]
    if seg_ids.shape[1] > 1:
        gt_embeds = embed_tokens(seg_ids[:, :-1])  # [1, N-1, H]
        input_embeds = torch.cat([latent_embed, gt_embeds], dim=1)  # [1, N, H]
    else:
        input_embeds = latent_embed  # [1, 1, H]

    outputs = translator.model(
        inputs_embeds=input_embeds,
        use_cache=False,
        return_dict=True,
    )

    logits = outputs.logits  # [1, N, V]
    return logits


# =============================================================================
# 段类型判定
# =============================================================================

def classify_decoded_segment(text: str) -> str:
    """判断 Translator 解码出的段文本类型。

    Returns:
        "RESULT" | "OBSERVE" | "REASON"
    """
    s = text.strip()
    if s.startswith("<result>") or "<result>" in s:
        return "RESULT"
    if "<observe" in s:
        return "OBSERVE"
    return "REASON"


# =============================================================================
# 单样本 OPD 训练
# =============================================================================

def train_one_sample_opd(
    sample: Dict[str, Any],
    latent_engine: LatentForwardEngine,
    translator,
    teacher_r: TeacherRForward,
    teacher_p: TeacherPForward,
    tokenizer,
    args: OPDArgs,
    student_device: torch.device,
    translator_device: torch.device,
    teacher_r_device: torch.device,
    teacher_p_device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """对一个样本执行完整的 OPD forward。

    On-Policy 流程：
      1. Student Phase1 + Phase2 → latent hiddens
      2. Translator 自回归解码每个 h_i → decoded_text（on-policy）
      3. 根据 decoded_text 类型，调用 Teacher_R 或 Teacher_P
      4. Translator teacher-forcing → student_logits
      5. KL(student_logits, teacher_logits)

    Args:
        sample: Collator 输出
        latent_engine: Student 的 LatentForwardEngine
        translator: Translator 模型（冻结）
        teacher_r: TeacherRForward 封装
        teacher_p: TeacherPForward 封装
        tokenizer: tokenizer
        args: OPD 参数
        student_device: Student GPU
        translator_device: Translator GPU
        teacher_r_device: Teacher_R GPU
        teacher_p_device: Teacher_P GPU

    Returns:
        loss dict 或 None（样本失败时）
    """
    # 解包样本
    phase1_input_ids = sample["phase1_input_ids"].unsqueeze(0).to(student_device)
    phase1_attention_mask = sample["phase1_attention_mask"].unsqueeze(0).to(student_device)
    gt_segment_ids = sample["gt_segment_ids"]
    gt_segment_types = sample["gt_segment_types"]
    answer_ids = sample["answer_ids"].unsqueeze(0).to(student_device)
    K = sample["num_latent_steps"]
    video_path = sample["video_path"]
    question = sample["question"]

    pixel_values_videos = sample.get("pixel_values_videos")
    video_grid_thw = sample.get("video_grid_thw")
    mm_token_type_ids = sample.get("mm_token_type_ids")
    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.unsqueeze(0).to(student_device)
    if video_grid_thw is not None:
        video_grid_thw = video_grid_thw.to(student_device)
    if mm_token_type_ids is not None:
        mm_token_type_ids = mm_token_type_ids.unsqueeze(0).to(student_device)

    # =========================================================================
    # Phase 1: Student forward 到 <think>
    # =========================================================================
    h_0, past_key_values = latent_engine.phase1_forward_efficient(
        input_ids=phase1_input_ids,
        attention_mask=phase1_attention_mask,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
    )

    # =========================================================================
    # Phase 2: 串行 K 步 latent forward
    # =========================================================================
    latent_hiddens, past_key_values = latent_engine.phase2_serial_forward(
        h_0=h_0,
        past_key_values=past_key_values,
        num_latent_steps=K,
    )

    # =========================================================================
    # Phase 3: Answer forward
    # =========================================================================
    answer_logits = latent_engine.phase3_forward(
        answer_ids=answer_ids,
        past_key_values=past_key_values,
    )

    # =========================================================================
    # On-Policy: Translator 解码 + 教师 KL 蒸馏
    # =========================================================================
    kl_losses = []
    context_for_teacher_r = ""  # Teacher_R 的累积上下文
    pending_pipeline_result = None  # 上一个 observe 解析得到的 pipeline 结果
    failed = False

    for i in range(K):
        h_i = latent_hiddens[i]  # [1, H]，有梯度

        # --- Step A: Translator 自回归解码（on-policy，no_grad） ---
        h_i_for_decode = h_i.detach().to(translator_device)
        decoded_text = translator_decode(
            translator, h_i_for_decode, tokenizer, max_len=args.max_decode_len
        )

        if not decoded_text.strip():
            # 解码为空，用 GT 兜底（避免浪费整条样本）
            decoded_text = sample["gt_segment_texts"][i]

        # --- Step B: 判断段类型 ---
        seg_type = classify_decoded_segment(decoded_text)

        # --- Step C: 获取 teacher logits ---
        teacher_logits = None

        if seg_type == "OBSERVE":
            # 解析 observe 指令
            try:
                observe_queries = parse_observe(decoded_text)
            except Exception:
                observe_queries = []

            if not observe_queries:
                # observe 格式无法解析 → 本样本失败
                failed = True
                break

            # 检查 observe type 是否支持
            for obs_q in observe_queries:
                if obs_q.type not in PIPELINES:
                    failed = True
                    break
            if failed:
                break

            # 调用 pipeline 工具处理视频
            try:
                pipeline_result = run_pipeline(observe_queries[0], video_path)
                pending_pipeline_result = pipeline_result
            except Exception:
                failed = True
                break

            # OBSERVE 段走 Teacher_R
            teacher_logits = teacher_r.get_segment_logits(
                question=question,
                context=context_for_teacher_r,
                segment_text=decoded_text,
            )
            # 累积上下文
            context_for_teacher_r += decoded_text

        elif seg_type == "RESULT":
            # RESULT 段走 Teacher_P
            if pending_pipeline_result is not None:
                teacher_logits = teacher_p.get_segment_logits(
                    video_path=pending_pipeline_result["video"],
                    perception_question=pending_pipeline_result["perception_question"],
                    result_text=decoded_text,
                )
                pending_pipeline_result = None  # 消费掉
            else:
                # 没有前置 observe（可能学生跳过了 observe 直接输出 result）
                # 用原始视频 + 通用问题兜底
                teacher_logits = teacher_p.get_segment_logits(
                    video_path=video_path,
                    perception_question=question,
                    result_text=decoded_text,
                )
            # 累积上下文（Teacher_R 需要看到 result 内容）
            context_for_teacher_r += decoded_text

        else:  # REASON
            # REASON 段走 Teacher_R
            teacher_logits = teacher_r.get_segment_logits(
                question=question,
                context=context_for_teacher_r,
                segment_text=decoded_text,
            )
            # 累积上下文
            context_for_teacher_r += decoded_text

        # --- Step D: Translator teacher-forcing → student logits（有梯度） ---
        if teacher_logits is not None:
            # 用 decoded_text 的 token ids 做 teacher-forcing
            decoded_ids = tokenizer.encode(decoded_text, add_special_tokens=False)
            if not decoded_ids:
                continue

            h_i_for_tf = h_i.to(translator_device)  # 保留梯度！
            student_logits = translator_teacher_forcing(
                translator, h_i_for_tf, decoded_ids, translator_device
            )

            # 对齐到同一设备计算 KL
            teacher_logits_aligned = teacher_logits.to(translator_device)
            kl = compute_kl_loss(student_logits, teacher_logits_aligned, args.temperature)
            kl_losses.append(kl)

    # =========================================================================
    # 如果失败，返回 None
    # =========================================================================
    if failed:
        return None

    # =========================================================================
    # 计算总 Loss
    # =========================================================================
    # L_kl: 所有段的平均 KL
    if kl_losses:
        l_kl = torch.stack(kl_losses).mean()
        # 移回 student_device
        l_kl = l_kl.to(student_device)
    else:
        l_kl = torch.tensor(0.0, device=student_device, requires_grad=True)

    # L_ans: answer CE
    l_ans = compute_answer_loss(answer_logits, answer_ids)

    # L_aux: 中间 latent 不退出
    l_aux = compute_aux_loss(latent_hiddens, latent_engine.exit_head)

    # L_think_end: 最后 latent 退出
    l_think_end = compute_think_end_loss(latent_hiddens, latent_engine.exit_head)

    # 总 loss
    total = (args.lambda_kl * l_kl + args.lambda_ans * l_ans
             + args.lambda_aux * l_aux + args.lambda_think_end * l_think_end)

    return {
        "l_kl": l_kl,
        "l_ans": l_ans,
        "l_aux": l_aux,
        "l_think_end": l_think_end,
        "total": total,
    }


# =============================================================================
# 模型加载
# =============================================================================

def load_student_from_ckpt(ckpt_dir: str, dtype: torch.dtype):
    """从 checkpoint 加载 Student 模型。"""
    from utils.qwen3vl_patch import apply_qwen3vl_patches
    apply_qwen3vl_patches()
    from transformers import AutoProcessor

    student_dir = os.path.join(ckpt_dir, "student")
    if not os.path.exists(student_dir):
        if os.path.exists(os.path.join(ckpt_dir, "config.json")):
            student_dir = ckpt_dir
        else:
            raise FileNotFoundError(f"Student checkpoint 不存在: {student_dir}")

    processor = AutoProcessor.from_pretrained(student_dir, trust_remote_code=True)
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            student_dir, torch_dtype=dtype, trust_remote_code=True)
    except Exception:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            student_dir, torch_dtype=dtype, trust_remote_code=True)
    return processor, model


def load_teacher_model(ckpt_dir: str, dtype: torch.dtype):
    """加载教师模型（冻结）。"""
    from utils.qwen3vl_patch import apply_qwen3vl_patches
    apply_qwen3vl_patches()
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as MC
    except:
        from transformers import AutoModelForVision2Seq as MC

    processor = AutoProcessor.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = MC.from_pretrained(ckpt_dir, torch_dtype=dtype, trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


# =============================================================================
# WandB
# =============================================================================

def maybe_init_wandb(args: OPDArgs) -> bool:
    if not is_main_process():
        return False
    if not args.wandb_project:
        return False
    try:
        import wandb
    except ImportError:
        return False
    name = args.wandb_run_name or f"opd-{int(time.time())}"
    wandb.init(project=args.wandb_project, name=name, mode=args.wandb_mode, config=vars(args))
    return True


def wandb_log(payload: dict, step: int):
    if not is_main_process():
        return
    try:
        import wandb
        if wandb.run is None:
            return
        wandb.log(payload, step=step)
    except Exception:
        pass


# =============================================================================
# Checkpoint 保存
# =============================================================================

def save_checkpoint(args: OPDArgs, student_model, processor, step: int, exit_head=None):
    """保存 Student + exit_head 的 checkpoint。"""
    if not is_main_process():
        return
    save_dir = Path(args.output_dir) / f"checkpoint-{step}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存 Student
    student_dir = save_dir / "student"
    student_dir.mkdir(exist_ok=True)
    target = student_model.module if hasattr(student_model, "module") else student_model
    target.save_pretrained(str(student_dir), safe_serialization=True)
    processor.save_pretrained(str(student_dir))

    # 保存 exit_head
    if exit_head is not None:
        torch.save(exit_head.state_dict(), str(save_dir / "exit_head.pt"))

    rank0_print(f"[Save] checkpoint-{step} 已保存到 {save_dir}")

    # 清理旧 checkpoint
    ckpts = sorted(
        [p for p in Path(args.output_dir).glob("checkpoint-*")
         if (p / "student").exists() and p.name.split("-")[-1].isdigit()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    while len(ckpts) > args.save_total_limit:
        old = ckpts.pop(0)
        import shutil
        shutil.rmtree(str(old), ignore_errors=True)


# =============================================================================
# 主训练流程
# =============================================================================

def main():
    args = parse_args()
    local_rank = setup_distributed()
    torch.manual_seed(args.seed)

    cfg_paths.ensure_dirs()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    maybe_init_wandb(args)

    dtype = torch.bfloat16 if args.bf16 else torch.float32

    # =========================================================================
    # 确定设备分配（4卡一组）
    # local_rank 0 → Student(GPU0), Translator(GPU1), Teacher_R(GPU2), Teacher_P(GPU3)
    # local_rank 1 → Student(GPU4), Translator(GPU5), Teacher_R(GPU6), Teacher_P(GPU7)
    # =========================================================================
    # 每个进程独占一组 4 卡
    group_id = local_rank  # 0 或 1
    base_gpu = group_id * 4
    student_device = torch.device(f"cuda:{base_gpu}")
    translator_device = torch.device(f"cuda:{base_gpu + 1}")
    teacher_r_device = torch.device(f"cuda:{base_gpu + 2}")
    teacher_p_device = torch.device(f"cuda:{base_gpu + 3}")

    rank0_print(f"[Init] world_size={get_world_size()} rank={get_rank()} local_rank={local_rank}")
    rank0_print(f"[Init] Group {group_id}: Student=GPU{base_gpu} Translator=GPU{base_gpu+1} "
                f"Teacher_R=GPU{base_gpu+2} Teacher_P=GPU{base_gpu+3}")

    # =========================================================================
    # 加载 Student
    # =========================================================================
    rank0_print("[Init] 加载 Student 模型...")
    if not args.model_path:
        args.model_path = str(cfg_paths.QWEN3_VL_4B_PATH)

    processor, student_model = load_student_from_ckpt(args.student_ckpt, dtype)
    student_model = student_model.to(student_device)
    student_model.train()
    for param in student_model.parameters():
        param.requires_grad_(True)

    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = args.fps
        processor.video_processor.max_frames = args.max_frames
        if args.video_max_pixels > 0:
            processor.video_processor.size = {
                "shortest_edge": min(args.video_max_pixels,
                                     processor.video_processor.size.get("shortest_edge", 131072)),
                "longest_edge": args.video_max_pixels,
            }

    # 加载 exit_head
    hidden_size = student_model.config.text_config.hidden_size
    exit_head = LatentExitHead(hidden_size)
    exit_head_path = os.path.join(args.student_ckpt, "exit_head.pt")
    if os.path.exists(exit_head_path):
        exit_head.load_state_dict(torch.load(exit_head_path, map_location="cpu"))
        rank0_print(f"  exit_head 已加载: {exit_head_path}")
    exit_head = exit_head.to(device=student_device, dtype=dtype)
    exit_head.train()
    for p in exit_head.parameters():
        p.requires_grad_(True)

    # 构造 LatentForwardEngine
    latent_engine = LatentForwardEngine(student_model, exit_head=exit_head)

    # =========================================================================
    # 加载 Translator（冻结）
    # =========================================================================
    rank0_print("[Init] 加载 Translator 模型（冻结）...")
    from training.translator_v5 import Translator
    translator = Translator(args.model_path, dtype=dtype)

    # 尝试加载 SFT 训练后的 Translator 权重
    translator_ckpt = os.path.join(args.student_ckpt, "translator", "translator_state_dict.pt")
    if os.path.exists(translator_ckpt):
        translator.load_state_dict(torch.load(translator_ckpt, map_location="cpu"))
        rank0_print(f"  Translator 从 checkpoint 加载: {translator_ckpt}")

    translator = translator.to(translator_device)
    translator.eval()
    for p in translator.parameters():
        p.requires_grad_(False)

    # =========================================================================
    # 加载 Teacher_R（冻结）
    # =========================================================================
    rank0_print("[Init] 加载 Teacher_R 模型...")
    teacher_r_proc, teacher_r_model = load_teacher_model(args.teacher_r_ckpt, dtype)
    teacher_r_model = teacher_r_model.to(teacher_r_device)
    teacher_r_fwd = TeacherRForward(
        model=teacher_r_model,
        tokenizer=teacher_r_proc.tokenizer,
        device=teacher_r_device,
        max_length=args.teacher_r_max_length,
    )

    # =========================================================================
    # 加载 Teacher_P（冻结）
    # =========================================================================
    rank0_print("[Init] 加载 Teacher_P 模型...")
    teacher_p_proc, teacher_p_model = load_teacher_model(args.teacher_p_ckpt, dtype)
    teacher_p_model = teacher_p_model.to(teacher_p_device)
    if hasattr(teacher_p_proc, "video_processor"):
        teacher_p_proc.video_processor.fps = args.fps
        teacher_p_proc.video_processor.max_frames = args.max_frames
    teacher_p_fwd = TeacherPForward(
        model=teacher_p_model,
        processor=teacher_p_proc,
        device=teacher_p_device,
        max_length=args.teacher_p_max_length,
        fps=args.fps,
    )

    # =========================================================================
    # 优化器
    # =========================================================================
    trainable_params = list(student_model.parameters()) + list(exit_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.student_lr, weight_decay=args.weight_decay
    )

    # =========================================================================
    # 数据
    # =========================================================================
    dataset = OPDDataset(args.train_jsonl, args.max_samples, args.seed)
    collator = OPDCollator(processor, args.max_length, args.max_seg_len, args.max_latent_steps)
    sampler = DistributedSampler(dataset, shuffle=True) if is_dist() else None
    dataloader = DataLoader(
        dataset, batch_size=1, sampler=sampler, collate_fn=collator,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # 学习率调度
    total_steps = len(dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: s / warmup_steps if s < warmup_steps
        else 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps)))
    )

    rank0_print(f"[Init] total_steps={total_steps} warmup={warmup_steps} "
                f"lr={args.student_lr} grad_accum={args.gradient_accumulation_steps}")
    rank0_print(f"[Init] lambda: kl={args.lambda_kl} ans={args.lambda_ans} "
                f"aux={args.lambda_aux} think_end={args.lambda_think_end} τ={args.temperature}")

    # =========================================================================
    # 训练循环
    # =========================================================================
    global_step = 0
    accum_loss = 0.0
    accum_kl = 0.0
    accum_ans = 0.0
    failed_count = 0
    total_count = 0

    for epoch in range(args.num_train_epochs):
        if sampler:
            sampler.set_epoch(epoch)
        rank0_print(f"\n[Epoch {epoch + 1}/{args.num_train_epochs}]")
        optimizer.zero_grad()

        pbar = tqdm(dataloader, disable=not is_main_process(), desc=f"Epoch {epoch+1}")
        for step, sample in enumerate(pbar):
            if sample is None:
                continue

            total_count += 1

            try:
                losses = train_one_sample_opd(
                    sample=sample,
                    latent_engine=latent_engine,
                    translator=translator,
                    teacher_r=teacher_r_fwd,
                    teacher_p=teacher_p_fwd,
                    tokenizer=processor.tokenizer,
                    args=args,
                    student_device=student_device,
                    translator_device=translator_device,
                    teacher_r_device=teacher_r_device,
                    teacher_p_device=teacher_p_device,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    warnings.warn(f"[OOM] step={step}")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise
            except Exception as e:
                rank0_print(f"[ERROR] step={step}: {e}")
                traceback.print_exc()
                continue

            if losses is None:
                # 样本失败（observe 解析失败等）
                failed_count += 1
                continue

            # 梯度累积
            loss = losses["total"] / args.gradient_accumulation_steps
            loss.backward()

            accum_loss += losses["total"].item()
            accum_kl += losses["l_kl"].item()
            accum_ans += losses["l_ans"].item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0 and is_main_process():
                    avg_loss = accum_loss / args.gradient_accumulation_steps
                    avg_kl = accum_kl / args.gradient_accumulation_steps
                    avg_ans = accum_ans / args.gradient_accumulation_steps
                    fail_rate = failed_count / max(1, total_count)
                    lr_now = scheduler.get_last_lr()[0]

                    rank0_print(
                        f"  step={global_step} loss={avg_loss:.4f} "
                        f"kl={avg_kl:.4f} ans={avg_ans:.4f} "
                        f"fail_rate={fail_rate:.2%} lr={lr_now:.2e}"
                    )
                    wandb_log({
                        "train/loss": avg_loss,
                        "train/kl": avg_kl,
                        "train/ans": avg_ans,
                        "train/fail_rate": fail_rate,
                        "train/lr": lr_now,
                    }, step=global_step)

                accum_loss = 0.0
                accum_kl = 0.0
                accum_ans = 0.0

                # 保存 checkpoint
                if args.save_steps and global_step % args.save_steps == 0:
                    save_checkpoint(args, student_model, processor, global_step, exit_head)

    # 最终保存
    save_checkpoint(args, student_model, processor, global_step, exit_head)

    # 统计
    rank0_print(f"\n[Done] OPD 训练完成。")
    rank0_print(f"  总样本: {total_count}, 失败: {failed_count} ({failed_count/max(1,total_count):.1%})")
    rank0_print(f"  总步数: {global_step}")

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
