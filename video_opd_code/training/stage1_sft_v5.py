"""Stage1-SFT 潜空间训练脚本（v5 最终版 · Coconut/VaLR 串行 latent）。

核心做法：
  Student 在 <think>...</think> 区间内，每步取 last hidden state 直接当下一步的 input embedding。
  每个 hidden state 代表一整段推理/感知内容。Translator 在训练时把 hidden 还原为文字，用于计算 CE loss。

三个 Loss：
  L_trans: Translator 还原 GT 段文字的 CE loss（梯度穿过 h_i → Student）
  L_ans:   Phase 3 中 answer 部分的标准 next-token CE loss
  L_aux:   中间 latent 位置压低 </think> 概率的辅助 loss

总 loss = 1.0 * L_trans + 1.0 * L_ans + 0.1 * L_aux
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
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
from training.losses import compute_aux_loss, compute_answer_loss, compute_total_loss, compute_think_end_loss


# =============================================================================
# 常量
# =============================================================================

MAX_SEG_LEN = 256  # 单段 GT 上限 token 数，超出报错中断
MAX_LATENT_STEPS = 32  # K 上限


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


def cleanup_distributed():
    if not is_dist():
        return
    try:
        dist.barrier()
    except Exception:
        pass
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    try:
        dist.destroy_process_group()
    except Exception:
        pass


# =============================================================================
# GT Trajectory 段切分
# =============================================================================

# 匹配 <result>...</result> 块
_RE_RESULT_BLOCK = re.compile(r"<result>(.*?)</result>", re.DOTALL)


def split_trajectory_into_segments(trajectory: str) -> List[Tuple[str, str]]:
    """将 GT trajectory 切分为交错的推理段和感知段。

    切分规则：
      - <result>...</result> 独立成一段 = 感知段 (type="perception")
      - 两个 <result> 之间的所有内容合并为一段 = 推理段 (type="reasoning")
      - 交错排列：推理、感知、推理、感知...推理

    Args:
        trajectory: <think>...</think> 之间的文本内容（不含 <think></think> 标签本身）

    Returns:
        segments: [(text, type), ...] 其中 type 为 "reasoning" 或 "perception"
    """
    segments: List[Tuple[str, str]] = []

    # 找到所有 <result>...</result> 的位置
    result_matches = list(_RE_RESULT_BLOCK.finditer(trajectory))

    if not result_matches:
        # 没有 <result> 块，整个 trajectory 就是一个推理段
        text = trajectory.strip()
        if text:
            segments.append((text, "reasoning"))
        return segments

    # 第一个 <result> 之前的内容 = 推理段
    first_start = result_matches[0].start()
    pre_text = trajectory[:first_start].strip()
    if pre_text:
        segments.append((pre_text, "reasoning"))

    for i, match in enumerate(result_matches):
        # <result>...</result> 内容 = 感知段
        result_content = f"<result>{match.group(1)}</result>"
        segments.append((result_content, "perception"))

        # 当前 </result> 到下一个 <result> 之间 = 推理段
        end_pos = match.end()
        if i + 1 < len(result_matches):
            next_start = result_matches[i + 1].start()
            between_text = trajectory[end_pos:next_start].strip()
        else:
            # 最后一个 </result> 之后的内容
            between_text = trajectory[end_pos:].strip()

        if between_text:
            segments.append((between_text, "reasoning"))

    return segments


def extract_think_content(trajectory: str) -> Tuple[str, str]:
    """从 trajectory 中提取 <think>...</think> 内容和 answer 部分。

    Args:
        trajectory: 完整的 trajectory 文本

    Returns:
        think_content: <think> 和 </think> 之间的内容
        answer_part: </think> 之后的所有内容（含 </think> 本身）
    """
    think_start = trajectory.find("<think>")
    think_end = trajectory.find("</think>")

    if think_start < 0 or think_end < 0:
        raise ValueError(f"trajectory 缺少 <think>...</think> 标签: {trajectory[:100]}")

    think_content = trajectory[think_start + len("<think>"):think_end]
    answer_part = trajectory[think_end:]  # 包含 </think>

    return think_content, answer_part


# =============================================================================
# Dataset
# =============================================================================

class Stage1SFTDatasetV5(Dataset):
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
# 模型加载
# =============================================================================

def load_processor_and_model(model_path: str, dtype: torch.dtype):
    """加载 Qwen3-VL 模型和 processor。"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    from utils.qwen3vl_patch import apply_qwen3vl_patches
    apply_qwen3vl_patches()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True)
    except Exception:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True)
    return processor, model


# =============================================================================
# Collator：构造训练所需的所有输入
# =============================================================================

class LatentSFTCollatorV5:
    """为 v5 串行 latent 训练构造 batch。

    每个样本产出：
      - phase1_input_ids: [system, user, video, <think>] 的 token ids
      - phase1_attention_mask: 对应的 attention mask
      - pixel_values_videos / video_grid_thw: 视频特征
      - gt_segments: K 段 GT 文字的 token ids（用于 Translator）
      - answer_ids: [</think>, <answer>, ..., </answer>, <|im_end|>] 的 token ids
      - num_latent_steps: K
    """

    def __init__(self, processor, max_length: int = 32768,
                 max_seg_len: int = MAX_SEG_LEN, max_latent_steps: int = MAX_LATENT_STEPS):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.max_seg_len = max_seg_len
        self.max_latent_steps = max_latent_steps

    def _build_phase1_messages(self, video_path: str, question: str) -> list:
        """构造 Phase 1 的 messages（到 <think> 为止）。"""
        from training.latent_sft_helpers import load_system_prompt
        sys_prompt = load_system_prompt("student")

        msgs = []
        if sys_prompt:
            msgs.append({"role": "system",
                         "content": [{"type": "text", "text": sys_prompt}]})
        msgs.append({"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": question},
        ]})
        # assistant 只到 <think>（不含 trajectory 内容）
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": "<think>"}]})
        return msgs

    def _tokenize_segment(self, text: str) -> List[int]:
        """将一段 GT 文字 tokenize 为 token ids。"""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return ids

    def __call__(self, batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """处理一个 batch（v5 设计中 batch_size 实际为 1，因为串行 latent 不好 batch）。

        Returns:
            dict 或 None（样本无效时返回 None）
        """
        results = []

        for sample in batch:
            video_path = sample["video_path"]
            question = sample["question"]
            trajectory = sample["trajectory"]

            try:
                # 1. 提取 think 内容和 answer 部分
                think_content, answer_part = extract_think_content(trajectory)

                # 2. 切分 GT 段
                segments = split_trajectory_into_segments(think_content)
                if not segments:
                    warnings.warn(f"[Collator] 跳过: trajectory 切分为空段: {trajectory[:80]}")
                    continue

                K = len(segments)
                if K > self.max_latent_steps:
                    warnings.warn(
                        f"[Collator] 跳过: 段数 K={K} > max_latent_steps={self.max_latent_steps}")
                    continue

                # 3. Tokenize 每段 GT 文字
                gt_segment_ids = []
                seg_too_long = False
                for seg_text, seg_type in segments:
                    ids = self._tokenize_segment(seg_text)
                    if len(ids) > self.max_seg_len:
                        raise RuntimeError(
                            f"段 token 数 ({len(ids)}) > max_seg_len ({self.max_seg_len})！\n"
                            f"段内容: {seg_text[:100]}...")
                    gt_segment_ids.append(ids)

                # 4. 构造 Phase 1 输入（到 <think> 为止）
                phase1_msgs = self._build_phase1_messages(video_path, question)

                phase1_inputs = None
                last_err = None
                for attempt in range(5):
                    try:
                        phase1_inputs = self.processor.apply_chat_template(
                            phase1_msgs,
                            tokenize=True,
                            add_generation_prompt=False,
                            return_dict=True,
                            return_tensors="pt",
                        )
                        break
                    except (BlockingIOError, OSError) as e:
                        msg = str(e)
                        if "Resource temporarily unavailable" not in msg \
                                and not isinstance(e, BlockingIOError):
                            raise
                        last_err = e
                        time.sleep(0.2 * (2 ** attempt))

                if phase1_inputs is None:
                    warnings.warn(
                        f"[Collator] 跳过 (视频解码失败): {video_path} | {last_err}")
                    continue

                phase1_input_ids = phase1_inputs["input_ids"][0]  # [L1]
                phase1_len = phase1_input_ids.shape[0]

                # 检查总长度是否超限（粗略估计）
                answer_ids = self.tokenizer.encode(answer_part, add_special_tokens=False)
                total_est = phase1_len + K + len(answer_ids)
                if total_est > self.max_length:
                    warnings.warn(
                        f"[Collator] 跳过: 估计总长 {total_est} > max_length {self.max_length}")
                    continue

                # 5. 构造 answer 部分的 token ids
                answer_ids_tensor = torch.tensor(answer_ids, dtype=torch.long)

                result = {
                    "phase1_input_ids": phase1_input_ids,
                    "phase1_attention_mask": torch.ones_like(phase1_input_ids),
                    "gt_segment_ids": gt_segment_ids,  # List[List[int]]
                    "answer_ids": answer_ids_tensor,
                    "num_latent_steps": K,
                    "segment_types": [t for _, t in segments],
                }

                # 视频特征
                if "pixel_values_videos" in phase1_inputs:
                    result["pixel_values_videos"] = phase1_inputs["pixel_values_videos"][0]
                if "video_grid_thw" in phase1_inputs:
                    result["video_grid_thw"] = phase1_inputs["video_grid_thw"]
                if "mm_token_type_ids" in phase1_inputs:
                    result["mm_token_type_ids"] = phase1_inputs["mm_token_type_ids"][0]

                results.append(result)

            except RuntimeError as e:
                # max_seg_len 超限 → 报错中断（不跳过）
                raise e
            except Exception as e:
                warnings.warn(f"[Collator] 跳过样本: {e}")
                continue

        if not results:
            return None

        # v5 设计中每个样本独立处理（串行 latent 不好 batch），返回第一个有效样本
        # 如果需要 batch > 1，后续可以扩展为 padding + 分别处理
        return results[0]


# =============================================================================
# 训练参数
# =============================================================================

@dataclass
class TrainArgs:
    train_jsonl: str
    output_dir: str
    model_path: str = str(cfg_paths.QWEN3_VL_4B_PATH)
    num_train_epochs: int = 2
    per_device_batch_size: int = 1  # v5 串行 latent 建议 batch=1
    gradient_accumulation_steps: int = 2
    student_lr: float = 2e-5
    translator_lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_length: int = 32768
    max_frames: int = 64
    fps: float = 1.0
    video_max_pixels: int = 0  # 视频 3D 体积上限（T×H×W），0 表示使用默认值(786432)
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    bf16: bool = True
    gradient_checkpointing: bool = False  # 关闭 GC 以获取有梯度的 KV cache
    deepspeed: Optional[str] = None
    num_workers: int = 2
    resume_from: Optional[str] = None
    max_samples: int = 0
    max_seg_len: int = MAX_SEG_LEN
    max_latent_steps: int = MAX_LATENT_STEPS
    lambda_trans: float = 1.0
    lambda_ans: float = 1.0
    lambda_aux: float = 1.0
    lambda_think_end: float = 1.0
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser("Stage1-SFT v5 (Coconut/VaLR serial latent)")
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_path", default=str(cfg_paths.QWEN3_VL_4B_PATH))
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--student_lr", type=float, default=2e-5)
    p.add_argument("--translator_lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_length", type=int, default=32768)
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--video_max_pixels", type=int, default=0,
                   help="视频 3D 体积上限(T×H×W)，控制视频 token 数量。0=使用默认值(786432)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--deepspeed", default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--resume_from", default=None)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--max_seg_len", type=int, default=MAX_SEG_LEN)
    p.add_argument("--max_latent_steps", type=int, default=MAX_LATENT_STEPS)
    p.add_argument("--lambda_trans", type=float, default=1.0)
    p.add_argument("--lambda_ans", type=float, default=1.0)
    p.add_argument("--lambda_aux", type=float, default=1.0)
    p.add_argument("--lambda_think_end", type=float, default=1.0)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    a = p.parse_args()
    return TrainArgs(
        train_jsonl=a.train_jsonl, output_dir=a.output_dir,
        model_path=a.model_path, num_train_epochs=a.num_train_epochs,
        per_device_batch_size=a.per_device_batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        student_lr=a.student_lr, translator_lr=a.translator_lr,
        weight_decay=a.weight_decay, warmup_ratio=a.warmup_ratio,
        max_length=a.max_length, max_frames=a.max_frames, fps=a.fps,
        video_max_pixels=a.video_max_pixels,
        seed=a.seed, logging_steps=a.logging_steps, save_steps=a.save_steps,
        save_total_limit=a.save_total_limit, bf16=a.bf16,
        gradient_checkpointing=not a.no_gradient_checkpointing,
        deepspeed=a.deepspeed, num_workers=a.num_workers,
        resume_from=a.resume_from, max_samples=a.max_samples,
        max_seg_len=a.max_seg_len, max_latent_steps=a.max_latent_steps,
        lambda_trans=a.lambda_trans, lambda_ans=a.lambda_ans, lambda_aux=a.lambda_aux,
        lambda_think_end=a.lambda_think_end,
        wandb_project=a.wandb_project, wandb_run_name=a.wandb_run_name,
        wandb_mode=a.wandb_mode,
    )


# =============================================================================
# WandB 工具
# =============================================================================

def maybe_init_wandb(args: TrainArgs) -> bool:
    if not is_main_process():
        return False
    if not args.wandb_project:
        return False
    try:
        import wandb
    except ImportError:
        rank0_print("[wandb] 未安装，跳过")
        return False
    name = args.wandb_run_name or f"sft-v5-latent-{int(time.time())}"
    wandb.init(project=args.wandb_project, name=name, mode=args.wandb_mode,
               config=vars(args))
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


def wandb_finish():
    if not is_main_process():
        return
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


# =============================================================================
# Checkpoint 保存
# =============================================================================

def save_checkpoint(args: TrainArgs, student_model, translator, processor, step: int,
                    exit_head=None):
    """保存 Student + Translator + exit_head 的 checkpoint。"""
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
        exit_head_path = save_dir / "exit_head.pt"
        torch.save(exit_head.state_dict(), str(exit_head_path))

    # 保存 Translator（已解冻参与训练，需要保存）
    translator_dir = save_dir / "translator"
    translator_dir.mkdir(exist_ok=True)
    trans_target = translator.module if hasattr(translator, "module") else translator
    torch.save(trans_target.state_dict(), str(translator_dir / "translator_state_dict.pt"))

    rank0_print(f"[Save] checkpoint-{step} 已保存到 {save_dir}")

    # 清理旧 checkpoint（只清理含 student/ 子目录的推理友好格式，不动 DeepSpeed 格式）
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
# 单步训练逻辑
# =============================================================================

def train_one_sample(
    sample: Dict[str, Any],
    latent_engine: LatentForwardEngine,
    translator,
    args: TrainArgs,
    device: torch.device,
    translator_device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """对一个样本执行完整的 Phase1 → Phase2 → Phase3 forward，返回三个 loss。

    Args:
        sample: Collator 输出的单样本 dict
        latent_engine: LatentForwardEngine 实例
        translator: Translator 模型（已解冻，参与训练）
        args: 训练参数
        device: Student 所在设备
        translator_device: Translator 所在设备（None 表示和 Student 同卡）

    Returns:
        {"l_trans": ..., "l_ans": ..., "l_aux": ..., "total": ...}
    """
    # 解包样本
    phase1_input_ids = sample["phase1_input_ids"].unsqueeze(0).to(device)  # [1, L1]
    phase1_attention_mask = sample["phase1_attention_mask"].unsqueeze(0).to(device)  # [1, L1]
    gt_segment_ids = sample["gt_segment_ids"]  # List[List[int]]
    answer_ids = sample["answer_ids"].unsqueeze(0).to(device)  # [1, L_ans]
    K = sample["num_latent_steps"]

    pixel_values_videos = sample.get("pixel_values_videos")
    video_grid_thw = sample.get("video_grid_thw")
    mm_token_type_ids = sample.get("mm_token_type_ids")
    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.unsqueeze(0).to(device)
    if video_grid_thw is not None:
        video_grid_thw = video_grid_thw.to(device)
    if mm_token_type_ids is not None:
        mm_token_type_ids = mm_token_type_ids.unsqueeze(0).to(device)

    # =========================================================================
    # Phase 1: forward 到 <think>，得到 h_0 和 KV cache
    # =========================================================================
    h_0, past_key_values = latent_engine.phase1_forward_efficient(
        input_ids=phase1_input_ids,
        attention_mask=phase1_attention_mask,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
    )

    # =========================================================================
    # Phase 2: 串行 K 步 latent forward，得到 h_1~h_K
    # =========================================================================
    latent_hiddens, past_key_values = latent_engine.phase2_serial_forward(
        h_0=h_0,
        past_key_values=past_key_values,
        num_latent_steps=K,
    )

    # =========================================================================
    # Phase 3: forward answer 部分，得到 logits
    # =========================================================================
    answer_logits = latent_engine.phase3_forward(
        answer_ids=answer_ids,
        past_key_values=past_key_values,
    )

    # =========================================================================
    # 计算三个 Loss
    # =========================================================================

    # L_trans: Translator 还原 GT 段文字（Translator 已解冻，忠实翻译 latent hidden 内容）
    trans_target = translator.module if hasattr(translator, "module") else translator
    # 如果 Translator 在不同卡上，需要把 latent_hiddens 移过去
    _trans_dev = translator_device if translator_device is not None else device
    if _trans_dev != device:
        latent_hiddens_for_trans = [h.to(_trans_dev) for h in latent_hiddens]
    else:
        latent_hiddens_for_trans = latent_hiddens
    l_trans = trans_target.forward_batch(
        latent_hiddens=latent_hiddens_for_trans,
        gt_segments=gt_segment_ids,
        tokenizer=None,
        max_seg_len=args.max_seg_len,
    )
    # 把 l_trans 移回 Student 设备以便求和
    if _trans_dev != device:
        l_trans = l_trans.to(device)

    # L_ans: answer 部分标准 CE
    # answer_ids 本身就是 label（shift 在 compute_answer_loss 内部处理）
    l_ans = compute_answer_loss(answer_logits, answer_ids)

    # L_aux: 中间 latent 不应退出（使用独立 exit_head）
    l_aux = compute_aux_loss(latent_hiddens, latent_engine.exit_head)

    # L_think_end: 最后一个 latent h_K 应退出（使用独立 exit_head）
    l_think_end = compute_think_end_loss(latent_hiddens, latent_engine.exit_head)

    # 总 loss
    total = compute_total_loss(
        l_trans, l_ans, l_aux, l_think_end,
        lambda_trans=args.lambda_trans,
        lambda_ans=args.lambda_ans,
        lambda_aux=args.lambda_aux,
        lambda_think_end=args.lambda_think_end,
    )

    return {
        "l_trans": l_trans,
        "l_ans": l_ans,
        "l_aux": l_aux,
        "l_think_end": l_think_end,
        "total": total,
    }


# =============================================================================
# 主训练流程
# =============================================================================

def main():
    args = parse_args()
    local_rank = setup_distributed()
    torch.manual_seed(args.seed)

    cfg_paths.ensure_dirs()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 加载 prompts
    if is_main_process():
        from training.latent_sft_helpers import load_all_prompts
        load_all_prompts(verbose=True)

    maybe_init_wandb(args)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    rank0_print(f"[Init] world_size={get_world_size()} rank={get_rank()} "
                f"local_rank={local_rank} dtype={dtype}")
    rank0_print(f"[Init] model_path={args.model_path}")
    rank0_print(f"[Init] max_length={args.max_length} max_seg_len={args.max_seg_len} "
                f"max_latent_steps={args.max_latent_steps}")
    rank0_print(f"[Init] lambda: trans={args.lambda_trans} ans={args.lambda_ans} "
                f"aux={args.lambda_aux} think_end={args.lambda_think_end}")

    # =========================================================================
    # 加载 Student 模型
    # =========================================================================
    rank0_print("[Init] 加载 Student 模型...")
    processor, student_model = load_processor_and_model(args.model_path, dtype)
    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = args.fps
        processor.video_processor.max_frames = args.max_frames
        # 通过 video_max_pixels 控制视频 token 预算
        # 降低 max_pixels 会让 smart_resize 缩小分辨率，从而减少视频 token 数
        if args.video_max_pixels > 0:
            processor.video_processor.size = {
                "shortest_edge": min(args.video_max_pixels, processor.video_processor.size.get("shortest_edge", 131072)),
                "longest_edge": args.video_max_pixels,
            }
            rank0_print(f"[Init] video_processor: size.longest_edge={args.video_max_pixels} "
                        f"(控制视频 token 预算)")

    # 全参数微调
    student_model.train()
    for param in student_model.parameters():
        param.requires_grad_(True)

    if args.gradient_checkpointing and hasattr(student_model, "gradient_checkpointing_enable"):
        student_model.gradient_checkpointing_enable()

    # =========================================================================
    # 加载 Translator 模型（解冻，参与训练）
    # =========================================================================
    rank0_print("[Init] 加载 Translator 模型（解冻模式，参与训练）...")
    from training.translator_v5 import Translator
    translator = Translator(args.model_path, dtype=dtype)
    # 解冻 Translator：让它学会从 Student hidden state 解码文字
    # 使用较低学习率（translator_lr）防止过拟合
    translator.train()
    for param in translator.parameters():
        param.requires_grad_(True)
    rank0_print("[Init] Translator 已解冻（train mode, requires_grad=True）")

    # =========================================================================
    # 构造 exit_head、latent_proj 和 LatentForwardEngine
    # =========================================================================
    # 获取 hidden_size（Qwen3VL 的 hidden_size 在 text_config 中）
    _hidden_size = student_model.config.text_config.hidden_size
    exit_head = LatentExitHead(_hidden_size)
    exit_head.train()
    for p in exit_head.parameters():
        p.requires_grad_(True)
    rank0_print(f"[Init] LatentExitHead 创建完成 (hidden_size={_hidden_size})")

    latent_engine = LatentForwardEngine(student_model, exit_head=exit_head)

    # =========================================================================
    # Dataset & DataLoader
    # =========================================================================
    dataset = Stage1SFTDatasetV5(
        args.train_jsonl, max_samples=args.max_samples, shuffle_seed=args.seed)

    sampler: Optional[DistributedSampler] = None
    if is_dist():
        sampler = DistributedSampler(
            dataset, num_replicas=get_world_size(), rank=get_rank(),
            shuffle=True, seed=args.seed)

    collator = LatentSFTCollatorV5(
        processor=processor, max_length=args.max_length,
        max_seg_len=args.max_seg_len, max_latent_steps=args.max_latent_steps)

    loader = DataLoader(
        dataset, batch_size=args.per_device_batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, collate_fn=collator,
        pin_memory=True, drop_last=True)

    steps_per_epoch = math.ceil(len(loader) / max(1, args.gradient_accumulation_steps))
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    rank0_print(f"[Init] dataset={len(dataset)} steps/epoch={steps_per_epoch} total={total_steps}")

    # =========================================================================
    # 设备分配：Student 放 local_rank*2 卡，Translator 放 local_rank*2+1 卡
    # 单机八卡 = 4 个并行训练进程，每个进程用 2 张 GPU
    # =========================================================================
    use_deepspeed = args.deepspeed is not None
    student_device = torch.device(f"cuda:{local_rank * 2}") if torch.cuda.is_available() else torch.device("cpu")
    translator_device = torch.device(f"cuda:{local_rank * 2 + 1}") if torch.cuda.is_available() else torch.device("cpu")
    device = student_device  # train_one_sample 中的 device 指 Student 所在卡
    rank0_print(f"[Init] 设备分配: Student → {student_device}, Translator → {translator_device}")

    if use_deepspeed:
        import deepspeed

        ds_config = json.load(open(args.deepspeed, "r"))
        ws = get_world_size()
        tbs = args.per_device_batch_size * args.gradient_accumulation_steps * ws
        ds_config["train_micro_batch_size_per_gpu"] = args.per_device_batch_size
        ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        ds_config["train_batch_size"] = tbs
        ds_config["gradient_clipping"] = (
            1.0 if ds_config.get("gradient_clipping") == "auto"
            else ds_config.get("gradient_clipping", 1.0))
        if "optimizer" in ds_config:
            op = ds_config["optimizer"].get("params", {})
            for k, v in [("lr", args.student_lr), ("weight_decay", args.weight_decay),
                         ("betas", [0.9, 0.95]), ("eps", 1e-8)]:
                if op.get(k) == "auto":
                    op[k] = v
        if "scheduler" in ds_config:
            sp = ds_config["scheduler"].get("params", {})
            for k, v in [("total_num_steps", total_steps), ("warmup_num_steps", warmup_steps),
                         ("warmup_max_lr", args.student_lr), ("warmup_min_lr", 0.0)]:
                if sp.get(k) == "auto":
                    sp[k] = v
        if "bf16" in ds_config and ds_config["bf16"].get("enabled") == "auto":
            ds_config["bf16"]["enabled"] = args.bf16
        if "fp16" in ds_config and ds_config["fp16"].get("enabled") == "auto":
            ds_config["fp16"]["enabled"] = False

        # 训练 Student + exit_head + Translator
        student_params = [
            {"params": [p for p in student_model.parameters() if p.requires_grad],
             "lr": args.student_lr, "weight_decay": args.weight_decay},
            {"params": list(exit_head.parameters()),
             "lr": args.student_lr, "weight_decay": 0.0},
            {"params": [p for p in translator.parameters() if p.requires_grad],
             "lr": args.translator_lr, "weight_decay": args.weight_decay},
        ]

        # 创建一个包装模型（Student + Translator + exit_head 都参与训练）
        class CombinedModel(torch.nn.Module):
            def __init__(self, student, trans, _exit_head):
                super().__init__()
                self.student = student
                self.translator = trans
                self.exit_head = _exit_head

        combined = CombinedModel(student_model, translator, exit_head)

        engine, optimizer, _, lr_sched = deepspeed.initialize(
            model=combined,
            model_parameters=student_params,
            config=ds_config,
        )
        device = engine.device
        student_device = device
        translator_device = device  # DeepSpeed 模式下都在同一卡
        student_model = engine.module.student
        translator = engine.module.translator
        exit_head = engine.module.exit_head
        latent_engine = LatentForwardEngine(student_model, exit_head=exit_head)

    else:
        # 原生 PyTorch 训练：Student 和 Translator 分卡
        student_model = student_model.to(student_device)
        exit_head = exit_head.to(device=student_device, dtype=dtype)
        translator = translator.to(translator_device)
        rank0_print(f"[Init] Student + exit_head → {student_device}")
        rank0_print(f"[Init] Translator → {translator_device}")

        # 优化器：所有参数统一管理
        optimizer = torch.optim.AdamW([
            {"params": [p for p in student_model.parameters() if p.requires_grad],
             "lr": args.student_lr, "weight_decay": args.weight_decay},
            {"params": list(exit_head.parameters()),
             "lr": args.student_lr, "weight_decay": 0.0},
            {"params": [p for p in translator.parameters() if p.requires_grad],
             "lr": args.translator_lr, "weight_decay": args.weight_decay},
        ], betas=(0.9, 0.95))

        from torch.optim.lr_scheduler import LambdaLR

        def _lr_lambda(s: int) -> float:
            if s < warmup_steps:
                return s / max(1, warmup_steps)
            prog = (s - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))

        lr_sched = LambdaLR(optimizer, _lr_lambda)
        engine = None

    # =========================================================================
    # 训练循环
    # =========================================================================
    global_step = 0
    student_model.train()
    translator.train()

    for epoch in range(args.num_train_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        if is_main_process():
            rank0_print(f"[Epoch {epoch+1}] 开始加载数据（首个 batch 需解码视频，请耐心等待）...")

        it = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}",
                  dynamic_ncols=True) if is_main_process() else loader

        running_total = 0.0
        running_trans = 0.0
        running_ans = 0.0
        running_aux = 0.0
        running_think_end = 0.0
        accum_count = 0

        if not use_deepspeed:
            optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(it):
            # batch 可能为 None（所有样本被跳过）
            if batch is None:
                continue

            try:
                losses = train_one_sample(
                    sample=batch,
                    latent_engine=latent_engine,
                    translator=translator if not use_deepspeed else engine.module.translator,
                    args=args,
                    device=device,
                    translator_device=translator_device,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    warnings.warn(f"[OOM] step={step}, 跳过此样本并清理显存")
                    torch.cuda.empty_cache()
                    continue
                raise e

            loss = losses["total"]

            if use_deepspeed:
                engine.backward(loss)
                engine.step()
                step_inc = 1
            else:
                scaled_loss = loss / args.gradient_accumulation_steps
                scaled_loss.backward()
                accum_count += 1
                step_inc = 0
                if accum_count == args.gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in student_model.parameters() if p.requires_grad],
                        1.0)
                    optimizer.step()
                    lr_sched.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    step_inc = 1

            # 记录 loss
            running_total += float(loss.detach().item())
            running_trans += float(losses["l_trans"].detach().item())
            running_ans += float(losses["l_ans"].detach().item())
            running_aux += float(losses["l_aux"].detach().item())
            running_think_end += float(losses["l_think_end"].detach().item())

            if step_inc:
                global_step += 1
                cur_lr = optimizer.param_groups[0]["lr"] if optimizer else 0.0

                wandb_log({
                    "train/loss_total": float(loss.detach().item()),
                    "train/loss_trans": float(losses["l_trans"].detach().item()),
                    "train/loss_ans": float(losses["l_ans"].detach().item()),
                    "train/loss_aux": float(losses["l_aux"].detach().item()),
                    "train/loss_think_end": float(losses["l_think_end"].detach().item()),
                    "train/lr": float(cur_lr),
                    "train/epoch": float(epoch + (step + 1) / max(1, len(loader))),
                    "train/K": batch["num_latent_steps"],
                }, step=global_step)

                if is_main_process() and global_step % args.logging_steps == 0:
                    n = args.logging_steps
                    avg_t = running_total / n
                    avg_tr = running_trans / n
                    avg_a = running_ans / n
                    avg_x = running_aux / n
                    avg_te = running_think_end / n
                    if hasattr(it, 'set_postfix'):
                        it.set_postfix(
                            total=f"{avg_t:.4f}",
                            trans=f"{avg_tr:.4f}",
                            ans=f"{avg_a:.4f}",
                            aux=f"{avg_x:.4f}",
                            te=f"{avg_te:.4f}",
                            lr=f"{cur_lr:.2e}",
                            step=global_step)
                    running_total = 0.0
                    running_trans = 0.0
                    running_ans = 0.0
                    running_aux = 0.0
                    running_think_end = 0.0

                # 保存 checkpoint
                if global_step % args.save_steps == 0:
                    if use_deepspeed:
                        engine.save_checkpoint(
                            args.output_dir, tag=f"checkpoint-{global_step}")
                        # 同时保存推理友好格式（student/ 子目录）
                        save_checkpoint(args, student_model, translator,
                                        processor, global_step, exit_head=exit_head)
                    else:
                        save_checkpoint(args, student_model, translator,
                                        processor, global_step, exit_head=exit_head)
                    if is_dist():
                        try:
                            dist.barrier()
                        except Exception:
                            pass

        # epoch 结束：flush 剩余梯度（不满 gradient_accumulation_steps 的部分）
        if not use_deepspeed and accum_count > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in student_model.parameters() if p.requires_grad],
                1.0)
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            rank0_print(f"[Epoch {epoch+1}] flush 剩余 {accum_count} 步梯度, global_step={global_step}")
            accum_count = 0

    # =========================================================================
    # 保存 final checkpoint
    # =========================================================================
    rank0_print("[Save] 保存 final checkpoint ...")
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if is_dist():
        dist.barrier()

    if use_deepspeed:
        engine.save_checkpoint(args.output_dir, tag="checkpoint-final")
        # 同时保存推理友好格式（student/ 子目录）
        save_checkpoint(args, student_model, translator, processor, global_step,
                        exit_head=exit_head)
    else:
        save_checkpoint(args, student_model, translator, processor, global_step,
                        exit_head=exit_head)

    rank0_print(f"[Done] Stage1-SFT-v5 训练结束，共 {global_step} 步")
    wandb_finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
