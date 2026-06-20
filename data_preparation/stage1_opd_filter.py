"""Stage 1-OPD 数据筛选。

筛选规则（与 docs/OPD_Video_Reasoning_Training_Plan.md 第六节一致）：

  分支 1 —— 不可校验（verifiable=False，描述类）:
      * 无法判断学生/教师是否答对
      * 信任教师，直接保留（不跑学生 forward，也不跑教师 forward）

  分支 2 —— 可校验（verifiable=True）:
      * 学生直答 == gt → 学生已会，丢
      * 学生失败 + 教师聚焦后 == gt → 保留
      * 学生失败 + 教师也答错       → 丢（数据本身太难）

训练信号始终是双教师 KL，与本脚本的"对错过滤"无关。
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.paths import (
    OPD_DATA_PATH, QWEN3_VL_4B_PATH, ensure_dirs,
)
from pipelines import PIPELINES

# 学生 / 教师 模型 (lazy)
_STUDENT = None
_STUDENT_PROCESSOR = None
_TEACHER_P = None
_TEACHER_P_PROCESSOR = None


def _load_qwen3vl(device: str, dtype: str = "bfloat16"):
    """加载 Qwen3-VL-4B（学生/教师同模型，但不同实例）"""
    if not QWEN3_VL_4B_PATH.exists():
        raise FileNotFoundError(
            f"Qwen3-VL-4B 路径不存在: {QWEN3_VL_4B_PATH}\n"
"请下载到该路径，或在 configs/paths.yaml 中调整 model_root / overrides。"
        )
    import torch
    # 优先用 Qwen3VL 类，没有则回退 AutoModel
    try:
        from transformers import (
            Qwen3VLForConditionalGeneration as VLClass,
            AutoProcessor,
        )
    except ImportError:
        from transformers import (
            AutoModelForCausalLM as VLClass,
            AutoProcessor,
        )
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    print(f"[VL] 加载 Qwen3-VL-4B -> {device} ({dtype})", flush=True)
    # 修复 transformers 5.10.0.dev0 Qwen3VLProcessor 的 <|placeholder|> bug
    try:
        from utils.qwen3vl_patch import apply_qwen3vl_patches
        apply_qwen3vl_patches()
    except Exception:
        pass
    processor = AutoProcessor.from_pretrained(str(QWEN3_VL_4B_PATH), trust_remote_code=True)
    model = VLClass.from_pretrained(
        str(QWEN3_VL_4B_PATH),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, processor


def get_student(device: str = "cuda:0"):
    global _STUDENT, _STUDENT_PROCESSOR
    if _STUDENT is None:
        _STUDENT, _STUDENT_PROCESSOR = _load_qwen3vl(device)
    return _STUDENT, _STUDENT_PROCESSOR


def get_teacher_p(device: str = "cuda:0"):
    global _TEACHER_P, _TEACHER_P_PROCESSOR
    if _TEACHER_P is None:
        _TEACHER_P, _TEACHER_P_PROCESSOR = _load_qwen3vl(device)
    return _TEACHER_P, _TEACHER_P_PROCESSOR


def vl_answer(model, processor, video_path: str, question: str,
              max_new_tokens: int = 256) -> str:
    """单条视频QA推理"""
    import torch
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": str(video_path)},
            {"type": "text", "text": question},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], videos=[str(video_path)],
                       return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ----------------------------------------------------------------------------
# 答案匹配 (按 verifiable type 决定校验方式)
# ----------------------------------------------------------------------------

def _parse_time_range(s: str):
    """解析 '3.0s-5.0s' / '3.0-5.0' -> (start, end)"""
    m = re.search(r"([0-9.]+)\s*s?\s*-\s*([0-9.]+)\s*s?", s)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _temporal_iou(pred: str, gt: str) -> float:
    a = _parse_time_range(pred)
    b = _parse_time_range(gt)
    if a is None or b is None:
        return 0.0
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _parse_bbox(s: str):
    """解析 '[x1,y1,x2,y2]'"""
    m = re.search(r"\[([\d.\s,]+)\]", s)
    if not m:
        return None
    try:
        nums = [float(x.strip()) for x in m.group(1).split(",")]
        if len(nums) != 4:
            return None
        return nums
    except ValueError:
        return None


def _bbox_iou(pred: str, gt) -> float:
    a = _parse_bbox(pred)
    if isinstance(gt, str):
        b = _parse_bbox(gt)
    else:
        b = list(gt) if gt is not None else None
    if a is None or b is None:
        return 0.0
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def matches(pred: str, sample: dict) -> bool:
    """根据 sample.type 与 verifiable 决定匹配方式"""
    if not sample.get("verifiable", False):
        # 描述类不可校验，永远返回False（外层逻辑会另行处理）
        return False
    typ = sample.get("type", "")
    gt = sample.get("gt_answer")
    if typ == "temporal_locate":
        return _temporal_iou(pred, str(gt)) >= 0.5
    if typ == "spatial_detect":
        return _bbox_iou(pred, gt) >= 0.5
    if typ == "tracking_overlay":
        # 时空IoU简化为 时间IoU + bbox IoU 平均
        if not isinstance(gt, dict):
            return False
        time_str = f"{gt['time'][0]:.1f}-{gt['time'][1]:.1f}"
        t_iou = _temporal_iou(pred, time_str)
        b_iou = _bbox_iou(pred, gt.get("bbox_start"))
        return (t_iou + b_iou) / 2 >= 0.3
    # 选择题/字符串：归一化后包含或相等
    p = str(pred).strip().lower()
    g = str(gt).strip().lower()
    return g in p or p == g


# ----------------------------------------------------------------------------
# 视觉聚焦
# ----------------------------------------------------------------------------

def build_focused_input(sample: dict) -> dict:
    """根据 sample.type 调用对应 pipeline，返回 {video, perception_question}"""
    typ = sample["type"]
    if typ not in PIPELINES:
        raise ValueError(f"未知 pipeline type: {typ}")
    params = sample.get("params", {})
    if not params:
        # 兼容老格式：从sample平铺字段构造
        params = {
            "target": sample.get("question", ""),
            "time": sample.get("time"),
            "frame": sample.get("frame"),
            "bbox": sample.get("bbox"),
            "objects": sample.get("objects"),
        }
        params = {k: v for k, v in params.items() if v}
    return PIPELINES[typ](sample["video"], **params)


# ----------------------------------------------------------------------------
# 数据读写
# ----------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_done_ids(out_path: Path) -> set:
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "sample_id" in obj:
                    done.add(obj["sample_id"])
            except json.JSONDecodeError:
                continue
    return done


# ----------------------------------------------------------------------------
# 多卡分片
# ----------------------------------------------------------------------------

def _shard(items, rank: int, world_size: int):
    return [x for i, x in enumerate(items) if i % world_size == rank]


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def filter_one(sample: dict, student_dev: str, teacher_dev: str) -> Optional[dict]:
    """对一条样本做双过滤。

    顺序重要：
      1) 不可校验 → 直接保留，**跳过学生 forward**（避免无意义计算）
      2) 可校验 → 学生直答 →
           - 答对 → 丢
           - 答错 → 跑教师 + 聚焦，教师答对保留、答错丢
    """
    # 1) 描述类样本：既不能判断对错，也不需要过滤，直接保留。
    if not sample.get("verifiable", False):
        return sample

    # 2) 可校验：先跑学生直答
    student, sproc = get_student(student_dev)
    student_pred = vl_answer(student, sproc, sample["video"], sample["question"])
    if matches(student_pred, sample):
        return None  # 学生已会

    # 学生失败：跑教师 + 视觉聚焦
    focused = build_focused_input(sample)
    teacher, tproc = get_teacher_p(teacher_dev)
    perception_q = focused.get("perception_question") or sample["question"]
    teacher_pred = vl_answer(teacher, tproc, focused["video"], perception_q)
    if matches(teacher_pred, sample):
        sample["_student_pred"] = student_pred
        sample["_teacher_pred"] = teacher_pred
        sample["_focused_video"] = focused["video"]
        return sample
    return None  # 教师也答错 → 数据太难，丢


def main():
    parser = argparse.ArgumentParser(description="Stage 1-OPD 数据筛选")
    parser.add_argument("--input", type=str, required=True,
                        help="候选池JSONL，每行需含 video/question/gt_answer/type/verifiable/params")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--student_device", type=str, default="cuda:0")
    parser.add_argument("--teacher_device", type=str, default="cuda:1")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world_size", type=int,
                        default=int(os.environ.get("WORLD_SIZE", 1)))
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"输入不存在: {in_path}")
    ensure_dirs()
    out_path = Path(args.output) if args.output else (
        OPD_DATA_PATH / f"stage1_opd_filtered.rank{args.rank}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = list(iter_jsonl(in_path))
    samples = _shard(samples, args.rank, args.world_size)
    done = _load_done_ids(out_path)
    print(f"[rank{args.rank}/{args.world_size}] 总{len(samples)}条 已完成{len(done)}条")

    only_rank0 = (args.rank == 0)
    pbar = tqdm(samples, desc=f"OPD-filter rank{args.rank}", disable=not only_rank0)

    with open(out_path, "a", encoding="utf-8") as f:
        kept = 0
        for s in pbar:
            sid = s.get("sample_id") or f"{s.get('video')}::{s.get('question')}"
            s["sample_id"] = sid
            if sid in done:
                continue
            res = filter_one(s, args.student_device, args.teacher_device)
            if res is not None:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
                kept += 1
        print(f"[rank{args.rank}] 保留 {kept} 条 -> {out_path}")


if __name__ == "__main__":
    main()
