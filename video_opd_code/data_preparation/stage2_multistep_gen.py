"""
Stage 2 多步推理训练数据生成。

设计思路：
  Stage 1 的 SFT 数据只有单次 observe/result，学生学会了每种工具的基本用法。
  Stage 2 需要多步推理链（2-4 次 observe/result），让学生学会：
    1. 分析复杂问题 → 拆解为多个子任务
    2. 按逻辑顺序调用不同工具
    3. 基于前序结果决定下一步观察

数据来源：
  - 规则组合（B类）：从已有单步数据中按规则组合出多步轨迹
  - LLM 生成（C类）：Qwen3-32B 基于视频标注生成复杂问题 + 多步推理链

组合模式（规则生成）：
  M-1: temporal_locate → temporal_clip（先定位时间，再描述该时段内容）
  M-2: temporal_locate → spatial_detect（先定位时间，再在该帧检测物体）
  M-3: spatial_detect → spatial_crop（先检测物体，再裁剪描述）
  M-4: spatial_detect → depth_overlay（先检测两物体，再比较深度）
  M-5: temporal_locate → spatial_detect → spatial_crop（三步：定位→检测→描述）
  M-6: temporal_locate → tracking_overlay（先定位事件，再追踪物体轨迹）

课程学习分级：
  Easy:   1 次 observe（复用 Stage 1 数据）
  Medium: 2 次 observe（M-1 ~ M-4, M-6）
  Hard:   3-4 次 observe（M-5 + LLM 生成）

输出格式: 与 Stage 1 相同的 JSONL，额外字段 difficulty / n_observe

用法:
    # 生成全部规则组合数据
    python -m data_preparation.stage2_multistep_gen --mode rule --n 5000
    # 生成 LLM 增强数据（需要 Qwen3-32B）
    python -m data_preparation.stage2_multistep_gen --mode llm --n 3000
    # 合并 Stage 1 + Stage 2 数据
    python -m data_preparation.stage2_multistep_gen --mode merge
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.paths import (  # noqa: E402
    SFT_DATA_PATH, OUTPUT_ROOT, QWEN3_32B_PATH, ensure_dirs,
)

# Stage 2 输出目录
STAGE2_DATA_PATH = OUTPUT_ROOT / "stage2_sft"


# ============================================================================
# 工具函数
# ============================================================================

def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _write_jsonl(items: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _load_stage1_by_type() -> dict[str, list[dict]]:
    """加载 Stage 1 SFT 数据，按 type 分组"""
    all_path = SFT_DATA_PATH / "stage1_sft_template_all.jsonl"
    if not all_path.exists():
        raise FileNotFoundError(
            f"Stage 1 合并文件不存在: {all_path}\n"
            "请先运行: python -m data_preparation.stage1_sft_template --subset merge"
        )
    items = _load_jsonl(all_path)
    by_type: dict[str, list[dict]] = {}
    for item in items:
        t = item.get("type", "raw")
        by_type.setdefault(t, []).append(item)
    return by_type


# ============================================================================
# 规则组合模式
# ============================================================================

def _combine_m1(loc_item: dict, clip_item: dict) -> Optional[dict]:
    """M-1: temporal_locate → temporal_clip
    先定位事件时间，再描述该时段内容。
    """
    # 从 loc_item 取时间定位信息
    loc_params = loc_item.get("params", {})
    target = loc_params.get("target", "")
    gt_answer_loc = loc_item.get("gt_answer", "")
    if not target or not gt_answer_loc:
        return None

    # 解析时间范围
    import re
    m = re.search(r"([0-9.]+)s?\s*-\s*([0-9.]+)s?", gt_answer_loc)
    if not m:
        return None
    start, end = float(m.group(1)), float(m.group(2))

    # 从 clip_item 取描述
    clip_gt = clip_item.get("gt_answer", "")
    if not clip_gt:
        return None

    video = loc_item["video"]
    question = f"What is happening when \"{target}\" occurs in the video? Describe the activity in detail."

    trajectory = (
        "<think>\n"
        f"[分析] 需要先定位\"{target}\"发生的时间，再描述该时段的具体内容。\n"
        f"<observe type=\"temporal_locate\" target=\"{target}\"/>\n"
        f"<result>It happens from {start:.1f}s to {end:.1f}s.</result>\n"
        f"[推理] 已定位到 {start:.1f}s-{end:.1f}s，现在需要详细描述这段时间内的活动。\n"
        f"<observe type=\"temporal_clip\" time=\"{start:.1f}-{end:.1f}\" target=\"activity within this time span\"/>\n"
        f"<result>{clip_gt}</result>\n"
        f"[结论] 在\"{target}\"发生时（{start:.1f}s-{end:.1f}s），视频中的活动是：{clip_gt}\n"
        "</think>\n"
        f"<answer>{clip_gt}</answer>"
    )

    return {
        "video": video,
        "question": question,
        "trajectory": trajectory,
        "gt_answer": clip_gt,
        "verifiable": False,
        "type": "multistep",
        "subtypes": ["temporal_locate", "temporal_clip"],
        "source": f"{loc_item.get('source', 'unknown')}+{clip_item.get('source', 'unknown')}",
        "params": {"target": target, "time": f"{start:.1f}-{end:.1f}"},
        "difficulty": "medium",
        "n_observe": 2,
    }


def _combine_m2(loc_item: dict, detect_item: dict) -> Optional[dict]:
    """M-2: temporal_locate → spatial_detect
    先定位事件时间，再在该帧检测特定物体。
    """
    loc_params = loc_item.get("params", {})
    target_event = loc_params.get("target", "")
    gt_answer_loc = loc_item.get("gt_answer", "")
    if not target_event or not gt_answer_loc:
        return None

    import re
    m = re.search(r"([0-9.]+)s?\s*-\s*([0-9.]+)s?", gt_answer_loc)
    if not m:
        return None
    start, end = float(m.group(1)), float(m.group(2))
    mid_time = (start + end) / 2

    # 从 detect_item 取物体信息
    detect_params = detect_item.get("params", {})
    detect_target = detect_params.get("target", "")
    detect_gt = detect_item.get("gt_answer", "")
    if not detect_target or not detect_gt:
        return None

    video = loc_item["video"]
    question = (
        f"When \"{target_event}\" happens in the video, "
        f"where is the {detect_target} located in the frame?"
    )

    trajectory = (
        "<think>\n"
        f"[分析] 需要先定位\"{target_event}\"发生的时间，再在该时刻的帧中检测{detect_target}的位置。\n"
        f"<observe type=\"temporal_locate\" target=\"{target_event}\"/>\n"
        f"<result>It happens from {start:.1f}s to {end:.1f}s.</result>\n"
        f"[推理] 事件发生在 {start:.1f}s-{end:.1f}s，取中间时刻 {mid_time:.1f}s 检测物体。\n"
        f"<observe type=\"spatial_detect\" frame=\"{mid_time:.1f}\" target=\"{detect_target}\"/>\n"
        f"<result>{detect_gt}</result>\n"
        f"[结论] 在\"{target_event}\"发生时（{mid_time:.1f}s），{detect_target}位于{detect_gt}。\n"
        "</think>\n"
        f"<answer>At {mid_time:.1f}s, {detect_gt}</answer>"
    )

    return {
        "video": video,
        "question": question,
        "trajectory": trajectory,
        "gt_answer": f"At {mid_time:.1f}s, {detect_gt}",
        "verifiable": False,
        "type": "multistep",
        "subtypes": ["temporal_locate", "spatial_detect"],
        "source": f"{loc_item.get('source', 'unknown')}+{detect_item.get('source', 'unknown')}",
        "params": {"target_event": target_event, "target_object": detect_target,
                   "frame": f"{mid_time:.1f}"},
        "difficulty": "medium",
        "n_observe": 2,
    }


def _combine_m3(detect_item: dict, crop_item: dict) -> Optional[dict]:
    """M-3: spatial_detect → spatial_crop
    先检测物体位置，再裁剪该区域详细描述。
    """
    detect_params = detect_item.get("params", {})
    detect_target = detect_params.get("target", "")
    detect_gt = detect_item.get("gt_answer", "")
    frame = detect_params.get("frame", "0.0")
    if not detect_target or not detect_gt:
        return None

    # 从 crop_item 取描述
    crop_gt = crop_item.get("gt_answer", "")
    crop_params = crop_item.get("params", {})
    bbox = crop_params.get("bbox", detect_gt)
    if not crop_gt:
        return None

    video = detect_item["video"]
    question = f"At {frame}s, describe the {detect_target} in detail."

    trajectory = (
        "<think>\n"
        f"[分析] 需要先定位{detect_target}在帧中的位置，再裁剪该区域进行详细描述。\n"
        f"<observe type=\"spatial_detect\" frame=\"{frame}\" target=\"{detect_target}\"/>\n"
        f"<result>The {detect_target} is located at {detect_gt}.</result>\n"
        f"[推理] 已定位到 {detect_gt}，现在裁剪该区域进行详细观察。\n"
        f"<observe type=\"spatial_crop\" frame=\"{frame}\" bbox=\"{bbox}\" target=\"detailed appearance of {detect_target}\"/>\n"
        f"<result>{crop_gt}</result>\n"
        f"[结论] {detect_target}位于{detect_gt}，其详细描述为：{crop_gt}\n"
        "</think>\n"
        f"<answer>{crop_gt}</answer>"
    )

    return {
        "video": video,
        "question": question,
        "trajectory": trajectory,
        "gt_answer": crop_gt,
        "verifiable": False,
        "type": "multistep",
        "subtypes": ["spatial_detect", "spatial_crop"],
        "source": f"{detect_item.get('source', 'unknown')}+{crop_item.get('source', 'unknown')}",
        "params": {"target": detect_target, "frame": frame, "bbox": bbox},
        "difficulty": "medium",
        "n_observe": 2,
    }


def _combine_m4(detect_item: dict, depth_item: dict) -> Optional[dict]:
    """M-4: spatial_detect → depth_overlay
    先检测两个物体，再比较它们的深度关系。
    """
    detect_params = detect_item.get("params", {})
    detect_target = detect_params.get("target", "")
    frame = detect_params.get("frame", "0.0")
    detect_gt = detect_item.get("gt_answer", "")
    if not detect_target or not detect_gt:
        return None

    # 从 depth_item 取深度比较结果
    depth_gt = depth_item.get("gt_answer", "")
    depth_params = depth_item.get("params", {})
    objects = depth_params.get("objects", "")
    if not depth_gt or not objects:
        return None

    video = depth_item["video"]
    question = f"At {frame}s, which object is closer to the camera?"

    trajectory = (
        "<think>\n"
        f"[分析] 需要先检测帧中的物体位置，再比较它们的深度关系。\n"
        f"<observe type=\"spatial_detect\" frame=\"{frame}\" target=\"{detect_target}\"/>\n"
        f"<result>The {detect_target} is located at {detect_gt}.</result>\n"
        f"[推理] 已定位物体，现在需要比较深度关系。\n"
        f"<observe type=\"depth_overlay\" frame=\"{frame}\" objects=\"{objects}\" target=\"depth relation\"/>\n"
        f"<result>{depth_gt}</result>\n"
        f"[结论] {depth_gt}\n"
        "</think>\n"
        f"<answer>{depth_gt}</answer>"
    )

    return {
        "video": video,
        "question": question,
        "trajectory": trajectory,
        "gt_answer": depth_gt,
        "verifiable": True,
        "type": "multistep",
        "subtypes": ["spatial_detect", "depth_overlay"],
        "source": f"{detect_item.get('source', 'unknown')}+{depth_item.get('source', 'unknown')}",
        "params": {"target": detect_target, "frame": frame, "objects": objects},
        "difficulty": "medium",
        "n_observe": 2,
    }


def _combine_m5(loc_item: dict, detect_item: dict, crop_item: dict) -> Optional[dict]:
    """M-5: temporal_locate → spatial_detect → spatial_crop（三步）
    先定位时间，再检测物体，最后裁剪描述。
    """
    loc_params = loc_item.get("params", {})
    target_event = loc_params.get("target", "")
    gt_answer_loc = loc_item.get("gt_answer", "")
    if not target_event or not gt_answer_loc:
        return None

    import re
    m = re.search(r"([0-9.]+)s?\s*-\s*([0-9.]+)s?", gt_answer_loc)
    if not m:
        return None
    start, end = float(m.group(1)), float(m.group(2))
    mid_time = (start + end) / 2

    detect_params = detect_item.get("params", {})
    detect_target = detect_params.get("target", "")
    detect_gt = detect_item.get("gt_answer", "")
    if not detect_target or not detect_gt:
        return None

    crop_gt = crop_item.get("gt_answer", "")
    crop_params = crop_item.get("params", {})
    bbox = crop_params.get("bbox", detect_gt)
    if not crop_gt:
        return None

    video = loc_item["video"]
    question = (
        f"When \"{target_event}\" happens, describe the {detect_target} in detail."
    )

    trajectory = (
        "<think>\n"
        f"[分析] 需要先定位事件时间，再找到{detect_target}，最后详细描述它。\n"
        f"<observe type=\"temporal_locate\" target=\"{target_event}\"/>\n"
        f"<result>It happens from {start:.1f}s to {end:.1f}s.</result>\n"
        f"[推理] 事件在 {start:.1f}s-{end:.1f}s，取 {mid_time:.1f}s 检测物体。\n"
        f"<observe type=\"spatial_detect\" frame=\"{mid_time:.1f}\" target=\"{detect_target}\"/>\n"
        f"<result>The {detect_target} is located at {detect_gt}.</result>\n"
        f"[推理] 已定位到 {detect_gt}，裁剪该区域详细观察。\n"
        f"<observe type=\"spatial_crop\" frame=\"{mid_time:.1f}\" bbox=\"{bbox}\" target=\"detailed appearance\"/>\n"
        f"<result>{crop_gt}</result>\n"
        f"[结论] 在\"{target_event}\"发生时，{detect_target}位于{detect_gt}，详细描述：{crop_gt}\n"
        "</think>\n"
        f"<answer>{crop_gt}</answer>"
    )

    return {
        "video": video,
        "question": question,
        "trajectory": trajectory,
        "gt_answer": crop_gt,
        "verifiable": False,
        "type": "multistep",
        "subtypes": ["temporal_locate", "spatial_detect", "spatial_crop"],
        "source": "combined_3step",
        "params": {"target_event": target_event, "target_object": detect_target,
                   "frame": f"{mid_time:.1f}", "bbox": bbox},
        "difficulty": "hard",
        "n_observe": 3,
    }


def _combine_m6(loc_item: dict, track_item: dict) -> Optional[dict]:
    """M-6: temporal_locate → tracking_overlay
    先定位事件时间，再追踪该时段内物体的运动轨迹。
    """
    loc_params = loc_item.get("params", {})
    target_event = loc_params.get("target", "")
    gt_answer_loc = loc_item.get("gt_answer", "")
    if not target_event or not gt_answer_loc:
        return None

    import re
    m = re.search(r"([0-9.]+)s?\s*-\s*([0-9.]+)s?", gt_answer_loc)
    if not m:
        return None
    start, end = float(m.group(1)), float(m.group(2))

    # 从 track_item 取追踪描述
    track_params = track_item.get("params", {})
    track_target = track_params.get("target", "")
    track_gt = track_item.get("gt_answer", "")
    if not track_target or not track_gt:
        return None

    video = loc_item["video"]
    question = (
        f"During \"{target_event}\", describe the movement trajectory of {track_target}."
    )

    trajectory = (
        "<think>\n"
        f"[分析] 需要先定位\"{target_event}\"的时间段，再追踪{track_target}的运动轨迹。\n"
        f"<observe type=\"temporal_locate\" target=\"{target_event}\"/>\n"
        f"<result>It happens from {start:.1f}s to {end:.1f}s.</result>\n"
        f"[推理] 事件在 {start:.1f}s-{end:.1f}s，现在追踪该时段内{track_target}的运动。\n"
        f"<observe type=\"tracking_overlay\" time=\"{start:.1f}-{end:.1f}\" target=\"{track_target}\"/>\n"
        f"<result>{track_gt}</result>\n"
        f"[结论] 在\"{target_event}\"期间（{start:.1f}s-{end:.1f}s），{track_target}的运动轨迹：{track_gt}\n"
        "</think>\n"
        f"<answer>{track_gt}</answer>"
    )

    return {
        "video": video,
        "question": question,
        "trajectory": trajectory,
        "gt_answer": track_gt,
        "verifiable": False,
        "type": "multistep",
        "subtypes": ["temporal_locate", "tracking_overlay"],
        "source": f"{loc_item.get('source', 'unknown')}+{track_item.get('source', 'unknown')}",
        "params": {"target_event": target_event, "target_object": track_target,
                   "time": f"{start:.1f}-{end:.1f}"},
        "difficulty": "medium",
        "n_observe": 2,
    }


# ============================================================================
# 规则组合生成器
# ============================================================================

def generate_rule_based(n: int, seed: int = 42) -> list[dict]:
    """从 Stage 1 数据中按规则组合生成多步推理样本。

    分配比例：
      M-1 (loc→clip):    25%
      M-2 (loc→detect):  20%
      M-3 (detect→crop): 20%
      M-4 (detect→depth):15%
      M-5 (loc→det→crop):10%  (hard)
      M-6 (loc→track):   10%
    """
    random.seed(seed)
    print("[Stage2-Rule] 加载 Stage 1 数据...")
    by_type = _load_stage1_by_type()

    # 检查各类型数据量
    type_counts = {k: len(v) for k, v in by_type.items()}
    print(f"  各类型数据量: {type_counts}")

    loc_items = by_type.get("temporal_locate", [])
    clip_items = by_type.get("temporal_clip", [])
    detect_items = by_type.get("spatial_detect", [])
    crop_items = by_type.get("spatial_crop", [])
    depth_items = by_type.get("depth_overlay", [])
    track_items = by_type.get("tracking_overlay", [])

    # 按比例分配
    n_m1 = int(n * 0.25)
    n_m2 = int(n * 0.20)
    n_m3 = int(n * 0.20)
    n_m4 = int(n * 0.15)
    n_m5 = int(n * 0.10)
    n_m6 = n - n_m1 - n_m2 - n_m3 - n_m4 - n_m5

    results = []

    # M-1: temporal_locate → temporal_clip
    print(f"  [M-1] 生成 {n_m1} 条 (loc→clip)...")
    random.shuffle(loc_items)
    random.shuffle(clip_items)
    for i in tqdm(range(min(n_m1, len(loc_items))), desc="M-1"):
        # 同视频优先匹配，否则随机配对（clip 描述作为泛化）
        loc = loc_items[i % len(loc_items)]
        # 尝试找同视频的 clip
        same_video_clips = [c for c in clip_items if c["video"] == loc["video"]]
        clip = random.choice(same_video_clips) if same_video_clips else random.choice(clip_items)
        item = _combine_m1(loc, clip)
        if item:
            results.append(item)

    # M-2: temporal_locate → spatial_detect
    print(f"  [M-2] 生成 {n_m2} 条 (loc→detect)...")
    for i in tqdm(range(min(n_m2, len(loc_items))), desc="M-2"):
        loc = loc_items[i % len(loc_items)]
        det = random.choice(detect_items) if detect_items else None
        if det:
            item = _combine_m2(loc, det)
            if item:
                results.append(item)

    # M-3: spatial_detect → spatial_crop
    print(f"  [M-3] 生成 {n_m3} 条 (detect→crop)...")
    random.shuffle(detect_items)
    for i in tqdm(range(min(n_m3, len(detect_items))), desc="M-3"):
        det = detect_items[i % len(detect_items)]
        # 同视频优先
        same_video_crops = [c for c in crop_items if c["video"] == det["video"]]
        crop = random.choice(same_video_crops) if same_video_crops else (
            random.choice(crop_items) if crop_items else None
        )
        if crop:
            item = _combine_m3(det, crop)
            if item:
                results.append(item)

    # M-4: spatial_detect → depth_overlay
    print(f"  [M-4] 生成 {n_m4} 条 (detect→depth)...")
    for i in tqdm(range(min(n_m4, len(detect_items))), desc="M-4"):
        det = detect_items[i % len(detect_items)]
        # 同视频优先
        same_video_depths = [d for d in depth_items if d["video"] == det["video"]]
        depth = random.choice(same_video_depths) if same_video_depths else (
            random.choice(depth_items) if depth_items else None
        )
        if depth:
            item = _combine_m4(det, depth)
            if item:
                results.append(item)

    # M-5: temporal_locate → spatial_detect → spatial_crop (3步)
    print(f"  [M-5] 生成 {n_m5} 条 (loc→detect→crop, hard)...")
    for i in tqdm(range(min(n_m5, len(loc_items))), desc="M-5"):
        loc = loc_items[i % len(loc_items)]
        det = random.choice(detect_items) if detect_items else None
        crop = random.choice(crop_items) if crop_items else None
        if det and crop:
            item = _combine_m5(loc, det, crop)
            if item:
                results.append(item)

    # M-6: temporal_locate → tracking_overlay
    print(f"  [M-6] 生成 {n_m6} 条 (loc→track)...")
    for i in tqdm(range(min(n_m6, len(loc_items))), desc="M-6"):
        loc = loc_items[i % len(loc_items)]
        track = random.choice(track_items) if track_items else None
        if track:
            item = _combine_m6(loc, track)
            if item:
                results.append(item)

    random.shuffle(results)
    print(f"[Stage2-Rule] 共生成 {len(results)} 条多步推理样本")
    print(f"  Medium(2步): {sum(1 for r in results if r['difficulty'] == 'medium')}")
    print(f"  Hard(3步):   {sum(1 for r in results if r['difficulty'] == 'hard')}")
    return results


# ============================================================================
# LLM 生成多步推理（需要 Qwen3-32B）
# ============================================================================

_LLM = None
_TOKENIZER = None


def _load_llm(dtype: str = "bfloat16"):
    """Lazy load Qwen3-32B"""
    global _LLM, _TOKENIZER
    if _LLM is not None:
        return _LLM, _TOKENIZER
    if not QWEN3_32B_PATH.exists():
        raise FileNotFoundError(
            f"Qwen3-32B 路径不存在: {QWEN3_32B_PATH}\n"
            "请下载到该路径，或在 configs/paths.yaml 中调整。"
        )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    print(f"[LLM] 加载 Qwen3-32B ({dtype}) ...", flush=True)
    _TOKENIZER = AutoTokenizer.from_pretrained(str(QWEN3_32B_PATH), trust_remote_code=True)
    _LLM = AutoModelForCausalLM.from_pretrained(
        str(QWEN3_32B_PATH),
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    _LLM.eval()
    return _LLM, _TOKENIZER


def _llm_generate(prompt: str, max_new_tokens: int = 768,
                  temperature: float = 0.7, top_p: float = 0.9) -> str:
    import torch
    model, tok = _load_llm()
    messages = [
        {"role": "system",
         "content": "你是严谨的视频QA数据标注员。你需要为视频问答任务生成多步推理轨迹。"},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


PROMPT_MULTISTEP = """你是视频QA数据标注员。给定一个视频的标注信息，生成一个需要**多步推理**的复杂问题和对应的推理轨迹。

视频信息:
- 视频路径: {video}
- 已知标注:
{annotations}

要求:
1. 生成一个需要 {n_steps} 步观察才能回答的复杂问题
2. 每步使用不同的工具（observe type），可选: temporal_locate, temporal_clip, spatial_detect, spatial_crop, depth_overlay, tracking_overlay, ocr_zoom
3. 每步的推理必须基于前一步的结果
4. 严格按以下格式输出:

QUESTION: (复杂问题，需要多步推理才能回答)
---
<think>
[分析] (分析问题，说明需要哪些步骤)
<observe type="TYPE1" 参数.../>
<result>(第一步观察结果)</result>
[推理] (基于第一步结果，说明下一步需要什么)
<observe type="TYPE2" 参数.../>
<result>(第二步观察结果)</result>
{extra_steps}[结论] (综合所有观察结果得出最终结论)
</think>
<answer>(最终答案)</answer>

规则:
- 每个 observe 的 type 必须不同（除非确实需要重复）
- 参数必须合理（frame 用秒数，bbox 用 [x1,y1,x2,y2]）
- result 写观察到的事实，不是推理结论
- 问题必须自然、有意义，不能是拼凑的
"""


def _build_annotation_text(items: list[dict]) -> str:
    """从同一视频的多条标注中构建描述文本"""
    lines = []
    for item in items[:5]:  # 最多取5条
        t = item.get("type", "raw")
        q = item.get("question", "")
        gt = item.get("gt_answer", "")
        lines.append(f"  - [{t}] Q: {q} → A: {gt}")
    return "\n".join(lines)


def generate_llm_based(n: int, seed: int = 42) -> list[dict]:
    """使用 Qwen3-32B 生成多步推理数据。

    策略：从 Stage 1 数据中找同一视频有多种标注的样本，
    让 LLM 基于这些标注生成需要多步推理的复杂问题。
    """
    random.seed(seed)
    print("[Stage2-LLM] 加载 Stage 1 数据...")
    by_type = _load_stage1_by_type()

    # 按视频分组
    by_video: dict[str, list[dict]] = {}
    for items in by_type.values():
        for item in items:
            v = item["video"]
            by_video.setdefault(v, []).append(item)

    # 筛选有多种标注的视频（至少2种不同type）
    rich_videos = []
    for video, items in by_video.items():
        types = set(item["type"] for item in items)
        if len(types) >= 2:
            rich_videos.append((video, items))

    print(f"  有多种标注的视频: {len(rich_videos)} 个")
    if not rich_videos:
        print("  [!] 没有足够的多标注视频，跳过 LLM 生成")
        return []

    random.shuffle(rich_videos)
    results = []
    errors = 0

    for i in tqdm(range(min(n, len(rich_videos))), desc="Stage2-LLM"):
        video, items = rich_videos[i % len(rich_videos)]
        types = set(item["type"] for item in items)
        n_steps = min(len(types), random.choice([2, 2, 2, 3, 3]))  # 偏向2步

        annotations = _build_annotation_text(items)
        extra_steps = ""
        if n_steps >= 3:
            extra_steps = "[推理] (基于前两步结果，说明第三步需要什么)\n<observe type=\"TYPE3\" 参数.../>\n<result>(第三步观察结果)</result>\n"

        prompt = PROMPT_MULTISTEP.format(
            video=video,
            annotations=annotations,
            n_steps=n_steps,
            extra_steps=extra_steps,
        )

        try:
            raw = _llm_generate(prompt)
            # 解析输出
            question, trajectory = _parse_llm_output(raw)
            if question and trajectory:
                # 验证格式
                from utils.parser import parse_observe, parse_result, split_segments
                obs = parse_observe(trajectory)
                res = parse_result(trajectory)
                if len(obs) >= 2 and len(obs) == len(res):
                    results.append({
                        "video": video,
                        "question": question,
                        "trajectory": trajectory,
                        "gt_answer": _extract_answer(trajectory),
                        "verifiable": False,
                        "type": "multistep",
                        "subtypes": [o.type for o in obs],
                        "source": "llm_generated",
                        "params": {},
                        "difficulty": "hard" if len(obs) >= 3 else "medium",
                        "n_observe": len(obs),
                    })
                else:
                    errors += 1
            else:
                errors += 1
        except Exception:
            errors += 1

    print(f"[Stage2-LLM] 生成 {len(results)} 条，失败 {errors} 条")
    return results


def _parse_llm_output(raw: str) -> tuple[str, str]:
    """解析 LLM 输出，提取 question 和 trajectory"""
    # 找 QUESTION: 行
    question = ""
    lines = raw.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("QUESTION:"):
            question = line.strip()[len("QUESTION:"):].strip()
            break

    # 找 <think>...</think><answer>...</answer>
    think_start = raw.find("<think>")
    answer_end = raw.rfind("</answer>")
    if think_start >= 0 and answer_end >= 0:
        trajectory = raw[think_start:answer_end + len("</answer>")]
    else:
        trajectory = ""

    return question, trajectory


def _extract_answer(trajectory: str) -> str:
    """从 trajectory 中提取 <answer>...</answer>"""
    from utils.parser import parse_answer
    return parse_answer(trajectory) or ""


# ============================================================================
# 合并 Stage 1 + Stage 2
# ============================================================================

def merge_all():
    """合并 Stage 1 单步 + Stage 2 多步数据，生成最终训练集。

    输出:
      stage2_sft_all.jsonl — 包含 easy(单步) + medium(2步) + hard(3+步)
      stage2_sft_curriculum.json — 课程学习配置
    """
    ensure_dirs()
    STAGE2_DATA_PATH.mkdir(parents=True, exist_ok=True)

    # 加载 Stage 1（作为 easy）
    stage1_path = SFT_DATA_PATH / "stage1_sft_template_all.jsonl"
    if not stage1_path.exists():
        raise FileNotFoundError(f"Stage 1 数据不存在: {stage1_path}")
    stage1_items = _load_jsonl(stage1_path)
    for item in stage1_items:
        item.setdefault("difficulty", "easy")
        item.setdefault("n_observe", 1)

    # 加载 Stage 2 规则生成
    rule_path = STAGE2_DATA_PATH / "stage2_rule_multistep.jsonl"
    rule_items = _load_jsonl(rule_path) if rule_path.exists() else []

    # 加载 Stage 2 LLM 生成
    llm_path = STAGE2_DATA_PATH / "stage2_llm_multistep.jsonl"
    llm_items = _load_jsonl(llm_path) if llm_path.exists() else []

    # 合并
    all_items = stage1_items + rule_items + llm_items
    random.shuffle(all_items)

    # 统计
    easy = sum(1 for x in all_items if x.get("difficulty") == "easy")
    medium = sum(1 for x in all_items if x.get("difficulty") == "medium")
    hard = sum(1 for x in all_items if x.get("difficulty") == "hard")
    total = len(all_items)

    print(f"[Stage2-Merge] 合并完成:")
    print(f"  Easy(1步):   {easy} ({100*easy/total:.1f}%)")
    print(f"  Medium(2步): {medium} ({100*medium/total:.1f}%)")
    print(f"  Hard(3+步):  {hard} ({100*hard/total:.1f}%)")
    print(f"  总计:        {total}")

    # 写出合并数据
    out_path = STAGE2_DATA_PATH / "stage2_sft_all.jsonl"
    _write_jsonl(all_items, out_path)
    print(f"  -> {out_path}")

    # 写出课程学习配置
    curriculum = {
        "description": "Stage 2 课程学习配置",
        "total_samples": total,
        "difficulty_distribution": {
            "easy": easy,
            "medium": medium,
            "hard": hard,
        },
        "phases": [
            {
                "name": "Phase 2.1 (前1/3 steps)",
                "sampling_weights": {"easy": 0.70, "medium": 0.25, "hard": 0.05},
            },
            {
                "name": "Phase 2.2 (中1/3 steps)",
                "sampling_weights": {"easy": 0.40, "medium": 0.40, "hard": 0.20},
            },
            {
                "name": "Phase 2.3 (后1/3 steps)",
                "sampling_weights": {"easy": 0.30, "medium": 0.35, "hard": 0.35},
            },
        ],
    }
    curriculum_path = STAGE2_DATA_PATH / "stage2_sft_curriculum.json"
    with open(curriculum_path, "w", encoding="utf-8") as f:
        json.dump(curriculum, f, ensure_ascii=False, indent=2)
    print(f"  -> {curriculum_path}")


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 多步推理训练数据生成"
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["rule", "llm", "merge"],
        help="生成模式: rule=规则组合, llm=LLM生成, merge=合并所有"
    )
    parser.add_argument("--n", type=int, default=5000,
                        help="生成数量（rule/llm 模式）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径（默认自动）")
    args = parser.parse_args()

    ensure_dirs()
    STAGE2_DATA_PATH.mkdir(parents=True, exist_ok=True)

    if args.mode == "rule":
        items = generate_rule_based(n=args.n, seed=args.seed)
        out_path = Path(args.output) if args.output else (
            STAGE2_DATA_PATH / "stage2_rule_multistep.jsonl"
        )
        _write_jsonl(items, out_path)
        print(f"[OK] 规则组合: {len(items)} -> {out_path}")

    elif args.mode == "llm":
        items = generate_llm_based(n=args.n, seed=args.seed)
        out_path = Path(args.output) if args.output else (
            STAGE2_DATA_PATH / "stage2_llm_multistep.jsonl"
        )
        _write_jsonl(items, out_path)
        print(f"[OK] LLM生成: {len(items)} -> {out_path}")

    elif args.mode == "merge":
        merge_all()


if __name__ == "__main__":
    main()
