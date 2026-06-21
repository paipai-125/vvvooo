"""
可视化 Teacher_P 输入 —— 真实调用 pipelines/ 下的工具函数生成工具产物。

  学生看到什么 → observe 触发什么 → 工具产物（Teacher_P 的视觉输入）→ Teacher_P 的文本问题

用法:
    # 真实调用（需要 GPU，加载 Grounding-DINO / Depth-Anything / SAM3）
    python -m tools.visualize_teacher_input --subset all --n 5

    # 仅跑不需要 GPU 的子集
    python -m tools.visualize_teacher_input --subset a1,a2,a4,a7 --n 5

    # 逐个试（先跑不需要 GPU 的，再跑需要 GPU 的）
    python -m tools.visualize_teacher_input --subset a3 --n 3

输出目录: {OUTPUT_ROOT}/visualize_teacher_input/
每个样本生成一个子目录，包含:
  - info.json            : 完整元信息（question, trajectory, gt_answer, params 等）
  - student_input.jpg    : 学生看到的视频帧（或首帧截图代表视频）
  - tool_output.*        : 工具产物（给 Teacher_P 的视觉输入，jpg 或 mp4）
  - teacher_prompt.txt   : Teacher_P 接收的完整文本 prompt

本脚本**真实调用** pipelines/ 下的 pipeline 函数：
  - A-1 temporal_locate  → temporal_locate_pipeline()  [纯 OpenCV, 无需 GPU]
  - A-2 temporal_clip    → temporal_clip_pipeline()    [ffmpeg, 无需 GPU]
  - A-3 spatial_detect   → spatial_detect_pipeline()   [Grounding-DINO, 需 GPU]
  - A-4 spatial_crop     → spatial_crop_pipeline()     [纯 PIL, 无需 GPU]
  - A-5 tracking_overlay → tracking_overlay_pipeline() [SAM3, 需 GPU]
  - A-6 depth_overlay    → depth_overlay_pipeline()    [Depth-Anything + G-DINO, 需 GPU]
  - A-7 ocr_zoom         → ocr_zoom_pipeline()         [纯 PIL, 无需 GPU]
  - A-8 raw              → 无工具调用
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# 项目路径
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.paths import OUTPUT_ROOT, SFT_DATA_PATH  # noqa: E402
from utils.video_utils import get_frame_at_time, parse_time_range  # noqa: E402

# 真实 pipeline 导入
from pipelines import (  # noqa: E402
    temporal_locate_pipeline,
    temporal_clip_pipeline,
    spatial_detect_pipeline,
    spatial_crop_pipeline,
    tracking_overlay_pipeline,
    depth_overlay_pipeline,
    ocr_zoom_pipeline,
)

# 输出根目录
VIS_OUTPUT_DIR = OUTPUT_ROOT / "visualize_teacher_input"

# 子集名称映射
SUBSET_FILE_MAP = {
    "a1": "stage1_sft_a1_temporal_locate.jsonl",
    "a2": "stage1_sft_a2_temporal_clip.jsonl",
    "a3": "stage1_sft_a3_spatial_detect.jsonl",
    "a4": "stage1_sft_a4_spatial_crop.jsonl",
    "a5": "stage1_sft_a5_tracking_describe.jsonl",
    "a6": "stage1_sft_a6_depth_overlay.jsonl",
    "a7": "stage1_sft_a7_ocr_zoom.jsonl",
    "a8": "stage1_sft_a8_raw_videoqa.jsonl",
}

# 需要 GPU 的子集
GPU_SUBSETS = {"a3", "a5", "a6"}


def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _get_video_first_frame(video_path: str) -> Optional[np.ndarray]:
    """获取视频第一帧 (RGB)，失败返回 None"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def _save_rgb_as_jpg(img_rgb: np.ndarray, path: Path, quality: int = 90):
    """保存 RGB numpy 数组为 JPG"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])


def _add_text_overlay(img_rgb: np.ndarray, text: str, position: str = "top") -> np.ndarray:
    """在图像顶部/底部添加文字条"""
    h, w = img_rgb.shape[:2]
    bar_h = max(40, h // 15)
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    font_scale = max(0.4, w / 1500.0)
    display_text = text[:150] + ("..." if len(text) > 150 else "")
    cv2.putText(bar, display_text, (10, bar_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    if position == "top":
        return np.vstack([bar, img_rgb])
    else:
        return np.vstack([img_rgb, bar])


def _copy_or_snapshot(tool_output_path: str, out_dir: Path) -> Path:
    """
    将 pipeline 产物复制到可视化目录。
    如果是视频(.mp4)，额外截取中间帧作为 tool_output_snapshot.jpg。
    返回复制后的路径。
    """
    src = Path(tool_output_path)
    if not src.exists():
        raise FileNotFoundError(f"Pipeline 产物不存在: {src}")

    suffix = src.suffix.lower()
    dst = out_dir / f"tool_output{suffix}"
    shutil.copy2(str(src), str(dst))

    # 如果是视频，额外截取中间帧
    if suffix in (".mp4", ".avi", ".mkv"):
        cap = cv2.VideoCapture(str(src))
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mid = total // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            cap.release()
            if ret:
                snapshot_path = out_dir / "tool_output_snapshot.jpg"
                cv2.imwrite(str(snapshot_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return dst


# ============================================================
# 各工具的可视化函数 —— 真实调用 pipeline
# ============================================================

def vis_a1_temporal_locate(sample: dict, out_dir: Path) -> bool:
    """A-1: 真实调用 temporal_locate_pipeline()
    
    pipeline 做的事: 在视频每帧叠加时间戳 + 底部进度条，输出新视频。
    不需要 GPU。
    """
    video_path = sample["video"]
    params = sample.get("params", {})
    target = params.get("target", "")

    if not Path(video_path).exists():
        print(f"  [跳过] 视频不存在: {video_path}")
        return False

    # 学生输入
    first_frame = _get_video_first_frame(video_path)
    if first_frame is None:
        print(f"  [跳过] 无法读取视频: {video_path}")
        return False
    student_frame = _add_text_overlay(first_frame, f"Q: {sample['question']}")
    _save_rgb_as_jpg(student_frame, out_dir / "student_input.jpg")

    # 真实调用 pipeline
    result = temporal_locate_pipeline(
        video_path=video_path,
        target=target,
    )

    # 复制产物
    _copy_or_snapshot(result["video"], out_dir)

    # Teacher_P prompt
    (out_dir / "teacher_prompt.txt").write_text(
        f"=== Pipeline 返回 ===\n"
        f"video: {result['video']}\n"
        f"perception_question: {result['perception_question']}\n"
        f"\n=== Teacher_P 接收的输入 ===\n"
        f"视觉输入: 叠加时间轴标注的完整视频\n"
        f"文本输入: {result['perception_question']}\n"
        f"\n=== GT Answer ===\n"
        f"{sample.get('gt_answer', '')}\n",
        encoding="utf-8",
    )
    return True


def vis_a2_temporal_clip(sample: dict, out_dir: Path) -> bool:
    """A-2: 真实调用 temporal_clip_pipeline()
    
    pipeline 做的事: ffmpeg 裁切 [start-1s, end+1s] 视频片段。
    不需要 GPU。
    """
    video_path = sample["video"]
    params = sample.get("params", {})
    time_str = params.get("time", "0.0-5.0")
    target = params.get("target", "")

    if not Path(video_path).exists():
        print(f"  [跳过] 视频不存在: {video_path}")
        return False

    # 学生输入
    first_frame = _get_video_first_frame(video_path)
    if first_frame is None:
        print(f"  [跳过] 无法读取视频: {video_path}")
        return False
    student_frame = _add_text_overlay(first_frame, f"Q: {sample['question']}")
    _save_rgb_as_jpg(student_frame, out_dir / "student_input.jpg")

    # 真实调用 pipeline
    result = temporal_clip_pipeline(
        video_path=video_path,
        target=target,
        time=time_str,
    )

    # 复制产物
    _copy_or_snapshot(result["video"], out_dir)

    # Teacher_P prompt
    (out_dir / "teacher_prompt.txt").write_text(
        f"=== Pipeline 返回 ===\n"
        f"video: {result['video']}\n"
        f"perception_question: {result['perception_question']}\n"
        f"\n=== Teacher_P 接收的输入 ===\n"
        f"视觉输入: 裁切后的视频片段 (time={time_str}, ±1s 上下文)\n"
        f"文本输入: {result['perception_question']}\n"
        f"\n=== GT Answer ===\n"
        f"{sample.get('gt_answer', '')}\n",
        encoding="utf-8",
    )
    return True


def vis_a3_spatial_detect(sample: dict, out_dir: Path) -> bool:
    """A-3: 真实调用 spatial_detect_pipeline()
    
    pipeline 做的事: 抽指定帧 → Grounding-DINO 检测所有候选物体 → 画框输出图像。
    需要 GPU (Grounding-DINO)。
    """
    video_path = sample["video"]
    params = sample.get("params", {})
    frame_time = params.get("frame", "0.0")
    target = params.get("target", "")

    if not Path(video_path).exists():
        print(f"  [跳过] 视频不存在: {video_path}")
        return False

    # 学生输入
    first_frame = _get_video_first_frame(video_path)
    if first_frame is None:
        print(f"  [跳过] 无法读取视频: {video_path}")
        return False
    student_frame = _add_text_overlay(first_frame, f"Q: {sample['question']}")
    _save_rgb_as_jpg(student_frame, out_dir / "student_input.jpg")

    # 真实调用 pipeline
    result = spatial_detect_pipeline(
        video_path=video_path,
        target=target,
        frame=frame_time,
    )

    # 复制产物（jpg 图像）
    _copy_or_snapshot(result["video"], out_dir)

    # Teacher_P prompt
    (out_dir / "teacher_prompt.txt").write_text(
        f"=== Pipeline 返回 ===\n"
        f"video: {result['video']}\n"
        f"perception_question: {result['perception_question']}\n"
        f"\n=== Teacher_P 接收的输入 ===\n"
        f"视觉输入: 帧@{frame_time}s + Grounding-DINO 候选框叠加\n"
        f"文本输入: {result['perception_question']}\n"
        f"\n=== GT Answer ===\n"
        f"{sample.get('gt_answer', '')}\n",
        encoding="utf-8",
    )
    return True


def vis_a4_spatial_crop(sample: dict, out_dir: Path) -> bool:
    """A-4: 真实调用 spatial_crop_pipeline()
    
    pipeline 做的事: 抽帧 → bbox 裁切 → 扩大20%边距 → Lanczos 放大到长边768。
    不需要 GPU。
    """
    video_path = sample["video"]
    params = sample.get("params", {})
    frame_time = params.get("frame", "0.0")
    bbox_str = params.get("bbox", "[0,0,100,100]")
    target = params.get("target", "")

    if not Path(video_path).exists():
        print(f"  [跳过] 视频不存在: {video_path}")
        return False

    # 学生输入
    first_frame = _get_video_first_frame(video_path)
    if first_frame is None:
        print(f"  [跳过] 无法读取视频: {video_path}")
        return False
    student_frame = _add_text_overlay(first_frame, f"Q: {sample['question']}")
    _save_rgb_as_jpg(student_frame, out_dir / "student_input.jpg")

    # 真实调用 pipeline
    result = spatial_crop_pipeline(
        video_path=video_path,
        target=target,
        frame=frame_time,
        bbox=bbox_str,
    )

    # 复制产物（jpg 图像）
    _copy_or_snapshot(result["video"], out_dir)

    # 额外保存原帧带框标注（方便对比）
    try:
        target_frame = get_frame_at_time(video_path, float(frame_time))
        bbox = json.loads(bbox_str) if isinstance(bbox_str, str) else bbox_str
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        img_bgr = cv2.cvtColor(target_frame, cv2.COLOR_RGB2BGR)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img_bgr, "crop region", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / "context_frame_with_bbox.jpg"), img_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    except Exception:
        pass

    # Teacher_P prompt
    (out_dir / "teacher_prompt.txt").write_text(
        f"=== Pipeline 返回 ===\n"
        f"video: {result['video']}\n"
        f"perception_question: {result['perception_question']}\n"
        f"\n=== Teacher_P 接收的输入 ===\n"
        f"视觉输入: 帧@{frame_time}s 的 {bbox_str} 区域裁切放大图\n"
        f"文本输入: {result['perception_question']}\n"
        f"\n=== GT Answer ===\n"
        f"{sample.get('gt_answer', '')}\n",
        encoding="utf-8",
    )
    return True


def vis_a5_tracking_overlay(sample: dict, out_dir: Path) -> bool:
    """A-5: 真实调用 tracking_overlay_pipeline()
    
    pipeline 做的事: 裁切时间段 → SAM3 文本 prompt 做 detect+segment+track → 每帧 mask 叠加。
    需要 GPU (SAM3)。
    """
    video_path = sample["video"]
    params = sample.get("params", {})
    time_str = params.get("time", "0.0-5.0")
    target = params.get("target", "")

    if not Path(video_path).exists():
        print(f"  [跳过] 视频不存在: {video_path}")
        return False

    # 学生输入
    first_frame = _get_video_first_frame(video_path)
    if first_frame is None:
        print(f"  [跳过] 无法读取视频: {video_path}")
        return False
    student_frame = _add_text_overlay(first_frame, f"Q: {sample['question']}")
    _save_rgb_as_jpg(student_frame, out_dir / "student_input.jpg")

    print("真实调用tracking_overlay_pipeline")
    # 真实调用 pipeline
    result = tracking_overlay_pipeline(
        video_path=video_path,
        target=target,
        time=time_str,
    )
    print(result)

    # 复制产物（mp4 视频 + 截取中间帧快照）
    _copy_or_snapshot(result["video"], out_dir)

    # Teacher_P prompt
    (out_dir / "teacher_prompt.txt").write_text(
        f"=== Pipeline 返回 ===\n"
        f"video: {result['video']}\n"
        f"perception_question: {result['perception_question']}\n"
        f"\n=== Teacher_P 接收的输入 ===\n"
        f"视觉输入: SAM3 mask 追踪可视化视频 (time={time_str})\n"
        f"文本输入: {result['perception_question']}\n"
        f"\n=== GT Answer ===\n"
        f"{sample.get('gt_answer', '')}\n",
        encoding="utf-8",
    )
    return True


def vis_a6_depth_overlay(sample: dict, out_dir: Path) -> bool:
    """A-6: 真实调用 depth_overlay_pipeline()
    
    pipeline 做的事: 抽帧 → Depth-Anything V2 深度估计 → 热力图叠加 → Grounding-DINO 检测物体画框。
    需要 GPU (Depth-Anything + Grounding-DINO)。
    """
    video_path = sample["video"]
    params = sample.get("params", {})
    frame_time = params.get("frame", "0.0")
    objects_str = params.get("objects", "")
    target = params.get("target", "")

    if not Path(video_path).exists():
        print(f"  [跳过] 视频不存在: {video_path}")
        return False

    # 学生输入
    first_frame = _get_video_first_frame(video_path)
    if first_frame is None:
        print(f"  [跳过] 无法读取视频: {video_path}")
        return False
    student_frame = _add_text_overlay(first_frame, f"Q: {sample['question']}")
    _save_rgb_as_jpg(student_frame, out_dir / "student_input.jpg")

    # 真实调用 pipeline
    result = depth_overlay_pipeline(
        video_path=video_path,
        target=target,
        frame=frame_time,
        objects=objects_str,
    )

    # 复制产物（jpg 图像）
    _copy_or_snapshot(result["video"], out_dir)

    # Teacher_P prompt
    (out_dir / "teacher_prompt.txt").write_text(
        f"=== Pipeline 返回 ===\n"
        f"video: {result['video']}\n"
        f"perception_question: {result['perception_question']}\n"
        f"\n=== Teacher_P 接收的输入 ===\n"
        f"视觉输入: 帧@{frame_time}s + Depth-Anything V2 深度热力图 + Grounding-DINO 物体框\n"
        f"文本输入: {result['perception_question']}\n"
        f"\n=== GT Answer ===\n"
        f"{sample.get('gt_answer', '')}\n",
        encoding="utf-8",
    )
    return True


def vis_a7_ocr_zoom(sample: dict, out_dir: Path) -> bool:
    """A-7: 真实调用 ocr_zoom_pipeline()
    
    pipeline 做的事: 抽帧 → bbox 裁切 → Lanczos 放大到长边1024。
    不需要 GPU。
    """
    video_path = sample["video"]
    params = sample.get("params", {})
    frame_time = params.get("frame", "0.0")
    bbox_str = params.get("bbox", "[0,0,100,100]")
    target = params.get("target", "")

    if not Path(video_path).exists():
        print(f"  [跳过] 视频不存在: {video_path}")
        return False

    # 学生输入
    first_frame = _get_video_first_frame(video_path)
    if first_frame is None:
        print(f"  [跳过] 无法读取视频: {video_path}")
        return False
    student_frame = _add_text_overlay(first_frame, f"Q: {sample['question']}")
    _save_rgb_as_jpg(student_frame, out_dir / "student_input.jpg")

    # 真实调用 pipeline
    result = ocr_zoom_pipeline(
        video_path=video_path,
        target=target,
        frame=frame_time,
        bbox=bbox_str,
    )

    # 复制产物（jpg 图像）
    _copy_or_snapshot(result["video"], out_dir)

    # 额外保存原帧带框标注
    try:
        target_frame = get_frame_at_time(video_path, float(frame_time))
        bbox = json.loads(bbox_str) if isinstance(bbox_str, str) else bbox_str
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        img_bgr = cv2.cvtColor(target_frame, cv2.COLOR_RGB2BGR)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(img_bgr, "OCR region", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / "context_frame_with_bbox.jpg"), img_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    except Exception:
        pass

    # Teacher_P prompt
    (out_dir / "teacher_prompt.txt").write_text(
        f"=== Pipeline 返回 ===\n"
        f"video: {result['video']}\n"
        f"perception_question: {result['perception_question']}\n"
        f"\n=== Teacher_P 接收的输入 ===\n"
        f"视觉输入: 帧@{frame_time}s 的 {bbox_str} 区域 Lanczos 放大图\n"
        f"文本输入: {result['perception_question']}\n"
        f"\n=== GT Answer ===\n"
        f"{sample.get('gt_answer', '')}\n",
        encoding="utf-8",
    )
    return True


def vis_a8_raw_videoqa(sample: dict, out_dir: Path) -> bool:
    """A-8: 无工具调用，学生直接推理。"""
    video_path = sample["video"]

    if not Path(video_path).exists():
        print(f"  [跳过] 视频不存在: {video_path}")
        return False

    # 学生输入
    first_frame = _get_video_first_frame(video_path)
    if first_frame is None:
        print(f"  [跳过] 无法读取视频: {video_path}")
        return False
    student_frame = _add_text_overlay(first_frame, f"Q: {sample['question'][:120]}")
    _save_rgb_as_jpg(student_frame, out_dir / "student_input.jpg")

    # A-8 无工具产物
    h, w = first_frame.shape[:2]
    placeholder = np.zeros((max(h // 2, 200), w, 3), dtype=np.uint8)
    cv2.putText(placeholder, "A-8: No tool output (raw VideoQA)", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(placeholder, "Student directly reasons without observe", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "tool_output.jpg"), placeholder,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Teacher prompt
    (out_dir / "teacher_prompt.txt").write_text(
        f"=== A-8 raw_videoqa: 无 Teacher_P 参与 ===\n"
        f"学生直接观看视频并推理回答，不调用任何工具。\n"
        f"\n=== 问题 ===\n"
        f"{sample['question']}\n"
        f"\n=== GT Answer ===\n"
        f"{sample.get('gt_answer', '')}\n"
        f"\n=== 说明 ===\n"
        f"A-8 用于训练模型的基础视频理解能力。\n"
        f"多选题 verifiable=True，开放式 verifiable=False。\n"
        f"source: {sample.get('source', 'unknown')}\n",
        encoding="utf-8",
    )
    return True


# ============================================================
# 主逻辑
# ============================================================

VIS_FUNCS = {
    "a1": vis_a1_temporal_locate,
    "a2": vis_a2_temporal_clip,
    "a3": vis_a3_spatial_detect,
    "a4": vis_a4_spatial_crop,
    "a5": vis_a5_tracking_overlay,
    "a6": vis_a6_depth_overlay,
    "a7": vis_a7_ocr_zoom,
    "a8": vis_a8_raw_videoqa,
}


def process_subset(subset: str, n: int, seed: int):
    """处理单个子集的可视化"""
    jsonl_file = SFT_DATA_PATH / SUBSET_FILE_MAP[subset]
    if not jsonl_file.exists():
        print(f"[跳过] {subset}: 文件不存在 {jsonl_file}")
        return

    items = _load_jsonl(jsonl_file)
    if not items:
        print(f"[跳过] {subset}: 文件为空")
        return

    rng = random.Random(seed)
    rng.shuffle(items)

    vis_func = VIS_FUNCS[subset]
    subset_dir = VIS_OUTPUT_DIR / subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    max_attempts = min(len(items), n * 5)  # 最多尝试 5 倍数量

    gpu_tag = " [GPU]" if subset in GPU_SUBSETS else " [CPU]"
    print(f"\n{'='*60}")
    print(f"[{subset}]{gpu_tag} 从 {len(items)} 条中抽取 {n} 个样本，真实调用 pipeline")
    print(f"{'='*60}")

    for item in items[:max_attempts]:
        if success_count >= n:
            break

        sample_dir = subset_dir / f"sample_{success_count:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整元信息
        info = {
            "video": item.get("video", ""),
            "question": item.get("question", ""),
            "trajectory": item.get("trajectory", ""),
            "gt_answer": item.get("gt_answer", ""),
            "verifiable": item.get("verifiable", False),
            "type": item.get("type", ""),
            "source": item.get("source", ""),
            "params": item.get("params", {}),
        }
        (sample_dir / "info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        try:
            ok = vis_func(item, sample_dir)
            if ok:
                success_count += 1
                print(f"  [{success_count}/{n}] ✅ {sample_dir.name} "
                      f"(video={Path(item.get('video', '')).name})")
            else:
                shutil.rmtree(sample_dir, ignore_errors=True)
        except Exception as e:
            print(f"  [!] 样本处理失败: {e}")
            traceback.print_exc()
            shutil.rmtree(sample_dir, ignore_errors=True)
            # 如果是模型加载失败，直接终止该子集
            if "权重未找到" in str(e) or "FileNotFoundError" in type(e).__name__:
                print(f"  [!!] 模型加载失败，终止 {subset} 子集")
                break

    print(f"[{subset}] 完成: {success_count}/{n} 个样本 -> {subset_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="可视化 Teacher_P 输入 —— 真实调用 pipelines/ 生成工具产物"
    )
    parser.add_argument(
        "--subset", type=str, default="all",
        help="要可视化的子集，逗号分隔（如 a1,a3,a6）或 'all'。"
             "不需要 GPU 的: a1,a2,a4,a7,a8。需要 GPU 的: a3,a5,a6。"
    )
    parser.add_argument("--n", type=int, default=5,
                        help="每个子集抽取的样本数（默认 5）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-only", action="store_true",
                        help="仅运行不需要 GPU 的子集 (a1,a2,a4,a7,a8)")
    args = parser.parse_args()

    if args.subset == "all":
        subsets = list(SUBSET_FILE_MAP.keys())
    else:
        subsets = [s.strip() for s in args.subset.split(",")]
        for s in subsets:
            if s not in SUBSET_FILE_MAP:
                print(f"[错误] 未知子集: {s}，可选: {list(SUBSET_FILE_MAP.keys())}")
                sys.exit(1)

    if args.cpu_only:
        subsets = [s for s in subsets if s not in GPU_SUBSETS]
        print(f"[--cpu-only] 仅运行不需要 GPU 的子集: {subsets}")

    VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {VIS_OUTPUT_DIR}")
    print(f"子集: {subsets}")
    print(f"每子集样本数: {args.n}")
    print(f"需要 GPU 的子集: {[s for s in subsets if s in GPU_SUBSETS]}")
    print(f"不需要 GPU 的子集: {[s for s in subsets if s not in GPU_SUBSETS]}")

    for subset in subsets:
        process_subset(subset, args.n, args.seed)

    print(f"\n{'='*60}")
    print(f"全部完成！请检查输出目录:")
    print(f"  {VIS_OUTPUT_DIR}")
    print(f"{'='*60}")
    print(f"\n每个样本目录包含:")
    print(f"  - info.json              : 完整元信息")
    print(f"  - student_input.jpg      : 学生看到的输入（视频首帧+问题）")
    print(f"  - tool_output.*          : 工具产物（Teacher_P 的视觉输入，jpg/mp4）")
    print(f"  - tool_output_snapshot.jpg : (仅视频产物) 中间帧快照")
    print(f"  - context_frame_with_bbox.jpg : (仅 a4/a7) 原帧带框标注")
    print(f"  - teacher_prompt.txt     : Teacher_P 接收的完整文本 prompt")


if __name__ == "__main__":
    main()
