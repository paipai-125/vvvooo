"""
temporal_locate pipeline
- 输入: 完整视频
- 处理: 在视频上叠加均匀时间轴候选标记（每秒打一个时间戳），便于Teacher_P做时间定位
- 输出: 处理后视频路径 + 给Teacher_P的perception_question
"""
from pathlib import Path

import cv2
import numpy as np

from utils.video_utils import load_video
from ._common import get_cache_dir, hash_key


def temporal_locate_pipeline(
    video_path: str,
    target: str = "",
    time=None,
    frame=None,
    bbox=None,
    objects=None,
) -> dict:
    if not target:
        raise ValueError("temporal_locate 需要参数 target")

    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"视频不存在: {src}")

    cache = get_cache_dir("temporal_locate")
    out_path = cache / f"{src.stem}_{hash_key(src, target)}.mp4"
    if out_path.exists():
        return {
            "video": str(out_path),
            "perception_question": _build_question(target),
        }

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"视频元信息异常: {src}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法打开写入器: {out_path}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, height / 720.0)
    thickness = max(1, int(font_scale * 2))

    frame_idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        t_sec = frame_idx / fps
        text = f"t={t_sec:.1f}s"
        # 左上角时间戳
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, (0, 0), (tw + 10, th + 10), (0, 0, 0), -1)
        cv2.putText(img, text, (5, th + 5), font, font_scale,
                    (0, 255, 255), thickness, cv2.LINE_AA)
        # 底部进度条
        bar_y = height - 8
        progress = frame_idx / max(total_frames - 1, 1)
        cv2.line(img, (0, bar_y), (width, bar_y), (50, 50, 50), 4)
        cv2.line(img, (0, bar_y), (int(width * progress), bar_y),
                 (0, 255, 0), 4)
        # 每秒一个刻度
        if total_frames > 0 and fps > 0:
            duration = total_frames / fps
            for sec in range(int(duration) + 1):
                x = int(width * (sec / max(duration, 1e-6)))
                cv2.line(img, (x, bar_y - 6), (x, bar_y + 6),
                         (255, 255, 255), 1)
        writer.write(img)
        frame_idx += 1

    cap.release()
    writer.release()

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"temporal_locate 输出失败: {out_path}")

    return {
        "video": str(out_path),
        "perception_question": _build_question(target),
    }


def _build_question(target: str) -> str:
    return (
        f'请观察视频中叠加的时间戳和进度条，告诉我"{target}"出现的时间区间，'
        f'格式: start-end (秒)。'
    )
