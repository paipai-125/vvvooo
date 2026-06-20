"""
视频处理基础工具函数。
依赖: opencv-python, numpy
"""
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def load_video(video_path: str, max_frames: int = 128, fps: float = 2.0) -> np.ndarray:
    """
    加载视频为numpy数组。
    
    Args:
        video_path: 视频文件路径
        max_frames: 最大帧数
        fps: 采样帧率
    
    Returns:
        np.ndarray of shape (N, H, W, 3), dtype uint8, RGB格式
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    # 计算采样帧的索引
    target_n_frames = min(int(duration * fps), max_frames)
    if target_n_frames <= 0:
        target_n_frames = min(total_frames, max_frames)

    indices = np.linspace(0, total_frames - 1, target_n_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"视频读取失败(0帧): {video_path}")

    return np.stack(frames)


def get_frame_at_time(video_path: str, time_sec: float) -> np.ndarray:
    """
    获取视频指定时刻的帧。
    
    Args:
        video_path: 视频路径
        time_sec: 时间（秒）
    
    Returns:
        np.ndarray (H, W, 3), RGB
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"无法读取视频 {video_path} 的第{frame_idx}帧(time={time_sec}s)")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def save_video_clip(
    video_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
    target_fps: Optional[float] = None,
) -> str:
    """
    使用ffmpeg裁切视频片段。
    
    Args:
        video_path: 原视频路径
        output_path: 输出路径
        start_sec: 开始时间（秒）
        end_sec: 结束时间（秒）
        target_fps: 目标帧率（None则保持原fps）
    
    Returns:
        输出文件路径
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(end_sec - start_sec),
        "-c:v", "libx264",
        "-preset", "fast",
    ]
    if target_fps is not None:
        cmd.extend(["-r", str(target_fps)])
    cmd.extend(["-an", str(output_path)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg裁切失败:\n命令: {' '.join(cmd)}\n错误: {result.stderr}"
        )

    return str(output_path)


def parse_time_range(time_str: str) -> Tuple[float, float]:
    """
    解析时间范围字符串。
    支持格式: "3.0-5.0", "3:00-4:30", "3.0s-5.0s"
    
    Returns:
        (start_sec, end_sec)
    """
    time_str = time_str.strip().replace("s", "")
    parts = time_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"无法解析时间范围: {time_str}")
    return _parse_single_time(parts[0]), _parse_single_time(parts[1])


def _parse_single_time(t: str) -> float:
    """解析单个时间字符串为秒数"""
    t = t.strip()
    if ":" in t:
        parts = t.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return float(t)
