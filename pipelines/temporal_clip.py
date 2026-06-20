"""
temporal_clip pipeline
- 输入: 视频 + 时间区间 (例如 "3.0-5.0")
- 处理: ffmpeg 裁切 [start-1s, end+1s], 输出更高帧率的小段
- 输出: 裁切后视频路径 + perception_question
"""
from pathlib import Path

from utils.video_utils import save_video_clip, parse_time_range
from ._common import get_cache_dir, hash_key


def temporal_clip_pipeline(
    video_path: str,
    target: str = "",
    time=None,
    frame=None,
    bbox=None,
    objects=None,
) -> dict:
    if time is None:
        raise ValueError("temporal_clip 需要参数 time (例 '3.0-5.0')")

    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"视频不存在: {src}")

    start_sec, end_sec = parse_time_range(time)
    if end_sec <= start_sec:
        raise ValueError(f"非法 time 区间: {time}")

    # 上下文 padding 1s
    ext_start = max(0.0, start_sec - 1.0)
    ext_end = end_sec + 1.0

    cache = get_cache_dir("temporal_clip")
    out_path = cache / f"{src.stem}_{hash_key(src, ext_start, ext_end)}.mp4"
    if not out_path.exists():
        save_video_clip(
            video_path=str(src),
            output_path=str(out_path),
            start_sec=ext_start,
            end_sec=ext_end,
            target_fps=8.0,
        )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"temporal_clip 输出失败: {out_path}")

    q = (
        f'这是从原视频 {start_sec:.1f}s-{end_sec:.1f}s 截出的片段（带前后1s上下文）。'
        f'请仔细观察并描述其中"{target}"相关的内容。'
    )
    return {"video": str(out_path), "perception_question": q}
