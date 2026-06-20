"""
raw pipeline: 不做特殊预处理，原视频原样返回。
"""
from pathlib import Path


def raw_pipeline(
    video_path: str,
    target: str = "",
    time=None,
    frame=None,
    bbox=None,
    objects=None,
) -> dict:
    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"视频不存在: {src}")

    target_str = target if target else "视频内容"
    q = f"请观看完整视频并回答关于「{target_str}」的问题。"
    return {"video": str(src), "perception_question": q}
