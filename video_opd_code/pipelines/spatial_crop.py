"""
spatial_crop pipeline
- 输入: 视频 + 帧时间 + bbox=[x1,y1,x2,y2]
- 处理: 抽帧后按 bbox 裁切并扩大 20% 边距
- 输出: 裁切后图像 + perception_question
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils.video_utils import get_frame_at_time
from ._common import get_cache_dir, hash_key


def _parse_bbox(bbox) -> list:
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return [float(x) for x in bbox]
    if isinstance(bbox, str):
        s = bbox.strip()
        if s.startswith("[") and s.endswith("]"):
            return [float(x) for x in json.loads(s)]
        parts = [p for p in s.replace(",", " ").split() if p]
        if len(parts) == 4:
            return [float(x) for x in parts]
    raise ValueError(f"无法解析 bbox: {bbox!r}")


def spatial_crop_pipeline(
    video_path: str,
    target: str = "",
    time=None,
    frame=None,
    bbox=None,
    objects=None,
) -> dict:
    if frame is None:
        raise ValueError("spatial_crop 需要参数 frame (秒)")
    if bbox is None:
        raise ValueError("spatial_crop 需要参数 bbox")

    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"视频不存在: {src}")

    t_sec = float(frame)
    box = _parse_bbox(bbox)

    img_rgb = get_frame_at_time(str(src), t_sec)
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        raise ValueError(f"非法 bbox: {box}")

    # 扩大 20% 边距
    pad_x, pad_y = bw * 0.2, bh * 0.2
    nx1 = int(max(0, x1 - pad_x))
    ny1 = int(max(0, y1 - pad_y))
    nx2 = int(min(w, x2 + pad_x))
    ny2 = int(min(h, y2 + pad_y))

    crop = img_rgb[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        raise RuntimeError(f"裁切结果为空，原 bbox={box}")

    # zoom: 长边放大到 768
    pil = Image.fromarray(crop)
    cw, ch = pil.size
    target_long = 768
    scale = target_long / max(cw, ch)
    if scale > 1.0:
        pil = pil.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)

    cache = get_cache_dir("spatial_crop")
    out_path = cache / f"{src.stem}_{hash_key(src, t_sec, box)}.jpg"
    pil.save(str(out_path), quality=95)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"spatial_crop 输出失败: {out_path}")

    target_str = target if target else "该区域"
    q = (
        f'这是原视频 {t_sec:.1f}s 帧的 [{int(x1)},{int(y1)},{int(x2)},{int(y2)}] '
        f'区域（扩大20%边距并放大）。请描述"{target_str}"的内容。'
    )
    return {"video": str(out_path), "perception_question": q}
