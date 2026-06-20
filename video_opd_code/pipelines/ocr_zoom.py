"""
ocr_zoom pipeline
- 输入: 视频 + 帧时间 + bbox=[x1,y1,x2,y2]
- 处理: 抽帧 -> 按 bbox 裁切 -> Lanczos 放大 (长边到 1024) 以便 OCR/小字识别
- 输出: 处理后图像 + perception_question
"""
import json
from pathlib import Path

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


def ocr_zoom_pipeline(
    video_path: str,
    target: str = "",
    time=None,
    frame=None,
    bbox=None,
    objects=None,
) -> dict:
    if frame is None:
        raise ValueError("ocr_zoom 需要参数 frame (秒)")
    if bbox is None:
        raise ValueError("ocr_zoom 需要参数 bbox")

    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"视频不存在: {src}")

    t_sec = float(frame)
    box = _parse_bbox(bbox)

    img_rgb = get_frame_at_time(str(src), t_sec)
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"非法 bbox: {box}")

    pil = Image.fromarray(img_rgb).crop((x1, y1, x2, y2))
    cw, ch = pil.size
    target_long = 1024
    scale = target_long / max(cw, ch)
    if scale > 1.0:
        pil = pil.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)

    cache = get_cache_dir("ocr_zoom")
    out_path = cache / f"{src.stem}_{hash_key(src, t_sec, box)}.jpg"
    pil.save(str(out_path), quality=95)
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"ocr_zoom 输出失败: {out_path}")

    target_str = target if target else "图中文字内容"
    q = (
        f'这是原视频 {t_sec:.1f}s 帧的 [{x1},{y1},{x2},{y2}] 区域（已 Lanczos 放大）。'
        f'请识别{target_str}。'
    )
    return {"video": str(out_path), "perception_question": q}
