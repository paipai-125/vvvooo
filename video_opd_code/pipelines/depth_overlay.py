"""
depth_overlay pipeline
- 输入: 视频 + 帧时间 + objects（多个物体的 bbox 坐标，逗号分隔，如 "[x1,y1,x2,y2],[x3,y3,x4,y4]"）
- 处理: 对指定帧用 Depth-Anything V2 做深度估计，
        把深度热力图叠加到原图，再直接根据 objects 中的 bbox 坐标画每个物体的边界框。
- 输出: 处理后图像 + perception_question
"""
from pathlib import Path

import cv2
import numpy as np
import json
import re

from utils.video_utils import get_frame_at_time
from ._common import (
    get_cache_dir,
    hash_key,
    get_depth_anything,
)


def _parse_objects(objects) -> list:
    """
    解析 objects 参数，支持两种格式：
    1. 字符串: "[x1,y1,x2,y2],[x3,y3,x4,y4]" 或 "[x1,y1,x2,y2], [x3,y3,x4,y4]"
    2. 列表: [[x1,y1,x2,y2], [x3,y3,x4,y4]]
    返回: [[x1,y1,x2,y2], [x3,y3,x4,y4], ...]
    """
    if isinstance(objects, (list, tuple)):
        # 已经是列表格式，直接返回
        return [list(map(int, obj)) for obj in objects]
    if isinstance(objects, str):
        s = objects.strip()
        # 尝试直接解析为 JSON 列表的列表
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list) and len(parsed) > 0:
                return [list(map(int, obj)) for obj in parsed]
        except json.JSONDecodeError:
            pass
        # 手动解析 "[x1,y1,x2,y2], [x3,y3,x4,y4]" 格式
        matches = re.findall(r'\[([\d,\s]+)\]', s)
        bboxes = []
        for match in matches:
            coords = [int(x.strip()) for x in match.split(',') if x.strip()]
            if len(coords) == 4:
                bboxes.append(coords)
        if bboxes:
            return bboxes
        # 尝试解析单个 bbox
        parts = [int(x.strip()) for x in s.replace('[', '').replace(']', '').split(',') if x.strip()]
        if len(parts) == 4:
            return [parts]
    raise ValueError(f"无法解析 objects: {objects!r}")


def depth_overlay_pipeline(
    video_path: str,
    target: str = "",
    time=None,
    frame=None,
    bbox=None,
    objects=None,
) -> dict:
    if frame is None:
        raise ValueError("depth_overlay 需要参数 frame (秒)")
    if not objects:
        raise ValueError("depth_overlay 需要参数 objects（物体的 bbox 坐标）")

    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"视频不存在: {src}")

    t_sec = float(frame)
    obj_bboxes = _parse_objects(objects)  # 直接解析 bbox 坐标

    cache = get_cache_dir("depth_overlay")
    out_path = cache / f"{src.stem}_{hash_key(src, t_sec, objects)}.jpg"

    img_rgb = get_frame_at_time(str(src), t_sec)
    h, w = img_rgb.shape[:2]

    # ------ Depth Anything ------
    import torch
    from PIL import Image
    da_model, da_proc = get_depth_anything()
    device = next(da_model.parameters()).device
    pil = Image.fromarray(img_rgb)
    inputs = da_proc(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = da_model(**inputs)
    depth = outputs.predicted_depth  # (1, H', W')
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1) if depth.ndim == 3 else depth,
        size=(h, w), mode="bicubic", align_corners=False,
    ).squeeze().detach().cpu().numpy()
    depth_min, depth_max = float(depth.min()), float(depth.max())
    if depth_max - depth_min < 1e-6:
        raise RuntimeError("深度估计退化（max≈min），疑似模型未正确加载")
    depth_norm = (depth - depth_min) / (depth_max - depth_min)
    depth_u8 = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.55, depth_color_rgb, 0.45, 0)

    # ------ 直接根据 objects 中的 bbox 坐标画边界框（不再使用 Grounding-DINO）------
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    for i, bbox in enumerate(obj_bboxes):
        x1, y1, x2, y2 = bbox
        # 确保坐标在图像范围内
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            print(f"  [depth_overlay] 警告: 非法 bbox {bbox}，跳过")
            continue
        cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 取 bbox 中心点深度（数值越小越近，按 Depth-Anything 习惯）
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        d_val = float(depth[cy, cx])
        cv2.putText(
            overlay_bgr, f"obj{i+1} d={d_val:.2f}",
            (x1, max(y1 - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

    if not cv2.imwrite(str(out_path), overlay_bgr):
        raise RuntimeError(f"无法写入 {out_path}")

    target_str = target if target else "物体之间的空间关系"
    q = (
        f'这是 {t_sec:.1f}s 帧叠加深度热力图（暖色越亮表示越远/近，参见 d 数值）'
        f'，并用绿色框标出 {len(obj_bboxes)} 个物体。请回答: {target_str}'
    )
    return {"video": str(out_path), "perception_question": q}
