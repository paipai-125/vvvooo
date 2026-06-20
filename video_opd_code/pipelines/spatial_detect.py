"""
spatial_detect pipeline
- 输入: 视频 + 帧时间 + target 描述
- 处理: 抽出指定帧, 用 Grounding-DINO 把所有候选物体高亮框出
- 输出: 处理后图像路径 + perception_question
"""
from pathlib import Path

import cv2
import numpy as np

from utils.video_utils import get_frame_at_time
from ._common import get_cache_dir, hash_key, get_grounding_dino


def spatial_detect_pipeline(
    video_path: str,
    target: str = "",
    time=None,
    frame=None,
    bbox=None,
    objects=None,
) -> dict:
    if frame is None:
        raise ValueError("spatial_detect 需要参数 frame (秒)")
    if not target:
        raise ValueError("spatial_detect 需要参数 target")

    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"视频不存在: {src}")

    t_sec = float(frame)
    cache = get_cache_dir("spatial_detect")
    out_path = cache / f"{src.stem}_{hash_key(src, t_sec, target)}.jpg"

    img_rgb = get_frame_at_time(str(src), t_sec)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # ---- Grounding-DINO 检测 ----
    import torch
    from PIL import Image
    model, processor = get_grounding_dino()
    device = next(model.parameters()).device

    pil_img = Image.fromarray(img_rgb)
    text_prompt = target.strip()
    if not text_prompt.endswith("."):
        text_prompt = text_prompt + "."

    inputs = processor(images=pil_img, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    h, w = img_rgb.shape[:2]
    target_sizes = torch.tensor([[h, w]], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.25,
        text_threshold=0.25,
        target_sizes=target_sizes,
    )[0]

    boxes = results["boxes"].detach().cpu().numpy().astype(int)
    scores = results["scores"].detach().cpu().numpy()
    labels = results.get("text_labels", results.get("labels", [""] * len(boxes)))

    # 绘制
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{labels[i] if i < len(labels) else 'obj'} {score:.2f}"
        cv2.putText(img_bgr, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    if not cv2.imwrite(str(out_path), img_bgr):
        raise RuntimeError(f"无法写入 {out_path}")

    q = (
        f'这是视频在 {t_sec:.1f}s 时的画面，已用绿色框标出所有候选物体（图像尺寸 {w}x{h}）。'
        f'请定位"{target}"，输出 [x1,y1,x2,y2] 像素坐标。'
    )
    return {"video": str(out_path), "perception_question": q}
