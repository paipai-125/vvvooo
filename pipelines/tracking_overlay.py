"""
tracking_overlay pipeline
- 输入: 视频 + time 区间 + target 描述
- 处理: 抽取时间段帧序列, 直接用 SAM 3 (facebook/sam3) 的视频文本 prompt
        做 detect+segment+track 一站式处理, 把每帧 mask/bbox 叠加到视频。
- 输出: 处理后视频 + perception_question

依赖: transformers >= 4.57.0 (Sam3VideoModel / Sam3VideoProcessor)
"""
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from utils.video_utils import save_video_clip, parse_time_range
from ._common import (
    get_cache_dir,
    hash_key,
    get_sam3_video,
)


# ---- 可视化用的固定调色板（最多支持 12 个并发实例，超过则循环复用）----
_PALETTE_BGR = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
    (255, 128, 0), (128, 255, 0), (0, 255, 128), (128, 128, 255),
]


def _extract_frames_to_array(video_path: Path, max_frames: int = 64):
    """
    将视频均匀采样为 RGB 帧数组 (T, H, W, 3) uint8，并返回采样到的实际帧数。
    SAM3 Sam3VideoProcessor.init_video_session 需要 list/ndarray 格式帧序列。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"视频帧数为0: {video_path}")
    n = min(total, max_frames)
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames_rgb = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, fr = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"读取第 {idx} 帧失败: {video_path}")
        frames_rgb.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames_rgb, axis=0)  # (T, H, W, 3) uint8


def tracking_overlay_pipeline(
    video_path: str,
    target: str = "",
    time=None,
    frame=None,
    bbox=None,
    objects=None,
) -> dict:
    if time is None:
        raise ValueError("tracking_overlay 需要参数 time")
    if not target:
        raise ValueError("tracking_overlay 需要参数 target")

    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"视频不存在: {src}")

    start_sec, end_sec = parse_time_range(time)
    if end_sec <= start_sec:
        raise ValueError(f"非法 time: {time}")

    cache = get_cache_dir("tracking_overlay")
    out_path = cache / f"{src.stem}_{hash_key(src, start_sec, end_sec, target)}.mp4"
    if out_path.exists():
        return {"video": str(out_path),
                "perception_question": _build_q(target, start_sec, end_sec)}

    tmp_root = Path(tempfile.mkdtemp(prefix="track_"))
    try:
        # 1. 裁切目标段（统一 fps=8）
        clip_path = tmp_root / "clip.mp4"
        save_video_clip(str(src), str(clip_path), start_sec, end_sec, target_fps=8.0)
        
        if not clip_path.exists():
            raise RuntimeError(f"裁切失败: {clip_path}")
        print("裁剪成功")

        # 2. 抽帧为 (T,H,W,3) RGB ndarray
        frames_rgb = _extract_frames_to_array(clip_path, max_frames=64)
        T, H, W, _ = frames_rgb.shape
        if T == 0:
            raise RuntimeError("未抽取到任何帧")
        print("抽帧成功")

        # 3. SAM 3 视频文本 prompt 跟踪
        import torch
        model, processor = get_sam3_video()
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # 直接使用原始文本作为提示
        text_prompt = target.strip()
        print(f"[tracking_overlay] 使用文本提示: '{text_prompt}'")

        # init_video_session 接受 list[np.ndarray] 或 np.ndarray (T,H,W,3)
        inference_session = processor.init_video_session(
            video=frames_rgb,
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=dtype,
        )
        
        # 添加文本提示
        inference_session = processor.add_text_prompt(
            inference_session=inference_session,
            text=text_prompt,
        )

        # 收集每帧的实例 mask（按 obj_id 聚合）
        # outputs["object_ids"]: (N,)  outputs["masks"]: (N, H, W) bool/0-1
        per_frame_results = {}
        frame_idx_to_obj_ids = {}  # 调试用
        
        with torch.no_grad():
            for model_outputs in model.propagate_in_video_iterator(
                inference_session=inference_session, max_frame_num_to_track=T,
                show_progress_bar=True,
            ):
                processed = processor.postprocess_outputs(inference_session, model_outputs)
                fidx = int(model_outputs.frame_idx)
                # masks: tensor (N, H, W)，object_ids: tensor (N,)
                masks = processed["masks"]
                obj_ids = processed["object_ids"]
                
                if hasattr(masks, "cpu"):
                    masks = masks.detach().cpu().numpy()
                if hasattr(obj_ids, "cpu"):
                    obj_ids = obj_ids.detach().cpu().numpy()
                
                frame_idx_to_obj_ids[fidx] = obj_ids
                
                if len(obj_ids) > 0:
                    per_frame_results[fidx] = (np.asarray(obj_ids).astype(int),
                                               np.asarray(masks).astype(np.uint8))

        # 调试输出
        non_empty_frames = [fidx for fidx, ids in frame_idx_to_obj_ids.items() if len(ids) > 0]
        print(f"[tracking_overlay] 检测到对象的帧: {non_empty_frames}")
        print(f"[tracking_overlay] 总共 {len(per_frame_results)} 帧有检测结果")
        
        if not per_frame_results:
            # 尝试用更通用的提示
            print(f"[tracking_overlay] 警告: 未检测到 '{target}'，尝试用更通用的提示...")
            # 可以在这里添加备选提示逻辑
            raise RuntimeError(
                f"SAM3 未跟踪到任何 '{target}' 实例。\n"
                f"候选提示已尝试: {candidate_prompts}\n"
                f"建议: 1) 检查视频中是否有该对象; 2) 尝试更通用的提示（如 'person'、'object'）; "
                f"3) 检查 SAM3 模型是否正确加载。"
            )

        # 4. 写出可视化视频
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, 8.0, (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"无法打开写入器: {out_path}")

        for i in range(T):
            img_bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR).copy()
            if i in per_frame_results:
                obj_ids, masks = per_frame_results[i]
                for k, oid in enumerate(obj_ids):
                    m = masks[k]
                    # masks 可能是 (H, W) 或 (1, H, W)
                    if m.ndim == 3:
                        m = m[0]
                    # 与原图尺寸对齐（SAM3 默认输出已为原图分辨率，但稳健起见）
                    if m.shape != (H, W):
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    if not m.any():
                        continue
                    color = _PALETTE_BGR[int(oid) % len(_PALETTE_BGR)]
                    # 创建彩色mask
                    mask_bool = m > 0
                    mask_color = np.zeros_like(img_bgr)
                    mask_color[mask_bool] = color
                    # 混合原图和mask
                    img_bgr = cv2.addWeighted(img_bgr, 0.6, mask_color, 0.4, 0)
                    ys, xs = np.where(m > 0)
                    if xs.size > 0 and ys.size > 0:
                        x1, y1 = int(xs.min()), int(ys.min())
                        x2, y2 = int(xs.max()), int(ys.max())
                        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img_bgr, f"id={int(oid)}", (x1, max(y1 - 6, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            cur_t = start_sec + (end_sec - start_sec) * i / max(T - 1, 1)
            cv2.putText(img_bgr, f"t={cur_t:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            writer.write(img_bgr)
        writer.release()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"tracking_overlay 输出失败: {out_path}")

    return {"video": str(out_path),
            "perception_question": _build_q(target, start_sec, end_sec)}


def _build_q(target: str, s: float, e: float) -> str:
    return (
        f'这是原视频 {s:.1f}s-{e:.1f}s 的片段，已对"{target}"做 mask 追踪可视化。'
        f'请描述其轨迹与活动（包括起止时间、起止位置）。'
    )
