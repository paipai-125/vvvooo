"""
批量跑 Depth Anything V2 预计算深度，替换 A-6 depth_overlay 的占位 gt_answer。

流程：
  1. 读取 stage1_sft_a6_depth_overlay.jsonl
  2. 对每条样本：提取视频帧 → 跑 Depth Anything V2 → 取两个 bbox 中心深度值
  3. 比较深度值，生成真实 gt_answer
  4. 输出更新后的 jsonl（覆盖原文件 or 写到新文件）

用法:
    # 8卡并行（默认）
    python -m tools.run_expert_models --task depth_a6
    # 指定GPU数量
    python -m tools.run_expert_models --task depth_a6 --num-gpus 4
    # 单卡
    python -m tools.run_expert_models --task depth_a6 --num-gpus 1

注意: 需要 GPU + Depth Anything V2 权重。支持单机多卡并行加速。
"""
from __future__ import annotations

import argparse
import json
import sys
import time as time_module
import os
from pathlib import Path
from typing import Optional
from multiprocessing import Process, Manager

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 项目路径
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.paths import SFT_DATA_PATH, DEPTH_ANYTHING_PATH  # noqa: E402
from utils.video_utils import get_frame_at_time  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _write_jsonl(items: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _parse_bbox_from_str(bbox_str: str) -> list[float]:
    """从 '[x1,y1,x2,y2]' 字符串解析 bbox"""
    s = bbox_str.strip().strip("[]")
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"无法解析 bbox: {bbox_str}")
    return parts


def load_depth_anything(device: str = "cuda:0"):
    """加载 Depth Anything V2 模型到指定设备"""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if not DEPTH_ANYTHING_PATH.exists():
        raise FileNotFoundError(
            f"Depth-Anything V2 权重未找到: {DEPTH_ANYTHING_PATH}\n"
            f"请下载到该目录，或在 configs/paths.yaml 中调整路径。"
        )

    processor = AutoImageProcessor.from_pretrained(str(DEPTH_ANYTHING_PATH))
    model = AutoModelForDepthEstimation.from_pretrained(
        str(DEPTH_ANYTHING_PATH)
    ).to(device).eval()

    return model, processor, device


def compute_depth_map(model, processor, device, img_rgb: np.ndarray) -> np.ndarray:
    """对单张 RGB 图像计算深度图，返回 (H, W) float32 数组。
    
    Depth Anything V2 输出：数值越大表示越远（disparity 的逆）。
    """
    h, w = img_rgb.shape[:2]
    pil = Image.fromarray(img_rgb)
    inputs = processor(images=pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    depth = outputs.predicted_depth  # (1, H', W') or (H', W')
    # 插值到原图尺寸
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1) if depth.ndim == 3 else depth,
        size=(h, w), mode="bicubic", align_corners=False,
    ).squeeze().cpu().numpy()

    return depth


def get_bbox_center_depth(depth_map: np.ndarray, bbox: list[float]) -> float:
    """取 bbox 中心点的深度值"""
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = bbox
    cx = int(max(0, min(w - 1, (x1 + x2) / 2)))
    cy = int(max(0, min(h - 1, (y1 + y2) / 2)))
    return float(depth_map[cy, cx])


def get_bbox_mean_depth(depth_map: np.ndarray, bbox: list[float]) -> float:
    """取 bbox 区域内的平均深度值（更鲁棒）"""
    h, w = depth_map.shape[:2]
    x1 = int(max(0, min(w - 1, bbox[0])))
    y1 = int(max(0, min(h - 1, bbox[1])))
    x2 = int(max(0, min(w, bbox[2])))
    y2 = int(max(0, min(h, bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return get_bbox_center_depth(depth_map, bbox)
    region = depth_map[y1:y2, x1:x2]
    return float(region.mean())


def _process_single_item(item: dict, model, processor, device: str,
                         use_mean: bool) -> dict | None:
    """处理单条样本，返回更新后的 item 或 None（失败时）"""
    try:
        video_path = item["video"]
        params = item.get("params", {})
        frame_time = float(params.get("frame", "0.0"))
        objects_str = params.get("objects", "")

        # 解析两个 bbox
        obj_parts = objects_str.split("],[")
        if len(obj_parts) != 2:
            obj_parts = objects_str.split(",")
            if len(obj_parts) == 8:
                bbox_a = [float(x.strip("[] ")) for x in obj_parts[:4]]
                bbox_b = [float(x.strip("[] ")) for x in obj_parts[4:]]
            else:
                return None
        else:
            bbox_a = _parse_bbox_from_str("[" + obj_parts[0].strip("[] ") + "]")
            bbox_b = _parse_bbox_from_str("[" + obj_parts[1].strip("[] ") + "]")

        # 计算深度图
        img_rgb = get_frame_at_time(video_path, frame_time)
        depth_map = compute_depth_map(model, processor, device, img_rgb)

        # 取深度值
        if use_mean:
            depth_a = get_bbox_mean_depth(depth_map, bbox_a)
            depth_b = get_bbox_mean_depth(depth_map, bbox_b)
        else:
            depth_a = get_bbox_center_depth(depth_map, bbox_a)
            depth_b = get_bbox_center_depth(depth_map, bbox_b)

        # 生成真实 gt_answer
        bbox_a_display = f"[{int(bbox_a[0])},{int(bbox_a[1])},{int(bbox_a[2])},{int(bbox_a[3])}]"
        bbox_b_display = f"[{int(bbox_b[0])},{int(bbox_b[1])},{int(bbox_b[2])},{int(bbox_b[3])}]"

        if depth_a < depth_b:
            new_gt = f"The object at {bbox_a_display} is closer to the camera (depth={depth_a:.2f} vs {depth_b:.2f})."
        elif depth_b < depth_a:
            new_gt = f"The object at {bbox_b_display} is closer to the camera (depth={depth_b:.2f} vs {depth_a:.2f})."
        else:
            new_gt = f"Both objects are at approximately the same depth ({depth_a:.2f})."

        # 更新
        old_gt = item.get("gt_answer", "")
        item["gt_answer"] = new_gt
        item["gt_answer_placeholder"] = old_gt
        item["depth_values"] = {"bbox_a": depth_a, "bbox_b": depth_b}

        # 更新 trajectory
        traj = item.get("trajectory", "")
        if old_gt and old_gt in traj:
            traj = traj.replace(old_gt, new_gt)
            item["trajectory"] = traj

        return item

    except Exception:
        return None


def _worker_fn(rank: int, num_gpus: int, items_shard: list[dict],
               use_mean: bool, result_dict: dict):
    """单个 GPU worker 进程的入口函数。
    
    Args:
        rank: GPU 编号 (0~num_gpus-1)
        num_gpus: 总 GPU 数
        items_shard: 该 worker 负责处理的数据分片
        use_mean: 是否使用区域平均深度
        result_dict: 共享字典，用于收集结果
    """
    device = f"cuda:{rank}"

    print(f"  [GPU {rank}] 加载模型到 {device}，负责 {len(items_shard)} 条样本")
    model, processor, _ = load_depth_anything(device)

    processed = []
    error_count = 0

    desc = f"GPU-{rank}" if num_gpus > 1 else "A-6 depth precompute"
    for item in tqdm(items_shard, desc=desc, position=rank, leave=True):
        result = _process_single_item(item, model, processor, device, use_mean)
        if result is not None:
            processed.append(result)
        else:
            # 保留原始 item（不更新）
            processed.append(item)
            error_count += 1

    result_dict[rank] = {"items": processed, "errors": error_count}
    print(f"  [GPU {rank}] ✅ 完成: {len(processed) - error_count} 成功, {error_count} 失败")


def run_depth_a6(output_path: Optional[str] = None, use_mean: bool = True,
                 num_gpus: int = 8):
    """批量跑 Depth Anything V2，替换 A-6 的占位 gt_answer。
    
    Args:
        output_path: 输出路径，None 则覆盖原文件
        use_mean: True 使用 bbox 区域平均深度，False 使用中心点深度
        num_gpus: 使用的 GPU 数量（默认 8）
    """
    input_file = SFT_DATA_PATH / "stage1_sft_a6_depth_overlay.jsonl"
    if not input_file.exists():
        raise FileNotFoundError(f"A-6 数据文件不存在: {input_file}")

    items = _load_jsonl(input_file)
    print(f"[A-6] 读取 {len(items)} 条样本: {input_file}")

    # 检测可用 GPU 数量
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("没有可用的 GPU！")
    num_gpus = min(num_gpus, available_gpus)
    print(f"[A-6] 使用 {num_gpus} 张 GPU 并行处理")

    t0 = time_module.time()

    if num_gpus == 1:
        # 单卡模式：直接在主进程跑
        print(f"[Depth-Anything V2] 加载模型到 cuda:0")
        model, processor, device = load_depth_anything("cuda:0")
        print(f"[Depth-Anything V2] ✅ 加载成功")

        updated_count = 0
        error_count = 0

        for i, item in enumerate(tqdm(items, desc="A-6 depth precompute")):
            result = _process_single_item(item, model, processor, device, use_mean)
            if result is not None:
                items[i] = result
                updated_count += 1
            else:
                if i < 5:
                    print(f"  [!] 样本 {i} 处理失败")
                error_count += 1

        print(f"\n[A-6] 完成! 耗时 {time_module.time() - t0:.1f}s")
        print(f"  更新: {updated_count}, 错误: {error_count}")

    else:
        # 多卡模式：数据分片 + 多进程
        # 均分数据
        shards = [[] for _ in range(num_gpus)]
        for i, item in enumerate(items):
            shards[i % num_gpus].append((i, item))

        # 使用 Manager 共享结果
        manager = Manager()
        result_dict = manager.dict()

        # 启动多进程
        processes = []
        for rank in range(num_gpus):
            shard_items = [item for _, item in shards[rank]]
            p = Process(
                target=_worker_fn,
                args=(rank, num_gpus, shard_items, use_mean, result_dict),
            )
            p.start()
            processes.append(p)

        # 等待所有进程完成
        for p in processes:
            p.join()

        # 合并结果（按原始顺序）
        # 重建索引映射
        total_errors = 0
        shard_results = {}
        for rank in range(num_gpus):
            if rank in result_dict:
                shard_results[rank] = result_dict[rank]["items"]
                total_errors += result_dict[rank]["errors"]
            else:
                print(f"  [!] GPU {rank} 没有返回结果")
                shard_results[rank] = [item for _, item in shards[rank]]
                total_errors += len(shards[rank])

        # 按原始顺序重组
        for rank in range(num_gpus):
            result_items = shard_results.get(rank, [])
            for j, (orig_idx, _) in enumerate(shards[rank]):
                if j < len(result_items):
                    items[orig_idx] = result_items[j]

        elapsed = time_module.time() - t0
        updated_count = len(items) - total_errors
        print(f"\n[A-6] 完成! 耗时 {elapsed:.1f}s ({num_gpus} GPUs)")
        print(f"  更新: {updated_count}, 错误: {total_errors}")

    # 写出
    if output_path:
        out = Path(output_path)
    else:
        out = input_file  # 覆盖原文件
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(items, out)
    print(f"[A-6] 已写入: {out}")

    # 统计深度翻转率
    flip_count = 0
    for item in items:
        old = item.get("gt_answer_placeholder", "")
        new = item.get("gt_answer", "")
        if old and new and old != new:
            flip_count += 1
    total_with_placeholder = sum(1 for it in items if it.get("gt_answer_placeholder"))
    if total_with_placeholder > 0:
        print(f"  深度翻转率（与 y 坐标占位不同）: {flip_count}/{total_with_placeholder} "
              f"({100*flip_count/total_with_placeholder:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="批量跑专家模型预计算（当前支持: depth_a6）。支持单机多卡并行。"
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["depth_a6"],
        help="要执行的任务"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出路径（默认覆盖原文件）"
    )
    parser.add_argument(
        "--use-center", action="store_true",
        help="使用 bbox 中心点深度（默认使用区域平均深度，更鲁棒）"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=8,
        help="使用的 GPU 数量（默认 8，自动 clamp 到实际可用数量）"
    )
    args = parser.parse_args()

    if args.task == "depth_a6":
        run_depth_a6(
            output_path=args.output,
            use_mean=not args.use_center,
            num_gpus=args.num_gpus,
        )


if __name__ == "__main__":
    main()
