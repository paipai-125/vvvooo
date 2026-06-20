"""
为双教师生成专属 SFT 训练数据。

核心思想：
  学生 SFT 数据中的每条 trajectory 包含完整的推理+感知链。
  我们需要从中拆分出：
    - Teacher_R 的训练数据：纯文本输入（问题+前序推理）→ 输出推理段
    - Teacher_P 的训练数据：视觉聚焦输入（工具产物）+ 感知问题 → 输出感知结果

三种角色的 SFT 数据对比：
  ┌─────────────┬──────────────────────────────┬────────────────────────────────┐
  │ 角色        │ 输入                          │ 监督目标                        │
  ├─────────────┼──────────────────────────────┼────────────────────────────────┤
  │ 学生        │ 视频 + 问题                   │ 完整 trajectory                 │
  │ Teacher_R   │ 纯文本: 问题 + 前序推理文本    │ 推理段（[分析]...<observe>...）  │
  │ Teacher_P   │ 视觉聚焦输入 + 感知问题       │ <result>内容                    │
  └─────────────┴──────────────────────────────┴────────────────────────────────┘

Teacher_R 的输入不包含视频！它是纯文本推理教师，只看问题和已有推理上下文。
Teacher_P 的输入不包含原始视频！它看的是工具产物（pipeline 预处理后的视觉输入）。

用法:
    python -m data_preparation.prepare_teacher_sft_data --role teacher_r
    python -m data_preparation.prepare_teacher_sft_data --role teacher_p
    python -m data_preparation.prepare_teacher_sft_data --role all
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.paths import SFT_DATA_PATH, ensure_dirs  # noqa: E402
from utils.parser import parse_observe, parse_result, split_segments  # noqa: E402


# ============================================================================
# 工具函数
# ============================================================================

def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _write_jsonl(items: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================================
# Teacher_R 数据生成
# ============================================================================

def generate_teacher_r_data(student_items: list[dict]) -> list[dict]:
    """从学生 SFT 数据中提取 Teacher_R 的训练数据。

    Teacher_R 是纯文本推理教师：
      - 输入包含完整 trajectory（含 <result>...</result> 内容）
      - 但只对推理段计算 loss，不对 <result>...</result> 段计算 loss
      - 这样 Teacher_R 能看到感知结果并基于此推理，但不学习预测感知内容

    训练时的 loss mask 策略：
      - <result>...</result> 区间内的 token：loss_mask = 0（不计算 loss）
      - 其余所有 token（推理段 + answer）：loss_mask = 1（计算 loss）

    数据格式：
      {
        "question": "...",
        "trajectory": "完整 trajectory（含 <result> 内容）",
        "no_loss_spans": [{"start": "<result>", "end": "</result>"}, ...],
        ...
      }
    """
    import re

    teacher_r_items = []

    for item in tqdm(student_items, desc="生成 Teacher_R 数据"):
        question = item.get("question", "")
        trajectory = item.get("trajectory", "")
        if not question or not trajectory:
            continue

        # 找到所有 <result>...</result> 的字符位置，用于标记 no-loss 区间
        no_loss_spans = []
        for m in re.finditer(r"<result>.*?</result>", trajectory, flags=re.DOTALL):
            no_loss_spans.append({"start": m.start(), "end": m.end()})

        teacher_r_items.append({
            "role": "teacher_r",
            "question": question,
            "trajectory": trajectory,  # 保留完整 trajectory（含 result 内容）
            "no_loss_spans": no_loss_spans,  # <result>...</result> 区间不计算 loss
            "gt_answer": item.get("gt_answer", ""),
            "type": item.get("type", ""),
            "source": item.get("source", ""),
            "video": item.get("video", ""),
            "has_video_input": False,
        })

    return teacher_r_items


# ============================================================================
# Teacher_P 数据生成
# ============================================================================

def _extract_observe_and_result(trajectory: str) -> list[dict]:
    """从 trajectory 中提取所有 (observe, result) 对。

    返回: [{"observe_text": str, "observe_attrs": dict, "result_text": str}]
    """
    pairs = []
    observes = parse_observe(trajectory)
    results = parse_result(trajectory)

    if len(observes) != len(results):
        return []

    for obs, res in zip(observes, results):
        pairs.append({
            "observe_text": obs.raw_text,
            "observe_attrs": {
                "type": obs.type,
                "target": obs.target,
                "time": obs.time,
                "frame": obs.frame,
                "bbox": obs.bbox,
                "objects": obs.objects,
            },
            "result_text": res,
        })

    return pairs


def _build_perception_question(obs_attrs: dict) -> str:
    """根据 observe 属性构造 Teacher_P 的感知问题。

    这个问题就是 Teacher_P 在 SFT 阶段学习回答的问题。
    """
    obs_type = obs_attrs.get("type", "raw")
    target = obs_attrs.get("target", "")
    frame = obs_attrs.get("frame", "")
    time = obs_attrs.get("time", "")
    bbox = obs_attrs.get("bbox", "")
    objects = obs_attrs.get("objects", "")

    if obs_type == "temporal_locate":
        return f'请定位视频中"{target}"发生的时间区间，格式: start_s-end_s。'
    elif obs_type == "temporal_clip":
        return f'这是视频 {time} 时间段的片段。请描述其中"{target}"相关的内容。'
    elif obs_type == "spatial_detect":
        return f'在 {frame}s 的画面中，请定位"{target}"的位置，输出 [x1,y1,x2,y2] 像素坐标。'
    elif obs_type == "spatial_crop":
        return f'这是 {frame}s 帧的 {bbox} 区域裁切放大图。请描述该区域的内容。'
    elif obs_type == "depth_overlay":
        return f'这是 {frame}s 帧叠加深度图，标注了物体位置。请判断哪个物体更靠近相机。'
    elif obs_type == "tracking_overlay":
        return f'这是 {time} 时间段内对"{target}"的追踪可视化。请描述其运动轨迹和活动。'
    elif obs_type == "ocr_zoom":
        return f'这是 {frame}s 帧的 {bbox} 区域放大图。请识别其中的文字内容。'
    else:
        return f'请观察并描述"{target}"。'


def generate_teacher_p_data(student_items: list[dict]) -> list[dict]:
    """从学生 SFT 数据中提取 Teacher_P 的训练数据。

    Teacher_P 是视觉感知教师：
      输入: 视觉聚焦输入（工具产物）+ 感知问题
      输出: <result> 的内容

    关键：Teacher_P 看的不是原始视频，而是 pipeline 预处理后的视觉输入！
    但在 SFT 阶段，我们用原始视频+帧时间作为近似（因为 pipeline 预处理
    需要 GPU 模型，SFT 数据生成时不方便跑）。

    实际 OPD 训练时，Teacher_P 的输入会经过真实 pipeline 预处理。
    """
    teacher_p_items = []

    for item in tqdm(student_items, desc="生成 Teacher_P 数据"):
        trajectory = item.get("trajectory", "")
        video = item.get("video", "")
        if not trajectory or not video:
            continue

        # 提取所有 observe-result 对
        pairs = _extract_observe_and_result(trajectory)
        if not pairs:
            continue

        for pair in pairs:
            obs_attrs = pair["observe_attrs"]
            result_text = pair["result_text"]
            perception_q = _build_perception_question(obs_attrs)

            teacher_p_items.append({
                "role": "teacher_p",
                "video": video,
                "perception_question": perception_q,
                "result_text": result_text,  # Teacher_P 的监督目标
                "observe_type": obs_attrs["type"],
                "observe_attrs": obs_attrs,
                "source": item.get("source", ""),
                "has_video_input": True,  # Teacher_P 看视觉聚焦输入
                # 以下字段用于 OPD 阶段的 pipeline 预处理
                "pipeline_params": {
                    "type": obs_attrs["type"],
                    "target": obs_attrs.get("target", ""),
                    "time": obs_attrs.get("time"),
                    "frame": obs_attrs.get("frame"),
                    "bbox": obs_attrs.get("bbox"),
                    "objects": obs_attrs.get("objects"),
                },
            })

    return teacher_p_items


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="为双教师生成专属 SFT 训练数据"
    )
    parser.add_argument(
        "--role", type=str, required=True,
        choices=["teacher_r", "teacher_p", "all"],
        help="生成哪个教师的数据"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="输入的学生 SFT 数据路径（默认: stage1_sft_template_all.jsonl）"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录（默认: SFT_DATA_PATH）"
    )
    args = parser.parse_args()

    ensure_dirs()
    out_dir = Path(args.output_dir) if args.output_dir else SFT_DATA_PATH

    # 加载学生 SFT 数据
    input_path = Path(args.input) if args.input else (
        SFT_DATA_PATH / "stage1_sft_template_all.jsonl"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    print(f"[加载] 学生 SFT 数据: {input_path}")
    student_items = _load_jsonl(input_path)
    print(f"  共 {len(student_items)} 条")

    if args.role in ("teacher_r", "all"):
        print("\n" + "=" * 60)
        print("[Teacher_R] 生成纯文本推理教师的 SFT 数据...")
        print("=" * 60)
        tr_items = generate_teacher_r_data(student_items)
        tr_path = out_dir / "stage1_sft_teacher_r.jsonl"
        _write_jsonl(tr_items, tr_path)
        print(f"[OK] Teacher_R: {len(tr_items)} 条 -> {tr_path}")

    if args.role in ("teacher_p", "all"):
        print("\n" + "=" * 60)
        print("[Teacher_P] 生成视觉感知教师的 SFT 数据...")
        print("=" * 60)
        tp_items = generate_teacher_p_data(student_items)
        tp_path = out_dir / "stage1_sft_teacher_p.jsonl"
        _write_jsonl(tp_items, tp_path)
        print(f"[OK] Teacher_P: {len(tp_items)} 条 -> {tp_path}")

        # 统计各 type 分布
        type_counts: dict[str, int] = {}
        for item in tp_items:
            t = item.get("observe_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"  各 type 分布: {type_counts}")

    print("\n[完成] 教师 SFT 数据生成完毕。")
    print("\n下一步:")
    print("  1. 训练学生:    bash scripts/run_stage1_sft_train.sh --role student")
    print("  2. 训练Teacher_R: bash scripts/run_stage1_sft_train.sh --role teacher_r")
    print("  3. 训练Teacher_P: bash scripts/run_stage1_sft_train.sh --role teacher_p")


if __name__ == "__main__":
    main()
