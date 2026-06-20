"""
预实验1：视觉聚焦有效性验证。

任务：在 Charades-STA 验证集上对比
    (a) 学生（Qwen3-VL-4B 基模）直接输入完整视频回答 temporal_locate
  (b) Teacher_P 输入「视觉聚焦预处理后」的视频回答同一题
评估指标：时间段 IoU。

输出：
  - 每条样本的预测与 IoU
  - 汇总：mean IoU / IoU>0.3 / IoU>0.5 / IoU>0.7 命中率
  - 学生 vs Teacher_P 的对比表
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

# 允许从项目根目录直接运行
_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs import paths as P  # noqa: E402
from pipelines import temporal_locate_pipeline  # noqa: E402


# ---------------- Charades-STA 数据加载 ---------------- #

def load_charades_sta_val(split_file: Path, video_dir: Path, limit: Optional[int] = None):
    """
    Charades-STA 标注格式（每行）:
        <video_id> <start> <end>##<query>
    例如:
        AO8RW 0.0 6.9##person turn a light on.
    """
    if not split_file.exists():
        raise FileNotFoundError(f"Charades-STA 标注文件不存在: {split_file}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Charades-STA 视频目录不存在: {video_dir}")

    samples = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "##" not in line:
                raise ValueError(f"Charades-STA 行格式不合法: {line}")
            head, query = line.split("##", 1)
            parts = head.strip().split()
            if len(parts) != 3:
                raise ValueError(f"Charades-STA 行格式不合法: {line}")
            vid, start, end = parts[0], float(parts[1]), float(parts[2])
            video_path = video_dir / f"{vid}.mp4"
            if not video_path.exists():
                # Charades 视频未下齐时直接报错
                raise FileNotFoundError(f"视频缺失: {video_path}")
            samples.append({
                "video_id": vid,
                "video_path": str(video_path),
                "query": query.strip().rstrip("."),
                "gt_start": start,
                "gt_end": end,
            })
            if limit is not None and len(samples) >= limit:
                break
    return samples


# ---------------- IoU ---------------- #

def temporal_iou(pred: tuple[float, float], gt: tuple[float, float]) -> float:
    p0, p1 = sorted(pred)
    g0, g1 = sorted(gt)
    inter = max(0.0, min(p1, g1) - max(p0, g0))
    union = max(p1, g1) - min(p0, g0)
    if union <= 0:
        return 0.0
    return inter / union


_TIME_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*s?\s*(?:-|to|~|至|到)\s*(\d+(?:\.\d+)?)\s*s?")


def parse_time_range_from_text(text: str) -> Optional[tuple[float, float]]:
    if not text:
        return None
    m = _TIME_RANGE_RE.search(text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


# ---------------- Qwen3-VL 推理封装 ---------------- #

class QwenVLRunner:
    """懒加载的 Qwen3-VL 推理器，支持视频输入。"""

    def __init__(self, model_path: Path, device: str = "cuda"):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Qwen3-VL 模型路径不存在: {model_path}")
        self.model_path = str(model_path)
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoProcessor
        # Qwen3-VL 的 ConditionalGeneration 类：优先 Qwen3VL，回退 Qwen2_5_VL
        try:
            from transformers import Qwen3VLForConditionalGeneration as _Cls  # type: ignore
        except ImportError:
            from transformers import Qwen2_5_VLForConditionalGeneration as _Cls  # type: ignore
        # 修复 transformers 5.10.0.dev0 Qwen3VLProcessor 的 <|placeholder|> bug
        try:
            from utils.qwen3vl_patch import apply_qwen3vl_patches
            apply_qwen3vl_patches()
        except Exception:
            pass
        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = _Cls.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()

    @torch.no_grad()
    def answer(self, video_path: str, question: str, max_new_tokens: int = 128) -> str:
        self._ensure_loaded()
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": question},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 视频帧由 processor 处理；不同版本 transformers 的视频字段名不同
        try:
            inputs = self._processor(
                text=[text],
                videos=[str(video_path)],
                return_tensors="pt",
                padding=True,
            )
        except TypeError:
            # 旧版用 video 单数
            inputs = self._processor(
                text=[text],
                video=[str(video_path)],
                return_tensors="pt",
                padding=True,
            )
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


# ---------------- 主流程 ---------------- #

QUESTION_TEMPLATE = '"{q}" 发生在视频什么时间？请回答时间段，格式如 "3.0s-7.0s"。'


def run_pre_experiment(
    annotation_file: Path,
    video_dir: Path,
    output_path: Path,
    limit: int,
    skip_focused: bool = False,
):
    samples = load_charades_sta_val(annotation_file, video_dir, limit=limit)
    print(f"[pre_experiment_focus] 加载 {len(samples)} 条 Charades-STA 样本")

    runner = QwenVLRunner(P.QWEN3_VL_4B_PATH)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(output_path, "w", encoding="utf-8")

    student_ious, teacher_ious = [], []

    for sample in tqdm(samples, desc="pre_experiment_focus"):
        gt = (sample["gt_start"], sample["gt_end"])
        question = QUESTION_TEMPLATE.format(q=sample["query"])

        # (a) 学生直答（完整原视频）
        student_ans = runner.answer(sample["video_path"], question)
        student_pred = parse_time_range_from_text(student_ans)
        student_iou = temporal_iou(student_pred, gt) if student_pred else 0.0
        student_ious.append(student_iou)

        # (b) Teacher_P + 视觉聚焦预处理
        teacher_ans, teacher_iou = "", 0.0
        if not skip_focused:
            focused = temporal_locate_pipeline(sample["video_path"], target=sample["query"])
            teacher_ans = runner.answer(focused["video"], focused["perception_question"])
            teacher_pred = parse_time_range_from_text(teacher_ans)
            teacher_iou = temporal_iou(teacher_pred, gt) if teacher_pred else 0.0
            teacher_ious.append(teacher_iou)

        record = {
            "video_id": sample["video_id"],
            "query": sample["query"],
            "gt": list(gt),
            "student_answer": student_ans,
            "student_iou": student_iou,
            "teacher_answer": teacher_ans,
            "teacher_iou": teacher_iou,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()

    fout.close()

    # 汇总
    def summary(ious: list[float]) -> dict:
        if not ious:
            return {"n": 0}
        n = len(ious)
        mean = sum(ious) / n
        return {
            "n": n,
            "mean_iou": round(mean, 4),
            "iou>0.3": round(sum(1 for x in ious if x > 0.3) / n, 4),
            "iou>0.5": round(sum(1 for x in ious if x > 0.5) / n, 4),
            "iou>0.7": round(sum(1 for x in ious if x > 0.7) / n, 4),
        }

    student_sum = summary(student_ious)
    teacher_sum = summary(teacher_ious)

    print("\n========== 预实验1: 视觉聚焦有效性 ==========")
    print(f"样本数: {len(samples)}")
    print(f"学生直答         : {student_sum}")
    print(f"Teacher_P+聚焦   : {teacher_sum}")
    if student_sum.get("mean_iou") is not None and teacher_sum.get("mean_iou") is not None:
        diff = teacher_sum["mean_iou"] - student_sum["mean_iou"]
        print(f"差值 (Teacher - 学生): mean_iou {diff:+.4f}")

    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"student": student_sum, "teacher_focused": teacher_sum}, f,
                  ensure_ascii=False, indent=2)
    print(f"详细记录: {output_path}")
    print(f"汇总: {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--annotation",
        type=str,
        default=str(P.CHARADES_STA_PATH / "charades_sta_test.txt"),
        help="Charades-STA 标注文件 (默认 test split)",
    )
    ap.add_argument(
        "--video_dir",
        type=str,
        default=str(P.CHARADES_STA_PATH / "videos"),
        help="Charades-STA 视频目录",
    )
    ap.add_argument("--limit", type=int, default=500, help="评测样本数量")
    ap.add_argument(
        "--output",
        type=str,
        default=str(P.OUTPUT_ROOT / "pre_experiment_focus" / "results.jsonl"),
    )
    ap.add_argument("--skip_focused", action="store_true", help="只跑学生直答，调试用")
    args = ap.parse_args()

    P.ensure_dirs()
    run_pre_experiment(
        annotation_file=Path(args.annotation),
        video_dir=Path(args.video_dir),
        output_path=Path(args.output),
        limit=args.limit,
        skip_focused=args.skip_focused,
    )


if __name__ == "__main__":
    main()
