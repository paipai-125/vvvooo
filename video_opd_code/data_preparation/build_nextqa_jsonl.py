#!/usr/bin/env python3
"""
NExT-QA 后处理脚本：parquet → train.jsonl

HF 仓库 lmms-lab/NExTQA 的实际结构：
  - OE/train-00000-of-00001.parquet  (开放式问答, 37523 条)
  - OE/validation-00000-of-00001.parquet (5343 条)
  - OE/test-00000-of-00001.parquet (9178 条)
  - MC/test-00000-of-00001.parquet  (多选, 8564 条)
  - videos/NExTVideo/<id>.mp4       (1570 个视频)

输出格式（与 download_datasets.sh 中定义一致）：
  每行一个 JSON:
  {
    "video": "videos/NExTVideo/<id>.mp4",   # 相对于 nextqa/ 的路径
    "question": "...",
    "choices": ["a0", "a1", "a2", "a3", "a4"],  # MC 有，OE 为空列表
    "gt_answer": "...",                          # MC 为选项文本，OE 为答案文本
    "type": "CW/CH/TN/TC/DC/DL/DO",
    "qid": "...",
    "split": "train/validation/test",
    "mode": "OE/MC"
  }

用法：
  cd /path/to/video_opd_code
  python -m data_preparation.build_nextqa_jsonl
"""

import json
import sys
from pathlib import Path

# 尝试导入 pandas（读 parquet 必须）
try:
    import pandas as pd
except ImportError:
    print("[ERROR] 需要 pandas + pyarrow 来读取 parquet 文件。")
    print("  pip install pandas pyarrow")
    sys.exit(1)

# 导入项目路径配置
try:
    from configs.paths import NEXTQA_PATH
except ImportError:
    # 兜底：如果不在代码目录运行
    NEXTQA_PATH = Path(__file__).resolve().parent.parent / "video_opd_data" / "datasets" / "nextqa"
    print(f"[WARN] 无法导入 configs.paths，使用默认路径: {NEXTQA_PATH}")


def find_parquet_files(base: Path) -> dict:
    """查找所有 parquet 文件，返回 {(mode, split): path}"""
    result = {}
    for mode_dir in ["OE", "MC"]:
        mode_path = base / mode_dir
        if not mode_path.exists():
            continue
        for pq in mode_path.glob("*.parquet"):
            # 文件名格式: train-00000-of-00001.parquet / test-00000-of-00001.parquet
            split_name = pq.name.split("-")[0]  # train / validation / test
            result[(mode_dir, split_name)] = pq
    return result


def process_oe_row(row: dict, split: str) -> dict:
    """处理 OE（开放式问答）的一行"""
    # OE 的 video 字段是字符串（视频ID）
    video_id = str(row["video"])
    # 视频路径：videos/NExTVideo/<id>.mp4
    video_path = f"videos/NExTVideo/{video_id}.mp4"

    return {
        "video": video_path,
        "question": row.get("question", ""),
        "choices": [],  # OE 没有选项
        "gt_answer": str(row.get("answer", "")),
        "type": row.get("type", ""),
        "qid": str(row.get("qid", "")),
        "split": split,
        "mode": "OE",
    }


def process_mc_row(row: dict, split: str) -> dict:
    """处理 MC（多选题）的一行"""
    # MC 的 video 字段是 int64
    video_id = str(int(row["video"]))
    video_path = f"videos/NExTVideo/{video_id}.mp4"

    # 收集选项 a0~a4
    choices = []
    for i in range(5):
        key = f"a{i}"
        if key in row and pd.notna(row[key]):
            choices.append(str(row[key]))

    # answer 是选项索引（int）
    gt_idx = int(row.get("answer", -1))
    gt_answer = choices[gt_idx] if 0 <= gt_idx < len(choices) else str(row.get("answer", ""))

    return {
        "video": video_path,
        "question": row.get("question", ""),
        "choices": choices,
        "gt_answer": gt_answer,
        "type": row.get("type", ""),
        "qid": str(row.get("qid", "")),
        "split": split,
        "mode": "MC",
    }


def main():
    base = NEXTQA_PATH
    out_path = base / "train.jsonl"

    print(f"[NExT-QA] 数据目录: {base}")
    print(f"[NExT-QA] 输出文件: {out_path}")

    # 查找 parquet 文件
    pq_files = find_parquet_files(base)
    if not pq_files:
        print("[ERROR] 未找到任何 parquet 文件！请确认 OE/ 和 MC/ 目录存在。")
        sys.exit(1)

    print(f"[NExT-QA] 找到 {len(pq_files)} 个 parquet 文件:")
    for (mode, split), path in sorted(pq_files.items()):
        print(f"  - {mode}/{split}: {path.name}")

    # 统计
    total = 0
    records = []

    for (mode, split), pq_path in sorted(pq_files.items()):
        df = pd.read_parquet(pq_path)
        count = len(df)
        print(f"  处理 {mode}/{split}: {count} 条...")

        for _, row in df.iterrows():
            if mode == "OE":
                rec = process_oe_row(row.to_dict(), split)
            else:
                rec = process_mc_row(row.to_dict(), split)
            records.append(rec)

        total += count

    # 写入 train.jsonl
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[NExT-QA] ✅ train.jsonl 生成完毕!")
    print(f"  总条数: {total}")
    print(f"  输出: {out_path}")

    # 额外校验：检查视频文件是否存在
    video_dir = base / "videos" / "NExTVideo"
    if video_dir.exists():
        existing_videos = set(p.name for p in video_dir.glob("*.mp4"))
        referenced_videos = set(Path(r["video"]).name for r in records)
        missing = referenced_videos - existing_videos
        if missing:
            print(f"\n[WARN] 有 {len(missing)} 个标注引用的视频不存在（前10个）:")
            for v in sorted(missing)[:10]:
                print(f"    {v}")
            print(f"  已有视频: {len(existing_videos)}, 标注引用: {len(referenced_videos)}")
        else:
            print(f"  视频校验: ✅ 所有 {len(referenced_videos)} 个引用视频均存在")
    else:
        print(f"\n[WARN] 视频目录不存在: {video_dir}")


if __name__ == "__main__":
    main()
