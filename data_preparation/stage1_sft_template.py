"""
Stage 1-SFT 来源A: 规则模板生成（无 LLM 成本，直接由数据集标注模板化）。

每条样本对应一种 <observe type="..."/> 工具，覆盖 7 个真实感知工具：
  A-1 temporal_locate    (Charades-STA + DiDeMo)        10K
  A-2 temporal_clip      (Charades-STA + DiDeMo)         3K
  A-3 spatial_detect     (VIPSeg + HC-STVG v2)            3K
  A-4 spatial_crop       (VIPSeg + HC-STVG v2)            2K
  A-5 tracking_describe  (VIPSeg + HC-STVG v2)            2K   verifiable=False
  A-6 depth_overlay      (VIPSeg 全景分割 124类)          2K   verifiable=False
  A-7 ocr_zoom           (TextVR)                        2K

注：
- A-1 中 ActivityNet 视频依赖 YouTube，本仓库不下载，故由 DiDeMo 替代。
- A-3/A-4/A-5 主力数据源为 VIPSeg（124类通用物体，bbox 指代），HC-STVG v2 就绪后
  作为补充（人的跟踪 + 自然语言描述）。VIPSeg 用类别名做 referring text，
  bbox 坐标做 spatial_crop 的输入。
- A-5 tracking_describe：VIPSeg 利用跨帧 instance_id 一致性构建 bbox 序列，
  gt_caption 为类别名+运动方向描述；HC-STVG 用自带的整段 caption。
  verifiable=False，推理时由 SAM3+DINO 在 pipeline 里给出框/高亮辅助 Teacher_P 描述。

输出格式: JSONL，每行一个样本，字段:
  video, question, trajectory, gt_answer, verifiable, type, source, params
"""
from __future__ import annotations

import argparse
import json
import os
import random

# 全局变量：控制 iter_*() 函数读取的数据集分割
# 可通过命令行参数 --data_split 修改（在 main() 中设置）
_DATA_SPLIT = "train"
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

from tqdm import tqdm

# 允许 `python -m data_preparation.stage1_sft_template` 与脚本直接运行
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.paths import (  # noqa: E402
    CHARADES_STA_PATH,
    CLEVRER_PATH,
    DIDEMO_PATH,
    HC_STVG_PATH,
    NEXTQA_PATH,
    STAR_PATH,
    VIPSEG_PATH,
    SFT_DATA_PATH,
    TEXTVR_PATH,
    ensure_dirs,
)
from utils.parser import parse_answer, parse_observe, parse_result, split_segments  # noqa: E402

# ----------------------------------------------------------------------------
# 模板函数：每个对应一种 <observe type=.../>
# ----------------------------------------------------------------------------

def wrap_temporal_locate(video_path: str, query: str, gt_start: float, gt_end: float,
                         source: str) -> dict:
    """A-1 temporal_locate"""
    trajectory = (
        "<think>\n"
        f"[Analyze] Need to localize when \"{query}\" happens.\n"
        f"<observe type=\"temporal_locate\" target=\"{query}\"/>\n"
        f"<result>It happens from {gt_start:.1f}s to {gt_end:.1f}s.</result>\n"
        "[Conclude] Time span confirmed.\n"
        "</think>\n"
        f"<answer>{gt_start:.1f}s-{gt_end:.1f}s</answer>"
    )
    return {
        "video": video_path,
        "question": f"At what time in the video does \"{query}\" happen?",
        "trajectory": trajectory,
        "gt_answer": f"{gt_start:.1f}s-{gt_end:.1f}s",
        "verifiable": True,
        "type": "temporal_locate",
        "source": source,
        "params": {"target": query},
    }


def wrap_temporal_clip(video_path: str, start: float, end: float, gt_caption: str,
                       source: str) -> dict:
    """A-2 temporal_clip"""
    trajectory = (
        "<think>\n"
        "[Analyze] Need to inspect what happens within the given time span.\n"
        f"<observe type=\"temporal_clip\" time=\"{start:.1f}-{end:.1f}\" target=\"activity within this time span\"/>\n"
        f"<result>{gt_caption}</result>\n"
        "[Conclude] Description finished.\n"
        "</think>\n"
        f"<answer>{gt_caption}</answer>"
    )
    return {
        "video": video_path,
        "question": f"Describe what happens between {start:.1f}s and {end:.1f}s in the video.",
        "trajectory": trajectory,
        "gt_answer": gt_caption,
        "verifiable": False,
        "type": "temporal_clip",
        "source": source,
        "params": {"time": f"{start:.1f}-{end:.1f}", "target": "activity within this time span"},
    }


def wrap_spatial_detect(video_path: str, frame_time: float, gt_bbox: list,
                        referring_text: str, source: str) -> dict:
    """A-3 spatial_detect"""
    b = [int(round(x)) for x in gt_bbox]
    bbox_str = f"[{b[0]},{b[1]},{b[2]},{b[3]}]"
    trajectory = (
        "<think>\n"
        f"[Analyze] Need to localize \"{referring_text}\" in the given frame.\n"
        f"<observe type=\"spatial_detect\" frame=\"{frame_time:.1f}\" target=\"{referring_text}\"/>\n"
        f"<result>{referring_text} is located at {bbox_str}.</result>\n"
        "[Conclude] Position confirmed.\n"
        "</think>\n"
        f"<answer>{bbox_str}</answer>"
    )
    return {
        "video": video_path,
        "question": f"At {frame_time:.1f}s, where is {referring_text} in the frame?",
        "trajectory": trajectory,
        "gt_answer": bbox_str,
        "verifiable": True,
        "type": "spatial_detect",
        "source": source,
        "params": {"frame": f"{frame_time:.1f}", "target": referring_text},
    }


def wrap_spatial_crop(video_path: str, frame_time: float, bbox: list,
                      gt_description: str, source: str) -> dict:
    """A-4 spatial_crop（反向：给定bbox要求描述）"""
    b = [int(round(x)) for x in bbox]
    bbox_str = f"[{b[0]},{b[1]},{b[2]},{b[3]}]"
    trajectory = (
        "<think>\n"
        "[Analyze] Need to inspect the content inside the given region.\n"
        f"<observe type=\"spatial_crop\" frame=\"{frame_time:.1f}\" bbox=\"{bbox_str}\" target=\"object inside this region\"/>\n"
        f"<result>{gt_description}</result>\n"
        "[Conclude] Description finished.\n"
        "</think>\n"
        f"<answer>{gt_description}</answer>"
    )
    return {
        "video": video_path,
        "question": f"At {frame_time:.1f}s, what is inside the region {bbox_str}?",
        "trajectory": trajectory,
        "gt_answer": gt_description,
        "verifiable": False,
        "type": "spatial_crop",
        "source": source,
        "params": {"frame": f"{frame_time:.1f}", "bbox": bbox_str},
    }


def wrap_tracking_describe(video_path: str, start: float, end: float,
                           target_desc: str, gt_caption: str,
                           source: str,
                           hint_bbox_start: list | None = None,
                           hint_bbox_end: list | None = None) -> dict:
    """A-5 tracking_describe（无监督的运动轨迹描述）

    pipeline 端 (`tracking_overlay`) 会用 SAM3+DINO 把 target 在每帧 mask/bbox 高亮，
    Teacher_P 据此自然语言描述运动轨迹。这里 gt 用数据集自带的整段 caption，
    verifiable=False，仅做格式监督。
    """
    time_str = f"{start:.1f}-{end:.1f}"
    # 可选：起止帧框作为角色锚定提示，写到 <observe> 的注释里以丰富 prompt
    obs_extra = ""
    if hint_bbox_start is not None and hint_bbox_end is not None:
        bs = [int(round(x)) for x in hint_bbox_start]
        be = [int(round(x)) for x in hint_bbox_end]
        obs_extra = (f' bbox_start="[{bs[0]},{bs[1]},{bs[2]},{bs[3]}]"'
                     f' bbox_end="[{be[0]},{be[1]},{be[2]},{be[3]}]"')
    trajectory = (
        "<think>\n"
        f"[Analyze] Need to describe the trajectory and activity of \"{target_desc}\" during this period.\n"
        f"<observe type=\"tracking_overlay\" time=\"{time_str}\" target=\"{target_desc}\"{obs_extra}/>\n"
        f"<result>{gt_caption}</result>\n"
        "[Conclude] Trajectory and activity summarized.\n"
        "</think>\n"
        f"<answer>{gt_caption}</answer>"
    )
    return {
        "video": video_path,
        "question": (f"During {start:.1f}s-{end:.1f}s of the video, what does \"{target_desc}\" do? "
                     f"Describe its movement trajectory and actions."),
        "trajectory": trajectory,
        "gt_answer": gt_caption,
        "verifiable": False,
        "type": "tracking_overlay",
        "source": source,
        "params": {"time": time_str, "target": target_desc},
    }


def wrap_depth_overlay(video_path: str, frame_time: float, objects: list,
                       gt_answer: str, question: str, source: str) -> dict:
    """A-6 depth_overlay（空间/深度关系问答）"""
    obj_csv = ",".join(objects)
    trajectory = (
        "<think>\n"
        f"[Analyze] Need to inspect the depth/spatial relation of objects in the given frame.\n"
        f"<observe type=\"depth_overlay\" frame=\"{frame_time:.1f}\" "
        f"objects=\"{obj_csv}\" target=\"{question}\"/>\n"
        f"<result>{gt_answer}</result>\n"
        "[Conclude] Spatial relation derived.\n"
        "</think>\n"
        f"<answer>{gt_answer}</answer>"
    )
    return {
        "video": video_path,
        "question": question,
        "trajectory": trajectory,
        "gt_answer": gt_answer,
        "verifiable": False,
        "type": "depth_overlay",
        "source": source,
        "params": {"frame": f"{frame_time:.1f}", "objects": obj_csv},
    }


def wrap_ocr_zoom(video_path: str, frame_time: float, gt_text: str,
                  source: str, bbox: list | None = None) -> dict:
    """A-7 ocr_zoom

    bbox 可选：如果有 bbox 则指定区域 OCR，否则全帧 OCR。
    """
    if bbox is not None:
        b = [int(round(x)) for x in bbox]
        bbox_str = f"[{b[0]},{b[1]},{b[2]},{b[3]}]"
        obs_attr = f' bbox="{bbox_str}"'
        question = f"What text is written in the region {bbox_str} at frame {frame_time:.1f}s?"
    else:
        bbox_str = None
        obs_attr = ""
        question = f"What text can you read in the video at {frame_time:.1f}s?"

    trajectory = (
        "<think>\n"
        "[Analyze] Need to read the text visible in the given frame.\n"
        f"<observe type=\"ocr_zoom\" frame=\"{frame_time:.1f}\"{obs_attr} target=\"text content\"/>\n"
        f"<result>{gt_text}</result>\n"
        "[Conclude] OCR finished.\n"
        "</think>\n"
        f"<answer>{gt_text}</answer>"
    )
    params: dict = {"frame": f"{frame_time:.1f}"}
    if bbox_str:
        params["bbox"] = bbox_str
    return {
        "video": video_path,
        "question": question,
        "trajectory": trajectory,
        "gt_answer": gt_text,
        "verifiable": True,
        "type": "ocr_zoom",
        "source": source,
        "params": params,
    }


# ----------------------------------------------------------------------------
# 数据集读取
# 标注文件均要求项目落地为简化版 JSONL（除 Charades-STA 保留官方txt）。
# 各 loader 在缺数据时直接 raise（禁止容错跳过）。
# ----------------------------------------------------------------------------

def _load_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _resolve_video(video_dir: Path, name: str) -> Path:
    """允许传入相对名或绝对路径；不存在则尝试同名其他扩展。"""
    if os.path.isabs(name):
        p = Path(name)
        if not p.exists():
            raise FileNotFoundError(f"视频不存在: {p}")
        return p
    p = video_dir / name
    if p.exists():
        return p
    # 尝试 stem.* 兜底
    stem = Path(name).stem
    alt = list(video_dir.glob(f"{stem}.*"))
    if alt:
        return alt[0]
    raise FileNotFoundError(f"视频不存在: {p}")


def iter_charades_sta(split: str = None) -> Iterator[dict]:
    """Charades-STA: 读取标准 JSONL 格式 ({split}.jsonl)

    标准 JSONL 格式：{video, query, start, end, duration}
    视频在 CHARADES_STA_PATH/videos/
    """
    if split is None:
        split = _DATA_SPLIT
    ann_file = CHARADES_STA_PATH / f"{split}.jsonl"
    if not ann_file.exists():
        raise FileNotFoundError(f"Charades-STA标注未找到: {ann_file}")
    video_dir = CHARADES_STA_PATH / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"Charades-STA视频目录未找到: {video_dir}")
    for item in _load_jsonl(ann_file):
        for k in ("video", "query", "start", "end"):
            if k not in item:
                raise KeyError(f"Charades-STA字段缺失{k}: {item}")
        video_path = _resolve_video(video_dir, item["video"])
        yield {
            "video": str(video_path),
            "query": str(item["query"]).strip(),
            "start": float(item["start"]),
            "end": float(item["end"]),
            "duration": float(item.get("duration", 0.0)),
        }


def iter_didemo(split: str = None) -> Iterator[dict]:
    """DiDeMo（替代 ActivityNet 用于 temporal_locate）。

    简化 JSONL：{video, query, start, end}
    视频在 DIDEMO_PATH/videos/，标注在 DIDEMO_PATH/{split}.jsonl
    DiDeMo 视频源自 Flickr/YFCC（非 YouTube），可直接下载。
    """
    if split is None:
        split = _DATA_SPLIT
    ann_file = DIDEMO_PATH / f"{split}.jsonl"
    if not ann_file.exists():
        raise FileNotFoundError(f"DiDeMo标注未找到: {ann_file}")
    video_dir = DIDEMO_PATH / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"DiDeMo视频目录未找到: {video_dir}")
    for item in _load_jsonl(ann_file):
        for k in ("video", "query", "start", "end"):
            if k not in item:
                raise KeyError(f"DiDeMo字段缺失{k}: {item}")
        video_path = _resolve_video(video_dir, item["video"])
        yield {
            "video": str(video_path),
            "query": str(item["query"]).strip(),
            "start": float(item["start"]),
            "end": float(item["end"]),
        }


def iter_hc_stvg(split: str = None) -> Iterator[dict]:
    """HC-STVG v2（spatial_detect/crop + tracking_describe 共用）。

    简化 JSONL：
      {video, start, end, fps, bbox_seq:[[x1,y1,x2,y2],...] (按帧/按秒采样),
       description}
    视频在 HC_STVG_PATH/videos/。
    """
    if split is None:
        split = _DATA_SPLIT
    ann_file = HC_STVG_PATH / f"{split}.jsonl"
    if not ann_file.exists():
        raise FileNotFoundError(f"HC-STVG标注未找到: {ann_file}")
    video_dir = HC_STVG_PATH / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"HC-STVG视频目录未找到: {video_dir}")
    for item in _load_jsonl(ann_file):
        for k in ("video", "start", "end", "bbox_seq", "description"):
            if k not in item:
                raise KeyError(f"HC-STVG样本缺字段{k}: {item}")
        bbox_seq = item["bbox_seq"]
        if not bbox_seq:
            raise ValueError(f"HC-STVG bbox_seq 为空: {item.get('video')}")
        video_path = _resolve_video(video_dir, item["video"])
        yield {
            "video": str(video_path),
            "start": float(item["start"]),
            "end": float(item["end"]),
            "fps": float(item.get("fps", 1.0)),
            "bbox_seq": [list(b) for b in bbox_seq],
            "description": str(item["description"]).strip(),
        }


def iter_vipseg(split: str = None) -> Iterator[dict]:
    """VIPSeg 视频全景分割数据集（depth_overlay + spatial）。

    train.jsonl 每行：{video_dir, video, frame, frame_time, frame_path, panoptic_path,
                       objects: [{name, bbox, category_id, instance_id, area}]}
    此函数读取 train.jsonl 并返回每条记录（含物体列表）。
    物体指代方式：直接用 bbox 坐标 [x1,y1,x2,y2]，无需类别唯一性约束。

    video 字段指向 2fps 合成的 mp4 视频（由 vipseg_frames_to_video.py 生成）。
    frame_time 是该帧在合成视频中的精确时间戳（秒）。
    """
    if split is None:
        split = _DATA_SPLIT
    ann_file = VIPSEG_PATH / f"{split}.jsonl"
    if not ann_file.exists():
        raise FileNotFoundError(f"VIPSeg标注未找到: {ann_file}")
    videos_mp4_dir = VIPSEG_PATH / "videos_mp4"
    if not videos_mp4_dir.exists():
        raise FileNotFoundError(
            f"VIPSeg 合成视频目录未找到: {videos_mp4_dir}\n"
            f"  请先运行: python scripts/vipseg_frames_to_video.py"
        )
    for item in _load_jsonl(ann_file):
        for k in ("video_dir", "video", "frame", "frame_time", "objects"):
            if k not in item:
                raise KeyError(f"VIPSeg字段缺失{k}: {item}")
        # 检查合成视频是否存在
        video_path = VIPSEG_PATH / item["video"]
        if not video_path.exists():
            continue
        objects = item["objects"]
        if not isinstance(objects, list) or len(objects) < 2:
            continue  # 至少需要 2 个物体才能出空间关系题
        yield {
            "video_dir": item["video_dir"],
            "video": str(video_path),
            "frame": item["frame"],
            "frame_time": float(item["frame_time"]),
            "frame_path": str(VIPSEG_PATH / item.get("frame_path", "")),
            "panoptic_path": str(VIPSEG_PATH / item.get("panoptic_path", "")),
            "objects": objects,
        }


def iter_textvr(split: str = None) -> Iterator[dict]:
    """TextVR（ocr_zoom）。

    实际 JSONL 格式：{video, captions: [str,...], duration: float}
    无逐帧 bbox 标注，frame_time 取视频中间帧。
    """
    if split is None:
        split = _DATA_SPLIT
    ann_file = TEXTVR_PATH / f"{split}.jsonl"
    if not ann_file.exists():
        raise FileNotFoundError(f"TextVR标注未找到: {ann_file}")
    video_dir = TEXTVR_PATH / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"TextVR视频目录未找到: {video_dir}")
    for item in _load_jsonl(ann_file):
        if "video" not in item:
            raise KeyError(f"TextVR字段缺失video: {item}")
        captions = item.get("captions", [])
        if not captions:
            continue  # 跳过无 caption 的条目
        duration = float(item.get("duration", 10.0))
        video_path = _resolve_video(video_dir, item["video"])
        yield {
            "video": str(video_path),
            "frame_time": duration / 2.0,  # 取视频中间帧
            "captions": captions,
            "duration": duration,
        }


# ----------------------------------------------------------------------------
# 生成函数
# ----------------------------------------------------------------------------

def _validate_sample(sample: dict) -> None:
    """格式校验：parser 能解析，且字段完整"""
    traj = sample["trajectory"]
    ans = parse_answer(traj)
    if ans is None:
        raise ValueError(f"模板样本缺少<answer>: {traj[:200]}")

    # A-8 raw_videoqa 不使用 <observe>/<result>，跳过工具标签校验
    if sample.get("type") == "raw_videoqa":
        return

    obs = parse_observe(traj)
    res = parse_result(traj)
    if len(obs) != 1:
        raise ValueError(f"模板样本应恰好1个<observe>，实得{len(obs)}: {traj[:200]}")
    if len(res) != 1:
        raise ValueError(f"模板样本应恰好1个<result>，实得{len(res)}: {traj[:200]}")
    split_segments(traj)


def _hcstvg_pick_frame_bbox(item: dict, rng: random.Random):
    """从 HC-STVG bbox_seq 中随机抽一帧 (frame_time, bbox)。"""
    n = len(item["bbox_seq"])
    if n == 0:
        raise ValueError("HC-STVG bbox_seq 为空")
    idx = rng.randrange(n)
    fps = max(item.get("fps", 1.0), 1e-3)
    # bbox_seq 按整段 [start, end] 内均匀采样保存（fps 表示 bbox 的采样率）
    frame_time = item["start"] + idx / fps
    if frame_time > item["end"]:
        frame_time = item["end"]
    return frame_time, item["bbox_seq"][idx]


# ---------- A-1 ----------
def gen_a1_temporal_locate(n_charades: int, n_didemo: int, seed: int, data_split: str = "train") -> list[dict]:
    rng = random.Random(seed)
    samples: list[dict] = []

    items = list(iter_charades_sta(data_split))
    rng.shuffle(items)
    for it in tqdm(items[:n_charades], desc="A-1 Charades-STA"):
        samples.append(wrap_temporal_locate(
            it["video"], it["query"], it["start"], it["end"],
            source="charades_sta",
        ))

    di_items = list(iter_didemo(data_split))
    rng.shuffle(di_items)
    for it in tqdm(di_items[:n_didemo], desc="A-1 DiDeMo"):
        samples.append(wrap_temporal_locate(
            it["video"], it["query"], it["start"], it["end"],
            source="didemo",
        ))
    return samples


# ---------- A-2 ----------
def gen_a2_temporal_clip(n_charades: int, n_didemo: int, seed: int, data_split: str = "train") -> list[dict]:
    """A-2 temporal_clip: 用 Charades-STA + DiDeMo 的标注片段做时间段描述。"""
    rng = random.Random(seed + 1)
    samples: list[dict] = []

    # Charades-STA: query 作为 gt_caption
    items = list(iter_charades_sta(data_split))
    rng.shuffle(items)
    for it in tqdm(items[:n_charades], desc="A-2 Charades-STA temporal_clip"):
        samples.append(wrap_temporal_clip(
            it["video"], it["start"], it["end"], it["query"],
            source="charades_sta",
        ))

    # DiDeMo: query 作为 gt_caption
    di_items = list(iter_didemo(data_split))
    rng.shuffle(di_items)
    for it in tqdm(di_items[:n_didemo], desc="A-2 DiDeMo temporal_clip"):
        samples.append(wrap_temporal_clip(
            it["video"], it["start"], it["end"], it["query"],
            source="didemo",
        ))
    return samples


# ---------- A-3 ----------
def gen_a3_spatial_detect(n_vipseg: int, n_hcstvg: int, seed: int, data_split: str = "train") -> list[dict]:
    """A-3 spatial_detect: VIPSeg（类别名→bbox）+ HC-STVG（描述→bbox）。

    VIPSeg: 给定帧时间 + 物体类别名，定位 bbox。
    HC-STVG: 给定帧时间 + 自然语言描述，定位 bbox。
    """
    rng = random.Random(seed + 2)
    samples = []

    # --- VIPSeg 部分 ---
    vipseg_items = list(iter_vipseg(data_split))
    rng.shuffle(vipseg_items)
    vipseg_count = 0
    for it in tqdm(vipseg_items, desc="A-3 VIPSeg spatial_detect"):
        if vipseg_count >= n_vipseg:
            break
        objects = it["objects"]
        # 随机选一个物体
        obj = rng.choice(objects)
        # 用类别名作为 referring text（如 "the cup", "the chair"）
        referring = f"the {obj['name']}"
        samples.append(wrap_spatial_detect(
            it["video"], it["frame_time"], obj["bbox"], referring,
            source="vipseg",
        ))
        vipseg_count += 1

    # --- HC-STVG 部分（如果可用）---
    if n_hcstvg > 0:
        try:
            hc_items = list(iter_hc_stvg(data_split))
            rng.shuffle(hc_items)
            for it in tqdm(hc_items[:n_hcstvg], desc="A-3 HC-STVG spatial_detect"):
                frame_time, bbox = _hcstvg_pick_frame_bbox(it, rng)
                samples.append(wrap_spatial_detect(
                    it["video"], frame_time, bbox, it["description"],
                    source="hc_stvg",
                ))
        except FileNotFoundError as e:
            print(f"[A-3] HC-STVG 不可用，跳过: {e}")

    return samples


# ---------- A-4 ----------
def gen_a4_spatial_crop(n_vipseg: int, n_hcstvg: int, seed: int, data_split: str = "train") -> list[dict]:
    """A-4 spatial_crop: VIPSeg（bbox→类别名）+ HC-STVG（bbox→描述）。

    VIPSeg: 给定帧时间 + bbox，回答该区域是什么物体。
    HC-STVG: 给定帧时间 + bbox，回答该区域的自然语言描述。
    """
    rng = random.Random(seed + 3)
    samples = []

    # --- VIPSeg 部分 ---
    vipseg_items = list(iter_vipseg(data_split))
    rng.shuffle(vipseg_items)
    vipseg_count = 0
    for it in tqdm(vipseg_items, desc="A-4 VIPSeg spatial_crop"):
        if vipseg_count >= n_vipseg:
            break
        objects = it["objects"]
        obj = rng.choice(objects)
        # gt_description 是类别名
        gt_desc = obj["name"]
        samples.append(wrap_spatial_crop(
            it["video"], it["frame_time"], obj["bbox"], gt_desc,
            source="vipseg",
        ))
        vipseg_count += 1

    # --- HC-STVG 部分（如果可用）---
    if n_hcstvg > 0:
        try:
            hc_items = list(iter_hc_stvg(data_split))
            rng.shuffle(hc_items)
            for it in tqdm(hc_items[:n_hcstvg], desc="A-4 HC-STVG spatial_crop"):
                frame_time, bbox = _hcstvg_pick_frame_bbox(it, rng)
                samples.append(wrap_spatial_crop(
                    it["video"], frame_time, bbox, it["description"],
                    source="hc_stvg",
                ))
        except FileNotFoundError as e:
            print(f"[A-4] HC-STVG 不可用，跳过: {e}")

    return samples


# ---------- A-5 ----------
def gen_a5_tracking_describe(n_vipseg: int, n_hcstvg: int, seed: int, data_split: str = "train") -> list[dict]:
    """A-5 tracking_describe: VIPSeg（跨帧 bbox 序列）+ HC-STVG（人的轨迹描述）。

    VIPSeg: 利用同一 instance_id 在不同帧中的 bbox 构建轨迹，
    gt_caption 为 "The {name} moves from [{bbox_start}] to [{bbox_end}]."
    HC-STVG: 用自带的整段 caption 作为 gt。
    """
    rng = random.Random(seed + 4)
    samples = []

    # --- VIPSeg 部分：跨帧 tracking ---
    # 需要从 train.jsonl 中找到同一视频的多帧，追踪同一 instance
    vipseg_items = list(iter_vipseg(data_split))
    # 按 video_dir 分组
    from collections import defaultdict
    video_frames: dict[str, list[dict]] = defaultdict(list)
    for it in vipseg_items:
        video_frames[it["video_dir"]].append(it)

    # 对每个视频，找跨帧出现的同一 instance
    vipseg_count = 0
    video_ids = list(video_frames.keys())
    rng.shuffle(video_ids)
    for vid in tqdm(video_ids, desc="A-5 VIPSeg tracking_describe"):
        if vipseg_count >= n_vipseg:
            break
        frames = sorted(video_frames[vid], key=lambda x: x["frame_time"])
        if len(frames) < 2:
            continue

        # 收集所有 (category_id, instance_id) 在各帧中的出现
        inst_appearances: dict[tuple, list[tuple]] = defaultdict(list)
        for fr in frames:
            for obj in fr["objects"]:
                key = (obj["category_id"], obj["instance_id"])
                inst_appearances[key].append((fr["frame_time"], obj["bbox"], obj["name"], fr["video"]))

        # 选出跨 ≥2 帧出现的 instance
        multi_frame_insts = [(k, v) for k, v in inst_appearances.items() if len(v) >= 2]
        if not multi_frame_insts:
            continue

        # 随机选一个 instance
        _, appearances = rng.choice(multi_frame_insts)
        appearances.sort(key=lambda x: x[0])  # 按时间排序
        start_time, bbox_start, name, video_path = appearances[0]
        end_time, bbox_end, _, _ = appearances[-1]

        # 构建 gt_caption
        bs_str = f"[{bbox_start[0]},{bbox_start[1]},{bbox_start[2]},{bbox_start[3]}]"
        be_str = f"[{bbox_end[0]},{bbox_end[1]},{bbox_end[2]},{bbox_end[3]}]"
        gt_caption = f"The {name} moves from {bs_str} to {be_str}."

        samples.append(wrap_tracking_describe(
            video_path=video_path,
            start=start_time, end=end_time,
            target_desc=f"the {name}",
            gt_caption=gt_caption,
            source="vipseg",
            hint_bbox_start=bbox_start,
            hint_bbox_end=bbox_end,
        ))
        vipseg_count += 1

    # --- HC-STVG 部分（如果可用）---
    if n_hcstvg > 0:
        try:
            hc_items = list(iter_hc_stvg(data_split))
            rng.shuffle(hc_items)
            for it in tqdm(hc_items[:n_hcstvg], desc="A-5 HC-STVG tracking_describe"):
                bs = it["bbox_seq"][0]
                be = it["bbox_seq"][-1]
                samples.append(wrap_tracking_describe(
                    video_path=it["video"],
                    start=it["start"], end=it["end"],
                    target_desc=it["description"],
                    gt_caption=it["description"],
                    source="hc_stvg",
                    hint_bbox_start=bs,
                    hint_bbox_end=be,
                ))
        except FileNotFoundError as e:
            print(f"[A-5] HC-STVG 不可用，跳过: {e}")

    return samples


# ---------- A-6 ----------
def gen_a6_depth_overlay(n: int, seed: int, data_split: str = "train") -> list[dict]:
    """VIPSeg 基于全景分割标注生成 depth_overlay 训练数据。

    物体指代方式：直接用 bbox 坐标 [x1,y1,x2,y2]。
    例如："Which is closer to the camera, the object at [120,80,200,160]
           or the object at [450,200,550,320]?"

    策略：从 VIPSeg 的帧中随机选取两个物体（用 bbox 指代），
    生成空间关系问答。实际深度比较在推理时由 Depth Anything V2 完成。
    训练时 gt_answer 使用 bbox 中心点的相对位置关系（左/右/上/下）作为
    占位监督信号，后续由 Depth Anything V2 离线预计算替换为真实深度答案。
    """
    rng = random.Random(seed + 5)
    items = list(iter_vipseg(data_split))
    rng.shuffle(items)
    samples = []

    for it in tqdm(items[:n], desc="A-6 VIPSeg depth_overlay"):
        objects = it["objects"]
        # 随机选两个不同物体
        if len(objects) < 2:
            continue
        obj_a, obj_b = rng.sample(objects, 2)

        # 用 bbox 坐标指代物体
        bbox_a = obj_a["bbox"]  # [x1, y1, x2, y2]
        bbox_b = obj_b["bbox"]
        bbox_a_str = f"[{bbox_a[0]},{bbox_a[1]},{bbox_a[2]},{bbox_a[3]}]"
        bbox_b_str = f"[{bbox_b[0]},{bbox_b[1]},{bbox_b[2]},{bbox_b[3]}]"

        # 物体名称（用于 gt_answer 中辅助说明）
        name_a = obj_a["name"]
        name_b = obj_b["name"]

        # gt_answer 占位：后续由 Depth Anything V2 离线预计算替换
        # 这里暂时用 bbox 中心 y 坐标作为粗略深度代理
        # （图像中 y 越大通常越近，但这只是占位逻辑）
        cy_a = (bbox_a[1] + bbox_a[3]) / 2
        cy_b = (bbox_b[1] + bbox_b[3]) / 2
        if cy_a > cy_b:
            gt_answer = f"The object at {bbox_a_str} ({name_a}) is closer."
        else:
            gt_answer = f"The object at {bbox_b_str} ({name_b}) is closer."

        # 使用合成视频路径和真实时间戳
        frame_time = it["frame_time"]
        question = (f"At {frame_time:.1f}s, which is closer to the camera, "
                    f"the object at {bbox_a_str} or the object at {bbox_b_str}?")

        samples.append(wrap_depth_overlay(
            video_path=it["video"],
            frame_time=frame_time,
            objects=[bbox_a_str, bbox_b_str],
            gt_answer=gt_answer,
            question=question,
            source="vipseg",
        ))
    return samples


# ---------- A-7 ----------
def gen_a7_ocr_zoom(n: int, seed: int, data_split: str = "train") -> list[dict]:
    rng = random.Random(seed + 6)
    items = list(iter_textvr(data_split))
    rng.shuffle(items)
    samples = []
    for it in tqdm(items[:n], desc="A-7 TextVR ocr_zoom"):
        # 从 captions 中随机选一条非空的作为 gt_text
        valid_caps = [c for c in it["captions"] if c.strip()]
        if not valid_caps:
            continue
        gt_text = rng.choice(valid_caps)
        samples.append(wrap_ocr_zoom(
            video_path=it["video"],
            frame_time=it["frame_time"],
            gt_text=gt_text,
            source="textvr",
            bbox=None,  # TextVR 无 bbox 标注
        ))
    return samples


# ---------- A-8 ----------
def iter_star(split: str = None) -> Iterator[dict]:
    """STAR 数据集（情境推理多选题，复用 Charades 视频）。

    train.jsonl: {video, question, choices: [str,...], gt_answer, type, qid}
    """
    if split is None:
        split = _DATA_SPLIT
    ann_file = STAR_PATH / f"{split}.jsonl"
    if not ann_file.exists():
        raise FileNotFoundError(f"STAR标注未找到: {ann_file}")
    video_dir = STAR_PATH / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"STAR视频目录未找到: {video_dir}")
    for item in _load_jsonl(ann_file):
        if "video" not in item or "question" not in item or "gt_answer" not in item:
            continue
        video_path = _resolve_video(video_dir, item["video"])
        yield {
            "video": str(video_path),
            "question": item["question"],
            "choices": item.get("choices", []),
            "gt_answer": item["gt_answer"],
            "type": item.get("type", ""),
            "qid": item.get("qid", ""),
        }


def iter_nextqa(split: str = None) -> Iterator[dict]:
    """NExT-QA 数据集（因果/时序推理多选题）。

    train.jsonl: {video, question, choices: [str,...], gt_answer, type, qid}
    注意：video 字段已包含 "videos/NExTVideo/xxx.mp4" 相对路径。
    """
    if split is None:
        split = _DATA_SPLIT
    ann_file = NEXTQA_PATH / f"{split}.jsonl"
    if not ann_file.exists():
        raise FileNotFoundError(f"NExT-QA标注未找到: {ann_file}")
    # video 字段已含 "videos/" 前缀，直接以 NEXTQA_PATH 为基础
    base_dir = NEXTQA_PATH
    for item in _load_jsonl(ann_file):
        if "video" not in item or "question" not in item or "gt_answer" not in item:
            continue
        video_path = _resolve_video(base_dir, item["video"])
        yield {
            "video": str(video_path),
            "question": item["question"],
            "choices": item.get("choices", []),
            "gt_answer": item["gt_answer"],
            "type": item.get("type", ""),
            "qid": item.get("qid", ""),
        }


def iter_clevrer(split: str = None) -> Iterator[dict]:
    """CLEVRER 数据集（合成因果/反事实推理）。

    train.jsonl: {video, question, choices: [str,...], gt_answer, type}
    注意：CLEVRER 部分题目 choices 为空（开放式问答）。
    视频在分片子目录中：videos/video_00000-01000/video_00000.mp4
    """
    if split is None:
        split = _DATA_SPLIT
    ann_file = CLEVRER_PATH / f"{split}.jsonl"
    if not ann_file.exists():
        raise FileNotFoundError(f"CLEVRER标注未找到: {ann_file}")
    video_dir = CLEVRER_PATH / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"CLEVRER视频目录未找到: {video_dir}")

    def _resolve_clevrer_video(name: str) -> Path:
        """根据视频编号找到对应的分片子目录。"""
        # name 格式: "video_00000.mp4"
        stem = Path(name).stem  # "video_00000"
        try:
            num = int(stem.split("_")[1])
        except (IndexError, ValueError):
            # 无法解析编号，尝试直接查找
            return _resolve_video(video_dir, name)
        # 计算分片目录：video_00000-01000, video_01000-02000, ...
        bucket_start = (num // 1000) * 1000
        bucket_end = bucket_start + 1000
        subdir = f"video_{bucket_start:05d}-{bucket_end:05d}"
        p = video_dir / subdir / name
        if p.exists():
            return p
        # 兜底：直接在 video_dir 下找
        return _resolve_video(video_dir, name)

    for item in _load_jsonl(ann_file):
        if "video" not in item or "question" not in item or "gt_answer" not in item:
            continue
        try:
            video_path = _resolve_clevrer_video(item["video"])
        except FileNotFoundError:
            continue  # 视频缺失则跳过
        yield {
            "video": str(video_path),
            "question": item["question"],
            "choices": item.get("choices", []),
            "gt_answer": item["gt_answer"],
            "type": item.get("type", ""),
        }


def wrap_raw_videoqa(video_path: str, question: str, choices: list,
                    gt_answer: str, source: str, qa_type: str = "") -> dict:
    """A-8 raw_videoqa（B类原始视频问答，不对应特定工具）。

    多选题：verifiable=True（精确匹配答案）
    开放式：verifiable=False
    """
    has_choices = bool(choices)
    if has_choices:
        # 格式化选项
        options_str = " ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
        full_question = f"{question}\nOptions: {options_str}"
        # 找到 gt_answer 对应的选项字母
        gt_letter = None
        for i, c in enumerate(choices):
            if c.strip() == gt_answer.strip():
                gt_letter = chr(65 + i)
                break
        if gt_letter:
            formatted_answer = f"({gt_letter}) {gt_answer}"
        else:
            formatted_answer = gt_answer
    else:
        full_question = question
        formatted_answer = gt_answer

    # A-8 不使用 <observe> 工具，直接推理回答
    trajectory = (
        "<think>\n"
        f"[Analyze] {question}\n"
        "[Reason] Based on the video content, I can determine the answer.\n"
        "</think>\n"
        f"<answer>{formatted_answer}</answer>"
    )
    return {
        "video": video_path,
        "question": full_question,
        "trajectory": trajectory,
        "gt_answer": formatted_answer,
        "verifiable": has_choices,
        "type": "raw_videoqa",
        "source": source,
        "params": {"qa_type": qa_type},
    }


def gen_a8_raw_videoqa(n_star: int, n_nextqa: int, n_clevrer: int, seed: int, data_split: str = "train") -> list[dict]:
    """A-8 raw_videoqa: STAR + NExT-QA + CLEVRER。"""
    rng = random.Random(seed + 7)
    samples = []

    # --- STAR ---
    try:
        star_items = list(iter_star(data_split))
        rng.shuffle(star_items)
        for it in tqdm(star_items[:n_star], desc="A-8 STAR raw_videoqa"):
            samples.append(wrap_raw_videoqa(
                it["video"], it["question"], it["choices"], it["gt_answer"],
                source="star", qa_type=it.get("type", ""),
            ))
    except FileNotFoundError as e:
        print(f"[A-8] STAR 不可用，跳过: {e}")

    # --- NExT-QA ---
    try:
        nqa_items = list(iter_nextqa(data_split))
        rng.shuffle(nqa_items)
        for it in tqdm(nqa_items[:n_nextqa], desc="A-8 NExT-QA raw_videoqa"):
            samples.append(wrap_raw_videoqa(
                it["video"], it["question"], it["choices"], it["gt_answer"],
                source="nextqa", qa_type=it.get("type", ""),
            ))
    except FileNotFoundError as e:
        print(f"[A-8] NExT-QA 不可用，跳过: {e}")

    # --- CLEVRER ---
    try:
        clv_items = list(iter_clevrer(data_split))
        rng.shuffle(clv_items)
        for it in tqdm(clv_items[:n_clevrer], desc="A-8 CLEVRER raw_videoqa"):
            samples.append(wrap_raw_videoqa(
                it["video"], it["question"], it["choices"], it["gt_answer"],
                source="clevrer", qa_type=it.get("type", ""),
            ))
    except FileNotFoundError as e:
        print(f"[A-8] CLEVRER 不可用，跳过: {e}")

    return samples


# ----------------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------------

def write_jsonl(samples: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


_SUBSETS = ("a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8")
_SUBSET_NAMES = (
    "a1_temporal_locate", "a2_temporal_clip", "a3_spatial_detect",
    "a4_spatial_crop", "a5_tracking_describe", "a6_depth_overlay",
    "a7_ocr_zoom", "a8_raw_videoqa",
)


def main():
    parser = argparse.ArgumentParser(description="Stage 1-SFT 来源A: 规则模板生成")
    parser.add_argument(
        "--subset",
        choices=("all", "merge") + _SUBSETS,
        default="all",
        help="生成的子集（merge=仅合并已有文件，不重新生成）",
    )
    parser.add_argument("--data_split", type=str, default="train",
                        choices=["train", "mini"],
                        help="使用的数据集分割：train=train.jsonl, mini=mini.jsonl（默认: train）")
    parser.add_argument("--n_charades", type=int, default=5000)
    parser.add_argument("--n_didemo", type=int, default=5000)
    parser.add_argument("--n_a2_charades", type=int, default=1500)
    parser.add_argument("--n_a2_didemo", type=int, default=1500)
    parser.add_argument("--n_a3_vipseg", type=int, default=2000)
    parser.add_argument("--n_a3_hcstvg", type=int, default=1000)
    parser.add_argument("--n_a4_vipseg", type=int, default=1500)
    parser.add_argument("--n_a4_hcstvg", type=int, default=500)
    parser.add_argument("--n_a5_vipseg", type=int, default=1500)
    parser.add_argument("--n_a5_hcstvg", type=int, default=500)
    parser.add_argument("--n_a6", type=int, default=2000)
    parser.add_argument("--n_a7", type=int, default=2000)
    parser.add_argument("--n_a8_star", type=int, default=5000)
    parser.add_argument("--n_a8_nextqa", type=int, default=5000)
    parser.add_argument("--n_a8_clevrer", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录，默认使用 configs.paths.SFT_DATA_PATH")
    args = parser.parse_args()

    # 设置全局变量，控制 iter_*() 函数读取的数据集分割
    global _DATA_SPLIT
    _DATA_SPLIT = args.data_split

    ensure_dirs()
    out_dir = Path(args.output_dir) if args.output_dir else SFT_DATA_PATH
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples: list[dict] = []

    def _do(name: str, samples: list[dict]):
        for s in samples:
            _validate_sample(s)
        path = out_dir / f"stage1_sft_{name}.jsonl"
        write_jsonl(samples, path)
        print(f"[OK] {name}: {len(samples)} -> {path}")
        all_samples.extend(samples)

    if args.subset in ("all", "a1"):
        _do("a1_temporal_locate",
            gen_a1_temporal_locate(args.n_charades, args.n_didemo, args.seed, args.data_split))
    if args.subset in ("all", "a2"):
        _do("a2_temporal_clip", gen_a2_temporal_clip(args.n_a2_charades, args.n_a2_didemo, args.seed, args.data_split))
    if args.subset in ("all", "a3"):
        _do("a3_spatial_detect", gen_a3_spatial_detect(args.n_a3_vipseg, args.n_a3_hcstvg, args.seed, args.data_split))
    if args.subset in ("all", "a4"):
        _do("a4_spatial_crop", gen_a4_spatial_crop(args.n_a4_vipseg, args.n_a4_hcstvg, args.seed, args.data_split))
    if args.subset in ("all", "a5"):
        _do("a5_tracking_describe", gen_a5_tracking_describe(args.n_a5_vipseg, args.n_a5_hcstvg, args.seed, args.data_split))
    if args.subset in ("all", "a6"):
        _do("a6_depth_overlay", gen_a6_depth_overlay(args.n_a6, args.seed, args.data_split))
    if args.subset in ("all", "a7"):
        _do("a7_ocr_zoom", gen_a7_ocr_zoom(args.n_a7, args.seed, args.data_split))
    if args.subset in ("all", "a8"):
        _do("a8_raw_videoqa", gen_a8_raw_videoqa(args.n_a8_star, args.n_a8_nextqa, args.n_a8_clevrer, args.seed, args.data_split))

    if args.subset == "all":
        merged_path = out_dir / "stage1_sft_template_all.jsonl"
        write_jsonl(all_samples, merged_path)
        print(f"[OK] merged: {len(all_samples)} -> {merged_path}")

    if args.subset == "merge":
        # 直接读取已有的各子集 JSONL 文件合并，不重新生成
        merged = []
        for name in _SUBSET_NAMES:
            p = out_dir / f"stage1_sft_{name}.jsonl"
            if p.exists():
                items = list(_load_jsonl(p))
                merged.extend(items)
                print(f"  [读取] {name}: {len(items)} 条 <- {p}")
            else:
                print(f"  [跳过] {name}: 文件不存在 {p}")
        merged_path = out_dir / "stage1_sft_template_all.jsonl"
        write_jsonl(merged, merged_path)
        print(f"[OK] merged: {len(merged)} -> {merged_path}")


if __name__ == "__main__":
    main()
