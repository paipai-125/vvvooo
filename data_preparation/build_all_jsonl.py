#!/usr/bin/env python3
"""
统一后处理脚本：为所有缺失 train.jsonl 的数据集生成标准格式。

用法：
  cd /path/to/video_opd_code
  python -m data_preparation.build_all_jsonl              # 处理全部
  python -m data_preparation.build_all_jsonl charades_sta star  # 只处理指定的
"""

import json
import sys
from pathlib import Path

# 全局 force 标志：跳过 "已存在" 检查，强制重新生成
_FORCE = False

try:
    from configs.paths import (
        CHARADES_STA_PATH, STAR_PATH, DIDEMO_PATH,
        VIPSEG_PATH, TEXTVR_PATH, NEXTQA_PATH, CLEVRER_PATH,
        HC_STVG_PATH
    )
except ImportError:
    print("[ERROR] 请在代码根目录运行: cd /path/to/video_opd_code")
    sys.exit(1)


# ============================================================
# 1. Charades-STA → train.jsonl
# ============================================================
def build_charades_sta():
    """
    Charades-STA train.json 格式:
    {video_id: {timestamps: [[s,e],...], sentences: [...], video_duration: float, ...}}
    输出: {video: "VIDEO_ID.mp4", query: "...", start: float, end: float}
    """
    dst = CHARADES_STA_PATH / "train.jsonl"
    if dst.exists() and not _FORCE:
        print(f"[charades_sta] train.jsonl 已存在，跳过。（用 --force 强制重新生成）")
        return

    # 尝试多种标注文件
    src = CHARADES_STA_PATH / "train.json"
    if not src.exists():
        # 尝试 charades_sta_train.txt 格式
        txt_src = CHARADES_STA_PATH / "charades_sta_train.txt"
        if txt_src.exists():
            _build_charades_sta_from_txt(txt_src, dst)
            return
        print(f"[charades_sta] 标注文件不存在: {src} 或 {txt_src}，跳过。")
        return

    with open(src) as f:
        data = json.load(f)

    count = 0
    with open(dst, "w", encoding="utf-8") as g:
        for vid, ann in data.items():
            timestamps = ann.get("timestamps", [])
            sentences = ann.get("sentences", [])
            duration = ann.get("video_duration", 0.0)
            for (start, end), sent in zip(timestamps, sentences):
                rec = {
                    "video": f"{vid}.mp4",
                    "query": sent,
                    "start": float(start),
                    "end": float(end),
                    "duration": float(duration),
                }
                g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1

    print(f"[charades_sta] ✅ train.jsonl 生成完毕: {count} 条 → {dst}")


def _build_charades_sta_from_txt(txt_src: Path, dst: Path):
    """从 charades_sta_train.txt 格式生成 (每行: VIDEO_ID START END##SENTENCE)"""
    count = 0
    with open(txt_src) as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式: VIDEO_ID START END##SENTENCE
            parts = line.split("##")
            if len(parts) != 2:
                continue
            time_part, sentence = parts
            tokens = time_part.strip().split()
            if len(tokens) < 3:
                continue
            vid = tokens[0]
            start = float(tokens[1])
            end = float(tokens[2])
            rec = {
                "video": f"{vid}.mp4",
                "query": sentence.strip(),
                "start": start,
                "end": end,
            }
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    print(f"[charades_sta] ✅ train.jsonl 生成完毕 (from txt): {count} 条 → {dst}")


# ============================================================
# 2. STAR → train.jsonl
# ============================================================
def build_star():
    """
    STAR_train.json: list of {video_id, question, choices, answer, question_type, question_id}
    输出: {video, question, choices, gt_answer, type, qid}
    """
    dst = STAR_PATH / "train.jsonl"
    if dst.exists() and not _FORCE:
        print(f"[star] train.jsonl 已存在，跳过。（用 --force 强制重新生成）")
        return

    src = STAR_PATH / "STAR_train.json"
    if not src.exists():
        print(f"[star] 标注文件不存在: {src}，跳过。")
        return

    with open(src) as f:
        data = json.load(f)

    count = 0
    with open(dst, "w", encoding="utf-8") as g:
        for item in data:
            choices = [c["choice"] for c in item.get("choices", [])]
            gt = item.get("answer", "")
            rec = {
                "video": f"{item['video_id']}.mp4",
                "question": item.get("question", ""),
                "choices": choices,
                "gt_answer": gt,
                "type": item.get("question_type", ""),
                "qid": item.get("question_id", ""),
            }
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"[star] ✅ train.jsonl 生成完毕: {count} 条 → {dst}")


# ============================================================
# 3. DiDeMo → train.jsonl
# ============================================================
def build_didemo():
    """
    DiDeMo raw_data/didemo_train.json 格式:
    [{description, times: [[s_idx, e_idx], ...], video: "USER@ID.avi", ...}, ...]
    时间单位是 5 秒块（每块 5s）。多人标注取中位。
    视频实际在 videos/train/ 目录下，统一为 .mp4 后缀。
    输出: {video: "train/xxx.mp4", query, start, end}
    """
    dst = DIDEMO_PATH / "train.jsonl"
    if dst.exists() and not _FORCE:
        print(f"[didemo] train.jsonl 已存在，跳过。（用 --force 强制重新生成）")
        return

    # 优先使用 raw_data/didemo_train.json（有 times 字段）
    src = DIDEMO_PATH / "raw_data" / "didemo_train.json"
    if not src.exists():
        src = DIDEMO_PATH / "didemo_train.json"
    if not src.exists():
        print(f"[didemo] 标注文件未找到，跳过。")
        return

    with open(src) as f:
        data = json.load(f)

    # 视频目录：videos/train/
    video_dir = DIDEMO_PATH / "videos" / "train"
    if not video_dir.exists():
        print(f"[didemo] 视频目录不存在: {video_dir}，跳过。")
        return

    count = 0
    skipped_no_times = 0
    skipped_no_video = 0
    with open(dst, "w", encoding="utf-8") as g:
        for item in data:
            times = item.get("times", [])
            if not times:
                skipped_no_times += 1
                continue
            # 5 秒块为单位，多人标注取中位
            starts = sorted([t[0] for t in times])
            ends = sorted([t[1] for t in times])
            s = starts[len(starts) // 2] * 5.0
            e = (ends[len(ends) // 2] + 1) * 5.0

            # 视频名映射：原始可能是 .avi/.mov/.mpg 等，实际都转为 .mp4
            raw_vid = item.get("video", "")
            vid_stem = Path(raw_vid).stem
            vid_mp4 = f"{vid_stem}.mp4"

            # 检查视频是否存在
            if not (video_dir / vid_mp4).exists():
                skipped_no_video += 1
                continue

            rec = {
                "video": f"train/{vid_mp4}",
                "query": item.get("description", ""),
                "start": float(s),
                "end": float(e),
            }
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"[didemo] ✅ train.jsonl 生成完毕: {count} 条 → {dst}")
    if skipped_no_times:
        print(f"[didemo] ⚠️ 跳过 {skipped_no_times} 条（无 times 字段）")
    if skipped_no_video:
        print(f"[didemo] ⚠️ 跳过 {skipped_no_video} 条（视频不存在）")





# ============================================================
# 5. VIPSeg → train.jsonl
# ============================================================
def build_vipseg():
    """
    VIPSeg 视频全景分割数据集（124类，2806视频）。
    从 panomasks 标注中解析 PNG mask，提取每个实例的 bbox + 类别。

    物体指代方式：**直接用 bbox 坐标**（如 [120,80,200,160]），
    无需类别唯一性约束，即使有多个同类物体也能无歧义指代。

    实际目录结构：
      vipseg/
      ├── videos_mp4/                    # 合成的 2fps MP4 视频（每帧 0.5s）
      ├── VIPSeg/
      │   ├── panomasks/                 # 全景分割标注（PNG，pixel_id = R+G*256+B*65536）
      │   ├── imgs/                      # 原始帧图片（jpg）
      │   ├── VIPSeg_720P/
      │   │   └── panoVIPSeg_categories.json  # 类别映射
      │   ├── train.txt / val.txt / test.txt
      │   └── label_num_dic_final.json

    输出 train.jsonl 每行:
      {video_dir, video, frame, frame_time, frame_path, panoptic_path,
       objects: [{name, bbox:[x1,y1,x2,y2], category_id, instance_id, area}]}
    """
    dst = VIPSEG_PATH / "train.jsonl"
    if dst.exists() and not _FORCE:
        print(f"[vipseg] train.jsonl 已存在，跳过。（用 --force 强制重新生成）")
        return

    # 检查必要文件（适配实际目录结构）
    panoptic_dir = VIPSEG_PATH / "VIPSeg" / "panomasks"
    videos_dir = VIPSEG_PATH / "VIPSeg" / "imgs"       # 帧图片目录
    videos_mp4_dir = VIPSEG_PATH / "videos_mp4"         # 合成的 2fps MP4 视频
    categories_file = VIPSEG_PATH / "VIPSeg" / "VIPSeg_720P" / "panoVIPSeg_categories.json"

    if not panoptic_dir.exists():
        print(f"[vipseg] panomasks 目录不存在: {panoptic_dir}，跳过。")
        return
    if not videos_dir.exists():
        print(f"[vipseg] imgs 目录不存在: {videos_dir}，跳过。")
        return
    if not videos_mp4_dir.exists():
        print(f"[vipseg] videos_mp4 目录不存在: {videos_mp4_dir}，请先运行 scripts/vipseg_frames_to_video.py")
        return

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        print("[vipseg] 需要 numpy 和 Pillow: pip install numpy Pillow")
        return

    # 加载类别映射
    cat_map = {}  # category_id -> category_name
    if categories_file.exists():
        with open(categories_file) as f:
            cats = json.load(f)
        for cat in cats:
            cat_map[cat["id"]] = cat.get("name", f"obj_{cat['id']}")
    else:
        print(f"[vipseg] 警告: 类别文件不存在 {categories_file}，将使用 category_id 作为名称")

    import random
    random.seed(42)

    # 遍历 panomasks 中的每个视频目录
    count = 0
    skipped_no_obj = 0
    skipped_no_mp4 = 0
    with open(dst, "w", encoding="utf-8") as g:
        for video_gt_dir in sorted(panoptic_dir.iterdir()):
            if not video_gt_dir.is_dir():
                continue
            video_id = video_gt_dir.name
            video_frames_dir = videos_dir / video_id
            if not video_frames_dir.exists():
                continue

            # 检查对应的 mp4 视频是否存在
            video_mp4_path = videos_mp4_dir / f"{video_id}.mp4"
            if not video_mp4_path.exists():
                skipped_no_mp4 += 1
                continue

            # 每个视频目录下有多个帧的 png 标注
            frame_files = sorted(video_gt_dir.glob("*.png"))
            if not frame_files:
                continue

            # 对每个视频取部分帧（避免数据量过大）
            # 每个视频最多取 5 帧
            sampled_frames = frame_files[::max(1, len(frame_files) // 5)][:5]

            # 构建帧名→时间戳映射（2fps 合成视频中，第 N 帧时间戳 = N * 0.5s）
            all_frame_names = sorted([f.stem for f in frame_files])
            frame_to_time = {name: idx * 0.5 for idx, name in enumerate(all_frame_names)}

            # 合成视频路径（相对于 VIPSEG_PATH）
            video_mp4_rel = f"videos_mp4/{video_id}.mp4"

            for frame_file in sampled_frames:
                frame_name = frame_file.stem  # 帧名（无后缀）
                # 检查对应的视频帧是否存在
                frame_img = video_frames_dir / f"{frame_name}.jpg"
                if not frame_img.exists():
                    frame_img = video_frames_dir / f"{frame_name}.png"
                if not frame_img.exists():
                    continue

                # 计算该帧在 2fps 合成视频中的时间戳
                frame_time = frame_to_time.get(frame_name, 0.0)

                # ---- 解析 panoptic mask，提取物体 bbox ----
                try:
                    pan_img = np.array(Image.open(frame_file))
                except Exception:
                    continue

                # VIPSeg panoptic PNG 编码：
                # 对于 RGB PNG: pixel_id = R + G*256 + B*256*256
                # 然后 category_id = pixel_id // 1000, instance_id = pixel_id % 1000
                if pan_img.ndim == 3:
                    # RGB → 单一 ID
                    pan_id = (pan_img[:, :, 0].astype(np.int32)
                              + pan_img[:, :, 1].astype(np.int32) * 256
                              + pan_img[:, :, 2].astype(np.int32) * 256 * 256)
                elif pan_img.ndim == 2:
                    # 灰度或 16-bit
                    pan_id = pan_img.astype(np.int32)
                else:
                    continue

                # 获取所有唯一 ID（排除 0 = 背景/void）
                unique_ids = np.unique(pan_id)
                unique_ids = unique_ids[unique_ids != 0]

                objects = []
                img_h, img_w = pan_id.shape
                min_area = img_h * img_w * 0.005  # 物体面积至少占图像 0.5%

                for uid in unique_ids:
                    cat_id = int(uid // 1000)
                    inst_id = int(uid % 1000)

                    # 跳过 stuff 类（inst_id == 0 通常是 stuff）
                    # 但 VIPSeg 中 stuff 也可能有 inst_id，保留面积足够大的
                    mask = (pan_id == uid)
                    area = int(mask.sum())
                    if area < min_area:
                        continue

                    # 计算 bbox [x1, y1, x2, y2]
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    y1, y2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
                    x1, x2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

                    cat_name = cat_map.get(cat_id, f"object_{cat_id}")

                    objects.append({
                        "name": cat_name,
                        "bbox": [x1, y1, x2, y2],
                        "category_id": cat_id,
                        "instance_id": inst_id,
                        "area": area,
                    })

                # 至少需要 2 个物体才能出空间关系题
                if len(objects) < 2:
                    skipped_no_obj += 1
                    continue

                rec = {
                    "video_dir": video_id,
                    "video": video_mp4_rel,
                    "frame": frame_name,
                    "frame_time": frame_time,
                    "frame_path": str(frame_img.relative_to(VIPSEG_PATH)),
                    "panoptic_path": str(frame_file.relative_to(VIPSEG_PATH)),
                    "objects": objects,
                }
                g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1

    print(f"[vipseg] ✅ train.jsonl 生成完毕: {count} 条 → {dst}")
    if skipped_no_obj:
        print(f"[vipseg] ⚠️ 跳过 {skipped_no_obj} 帧（物体数 < 2）")
    if skipped_no_mp4:
        print(f"[vipseg] ⚠️ 跳过 {skipped_no_mp4} 个视频（无对应 mp4）")


# ============================================================
# 6. TextVR → train.jsonl
# ============================================================
def build_textvr():
    """
    TextVR_train.json 格式:
    [{"path": "Domain/VIDEO.mp4", "captions_info": [{"caption": "...", ...}], ...}, ...]
    视频已扁平化到 videos/ 目录（去掉 domain 前缀）。
    输出: {video: "VIDEO.mp4", captions: ["...", ...], duration: float}
    """
    dst = TEXTVR_PATH / "train.jsonl"
    if dst.exists() and not _FORCE:
        print(f"[textvr] train.jsonl 已存在，跳过。（用 --force 强制重新生成）")
        return

    # 兼容多种命名
    src = TEXTVR_PATH / "TextVR_train.json"
    if not src.exists():
        src = TEXTVR_PATH / "train.json"
    if not src.exists():
        print(f"[textvr] 标注文件不存在: {TEXTVR_PATH / 'TextVR_train.json'}，跳过。")
        return

    with open(src) as f:
        data = json.load(f)

    videos_dir = TEXTVR_PATH / "videos"
    count = 0
    skipped = 0
    with open(dst, "w", encoding="utf-8") as g:
        for item in data:
            # path 形如 "Sports/G7ll8RnSgvU_00300_00310.mp4"，取文件名部分
            raw_path = item.get("path", "")
            video_name = Path(raw_path).name  # 去掉 domain 前缀

            # 检查视频是否存在
            if not (videos_dir / video_name).exists():
                skipped += 1
                continue

            captions = [c["caption"] for c in item.get("captions_info", [])]
            if not captions:
                skipped += 1
                continue

            # 取第一个 caption 的 duration 作为视频时长
            duration = 0.0
            cap_infos = item.get("captions_info", [])
            if cap_infos and "caption_info" in cap_infos[0]:
                duration = float(cap_infos[0]["caption_info"].get("duration", 0.0))

            rec = {
                "video": video_name,
                "captions": captions,
                "duration": duration,
            }
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"[textvr] ✅ train.jsonl 生成完毕: {count} 条 → {dst}")
    if skipped:
        print(f"[textvr] ⚠️ 跳过 {skipped} 条（视频不存在或无 caption）")


# ============================================================
# 7. NExT-QA → train.jsonl（验证已有文件）
# ============================================================
def build_nextqa():
    """
    NExT-QA train.jsonl 已由之前的流程生成（23085条）。
    格式: {video, question, choices, gt_answer, type, qid, split, mode}
    此函数仅验证文件存在性和完整性。
    """
    dst = NEXTQA_PATH / "train.jsonl"
    if dst.exists():
        with open(dst) as f:
            n = sum(1 for _ in f)
        print(f"[nextqa] ✅ train.jsonl 已存在: {n} 条 → {dst}")
        return

    print(f"[nextqa] ❌ train.jsonl 不存在: {dst}")
    print(f"[nextqa] 请确认 NExT-QA 数据集已正确下载并生成 train.jsonl")


# ============================================================
# 8. HC-STVG v2 → train.jsonl
# ============================================================
def build_hc_stvg():
    """
    HC-STVG v2 train_v2.json 格式:
    {"VIDEO.mp4": {img_size, Chinese, img_num, st_time, ed_time, st_frame, ed_offset, English, bbox:[[x,y,w,h],...], ed_frame}, ...}

    bbox 格式为 [x, y, w, h]，需转换为 [x1, y1, x2, y2]。
    每条标注对应一个视频中的一个人物轨迹片段。
    fps 通过 img_num / (视频总时长) 推算，约 30fps。

    输出: {video: "xxx.mp4", start: float, end: float, fps: float,
           bbox_seq: [[x1,y1,x2,y2],...], description: "..."}
    """
    dst = HC_STVG_PATH / "train.jsonl"
    if dst.exists() and not _FORCE:
        print(f"[hc_stvg] train.jsonl 已存在，跳过。（用 --force 强制重新生成）")
        return

    src = HC_STVG_PATH / "train_v2.json"
    if not src.exists():
        print(f"[hc_stvg] 标注文件不存在: {src}，跳过。")
        return

    video_dir = HC_STVG_PATH / "videos"
    if not video_dir.exists():
        print(f"[hc_stvg] 视频目录不存在: {video_dir}，跳过。")
        return

    with open(src) as f:
        data = json.load(f)

    existing_videos = set(p.name for p in video_dir.iterdir() if p.is_file())

    count = 0
    skipped_no_video = 0
    skipped_no_bbox = 0
    with open(dst, "w", encoding="utf-8") as g:
        for video_key, ann in data.items():
            # 视频名匹配：标注中可能是 .mkv，实际文件都是 .mp4
            stem = video_key.rsplit('.', 1)[0]
            if video_key in existing_videos:
                video_name = video_key
            elif f"{stem}.mp4" in existing_videos:
                video_name = f"{stem}.mp4"
            else:
                skipped_no_video += 1
                continue

            # 提取字段
            bbox_raw = ann.get("bbox", [])
            if not bbox_raw:
                skipped_no_bbox += 1
                continue

            st_time = float(ann.get("st_time", 0))
            ed_time = float(ann.get("ed_time", 0))
            img_num = int(ann.get("img_num", 1))

            # 推算 fps：img_num 是视频总帧数
            # 视频总时长 ≈ ed_time + ed_offset（但 ed_offset 不太可靠）
            # 更准确：fps = (ed_frame - st_frame + 1) / (ed_time - st_time)
            st_frame = int(ann.get("st_frame", 0))
            ed_frame = int(ann.get("ed_frame", st_frame + len(bbox_raw) - 1))
            duration = ed_time - st_time
            if duration > 0 and (ed_frame - st_frame) > 0:
                fps = (ed_frame - st_frame + 1) / duration
            else:
                fps = 30.0  # 默认 30fps

            # bbox: [x, y, w, h] → [x1, y1, x2, y2]
            bbox_seq = []
            for b in bbox_raw:
                if len(b) == 4:
                    x, y, w, h = b
                    bbox_seq.append([x, y, x + w, y + h])
                else:
                    bbox_seq.append(b)  # 保留原样

            # 描述：优先用 English
            description = ann.get("English", ann.get("Chinese", "")).strip()
            if not description:
                description = "A person performing an action."

            rec = {
                "video": video_name,
                "start": st_time,
                "end": ed_time,
                "fps": round(fps, 2),
                "bbox_seq": bbox_seq,
                "description": description,
            }
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"[hc_stvg] ✅ train.jsonl 生成完毕: {count} 条 → {dst}")
    if skipped_no_video:
        print(f"[hc_stvg] ⚠️ 跳过 {skipped_no_video} 条（视频不存在）")
    if skipped_no_bbox:
        print(f"[hc_stvg] ⚠️ 跳过 {skipped_no_bbox} 条（无 bbox）")


# ============================================================
# 9. CLEVRER → train.jsonl（验证已有文件）
# ============================================================
def build_clevrer():
    """
    CLEVRER train.jsonl 已由之前的流程生成（152572条）。
    格式: {video, question, choices, gt_answer, type}
    此函数仅验证文件存在性和完整性。
    """
    dst = CLEVRER_PATH / "train.jsonl"
    if dst.exists():
        with open(dst) as f:
            n = sum(1 for _ in f)
        print(f"[clevrer] ✅ train.jsonl 已存在: {n} 条 → {dst}")
        return

    print(f"[clevrer] ❌ train.jsonl 不存在: {dst}")
    print(f"[clevrer] 请确认 CLEVRER 数据集已正确下载并生成 train.jsonl")


# ============================================================
# 调度
# ============================================================
REGISTRY = {
    "charades_sta": build_charades_sta,
    "star": build_star,
    "didemo": build_didemo,
    "vipseg": build_vipseg,
    "textvr": build_textvr,
    "nextqa": build_nextqa,
    "clevrer": build_clevrer,
    "hc_stvg": build_hc_stvg,
}


def main():
    global _FORCE
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--force" in sys.argv:
        _FORCE = True
        print("[MODE] --force: 强制重新生成所有 train.jsonl\n")

    if args:
        keys = args
    else:
        keys = list(REGISTRY.keys())

    for key in keys:
        if key not in REGISTRY:
            print(f"[ERROR] 未知数据集: {key}（可选: {', '.join(REGISTRY.keys())}）")
            continue
        print(f"\n{'='*50}")
        print(f"[处理] {key}")
        print(f"{'='*50}")
        try:
            REGISTRY[key]()
        except Exception as e:
            print(f"[ERROR] {key} 处理失败: {e}")


if __name__ == "__main__":
    main()
