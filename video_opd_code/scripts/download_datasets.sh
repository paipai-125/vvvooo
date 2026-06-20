#!/usr/bin/env bash
# ============================================================
# Video-OPD 数据集统一下载脚本（零 YouTube 依赖）
# ------------------------------------------------------------
# 8 个数据集对应的感知能力：
#   1) Charades-STA   temporal_locate ★            (~13 GB, AI2 直链)
#   2) STAR           raw VideoQA ★ (复用Charades) (~100 MB, GitHub release)
#   3) NExT-QA        raw VideoQA ★                (~50 GB, HF lmms-lab)
#   4) CLEVRER        raw VideoQA ★ (因果)         (~52 GB, MIT 直链)
#   5) HC-STVG v2     spatial_detect/crop + tracking_describe ★
#                                                  (~25 GB, OneDrive 直链)
#   6) DiDeMo         temporal_locate ★ (替 ActivityNet)
#                                                  (~6 GB, Flickr 来源)
#   7) TextVR         ocr_zoom ★                   (~20 GB, HF 直链)
#   9) VIPSeg         spatial_detect/crop + depth_overlay ★ (124类全景分割)
#                                                  (~30 GB, Google Drive)
#
# 用法:
#   bash scripts/download_datasets.sh                # 下载全部
#   bash scripts/download_datasets.sh charades_sta   # 单独下某一个
#   bash scripts/download_datasets.sh charades_sta hc_stvg textvr   # 多选
# ============================================================

# 注意: 不要用 set -e；单个数据集失败不应阻断其他数据集。
set -uo pipefail

cd "$(dirname "$0")/.."

# ---- 路径 ----
DATASET_ROOT="$(python -m configs.paths dataset_root)"
echo "[download_datasets] DATASET_ROOT = ${DATASET_ROOT}"
mkdir -p "${DATASET_ROOT}"

# ---- 全局状态 ----
declare -a OK_LIST=()
declare -a FAIL_LIST=()

# ---- 通用 curl 重试包装：成功返回 0，失败返回非 0 但不退出脚本 ----
# 用法: try_curl <out_path> <url> [<url> <url> ...]   多个 URL 作为 fallback
try_curl() {
    local out="$1"; shift
    local urls=("$@")
    local u
    for u in "${urls[@]}"; do
        echo "  [try_curl] -> ${u}"
        if curl -L --fail --retry 3 --connect-timeout 15 -o "${out}" "${u}"; then
            echo "  [try_curl] OK: ${out}"
            return 0
        fi
        echo "  [try_curl] FAILED: ${u}" >&2
    done
    return 1
}

# ---- 工具检查 ----
need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[ERROR] 需要命令 '$1'，请先安装。" >&2
        return 1
    fi
}
need_cmd curl
need_cmd wget || true   # 优先用 curl，wget 兜底
need_cmd unzip
need_cmd tar
need_cmd python

HF_DOWNLOAD() {
    # HF_DOWNLOAD <repo_id> <local_dir> [--repo-type dataset|model] [extra args...]
    local repo="$1"; shift
    local out="$1"; shift
    mkdir -p "${out}"
    hf download "${repo}" --local-dir "${out}" "$@"
}

# ============================================================
# 1) Charades-STA
#    视频源：AI2 官方直链（480p, ~13GB）
#    标注：charades_sta_train/test.txt（~1MB，GitHub jiyanggao）
# ============================================================
dl_charades_sta() {
    local d="${DATASET_ROOT}/charades_sta"
    mkdir -p "${d}/videos"
    echo "[1/9] Charades-STA -> ${d}"

    # 标注（jiyanggao/TALL 仓库的 exp_data/Charades/ 路径已不存在，改用多个公开 mirror）
    if [[ ! -f "${d}/charades_sta_train.txt" ]]; then
        try_curl "${d}/charades_sta_train.txt" \
            "https://raw.githubusercontent.com/Tangkfan/CICR/main/data/charades_sta/charades_sta_train.txt" \
            "https://raw.githubusercontent.com/Soldelli/MAD/main/data/charades_sta/charades_sta_train.txt" \
            "https://raw.githubusercontent.com/jiyanggao/TALL/master/exp_data/Charades/charades_sta_train.txt" \
            || { echo "[ERROR] charades_sta_train.txt 所有 mirror 都失败" >&2; return 1; }
    fi
    if [[ ! -f "${d}/charades_sta_test.txt" ]]; then
        try_curl "${d}/charades_sta_test.txt" \
            "https://raw.githubusercontent.com/Tangkfan/CICR/main/data/charades_sta/charades_sta_test.txt" \
            "https://raw.githubusercontent.com/Soldelli/MAD/main/data/charades_sta/charades_sta_test.txt" \
            "https://raw.githubusercontent.com/jiyanggao/TALL/master/exp_data/Charades/charades_sta_test.txt" \
            || { echo "[ERROR] charades_sta_test.txt 所有 mirror 都失败" >&2; return 1; }
    fi

    # 视频（480p 版本约 13GB；有就跳过）
    if [[ -z "$(ls -A "${d}/videos" 2>/dev/null || true)" ]]; then
        local zip="${d}/Charades_v1_480.zip"
        if [[ ! -f "${zip}" ]]; then
            try_curl "${zip}" \
                "https://ai2-public-datasets.s3.amazonaws.com/charades/Charades_v1_480.zip" \
                || { echo "[ERROR] Charades_v1_480.zip 下载失败" >&2; return 1; }
        fi
        unzip -q "${zip}" -d "${d}/videos_tmp" || { echo "[ERROR] unzip 失败" >&2; return 1; }
        # zip 解出来是 Charades_v1_480/<vid>.mp4，我们扁平化到 videos/
        mv "${d}/videos_tmp/Charades_v1_480/"*.mp4 "${d}/videos/"
        rm -rf "${d}/videos_tmp" "${zip}"
    fi
    echo "[OK] Charades-STA done."
    return 0
}

# ============================================================
# 2) STAR Benchmark（复用 Charades 视频）
#    标注：约 100MB（来自官方 release）
# ============================================================
dl_star() {
    local d="${DATASET_ROOT}/star"
    mkdir -p "${d}"
    echo "[2/9] STAR -> ${d}"

    # 视频通过软链复用 Charades
    if [[ ! -e "${d}/videos" ]]; then
        ln -s "${DATASET_ROOT}/charades_sta/videos" "${d}/videos"
    fi

    # 标注（官方 GitHub release: csv 与 splits）
    if [[ ! -f "${d}/STAR_train.json" ]]; then
        curl -L --fail -o "${d}/STAR_train.json" \
            https://raw.githubusercontent.com/csbobby/STAR_Benchmark/main/STAR_train.json
    fi
    if [[ ! -f "${d}/STAR_val.json" ]]; then
        curl -L --fail -o "${d}/STAR_val.json" \
            https://raw.githubusercontent.com/csbobby/STAR_Benchmark/main/STAR_val.json
    fi
    if [[ ! -f "${d}/STAR_test.json" ]]; then
        curl -L --fail -o "${d}/STAR_test.json" \
            https://raw.githubusercontent.com/csbobby/STAR_Benchmark/main/STAR_test.json
    fi

    # 简化为 train.jsonl（loader 期望的格式：video, question, choices, gt_answer, type）
    if [[ ! -f "${d}/train.jsonl" ]]; then
        python - <<'PY'
import json, os
from configs.paths import STAR_PATH
src = STAR_PATH / "STAR_train.json"
dst = STAR_PATH / "train.jsonl"
with open(src) as f:
    data = json.load(f)
with open(dst, "w", encoding="utf-8") as g:
    for item in data:
        # 官方字段: video_id, question, choices(list of {choice_id, choice}), answer, question_type
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
print("[STAR] train.jsonl ready:", dst)
PY
    fi
    echo "[OK] STAR done."
}

# ============================================================
# 3) NExT-QA
#    HF 镜像：lmms-lab/NExTQA（含视频与标注）
# ============================================================
dl_nextqa() {
    local d="${DATASET_ROOT}/nextqa"
    mkdir -p "${d}"
    echo "[3/9] NExT-QA -> ${d}"
    if [[ ! -d "${d}/videos" ]]; then
        HF_DOWNLOAD "lmms-lab/NExTQA" "${d}" --repo-type dataset
    fi
    # 简化为 train.jsonl（兼容 parquet 和 csv 两种格式）
    if [[ ! -f "${d}/train.jsonl" ]]; then
        python -m data_preparation.build_nextqa_jsonl
    fi
    echo "[OK] NExT-QA done."
}

# ============================================================
# 4) CLEVRER（合成视频）
# ============================================================
dl_clevrer() {
    local d="${DATASET_ROOT}/clevrer"
    mkdir -p "${d}/videos"
    echo "[4/9] CLEVRER -> ${d}"
    if [[ -z "$(ls -A "${d}/videos" 2>/dev/null || true)" ]]; then
        local v="${d}/video_train.zip"
        local q="${d}/train.json"
        [[ -f "${v}" ]] || curl -L --fail -o "${v}" \
            http://data.csail.mit.edu/clevrer/videos/train/video_train.zip
        [[ -f "${q}" ]] || curl -L --fail -o "${q}" \
            http://data.csail.mit.edu/clevrer/questions/train.json
        unzip -q "${v}" -d "${d}/videos_tmp"
        # 解出来是 video_train/video_xxxxx.mp4，扁平化
        find "${d}/videos_tmp" -name "*.mp4" -exec mv {} "${d}/videos/" \;
        rm -rf "${d}/videos_tmp" "${v}"
    fi
    # 简化为 train.jsonl
    if [[ ! -f "${d}/train.jsonl" ]]; then
        python - <<'PY'
import json
from configs.paths import CLEVRER_PATH
src = CLEVRER_PATH / "train.json"
dst = CLEVRER_PATH / "train.jsonl"
with open(src) as f:
    data = json.load(f)
with open(dst, "w", encoding="utf-8") as g:
    for scene in data:
        vid = scene.get("video_filename", f"video_{scene['scene_index']:05d}.mp4")
        for q in scene.get("questions", []):
            qtype = q.get("question_type", "")
            choices = [c["choice"] for c in q.get("choices", [])] if "choices" in q else []
            if "answer" in q:
                gt = str(q["answer"])
            elif choices:
                # 多选：找 correct=true
                corr = [c["choice"] for c in q["choices"] if c.get("answer") == "correct"]
                gt = corr[0] if corr else ""
            else:
                gt = ""
            rec = {
                "video": vid,
                "question": q.get("question", ""),
                "choices": choices,
                "gt_answer": gt,
                "type": qtype,
            }
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("[CLEVRER] train.jsonl ready:", dst)
PY
    fi
    echo "[OK] CLEVRER done."
}

# ============================================================
# 5) HC-STVG v2（spatial_detect/crop + tracking_describe）
#    OneDrive 公开链接 + GitHub 标注
# ============================================================
dl_hc_stvg() {
    local d="${DATASET_ROOT}/hc_stvg"
    mkdir -p "${d}/videos"
    echo "[5/9] HC-STVG v2 -> ${d}"
    if [[ -z "$(ls -A "${d}/videos" 2>/dev/null || true)" ]]; then
        cat <<'TIP'
[HC-STVG] 视频需手动从官方 OneDrive 下载（25GB 左右），URL 见
  https://github.com/tzhhhh123/HC-STVG
下完后请把 v2_video/ 下的所有 .mp4 放到 ${HC_STVG_PATH}/videos/，
再重新执行 `bash scripts/download_datasets.sh hc_stvg` 处理标注。
TIP
    fi
    # 标注（HC-STVG v2 的 anno 是 GitHub 上的）
    for split in train val; do
        if [[ ! -f "${d}/${split}_v2.json" ]]; then
            curl -L --fail -o "${d}/${split}_v2.json" \
                "https://raw.githubusercontent.com/tzhhhh123/HC-STVG/main/anno_v2/${split}_v2.json" \
                || echo "[WARN] HC-STVG ${split}_v2.json 下载失败，请手动放置。"
        fi
    done
    # 简化为 train.jsonl: video, start, end, fps, bbox_seq, description
    if [[ ! -f "${d}/train.jsonl" ]]; then
        python - <<'PY'
import json
from pathlib import Path
from configs.paths import HC_STVG_PATH
src = HC_STVG_PATH / "train_v2.json"
if not src.exists():
    print("[HC-STVG] 暂无 train_v2.json，跳过 train.jsonl 生成。")
else:
    with open(src) as f:
        data = json.load(f)
    out = HC_STVG_PATH / "train.jsonl"
    with open(out, "w", encoding="utf-8") as g:
        for vid, item in data.items():
            # 官方字段: img_num, st_frame, ed_frame, fps, bbox (每帧4维), caption
            fps = float(item.get("fps", 25.0))
            st = item["st_frame"] / fps
            ed = item["ed_frame"] / fps
            bbox_seq = item.get("bbox", [])  # list of [x1,y1,x2,y2]
            rec = {
                "video": f"{vid}.mp4",
                "start": st, "end": ed, "fps": fps,
                "bbox_seq": bbox_seq,
                "description": item.get("caption", ""),
            }
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("[HC-STVG] train.jsonl ready:", out)
PY
    fi
    echo "[OK] HC-STVG done."
}

# ============================================================
# 6) DiDeMo（temporal_locate 替代 ActivityNet）
#    视频源：HF datasets（Flickr/YFCC，非 YouTube）
# ============================================================
dl_didemo() {
    local d="${DATASET_ROOT}/didemo"
    mkdir -p "${d}"
    echo "[6/9] DiDeMo -> ${d}"
    if [[ ! -d "${d}/videos" ]]; then
        # HF 镜像（社区维护）：包含视频与 split json
        HF_DOWNLOAD "lmms-lab/DiDeMo" "${d}" --repo-type dataset || \
            HF_DOWNLOAD "friedrichor/DiDeMo" "${d}" --repo-type dataset
    fi
    # 转 train.jsonl
    if [[ ! -f "${d}/train.jsonl" ]]; then
        python - <<'PY'
import json
from pathlib import Path
from configs.paths import DIDEMO_PATH
# 兼容多种 split 文件命名
cand = list(DIDEMO_PATH.rglob("train_data.json")) \
    + list(DIDEMO_PATH.rglob("didemo_train.json")) \
    + list(DIDEMO_PATH.rglob("train.json"))
if not cand:
    raise FileNotFoundError("DiDeMo train split 未找到（接受 train_data.json / didemo_train.json / train.json）")
src = cand[0]
with open(src) as f:
    data = json.load(f)
out = DIDEMO_PATH / "train.jsonl"
with open(out, "w", encoding="utf-8") as g:
    for item in data:
        # 官方字段：video, description, times（5 secs 一段，list of [start_idx, end_idx]）
        # 取标注一致的中位段作为时间区间
        times = item.get("times", [])
        if not times:
            continue
        # 5 秒块为单位，多人标注取中位
        starts = sorted([t[0] for t in times])
        ends   = sorted([t[1] for t in times])
        s = starts[len(starts)//2] * 5.0
        e = (ends[len(ends)//2] + 1) * 5.0
        rec = {
            "video": item["video"] if str(item["video"]).endswith(".mp4") else f"{item['video']}.mp4",
            "query": item.get("description", ""),
            "start": float(s), "end": float(e),
        }
        g.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("[DiDeMo] train.jsonl ready:", out)
PY
    fi
    echo "[OK] DiDeMo done."
}

# ============================================================
# 7) TextVR（ocr_zoom）
# ============================================================
dl_textvr() {
    local d="${DATASET_ROOT}/textvr"
    mkdir -p "${d}"
    echo "[8/9] TextVR -> ${d}"
    if [[ ! -d "${d}/videos" ]]; then
        # HF 镜像
        HF_DOWNLOAD "ChartMimic/TextVR" "${d}" --repo-type dataset || \
            HF_DOWNLOAD "nineninesix/textvr" "${d}" --repo-type dataset || \
            cat <<'TIP'
[TextVR] HF 镜像未找到，请到官方 GitHub https://github.com/callsys/TextVR
按 README 下载视频与标注，放到 ${TEXTVR_PATH}/{videos,train.json}。
TIP
    fi
    # 转 train.jsonl: video, frame_time, bbox, text
    if [[ ! -f "${d}/train.jsonl" ]]; then
        python - <<'PY'
import json
from pathlib import Path
from configs.paths import TEXTVR_PATH
cand = list(TEXTVR_PATH.rglob("TextVR_train.json")) + list(TEXTVR_PATH.rglob("train.json"))
if not cand:
    print("[TextVR] 标注未找到，跳过 train.jsonl 生成。")
else:
    src = cand[0]
    with open(src) as f:
        data = json.load(f)
    out = TEXTVR_PATH / "train.jsonl"
    with open(out, "w", encoding="utf-8") as g:
        for item in data if isinstance(data, list) else data.get("data", []):
            # TextVR 官方每条带 video, fps, transcripts: [{frame, bbox, text}]
            video = item.get("video", item.get("video_id", ""))
            fps = float(item.get("fps", 25.0))
            for tr in item.get("transcripts", item.get("annotations", [])):
                fr = tr.get("frame", tr.get("frame_id", 0))
                bbox = tr.get("bbox") or tr.get("box")
                text = tr.get("text", tr.get("transcription", ""))
                if not bbox or not text:
                    continue
                rec = {
                    "video": video if str(video).endswith(".mp4") else f"{video}.mp4",
                    "frame_time": float(fr) / max(fps, 1e-3),
                    "bbox": list(bbox),
                    "text": text,
                }
                g.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("[TextVR] train.jsonl ready:", out)
PY
    fi
    echo "[OK] TextVR done."
}

# ============================================================
# 9) VIPSeg（视频全景分割，124类，3536视频）
#    用于 spatial_detect / spatial_crop / depth_overlay
#    数据来源：https://github.com/VIPSeg-Dataset/VIPSeg-Dataset
#    视频+标注需从 Google Drive 下载（详见手动下载指南）
# ============================================================
dl_vipseg() {
    local d="${DATASET_ROOT}/vipseg"
    mkdir -p "${d}"
    echo "[9/9] VIPSeg -> ${d}"
    if [[ ! -d "${d}/videos" ]]; then
        cat <<'TIP'
[VIPSeg] 需手动下载。请参考 Video-OPD数据集手动下载指南.md 中 VIPSeg 章节：
  1. 从 GitHub release 或 Google Drive 下载 VIPSeg_720p.zip
  2. 解压后将视频帧目录放到 ${d}/videos/
  3. 将 panoptic_gt_VIPSeg/ 标注放到 ${d}/panoptic_gt/
TIP
    fi
    if [[ ! -f "${d}/train.jsonl" ]]; then
        echo "[VIPSeg] train.jsonl 需通过 data_preparation/build_all_jsonl.py vipseg 生成"
    fi
    echo "[OK] VIPSeg done."
}

# ============================================================
# 调度
# ============================================================
declare -A REGISTRY=(
    [charades_sta]=dl_charades_sta
    [star]=dl_star
    [nextqa]=dl_nextqa
    [clevrer]=dl_clevrer
    [hc_stvg]=dl_hc_stvg
    [didemo]=dl_didemo
    [textvr]=dl_textvr
    [vipseg]=dl_vipseg
)

ORDER=(charades_sta star didemo hc_stvg textvr vipseg nextqa clevrer)

run_one() {
    local key="$1"
    local fn="${REGISTRY[$key]:-}"
    if [[ -z "${fn}" ]]; then
        echo "[ERROR] 未知数据集: ${key}（可选: ${!REGISTRY[*]}）" >&2
        FAIL_LIST+=("${key}(unknown)")
        return 1
    fi
    echo
    echo "========================================"
    echo "[run] ${key}"
    echo "========================================"
    if "${fn}"; then
        OK_LIST+=("${key}")
    else
        FAIL_LIST+=("${key}")
        echo "[FAIL] 数据集 ${key} 处理失败，继续下一个。" >&2
    fi
}

if [[ $# -eq 0 ]]; then
    for k in "${ORDER[@]}"; do run_one "${k}"; done
else
    for k in "$@"; do run_one "${k}"; done
fi

echo
echo "==================== SUMMARY ===================="
echo "  OK   (${#OK_LIST[@]}): ${OK_LIST[*]:-<none>}"
echo "  FAIL (${#FAIL_LIST[@]}): ${FAIL_LIST[*]:-<none>}"
echo "================================================="
echo
echo "[download_datasets] 结束。请运行 \`bash scripts/check_datasets.sh\` 详细校验。"
if [[ ${#FAIL_LIST[@]} -gt 0 ]]; then
    exit 1
fi
