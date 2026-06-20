#!/usr/bin/env bash
# ============================================================
# Video-OPD 数据集完整性校验
# ------------------------------------------------------------
# 检查每个数据集：
#   1) 标注文件存在
#   2) 视频目录存在且非空
#   3) 简化 train.jsonl 已就绪（loader 能直接吃）
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET_ROOT="$(python -m configs.paths dataset_root)"
echo "[check_datasets] DATASET_ROOT=${DATASET_ROOT}"

ok=0; warn=0; bad=0

# 通用单数据集检查
# args: <key> <ann_relative> <videos_relative_or_link> [more annotation files]
check_one() {
    local key="$1"; shift
    local ann="$1"; shift
    local vdir="$1"; shift
    local d="${DATASET_ROOT}/${key}"

    local pass=1 msg=""
    if [[ ! -e "${d}/${ann}" ]]; then
        pass=0; msg="缺失 ${ann}"
    elif [[ ! -d "${d}/${vdir}" && ! -L "${d}/${vdir}" ]]; then
        pass=0; msg="缺失 ${vdir}/ 目录"
    elif [[ -z "$(ls -A "${d}/${vdir}" 2>/dev/null || true)" ]]; then
        pass=0; msg="${vdir}/ 为空"
    fi

    # 额外文件检查
    for extra in "$@"; do
        if [[ ! -e "${d}/${extra}" ]]; then
            pass=2; msg="（提示）建议生成 ${extra}"
        fi
    done

    case "${pass}" in
        1) echo "  [OK]   ${key}  ($(du -sh -L "${d}" 2>/dev/null | awk '{print $1}'))"; ((ok+=1)) ;;
        2) echo "  [WARN] ${key}  ${msg}"; ((warn+=1)) ;;
        0) echo "  [MISS] ${key}  ${msg} (${d})"; ((bad+=1)) ;;
    esac
}

echo
echo "=== 主数据集（必须有视频 + jsonl）==="
check_one charades_sta train.jsonl            videos
check_one star         STAR_train.json        videos train.jsonl
check_one nextqa       train.jsonl            videos
check_one clevrer      train.jsonl            videos
check_one hc_stvg      train_v2.json          videos train.jsonl
check_one didemo       train.jsonl            videos
check_one textvr       train.jsonl            videos
check_one vipseg        panoVIPSeg_categories.json videos train.jsonl

echo
echo "=== 仅标注（视频源自 YouTube/VidOR，本仓库不下载）==="
for legacy in activitynet_captions vidstg; do
    if [[ -d "${DATASET_ROOT}/${legacy}" ]]; then
        echo "  [INFO] ${legacy}  存在标注，视频已主动跳过（详见 data/datasets/_TODO_DATASETS.md）"
    else
        echo "  [SKIP] ${legacy}  未下载，符合预期"
    fi
done

echo
echo "汇总: OK=${ok}  WARN=${warn}  MISS=${bad}"
if (( bad > 0 )); then
    echo "[check_datasets] 有数据集缺失，请先 bash scripts/download_datasets.sh <key> 补齐。" >&2
    exit 1
fi
echo "[check_datasets] 全部数据集就绪。"
