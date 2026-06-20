#!/usr/bin/env bash
# 预实验1: 视觉聚焦有效性验证
# 在 Charades-STA 验证集上对比: 学生直答 vs Teacher_P+聚焦
#
# 用法:
#   bash scripts/run_pre_experiment_focus.sh
#   LIMIT=100 bash scripts/run_pre_experiment_focus.sh
#   bash scripts/run_pre_experiment_focus.sh --skip_focused

set -euo pipefail

cd "$(dirname "$0")/.."

# 路径来源：configs/paths.yaml
DATA_ROOT="$(python -m configs.paths data_root)"

ANNOTATION="${ANNOTATION:-${DATA_ROOT}/datasets/charades_sta/charades_sta_test.txt}"
VIDEO_DIR="${VIDEO_DIR:-${DATA_ROOT}/datasets/charades_sta/videos}"
OUTPUT="${OUTPUT:-${DATA_ROOT}/outputs/pre_experiment_focus/results.jsonl}"
LIMIT="${LIMIT:-500}"

mkdir -p "$(dirname "${OUTPUT}")"

CMD=(python -m evaluation.pre_experiment_focus
    --annotation "${ANNOTATION}"
    --video_dir  "${VIDEO_DIR}"
    --output     "${OUTPUT}"
    --limit      "${LIMIT}"
)

# 透传额外参数，如 --skip_focused
CMD+=("$@")

echo "[run_pre_experiment_focus] CMD: ${CMD[*]}"
"${CMD[@]}"
