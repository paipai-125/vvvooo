#!/usr/bin/env bash
# Stage 1-SFT 来源A: 规则模板生成 (zero-LLM)
# 用法:
#   bash scripts/run_stage1_sft_template.sh
#   bash scripts/run_stage1_sft_template.sh a1   # 只跑 temporal_locate
#   subset 可选: all, a1, a2, a3, a4, a5, a6, a7
#
# 路径来源：configs/paths.yaml（请先 cp configs/paths.example.yaml configs/paths.yaml 并填好）
#
# 环境变量(可选, 用于覆盖默认):
#   OUTPUT_DIR  输出目录, 默认 <data_root>/outputs/stage1_sft（由 paths.yaml 决定）

set -euo pipefail

cd "$(dirname "$0")/.."

# 触发一次 paths.yaml 校验：缺失/字段错时此处直接退出
python -m configs.paths data_root >/dev/null

SUBSET="${1:-all}"

CMD=(python -m data_preparation.stage1_sft_template
    --subset      "${SUBSET}"
    --n_charades  "${N_CHARADES:-5000}"
    --n_didemo    "${N_DIDEMO:-5000}"
    --n_a2        "${N_A2:-3000}"
    --n_a3        "${N_A3:-3000}"
    --n_a4        "${N_A4:-2000}"
    --n_a5        "${N_A5:-2000}"
    --n_a6        "${N_A6:-2000}"
    --n_a7        "${N_A7:-2000}"
    --seed        "${SEED:-42}"
)

if [[ -n "${OUTPUT_DIR:-}" ]]; then
    CMD+=(--output_dir "${OUTPUT_DIR}")
fi

echo "[run_stage1_sft_template] subset=${SUBSET}"
echo "[run_stage1_sft_template] CMD: ${CMD[*]}"
"${CMD[@]}"
