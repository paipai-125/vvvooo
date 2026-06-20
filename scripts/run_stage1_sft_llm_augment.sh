#!/usr/bin/env bash
# Stage 1-SFT 来源C: Qwen3-32B 文本LLM补写推理文字
# 用法:
#   bash scripts/run_stage1_sft_llm_augment.sh
#   bash scripts/run_stage1_sft_llm_augment.sh c1
#
# 注意: 需要本地有 Qwen3-32B 权重(在 configs/paths.py 的 QWEN3_32B_PATH)

set -euo pipefail

cd "$(dirname "$0")/.."

SUBSET="${1:-all}"

CMD=(python -m data_preparation.stage1_sft_llm_augment
    --subset "${SUBSET}"
    --n_c1_nextqa "${N_C1_NEXTQA:-3000}"
    --n_c1_star   "${N_C1_STAR:-2000}"
    --n_c2        "${N_C2:-3000}"
    --n_c3        "${N_C3:-2000}"
)

if [[ -n "${OUTPUT_DIR:-}" ]]; then
    CMD+=(--output_dir "${OUTPUT_DIR}")
fi

echo "[run_stage1_sft_llm_augment] subset=${SUBSET}"
echo "[run_stage1_sft_llm_augment] CMD: ${CMD[*]}"
"${CMD[@]}"
