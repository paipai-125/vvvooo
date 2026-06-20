#!/usr/bin/env bash
# Stage1-SFT v5 训练脚本（Coconut/VaLR 串行 latent）
set -euo pipefail
cd "$(dirname "$0")/.."

# wandb 免交互登录
export WANDB_API_KEY="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"

DATA_ROOT="$(python -m configs.paths data_root)"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_ROOT}/outputs/stage1_sft/stage1_sft_template_all.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/outputs/checkpoints/stage1_sft_v5_latent}"
DEEPSPEED="${DEEPSPEED:-}"  # 分卡方案下不使用 DeepSpeed
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"  # 4进程×2卡/进程=8卡
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

mkdir -p "${OUTPUT_DIR}"

CMD=(torchrun
    --nproc_per_node="${NPROC_PER_NODE}"
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    -m training.stage1_sft_v5
    --train_jsonl "${TRAIN_JSONL}"
    --output_dir  "${OUTPUT_DIR}"
    --num_train_epochs           "${NUM_TRAIN_EPOCHS:-1}"
    --per_device_batch_size      "${PER_DEVICE_BATCH_SIZE:-1}"
    --gradient_accumulation_steps "${GRAD_ACCUM:-1}"
    --student_lr                 "${STUDENT_LR:-2e-5}"
    --translator_lr              "${TRANSLATOR_LR:-2e-5}"
    --weight_decay               "${WEIGHT_DECAY:-0.01}"
    --warmup_ratio               "${WARMUP_RATIO:-0.03}"
    --max_length                 "${MAX_LENGTH:-32768}"
    --max_frames                 "${MAX_FRAMES:-64}"
    --fps                        "${FPS:-1.0}"
    --video_max_pixels           "${VIDEO_MAX_PIXELS:-0}"
    --logging_steps              "${LOGGING_STEPS:-10}"
    --save_steps                 "${SAVE_STEPS:-500}"
    --save_total_limit           "${SAVE_TOTAL_LIMIT:-3}"
    --num_workers                "${NUM_WORKERS:-2}"
    --max_seg_len                "${MAX_SEG_LEN:-256}"
    --max_latent_steps           "${MAX_LATENT_STEPS:-32}"
    --lambda_trans               "${LAMBDA_TRANS:-1.0}"
    --lambda_ans                 "${LAMBDA_ANS:-1.0}"
    --lambda_aux                 "${LAMBDA_AUX:-1.0}"
    --bf16
)

if [[ -n "${MODEL_PATH:-}" ]]; then
    CMD+=(--model_path "${MODEL_PATH}")
fi
if [[ -n "${DEEPSPEED}" && -f "${DEEPSPEED}" ]]; then
    CMD+=(--deepspeed "${DEEPSPEED}")
fi
if [[ -n "${MAX_SAMPLES:-}" && "${MAX_SAMPLES:-0}" -gt 0 ]]; then
    CMD+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
    CMD+=(--resume_from "${RESUME_FROM}")
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    CMD+=(--wandb_project "${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
    CMD+=(--wandb_run_name "${WANDB_RUN_NAME}")
fi
if [[ -n "${WANDB_MODE:-}" ]]; then
    CMD+=(--wandb_mode "${WANDB_MODE}")
fi

echo "====== Stage1-SFT v5 (Coconut/VaLR serial latent) ======"
echo "  TRAIN_JSONL=${TRAIN_JSONL}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  NPROC=${NPROC_PER_NODE} DEEPSPEED=${DEEPSPEED}"
echo "  MAX_LENGTH=${MAX_LENGTH:-32768} MAX_SEG_LEN=${MAX_SEG_LEN:-256}"
echo "  MAX_FRAMES=${MAX_FRAMES:-64} FPS=${FPS:-1.0} VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-0}(0=default)"
echo "  STUDENT_LR=${STUDENT_LR:-2e-5} TRANSLATOR_LR=${TRANSLATOR_LR:-2e-5}"
echo "  LAMBDA: trans=${LAMBDA_TRANS:-1.0} ans=${LAMBDA_ANS:-1.0} aux=${LAMBDA_AUX:-1.0}"
echo "========================================================="
"${CMD[@]}"
