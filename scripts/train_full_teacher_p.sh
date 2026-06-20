#!/usr/bin/env bash
# =============================================================================
# 全量训练 —— Teacher_P (感知教师, 标准 CE, 看视频)
#
# 与 student / teacher_r 完全独立，可以单独 Ctrl-C 终止。
# Teacher_P 看视频，预期速度与 student 相当（视频解码主导）。
#
# 用法：
#   bash scripts/train_full_teacher_p.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------- wandb ----------
export WANDB_API_KEY="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"
WANDB_PROJECT="${WANDB_PROJECT:-video-opd-stage1-sft}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-teacher_p-full}"

# ---------- 路径 ----------
DATA_ROOT="$(python -m configs.paths data_root)"
SFT_DIR="${DATA_ROOT}/outputs/stage1_sft"
TRAIN_JSONL="${TRAIN_JSONL:-${SFT_DIR}/stage1_sft_teacher_p.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/outputs/checkpoints/stage1_sft_teacher_p}"

# ---------- 超参 ----------
NPROC="${NPROC:-8}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-40960}"
MAX_FRAMES="${MAX_FRAMES:-64}"
FPS="${FPS:-1.0}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEEPSPEED="${DEEPSPEED:-scripts/ds_zero2.json}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${TRAIN_JSONL}" ]]; then
    echo "[ERROR] Teacher_P 训练数据不存在: ${TRAIN_JSONL}"
    echo "       请先运行: bash scripts/run_sft_full.sh"
    exit 1
fi

echo "============================================================"
echo "[Train Full] role=teacher_p"
echo "  TRAIN_JSONL: ${TRAIN_JSONL} ($(wc -l < "${TRAIN_JSONL}") 条)"
echo "  OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "  WANDB:       project=${WANDB_PROJECT} run=${WANDB_RUN_NAME}"
echo "  GPUS=${NPROC} BS=${BATCH_SIZE} GA=${GRAD_ACCUM} LR=${LR}"
echo "  SAVE_STEPS=${SAVE_STEPS} (只保留最新 1 份)"
echo "============================================================"

torchrun --nproc_per_node="${NPROC}" \
    -m training.stage1_sft_train \
    --role teacher_p \
    --train_jsonl "${TRAIN_JSONL}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --max_length "${MAX_LENGTH}" \
    --max_frames "${MAX_FRAMES}" \
    --fps "${FPS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --keep_only_latest_ckpt \
    --num_workers "${NUM_WORKERS}" \
    --bf16 \
    --no_save_final \
    --deepspeed "${DEEPSPEED}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}"

