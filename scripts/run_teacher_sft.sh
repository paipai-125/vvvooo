#!/usr/bin/env bash
# Teacher SFT: 数据准备 + 训练 + 推理检查
# 用法: bash scripts/run_teacher_sft.sh teacher_r
#       bash scripts/run_teacher_sft.sh teacher_p
#       bash scripts/run_teacher_sft.sh all
set -euo pipefail
cd "$(dirname "$0")/.."

ROLE="${1:-teacher_r}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
WANDB_PROJECT="${WANDB_PROJECT:-video-opd-teacher-sft}"
DATA_ROOT="/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/video_opd_data"
SFT_DIR="${DATA_ROOT}/outputs/stage1_sft"
STUDENT_JSONL="${SFT_DIR}/stage1_sft_template_all.jsonl"
CKPT_ROOT="${DATA_ROOT}/outputs/checkpoints"
NPROC="${NPROC:-8}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-5}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
SAVE_STEPS="${SAVE_STEPS:-100}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

train_teacher_r() {
    local JSONL="${SFT_DIR}/stage1_sft_teacher_r.jsonl"
    local OUTDIR="${CKPT_ROOT}/stage1_sft_teacher_r"
    echo "[1/3] 生成 Teacher_R 数据..."
    python -m data_preparation.prepare_teacher_sft_data --role teacher_r --input "${STUDENT_JSONL}" --output_dir "${SFT_DIR}"
    echo "  -> ${JSONL} ($(wc -l < "${JSONL}") 条)"
    echo "[2/3] 训练 Teacher_R (纯文本, max_length=8192)..."
    torchrun --nproc_per_node="${NPROC}" -m training.teacher_sft \
        --role teacher_r --train_jsonl "${JSONL}" --output_dir "${OUTDIR}" \
        --epochs "${EPOCHS}" --lr "${LR}" --max_length 8192 \
        --grad_accum "${GRAD_ACCUM}" --save_steps "${SAVE_STEPS}" \
        --max_samples "${MAX_SAMPLES}" --gradient_checkpointing \
        --wandb_project "${WANDB_PROJECT}" --wandb_run_name "teacher_r-sft"
    echo "[3/3] Teacher_R 推理检查..."
    python -m training.teacher_sft --role teacher_r --train_jsonl "${JSONL}" \
        --output_dir "${OUTDIR}" --inference_only --max_samples 5
}

train_teacher_p() {
    local JSONL="${SFT_DIR}/stage1_sft_teacher_p.jsonl"
    local OUTDIR="${CKPT_ROOT}/stage1_sft_teacher_p"
    echo "[1/3] 生成 Teacher_P 数据..."
    python -m data_preparation.prepare_teacher_sft_data --role teacher_p --input "${STUDENT_JSONL}" --output_dir "${SFT_DIR}"
    echo "  -> ${JSONL} ($(wc -l < "${JSONL}") 条)"
    echo "[2/3] 训练 Teacher_P (视频, max_length=32768)..."
    torchrun --nproc_per_node="${NPROC}" -m training.teacher_sft \
        --role teacher_p --train_jsonl "${JSONL}" --output_dir "${OUTDIR}" \
        --epochs "${EPOCHS}" --lr "${LR}" --max_length 32768 \
        --max_frames 64 --fps 1.0 \
        --grad_accum "${GRAD_ACCUM}" --save_steps "${SAVE_STEPS}" \
        --max_samples "${MAX_SAMPLES}" --gradient_checkpointing \
        --wandb_project "${WANDB_PROJECT}" --wandb_run_name "teacher_p-sft"
    echo "[3/3] Teacher_P 推理检查..."
    python -m training.teacher_sft --role teacher_p --train_jsonl "${JSONL}" \
        --output_dir "${OUTDIR}" --inference_only --max_samples 5
}

case "${ROLE}" in
    teacher_r) train_teacher_r ;;
    teacher_p) train_teacher_p ;;
    all) train_teacher_r; echo ""; train_teacher_p ;;
    *) echo "用法: bash scripts/run_teacher_sft.sh [teacher_r|teacher_p|all]"; exit 1 ;;
esac
echo "[Done] Teacher SFT 完成。"
