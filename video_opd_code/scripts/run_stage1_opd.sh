#!/usr/bin/env bash
# Stage1-OPD 训练脚本（四模型协同蒸馏 · On-Policy Distillation）
# 8卡 = 2组×4卡/组，每组：Student + Translator + Teacher_R + Teacher_P
set -euo pipefail
cd "$(dirname "$0")/.."

# wandb 免交互登录
export WANDB_API_KEY="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"

DATA_ROOT="$(python -m configs.paths data_root)"
CKPT_ROOT="${DATA_ROOT}/outputs/checkpoints"

# 模型 checkpoint 路径
STUDENT_CKPT="${STUDENT_CKPT:-${CKPT_ROOT}/stage1_sft_v5_latent/checkpoint-9000}"
TEACHER_R_CKPT="${TEACHER_R_CKPT:-${CKPT_ROOT}/stage1_sft_teacher_r/final}"
TEACHER_P_CKPT="${TEACHER_P_CKPT:-${CKPT_ROOT}/stage1_sft_teacher_p/final}"

# 训练数据
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_ROOT}/outputs/stage1_sft/stage1_sft_template_all.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${CKPT_ROOT}/stage1_opd}"

# 分布式参数：2个进程，每个进程管理一组4卡
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

mkdir -p "${OUTPUT_DIR}"

CMD=(torchrun
    --nproc_per_node="${NPROC_PER_NODE}"
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    -m training.stage1_opd
    --train_jsonl "${TRAIN_JSONL}"
    --output_dir  "${OUTPUT_DIR}"
    --student_ckpt "${STUDENT_CKPT}"
    --teacher_r_ckpt "${TEACHER_R_CKPT}"
    --teacher_p_ckpt "${TEACHER_P_CKPT}"
    --num_train_epochs           "${NUM_TRAIN_EPOCHS:-2}"
    --gradient_accumulation_steps "${GRAD_ACCUM:-4}"
    --student_lr                 "${STUDENT_LR:-5e-6}"
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
    --max_decode_len             "${MAX_DECODE_LEN:-256}"
    --lambda_kl                  "${LAMBDA_KL:-1.0}"
    --lambda_ans                 "${LAMBDA_ANS:-1.0}"
    --lambda_aux                 "${LAMBDA_AUX:-0.1}"
    --lambda_think_end           "${LAMBDA_THINK_END:-1.0}"
    --temperature                "${TEMPERATURE:-2.0}"
    --teacher_r_max_length       "${TEACHER_R_MAX_LENGTH:-8192}"
    --teacher_p_max_length       "${TEACHER_P_MAX_LENGTH:-32768}"
    --bf16
)

if [[ -n "${MODEL_PATH:-}" ]]; then
    CMD+=(--model_path "${MODEL_PATH}")
fi
if [[ -n "${MAX_SAMPLES:-}" && "${MAX_SAMPLES:-0}" -gt 0 ]]; then
    CMD+=(--max_samples "${MAX_SAMPLES}")
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

echo "====== Stage1-OPD (On-Policy Distillation · 4-Model) ======"
echo "  STUDENT_CKPT=${STUDENT_CKPT}"
echo "  TEACHER_R_CKPT=${TEACHER_R_CKPT}"
echo "  TEACHER_P_CKPT=${TEACHER_P_CKPT}"
echo "  TRAIN_JSONL=${TRAIN_JSONL}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  NPROC=${NPROC_PER_NODE} (每进程管理4卡)"
echo "  MAX_LENGTH=${MAX_LENGTH:-32768}"
echo "  STUDENT_LR=${STUDENT_LR:-5e-6} τ=${TEMPERATURE:-2.0}"
echo "  LAMBDA: kl=${LAMBDA_KL:-1.0} ans=${LAMBDA_ANS:-1.0} aux=${LAMBDA_AUX:-0.1} think_end=${LAMBDA_THINK_END:-1.0}"
echo "============================================================"
"${CMD[@]}"
