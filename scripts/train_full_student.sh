#!/usr/bin/env bash
# =============================================================================
# 全量训练 —— 学生 (Coconut 式潜空间 SFT)
#
# 与 teacher_r / teacher_p 完全独立，可以单独 Ctrl-C 终止。
#
# 特性：
#   - 自动用 WANDB_API_KEY 免交互登录 wandb
#   - 每 SAVE_STEPS 步保存一次 ckpt，只保留最新一份（--keep_only_latest_ckpt）
#   - 训练结束默认 --no_save_final（避免 NCCL OOM；ckpt 已经周期性保存）
#
# 用法：
#   bash scripts/train_full_student.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------- wandb ----------
export WANDB_API_KEY="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"
WANDB_PROJECT="${WANDB_PROJECT:-video-opd-stage1-sft}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-student-full}"

# 自动检测/安装 wandb（避免 [wandb] 未安装 wandb 的提示）
if ! python -c "import wandb" 2>/dev/null; then
    echo "[wandb] 未检测到 wandb，开始 pip install ..."
    pip install --quiet wandb || {
        echo "[wandb][WARN] 安装失败，将继续训练但 loss 不会上传到 wandb"
    }
fi

# ---------- 路径 ----------
DATA_ROOT="$(python -m configs.paths data_root)"
SFT_DIR="${DATA_ROOT}/outputs/stage1_sft"
TRAIN_JSONL="${TRAIN_JSONL:-${SFT_DIR}/stage1_sft_template_all.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/outputs/checkpoints/stage1_sft_student}"

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
LATENT_WARMUP="${LATENT_WARMUP:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEEPSPEED="${DEEPSPEED:-scripts/ds_zero2.json}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${TRAIN_JSONL}" ]]; then
    echo "[ERROR] 学生训练数据不存在: ${TRAIN_JSONL}"
    echo "       请先运行: bash scripts/run_sft_full.sh"
    exit 1
fi

echo "============================================================"
echo "[Train Full] role=student"
echo "  TRAIN_JSONL: ${TRAIN_JSONL} ($(wc -l < "${TRAIN_JSONL}") 条)"
echo "  OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "  WANDB:       project=${WANDB_PROJECT} run=${WANDB_RUN_NAME}"
echo "  GPUS=${NPROC} BS=${BATCH_SIZE} GA=${GRAD_ACCUM} LR=${LR}"
echo "  SAVE_STEPS=${SAVE_STEPS} (只保留最新 1 份)"
echo "============================================================"

# 后台运行模式：BG=1 bash scripts/train_full_student.sh
# 这样 ssh 连接断开也不会杀进程，日志写到 OUTPUT_DIR/train.log
TORCHRUN_CMD=(
    torchrun --nproc_per_node="${NPROC}"
    -m training.stage1_sft_train
    --role student
    --train_jsonl "${TRAIN_JSONL}"
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs "${EPOCHS}"
    --per_device_batch_size "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --learning_rate "${LR}"
    --max_length "${MAX_LENGTH}"
    --max_frames "${MAX_FRAMES}"
    --fps "${FPS}"
    --logging_steps "${LOGGING_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --keep_only_latest_ckpt
    --latent_warmup_steps "${LATENT_WARMUP}"
    --num_workers "${NUM_WORKERS}"
    --bf16
    --no_save_final
    --deepspeed "${DEEPSPEED}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_run_name "${WANDB_RUN_NAME}"
)

if [[ "${BG:-0}" == "1" ]]; then
    LOG_FILE="${OUTPUT_DIR}/train.log"
    echo "[BG] 后台运行, 日志: ${LOG_FILE}"
    echo "[BG] 实时查看: tail -f ${LOG_FILE}"
    nohup "${TORCHRUN_CMD[@]}" > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "[BG] PID=${PID} (kill 命令: kill ${PID})"
else
    "${TORCHRUN_CMD[@]}"
fi

