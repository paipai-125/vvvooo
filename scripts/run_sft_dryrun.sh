#!/usr/bin/env bash
# =============================================================================
# SFT 试跑脚本 —— 100 条数据完整走通 student / teacher_r / teacher_p
#
# 目的：验证整个 SFT 训练流程能跑通（数据加载、模型前向、loss 计算、梯度更新、保存）
#
# 用法：
#   bash scripts/run_sft_dryrun.sh
#
# 预计耗时：~10-15 分钟（8卡，100条数据，1 epoch）
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================================"
echo "[SFT DryRun] 开始试跑 — $(date)"
echo "============================================================"

# --- 路径配置 ---
DATA_ROOT="$(python -m configs.paths data_root)"
SFT_DIR="${DATA_ROOT}/outputs/stage1_sft"
DRYRUN_DIR="${DATA_ROOT}/outputs/dryrun_sft"
FULL_JSONL="${SFT_DIR}/stage1_sft_template_all.jsonl"

# 试跑数据路径
STUDENT_JSONL="${DRYRUN_DIR}/dryrun_student_100.jsonl"
TEACHER_R_JSONL="${DRYRUN_DIR}/dryrun_teacher_r.jsonl"
TEACHER_P_JSONL="${DRYRUN_DIR}/dryrun_teacher_p.jsonl"

# 输出 checkpoint 路径
STUDENT_OUT="${DRYRUN_DIR}/ckpt_student"
TEACHER_R_OUT="${DRYRUN_DIR}/ckpt_teacher_r"
TEACHER_P_OUT="${DRYRUN_DIR}/ckpt_teacher_p"

mkdir -p "${DRYRUN_DIR}"

# --- Step 1: 随机采样 100 条学生数据 ---
echo ""
echo "============================================================"
echo "[Step 1] 从 ${FULL_JSONL} 中随机采样 100 条"
echo "============================================================"

if [[ ! -f "${FULL_JSONL}" ]]; then
    echo "[ERROR] 学生 SFT 数据不存在: ${FULL_JSONL}"
    exit 1
fi

python -c "
import json, random, sys
random.seed(42)
input_path = '${FULL_JSONL}'
output_path = '${STUDENT_JSONL}'
with open(input_path, 'r') as f:
    lines = f.readlines()
sampled = random.sample(lines, min(100, len(lines)))
with open(output_path, 'w') as f:
    f.writelines(sampled)
print(f'  [OK] 采样 {len(sampled)} 条 -> {output_path}')
"

# --- Step 2: 生成教师数据 ---
echo ""
echo "============================================================"
echo "[Step 2] 生成教师 SFT 数据（从采样的 100 条中提取）"
echo "============================================================"

python -m data_preparation.prepare_teacher_sft_data \
    --role all \
    --input "${STUDENT_JSONL}" \
    --output_dir "${DRYRUN_DIR}"

# 重命名输出文件（prepare_teacher_sft_data 输出固定文件名）
if [[ -f "${DRYRUN_DIR}/stage1_sft_teacher_r.jsonl" ]]; then
    mv "${DRYRUN_DIR}/stage1_sft_teacher_r.jsonl" "${TEACHER_R_JSONL}"
fi
if [[ -f "${DRYRUN_DIR}/stage1_sft_teacher_p.jsonl" ]]; then
    mv "${DRYRUN_DIR}/stage1_sft_teacher_p.jsonl" "${TEACHER_P_JSONL}"
fi

echo ""
echo "  学生数据:    $(wc -l < "${STUDENT_JSONL}") 条"
echo "  Teacher_R:   $(wc -l < "${TEACHER_R_JSONL}") 条"
echo "  Teacher_P:   $(wc -l < "${TEACHER_P_JSONL}") 条"

# --- 训练超参（试跑用小参数）---
NPROC=8
EPOCHS=1
BATCH_SIZE=1
GRAD_ACCUM=2
LR=1e-5
MAX_LENGTH=40960
MAX_FRAMES=64
FPS=1.0
LOGGING_STEPS=5
SAVE_STEPS=9999  # 试跑不保存中间 checkpoint
LATENT_WARMUP=5  # 前5步标准CE，之后切潜空间
NUM_WORKERS=4    # 每卡 DataLoader workers; 依赖 torchcodec 避免并发冲突
DEEPSPEED="scripts/ds_zero2.json"

# --- Step 3: 训练学生（潜空间 SFT）---
echo ""
echo "============================================================"
echo "[Step 3] 训练学生 (Coconut-style latent space SFT)"
echo "  数据: ${STUDENT_JSONL}"
echo "  输出: ${STUDENT_OUT}"
echo "============================================================"

torchrun --nproc_per_node=${NPROC} \
    -m training.stage1_sft_train \
    --role student \
    --train_jsonl "${STUDENT_JSONL}" \
    --output_dir "${STUDENT_OUT}" \
    --num_train_epochs ${EPOCHS} \
    --per_device_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --max_length ${MAX_LENGTH} \
    --max_frames ${MAX_FRAMES} \
    --fps ${FPS} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --latent_warmup_steps ${LATENT_WARMUP} \
    --num_workers ${NUM_WORKERS} \
    --bf16 \
    --no_save_final \
    --deepspeed "${DEEPSPEED}"

echo ""
echo "  [OK] 学生训练完成 ✅"

# --- Step 4: 训练 Teacher_R（纯文本推理教师）---
echo ""
echo "============================================================"
echo "[Step 4] 训练 Teacher_R (纯文本推理教师, 标准 CE)"
echo "  数据: ${TEACHER_R_JSONL}"
echo "  输出: ${TEACHER_R_OUT}"
echo "============================================================"

torchrun --nproc_per_node=${NPROC} \
    -m training.stage1_sft_train \
    --role teacher_r \
    --train_jsonl "${TEACHER_R_JSONL}" \
    --output_dir "${TEACHER_R_OUT}" \
    --num_train_epochs ${EPOCHS} \
    --per_device_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --max_length ${MAX_LENGTH} \
    --max_frames ${MAX_FRAMES} \
    --fps ${FPS} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --bf16 \
    --no_save_final \
    --deepspeed "${DEEPSPEED}"

echo ""
echo "  [OK] Teacher_R 训练完成 ✅"

# --- Step 5: 训练 Teacher_P（视觉感知教师）---
echo ""
echo "============================================================"
echo "[Step 5] 训练 Teacher_P (视觉感知教师, 标准 CE)"
echo "  数据: ${TEACHER_P_JSONL}"
echo "  输出: ${TEACHER_P_OUT}"
echo "============================================================"

torchrun --nproc_per_node=${NPROC} \
    -m training.stage1_sft_train \
    --role teacher_p \
    --train_jsonl "${TEACHER_P_JSONL}" \
    --output_dir "${TEACHER_P_OUT}" \
    --num_train_epochs ${EPOCHS} \
    --per_device_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --max_length ${MAX_LENGTH} \
    --max_frames ${MAX_FRAMES} \
    --fps ${FPS} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --bf16 \
    --no_save_final \
    --deepspeed "${DEEPSPEED}"

echo ""
echo "  [OK] Teacher_P 训练完成 ✅"

# --- 完成 ---
echo ""
echo "============================================================"
echo "[SFT DryRun] 全部完成！ — $(date)"
echo "============================================================"
echo ""
echo "输出目录: ${DRYRUN_DIR}"
echo "  学生 ckpt:    ${STUDENT_OUT}"
echo "  Teacher_R:    ${TEACHER_R_OUT}"
echo "  Teacher_P:    ${TEACHER_P_OUT}"
echo ""
echo "下一步："
echo "  1. 检查 loss 是否正常下降"
echo "  2. 检查 checkpoint 是否正确保存"
echo "  3. 确认无误后，用完整数据训练："
echo "     ROLE=student bash scripts/run_stage1_sft_train.sh"
echo "     ROLE=teacher_r bash scripts/run_stage1_sft_train.sh"
echo "     ROLE=teacher_p bash scripts/run_stage1_sft_train.sh"
