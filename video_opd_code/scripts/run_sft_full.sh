#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="$(python -m configs.paths data_root)"
SFT_DIR="${DATA_ROOT}/outputs/stage1_sft"
STUDENT_JSONL="${SFT_DIR}/stage1_sft_template_all.jsonl"
TEACHER_R_JSONL="${SFT_DIR}/stage1_sft_teacher_r.jsonl"
TEACHER_P_JSONL="${SFT_DIR}/stage1_sft_teacher_p.jsonl"

echo "[Full SFT Data Prep] $(date)"
echo "  STUDENT:   ${STUDENT_JSONL}"
echo "  TEACHER_R: ${TEACHER_R_JSONL}"
echo "  TEACHER_P: ${TEACHER_P_JSONL}"

if [[ ! -f "${STUDENT_JSONL}" ]]; then
    echo "[ERROR] 学生全量数据不存在: ${STUDENT_JSONL}"
    exit 1
fi

if [[ ! -f "${TEACHER_R_JSONL}" || ! -f "${TEACHER_P_JSONL}" ]]; then
    echo "[Step] 从全量学生数据派生 teacher_r / teacher_p ..."
    python -m data_preparation.prepare_teacher_sft_data \
        --role all --input "${STUDENT_JSONL}" --output_dir "${SFT_DIR}"
else
    echo "[Step] teacher_r / teacher_p 已存在, 跳过派生"
fi

echo ""
echo "[OK] 数据准备完成。统计："
echo "  STUDENT:   $(wc -l < "${STUDENT_JSONL}") 条"
echo "  TEACHER_R: $(wc -l < "${TEACHER_R_JSONL}") 条"
echo "  TEACHER_P: $(wc -l < "${TEACHER_P_JSONL}") 条"
echo ""
echo "下一步：分别运行三个角色的训练（每条都是独立命令，可单独 Ctrl-C 终止）"
echo "  bash scripts/train_full_student.sh"
echo "  bash scripts/train_full_teacher_r.sh"
echo "  bash scripts/train_full_teacher_p.sh"
