#!/usr/bin/env bash
# Stage 1-OPD 数据筛选: 学生失败 ∩ 教师成功
# 用法:
#   bash scripts/run_stage1_opd_filter.sh /path/to/candidate.jsonl
#   bash scripts/run_stage1_opd_filter.sh /path/to/candidate.jsonl /path/to/output.jsonl
#
# 多卡分片: 设置 RANK / WORLD_SIZE 环境变量分多进程跑
#   WORLD_SIZE=8 RANK=0 STUDENT_DEVICE=cuda:0 TEACHER_DEVICE=cuda:1 bash ... &
#   WORLD_SIZE=8 RANK=1 STUDENT_DEVICE=cuda:2 TEACHER_DEVICE=cuda:3 bash ... &
#   ...

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <input.jsonl> [output.jsonl]"
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-}"

CMD=(python -m data_preparation.stage1_opd_filter
    --input "${INPUT}"
    --student_device "${STUDENT_DEVICE:-cuda:0}"
    --teacher_device "${TEACHER_DEVICE:-cuda:1}"
    --rank "${RANK:-0}"
    --world_size "${WORLD_SIZE:-1}"
)

if [[ -n "${OUTPUT}" ]]; then
    CMD+=(--output "${OUTPUT}")
fi

echo "[run_stage1_opd_filter] CMD: ${CMD[*]}"
"${CMD[@]}"
