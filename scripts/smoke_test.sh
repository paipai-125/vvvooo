#!/usr/bin/env bash
# 冒烟测试: 验证 import 与基本路径无误, 不真正加载模型
# 用法: bash scripts/smoke_test.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "[smoke_test] python = $(which python)"
echo "[smoke_test] cwd    = $(pwd)"

python - <<'PY'
import importlib, sys
mods = [
    "configs", "configs.paths",
    "utils", "utils.parser", "utils.video_utils",
    "pipelines",
    "pipelines.temporal_locate", "pipelines.temporal_clip",
    "pipelines.spatial_detect", "pipelines.spatial_crop",
    "pipelines.depth_overlay", "pipelines.tracking_overlay",
    "pipelines.ocr_zoom", "pipelines.raw",
    "data_preparation",
    "data_preparation.stage1_sft_template",
    "data_preparation.stage1_sft_llm_augment",
    "data_preparation.stage1_opd_filter",
    "training",
    "training.stage1_sft_train",
    "training.stage1_opd_train",
    "evaluation",
    "evaluation.pre_experiment_focus",
]
for m in mods:
    importlib.import_module(m)
    print(f"  OK  {m}")
print("[smoke_test] all imports OK")
PY
