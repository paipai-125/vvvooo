#!/usr/bin/env bash
# ============================================================
# 一键创建 conda 环境 + 安装全部依赖
# 用法：
#   bash setup_env.sh                       # 默认 env_name=video_opd, python=3.11
#   ENV_NAME=my_env PYTHON_VERSION=3.11 bash setup_env.sh
# ============================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-video_opd}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"
CUDA_TAG="${CUDA_TAG:-cu124}"

# 切到脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "[setup_env] 工作目录: ${SCRIPT_DIR}"
echo "[setup_env] ENV_NAME=${ENV_NAME} PYTHON=${PYTHON_VERSION} CUDA=${CUDA_TAG}"

# ---------- 1. 创建 / 复用 conda 环境 ----------
if ! command -v conda >/dev/null 2>&1; then
    echo "[setup_env] 错误: 找不到 conda，请先安装 Miniconda/Anaconda" >&2
    exit 1
fi

# 让 conda activate 在脚本中可用
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[setup_env] 复用已存在的 conda 环境: ${ENV_NAME}"
else
    echo "[setup_env] 创建 conda 环境: ${ENV_NAME} (python=${PYTHON_VERSION})"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

# ---------- 2. 安装系统级依赖（ffmpeg）----------
# 使用 conda-forge 装 ffmpeg，避免污染系统
echo "[setup_env] 安装 ffmpeg (conda-forge)"
conda install -y -c conda-forge ffmpeg

# ---------- 3. 安装 PyTorch ----------
echo "[setup_env] 安装 PyTorch ${TORCH_VERSION} (${CUDA_TAG})"
pip install --upgrade pip
pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# ---------- 4. 安装 requirements.txt ----------
# 用 --upgrade-strategy only-if-needed 避免 pip 主动把已经装好的 torch 升级
# 到 cu128 wheel（cu128 wheel 需要 NVIDIA driver >= 12.6，本服务器 driver=12.4）。
echo "[setup_env] 安装 requirements.txt (only-if-needed)"
pip install --upgrade-strategy only-if-needed -r "${SCRIPT_DIR}/requirements.txt"

# ---------- 4.1 守卫：确保 torch 仍是 ${CUDA_TAG} 版本 ----------
# 若 requirements.txt 中某个依赖不小心升级了 torch，会导致 driver 太旧报错。
# 这里强制重装一次正确的 torch wheel（无副作用：版本相同则只是重新校验）。
echo "[setup_env] 守卫: 强制对齐 torch==${TORCH_VERSION}+${CUDA_TAG}"
pip install --no-deps --force-reinstall \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# ---------- 5. SAM 3（由 transformers >= 4.57 内置加载，无需源码包） ----------
# tracking_overlay pipeline 现已直接走 transformers 的 Sam3VideoModel。
# 只需把 facebook/sam3 权重放到 <model_root>/sam3/ 即可
# （<model_root> 由 configs/paths.yaml 中的 model_root 字段决定），
# 详见 README.md "模型权重下载" 章节。
echo "[setup_env] SAM 3 通过 transformers (>=4.57) 内置加载，跳过源码安装"

# ---------- 5.1 可选：vLLM（来源 C 数据增强用，默认不装） ----------
# vllm 高版本会强行把 torch 升到 cu128，与本服务器 driver=12.4 不兼容。
# 仅当用户确认 driver >= 12.6 时手动开启：
#   INSTALL_VLLM=1 bash setup_env.sh
if [[ "${INSTALL_VLLM:-0}" == "1" ]]; then
    echo "[setup_env] (可选) 安装 vLLM 0.6.3（与 torch 2.4.1 兼容的最后一个版本）"
    pip install --no-deps "vllm==0.6.3" || {
        echo "[setup_env] 警告: vLLM 安装失败，可改用 transformers 路径跑 stage1_sft_llm_augment.py" >&2
    }
else
    echo "[setup_env] 跳过 vLLM 安装（如需启用: INSTALL_VLLM=1 bash setup_env.sh）"
fi

# ---------- 6. 校验 ----------
echo "[setup_env] 校验 import"
python - <<'PY'
import importlib, sys
for m in ["torch", "transformers", "accelerate", "deepspeed", "cv2",
         "PIL", "numpy", "tqdm", "einops", "decord"]:
    importlib.import_module(m)
    print(f"  OK  {m}")
import torch
ok = torch.cuda.is_available()
print(f"  torch={torch.__version__}  cuda_built={torch.version.cuda}  "
      f"available={ok}  device_count={torch.cuda.device_count()}")
if not ok:
    print(
        "\n[setup_env][FATAL] torch.cuda.is_available()=False\n"
        "  常见原因：torch 的 CUDA wheel 比本机 NVIDIA driver 更新。\n"
        "  立即修复（强制对齐到 cu124）：\n"
        "    pip install --no-deps --force-reinstall \\\n"
        "        torch==2.4.1 torchvision==0.19.1 \\\n"
        "        --index-url https://download.pytorch.org/whl/cu124\n"
        "  并检查是否曾安装过 vllm（vllm 会拖动 torch 到 cu128）：\n"
        "    pip uninstall -y vllm xformers triton\n",
        file=sys.stderr,
    )
    sys.exit(1)
PY

echo ""
echo "[setup_env] 完成！激活方式: conda activate ${ENV_NAME}"

# ---------- 7. 自动准备 configs/paths.yaml ----------
PATHS_YAML="${SCRIPT_DIR}/configs/paths.yaml"
PATHS_EXAMPLE="${SCRIPT_DIR}/configs/paths.example.yaml"
if [[ ! -f "${PATHS_YAML}" ]]; then
    if [[ -f "${PATHS_EXAMPLE}" ]]; then
        cp "${PATHS_EXAMPLE}" "${PATHS_YAML}"
        echo "[setup_env] 已从模板生成 configs/paths.yaml，请按需编辑里面的 data_root / model_root"
    else
        echo "[setup_env][WARN] 找不到 configs/paths.example.yaml，请手动创建 configs/paths.yaml" >&2
    fi
else
    echo "[setup_env] 检测到 configs/paths.yaml 已存在，未覆盖"
fi

echo "[setup_env] 下一步:"
echo "    1) 编辑 configs/paths.yaml，填入你的 data_root / model_root（绝对路径）"
echo "    2) 下载模型权重和数据集（详见 README.md）"
echo "    3) 跑冒烟测试: bash scripts/smoke_test.sh"
