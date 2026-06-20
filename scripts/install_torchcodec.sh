#!/usr/bin/env bash
# =============================================================================
# 安装 torchcodec —— Qwen3-VL 视频解码的官方推荐后端
# =============================================================================
# 背景:
#   transformers 默认用 torchcodec 解视频; 没装时回落到 torchvision (内部走 PyAV)。
#   torchvision/PyAV 在 8 卡 + 多 worker 并发下会触发 libswscale 全局资源争用:
#       av.error.BlockingIOError: Resource temporarily unavailable
#       Failed initializing scaling graph fmt:yuv420p -> fmt:rgb24
#   torchcodec 直接调 FFmpeg C API + 张量管道, 不走 swscaler 的 Python 全局上下文,
#   多进程并发不会冲突, 是根治方案。
#
# 关键限制 (动态链接 ffmpeg):
#   截至本脚本编写时, **所有** torchcodec 版本 (0.2 ~ 0.7) 都只支持
#       FFmpeg 4 / 5 / 6 / 7   (libavutil 56 / 57 / 58 / 59)
#   完全不识别 FFmpeg 8 (libavutil.so.60)。
#
#   而你当前环境是 conda 装的 ffmpeg 8.0.1, 所以必须把 ffmpeg 降到 7.x。
#
# 本脚本流程:
#   1. 卸载/降级当前 conda ffmpeg 8 -> ffmpeg 7
#   2. 安装 torchcodec 0.2.1 (官方文档说该版本针对 PyTorch 2.6 构建)
#   3. import 验证
#   4. transformers.is_torchcodec_available() 验证
#
# 使用:
#   bash scripts/install_torchcodec.sh
# =============================================================================

set -uo pipefail

CONDA_ENV_PREFIX="${CONDA_PREFIX:-/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/root/miniconda3/envs/video_opd}"

echo "[install_torchcodec] 当前 PyTorch:"
python -c "import torch; print(' torch =', torch.__version__, '| cuda =', torch.version.cuda)"

echo "[install_torchcodec] 当前 ffmpeg:"
ffmpeg -version 2>&1 | head -1 || echo "(没找到 ffmpeg)"

echo "[install_torchcodec] 当前 libavutil:"
ls "$CONDA_ENV_PREFIX"/lib/libavutil.so.* 2>/dev/null | head -5

verify_torchcodec() {
    python - <<'PY'
import sys
try:
    import torchcodec
    from torchcodec.decoders import VideoDecoder  # noqa: F401
    print(f"[OK] torchcodec {getattr(torchcodec, '__version__', '?')} 加载成功")
    sys.exit(0)
except Exception as e:
    print(f"[FAIL] torchcodec 导入失败:\n{e}", file=sys.stderr)
    sys.exit(1)
PY
}

# -----------------------------------------------------------------------------
# Step 0: 检查当前 ffmpeg 是否已经是 7 (libavutil.so.59)
# -----------------------------------------------------------------------------
HAS_FFMPEG7=0
if ls "$CONDA_ENV_PREFIX"/lib/libavutil.so.59* >/dev/null 2>&1; then
    HAS_FFMPEG7=1
    echo "[install_torchcodec] ✓ 已存在 libavutil.so.59 (ffmpeg 7), 跳过 ffmpeg 降级"
fi

# -----------------------------------------------------------------------------
# Step 1: 把 conda ffmpeg 8 -> ffmpeg 7 (如果尚未做)
# -----------------------------------------------------------------------------
if [ "$HAS_FFMPEG7" -eq 0 ]; then
    echo
    echo "[install_torchcodec] Step 1) 用 conda 把 ffmpeg 8 降级到 ffmpeg 7"
    echo "                              (torchcodec 至今不支持 ffmpeg 8)"
    echo "                              这一步可能比较久, 请耐心等待 ..."

    # conda-forge 的 ffmpeg 7.x 系列是经过 torchcodec 测试的标准版本
    # 不要用 --force-reinstall, conda 会自己处理依赖
    conda install -y -n "$(basename "$CONDA_ENV_PREFIX")" \
        -c conda-forge "ffmpeg=7.*" 2>&1 | tail -20

    echo
    echo "[install_torchcodec]   降级后的 ffmpeg:"
    ffmpeg -version 2>&1 | head -1 || echo "(没找到 ffmpeg)"
    echo "[install_torchcodec]   降级后的 libavutil:"
    ls "$CONDA_ENV_PREFIX"/lib/libavutil.so.* 2>/dev/null | head -5

    if ! ls "$CONDA_ENV_PREFIX"/lib/libavutil.so.59* >/dev/null 2>&1; then
        echo
        echo "[install_torchcodec] ❌ ffmpeg 降级失败: 仍未找到 libavutil.so.59"
        echo "                      请手动检查 conda 是否能连到 conda-forge:"
        echo "                          conda search -c conda-forge 'ffmpeg=7.*'"
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# Step 2: 安装 torchcodec 0.2.1 (PyTorch 2.6 配套版本)
# -----------------------------------------------------------------------------
echo
echo "[install_torchcodec] Step 2) 卸载旧 torchcodec, 重装 0.2.1 (PyTorch 2.6 配套)"
pip uninstall -y torchcodec 2>/dev/null || true
pip install --no-cache-dir "torchcodec==0.2.1" 2>&1 | tail -5

# -----------------------------------------------------------------------------
# Step 3: 验证
# -----------------------------------------------------------------------------
echo
echo "[install_torchcodec] Step 3) 验证 torchcodec import ..."
if ! verify_torchcodec; then
    echo
    echo "[install_torchcodec] ❌ torchcodec 0.2.1 仍无法加载, 请把上面的报错贴给开发者排查"
    exit 1
fi

echo
echo "[install_torchcodec] Step 4) 验证 transformers 能识别 torchcodec ..."
python - <<'PY'
try:
    from transformers.utils.import_utils import is_torchcodec_available
    avail = is_torchcodec_available()
    print(f"  transformers.is_torchcodec_available() = {avail}")
    if not avail:
        raise RuntimeError("transformers 居然还没识别到, 异常")
except Exception as e:
    print(f"[WARN] transformers 检测时报错: {e}")
PY

# -----------------------------------------------------------------------------
# Step 5: 端到端冒烟测试 — 实际解一个测试视频
# -----------------------------------------------------------------------------
echo
echo "[install_torchcodec] Step 5) 冒烟测试: 用 torchcodec 解一段测试视频 ..."
python - <<'PY'
import os, sys
test_video = "/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/video_opd_data/datasets/charades_sta/videos/001YG.mp4"
# Agent 视角的等价路径回退
if not os.path.exists(test_video):
    test_video = test_video.replace(
        "/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng",
        "/apdcephfs/aigc/group_2/user_sleepfeng",
    )
if not os.path.exists(test_video):
    print(f"[SKIP] 找不到测试视频: {test_video}")
    sys.exit(0)
try:
    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(test_video)
    frame = dec[0]
    print(f"[OK] 解码成功. metadata = {dec.metadata}")
    print(f"     frame.shape = {tuple(frame.shape)}, dtype = {frame.dtype}")
except Exception as e:
    print(f"[FAIL] 解码报错: {e}", file=sys.stderr)
    sys.exit(1)
PY

echo
echo "[install_torchcodec] ✅ 全部就绪. 下次跑 SFT/OPD 时, transformers 会自动用 torchcodec,"
echo "                      不再触发 PyAV swscaler 并发冲突。"
