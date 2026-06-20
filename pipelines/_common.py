"""
pipelines 公共工具：
- 缓存目录管理
- 外部模型 lazy loading 单例（Grounding-DINO / Depth-Anything / SAM）
"""
import hashlib
import threading
from pathlib import Path
from typing import Tuple

from configs.paths import (
    OUTPUT_ROOT,
    GROUNDING_DINO_PATH,
    DEPTH_ANYTHING_PATH,
    SAM3_PATH,
)


# pipeline 中间产物缓存目录
PIPELINE_CACHE_DIR = OUTPUT_ROOT / "pipeline_cache"


def get_cache_dir(subdir: str) -> Path:
    """获取/创建pipeline中间产物缓存子目录"""
    p = PIPELINE_CACHE_DIR / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


def hash_key(*parts) -> str:
    """根据输入参数生成稳定哈希，用作缓存文件名"""
    h = hashlib.md5()
    for x in parts:
        h.update(str(x).encode("utf-8"))
    return h.hexdigest()[:16]


# ----------------------- Lazy Loaders -----------------------

_GD_LOCK = threading.Lock()
_GD_MODEL = None
_GD_PROCESSOR = None


def get_grounding_dino() -> Tuple[object, object]:
    """加载 Grounding-DINO（HuggingFace transformers）"""
    global _GD_MODEL, _GD_PROCESSOR
    if _GD_MODEL is not None:
        return _GD_MODEL, _GD_PROCESSOR
    with _GD_LOCK:
        if _GD_MODEL is not None:
            return _GD_MODEL, _GD_PROCESSOR
        if not GROUNDING_DINO_PATH.exists():
            raise FileNotFoundError(
                f"Grounding-DINO 权重未找到: {GROUNDING_DINO_PATH}\n"
f"请下载到该目录，或在 configs/paths.yaml 中调整 model_root / overrides"
            )
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _GD_PROCESSOR = AutoProcessor.from_pretrained(str(GROUNDING_DINO_PATH))
        _GD_MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(GROUNDING_DINO_PATH)
        ).to(device).eval()
        return _GD_MODEL, _GD_PROCESSOR


_DA_LOCK = threading.Lock()
_DA_MODEL = None
_DA_PROCESSOR = None


def get_depth_anything() -> Tuple[object, object]:
    """加载 Depth-Anything-V2"""
    global _DA_MODEL, _DA_PROCESSOR
    if _DA_MODEL is not None:
        return _DA_MODEL, _DA_PROCESSOR
    with _DA_LOCK:
        if _DA_MODEL is not None:
            return _DA_MODEL, _DA_PROCESSOR
        if not DEPTH_ANYTHING_PATH.exists():
            raise FileNotFoundError(
                f"Depth-Anything 权重未找到: {DEPTH_ANYTHING_PATH}"
            )
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _DA_PROCESSOR = AutoImageProcessor.from_pretrained(str(DEPTH_ANYTHING_PATH))
        _DA_MODEL = AutoModelForDepthEstimation.from_pretrained(
            str(DEPTH_ANYTHING_PATH)
        ).to(device).eval()
        return _DA_MODEL, _DA_PROCESSOR


_SAM_LOCK = threading.Lock()
_SAM3_MODEL = None
_SAM3_PROCESSOR = None


def get_sam3_video() -> Tuple[object, object]:
    """
    加载 SAM 3 视频模型 (transformers 内置 Sam3VideoModel)。
    SAM3 支持文本 prompt 直接做 detect+segment+track，
    不再需要先用 Grounding-DINO 拿首帧 box。

    返回: (model, processor)
    要求 transformers >= 4.57.0（已集成 facebook/sam3）。
    """
    global _SAM3_MODEL, _SAM3_PROCESSOR
    if _SAM3_MODEL is not None:
        return _SAM3_MODEL, _SAM3_PROCESSOR
    with _SAM_LOCK:
        if _SAM3_MODEL is not None:
            return _SAM3_MODEL, _SAM3_PROCESSOR
        if not SAM3_PATH.exists():
            raise FileNotFoundError(
                f"SAM3 权重未找到: {SAM3_PATH}\n"
                f"请将 facebook/sam3 权重放在 {SAM3_PATH}，或软链已有目录到此处。"
            )
        # 必备文件检查（HuggingFace 格式）
        for fname in ["config.json", "model.safetensors", "processor_config.json"]:
            fp = SAM3_PATH / fname
            if not fp.exists():
                raise FileNotFoundError(
                    f"SAM3 目录缺少 {fname}: {fp}\n"
                    f"请确认权重为 HuggingFace 格式 (facebook/sam3)。"
                )
        import torch
        try:
            from transformers import Sam3VideoModel, Sam3VideoProcessor
        except ImportError as e:
            raise ImportError(
                "未找到 Sam3VideoModel / Sam3VideoProcessor。\n"
                "请升级 transformers >= 4.57.0：pip install -U 'transformers>=4.57.0'"
            ) from e
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        _SAM3_PROCESSOR = Sam3VideoProcessor.from_pretrained(str(SAM3_PATH))
        _SAM3_MODEL = Sam3VideoModel.from_pretrained(str(SAM3_PATH)).to(device, dtype=dtype).eval()
        return _SAM3_MODEL, _SAM3_PROCESSOR
