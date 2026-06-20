"""
路径配置 —— 单一事实来源是 configs/paths.yaml。

所有数据集 / 模型 / 输出路径都从 yaml 派生：
  - data_root:  数据根目录（数据集 + 输出都在这下面）
  - model_root: 模型权重根目录（默认 = data_root/models）
  - overrides:  可选，单独覆盖某个模型路径（key 是模型短名）

如果 paths.yaml 不存在或字段缺失，启动时直接 raise，绝不容错跳过。

也可作为命令行入口供 shell 脚本读取路径，例如：
    DATA_ROOT="$(python -m configs.paths data_root)"
    MODEL_ROOT="$(python -m configs.paths model_root)"
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# 代码根目录
CODE_ROOT = Path(__file__).resolve().parent.parent

# yaml 真实配置 / 模板
_PATHS_YAML = CODE_ROOT / "configs" / "paths.yaml"
_PATHS_EXAMPLE_YAML = CODE_ROOT / "configs" / "paths.example.yaml"


def _load_yaml() -> Dict[str, Any]:
    """读取 paths.yaml；不存在直接报错并提示如何创建。"""
    if not _PATHS_YAML.exists():
        raise FileNotFoundError(
            f"路径配置文件不存在: {_PATHS_YAML}\n"
            f"请基于模板创建一份私人配置:\n"
            f"    cp {_PATHS_EXAMPLE_YAML.relative_to(CODE_ROOT)} "
            f"{_PATHS_YAML.relative_to(CODE_ROOT)}\n"
            f"然后用编辑器把 data_root / model_root 改成你机器上的真实路径。"
        )
    with _PATHS_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"{_PATHS_YAML} 解析结果不是字典: type={type(cfg).__name__}"
        )
    return cfg


def _require(cfg: Dict[str, Any], key: str) -> str:
    if key not in cfg or cfg[key] in (None, ""):
        raise KeyError(
            f"{_PATHS_YAML} 缺少必填字段 '{key}'。"
            f"请参考 {_PATHS_EXAMPLE_YAML.name} 补全。"
        )
    val = cfg[key]
    if not isinstance(val, str):
        raise TypeError(
            f"{_PATHS_YAML} 字段 '{key}' 必须是字符串路径，实际为 {type(val).__name__}: {val!r}"
        )
    return val


_CFG = _load_yaml()

# ========== 根路径 ==========
DATA_ROOT = Path(_require(_CFG, "data_root")).expanduser().resolve()

# model_root 不显式指定时，默认 data_root/models
_model_root_raw = _CFG.get("model_root")
if _model_root_raw in (None, ""):
    MODEL_ROOT = DATA_ROOT / "models"
else:
    if not isinstance(_model_root_raw, str):
        raise TypeError(
            f"{_PATHS_YAML} 字段 'model_root' 必须是字符串路径，"
            f"实际为 {type(_model_root_raw).__name__}: {_model_root_raw!r}"
        )
    MODEL_ROOT = Path(_model_root_raw).expanduser().resolve()

# overrides: 单独覆盖某些模型路径
_OVERRIDES: Dict[str, str] = _CFG.get("overrides") or {}
if not isinstance(_OVERRIDES, dict):
    raise TypeError(
        f"{_PATHS_YAML} 字段 'overrides' 必须是字典，"
        f"实际为 {type(_OVERRIDES).__name__}"
    )


def _model_path(short_name: str) -> Path:
    """对外解析模型路径：优先看 overrides，否则用 MODEL_ROOT/<short_name>。"""
    if short_name in _OVERRIDES:
        v = _OVERRIDES[short_name]
        if not isinstance(v, str) or not v:
            raise ValueError(
                f"{_PATHS_YAML} overrides.{short_name} 必须是非空字符串路径，得到 {v!r}"
            )
        return Path(v).expanduser().resolve()
    return MODEL_ROOT / short_name


# ========== 各模型路径 ==========
QWEN3_VL_4B_PATH = _model_path("Qwen3-VL-4B-Instruct")
GROUNDING_DINO_PATH = _model_path("grounding-dino-base")
SAM3_PATH = _model_path("sam3")
DEPTH_ANYTHING_PATH = _model_path("depth-anything-v2-large")
QWEN3_32B_PATH = _model_path("Qwen3-32B")
CUBE_RCNN_PATH = _model_path("cube-rcnn")              # 物体 3D bbox + 朝向估计
PERSPECTIVE_FIELDS_PATH = _model_path("perspective-fields")  # 相机视角估计（pitch/roll/FOV）

# ========== 数据集路径 ==========
DATASET_ROOT = DATA_ROOT / "datasets"
# —— 已规划且可下视频（≤100GB / 零 YouTube）——
CHARADES_STA_PATH = DATASET_ROOT / "charades_sta"   # temporal_locate ★
HC_STVG_PATH      = DATASET_ROOT / "hc_stvg"         # tracking_describe ★ + spatial_detect/crop ★
NEXTQA_PATH       = DATASET_ROOT / "nextqa"          # raw VideoQA ★
STAR_PATH         = DATASET_ROOT / "star"            # raw VideoQA ★（复用 Charades 视频）
CLEVRER_PATH      = DATASET_ROOT / "clevrer"         # raw VideoQA ★（因果/反事实）
# —— 新增：填补能力洞 ——
DIDEMO_PATH       = DATASET_ROOT / "didemo"          # temporal_locate ★（替代 ActivityNet）
TEXTVR_PATH       = DATASET_ROOT / "textvr"          # ocr_zoom ★
VIPSEG_PATH       = DATASET_ROOT / "vipseg"          # spatial_detect/crop + depth_overlay ★（124类全景分割）
# —— 仅标注（视频源是 YouTube/VidOR，本机不下视频，仅保留路径占位）——
ACTIVITYNET_PATH  = DATASET_ROOT / "activitynet_captions"  # 仅标注，视频缺失 → 由 DiDeMo 替代
VIDSTG_PATH       = DATASET_ROOT / "vidstg"                # 仅标注，视频缺失 → 由 HC-STVG 替代

# ========== 输出路径 ==========
OUTPUT_ROOT = DATA_ROOT / "outputs"
SFT_DATA_PATH = OUTPUT_ROOT / "stage1_sft"
OPD_DATA_PATH = OUTPUT_ROOT / "stage1_opd"
CHECKPOINTS_PATH = OUTPUT_ROOT / "checkpoints"


def ensure_dirs():
    """创建必要的输出目录"""
    for p in [OUTPUT_ROOT, SFT_DATA_PATH, OPD_DATA_PATH, CHECKPOINTS_PATH]:
        p.mkdir(parents=True, exist_ok=True)


def validate_model_paths():
    """检查模型路径是否存在，不存在直接报错"""
    missing = []
    for name, path in [
        ("Qwen3-VL-4B-Instruct", QWEN3_VL_4B_PATH),
        ("Grounding-DINO", GROUNDING_DINO_PATH),
        ("Depth-Anything-V2", DEPTH_ANYTHING_PATH),
        ("Cube-RCNN", CUBE_RCNN_PATH),
        ("PerspectiveFields", PERSPECTIVE_FIELDS_PATH),
    ]:
        if not path.exists():
            missing.append(f"  {name}: {path}")
    if missing:
        raise FileNotFoundError(
            "以下模型权重未找到，请先下载到指定路径:\n" + "\n".join(missing)
            + f"\n\n提示: 编辑 {_PATHS_YAML} 中的 model_root 或 overrides。"
        )


def validate_dataset_paths(required_datasets: list):
    """检查数据集路径是否存在"""
    mapping = {
        "charades_sta": CHARADES_STA_PATH,
        "activitynet": ACTIVITYNET_PATH,
        "vidstg": VIDSTG_PATH,
        "hc_stvg": HC_STVG_PATH,
        "nextqa": NEXTQA_PATH,
        "star": STAR_PATH,
        "clevrer": CLEVRER_PATH,
        "didemo": DIDEMO_PATH,

        "textvr": TEXTVR_PATH,
        "vipseg": VIPSEG_PATH,
    }
    missing = []
    for name in required_datasets:
        path = mapping.get(name)
        if path is None:
            raise ValueError(f"未知数据集名: {name}")
        if not path.exists():
            missing.append(f"  {name}: {path}")
    if missing:
        raise FileNotFoundError(
            "以下数据集未找到:\n" + "\n".join(missing)
            + f"\n\n提示: 编辑 {_PATHS_YAML} 中的 data_root，"
              f"或把数据集放到 {DATASET_ROOT}/ 下。"
        )


# ============================================================
# CLI 入口：供 shell 脚本读取 yaml 中的路径
#   python -m configs.paths data_root
#   python -m configs.paths model_root
#   python -m configs.paths qwen3_vl_4b
#   ...
# ============================================================
_CLI_KEYS = {
    "data_root": lambda: DATA_ROOT,
    "model_root": lambda: MODEL_ROOT,
    "dataset_root": lambda: DATASET_ROOT,
    "output_root": lambda: OUTPUT_ROOT,
    "sft_data": lambda: SFT_DATA_PATH,
    "opd_data": lambda: OPD_DATA_PATH,
    "checkpoints": lambda: CHECKPOINTS_PATH,
    "qwen3_vl_4b": lambda: QWEN3_VL_4B_PATH,
    "qwen3_32b": lambda: QWEN3_32B_PATH,
    "grounding_dino": lambda: GROUNDING_DINO_PATH,
    "sam3": lambda: SAM3_PATH,
    "depth_anything": lambda: DEPTH_ANYTHING_PATH,
    "cube_rcnn": lambda: CUBE_RCNN_PATH,
    "perspective_fields": lambda: PERSPECTIVE_FIELDS_PATH,
}


def _cli_main(argv):
    if len(argv) != 2:
        sys.stderr.write(
            "用法: python -m configs.paths <key>\n"
            "可用 key: " + ", ".join(sorted(_CLI_KEYS.keys())) + "\n"
        )
        sys.exit(2)
    key = argv[1]
    if key not in _CLI_KEYS:
        sys.stderr.write(
            f"未知 key: {key}\n"
            "可用 key: " + ", ".join(sorted(_CLI_KEYS.keys())) + "\n"
        )
        sys.exit(2)
    sys.stdout.write(str(_CLI_KEYS[key]()) + "\n")


if __name__ == "__main__":
    _cli_main(sys.argv)
