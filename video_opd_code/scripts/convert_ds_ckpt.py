"""将 DeepSpeed ZeRO-2 checkpoint 转换为推理格式。

DeepSpeed ZeRO-2 保存的 checkpoint 结构：
  checkpoint-final/
    mp_rank_00_model_states.pt  (包含完整模型参数)
    bf16_zero_pp_rank_*_optim_states.pt  (优化器状态，推理不需要)

转换后的结构：
  checkpoint-final/
    student/   (HuggingFace 格式，可直接 from_pretrained)
    translator/
      translator_state_dict.pt

用法：
  python scripts/convert_ds_ckpt.py \
    --ds_ckpt_dir /path/to/checkpoint-final \
    --model_path /path/to/Qwen3-VL-4B-Instruct \
    --output_dir /path/to/checkpoint-final  (默认原地输出)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))


def parse_args():
    p = argparse.ArgumentParser("DeepSpeed checkpoint → 推理格式转换")
    p.add_argument("--ds_ckpt_dir", required=True,
                   help="DeepSpeed checkpoint 目录（含 mp_rank_00_model_states.pt）")
    p.add_argument("--model_path", default=None,
                   help="基础模型路径（用于加载 config/tokenizer）。不指定则自动从 configs.paths 获取")
    p.add_argument("--output_dir", default=None,
                   help="输出目录（默认与 ds_ckpt_dir 相同）")
    return p.parse_args()


def main():
    args = parse_args()

    ds_ckpt_dir = Path(args.ds_ckpt_dir)
    output_dir = Path(args.output_dir) if args.output_dir else ds_ckpt_dir

    # 确定基础模型路径
    if args.model_path:
        model_path = args.model_path
    else:
        from configs import paths as cfg_paths
        model_path = str(cfg_paths.QWEN3_VL_4B_PATH)

    # 找到模型状态文件
    model_state_file = ds_ckpt_dir / "mp_rank_00_model_states.pt"
    if not model_state_file.exists():
        raise FileNotFoundError(f"找不到模型状态文件: {model_state_file}")

    print(f"[1/4] 加载 DeepSpeed 模型状态: {model_state_file}")
    print(f"       文件大小: {model_state_file.stat().st_size / 1024**3:.2f} GB")

    # 加载模型状态
    state = torch.load(str(model_state_file), map_location="cpu")

    # DeepSpeed 保存格式: state["module"] 包含模型参数
    if "module" in state:
        full_state_dict = state["module"]
    else:
        # 有些版本直接是 state dict
        full_state_dict = state

    print(f"       总参数数量: {len(full_state_dict)}")

    # 分离 Student 和 Translator 的参数
    # CombinedModel 结构: module.student.xxx 和 module.translator.xxx
    # 加载后 key 可能是 student.xxx 或 module.student.xxx
    student_state_dict = {}
    translator_state_dict = {}
    unknown_keys = []

    for key, value in tqdm(full_state_dict.items(), desc="[2/4] 分离参数"):
        if key.startswith("student."):
            # 去掉 "student." 前缀
            new_key = key[len("student."):]
            student_state_dict[new_key] = value
        elif key.startswith("translator."):
            # 去掉 "translator." 前缀
            new_key = key[len("translator."):]
            translator_state_dict[new_key] = value
        elif key.startswith("module.student."):
            new_key = key[len("module.student."):]
            student_state_dict[new_key] = value
        elif key.startswith("module.translator."):
            new_key = key[len("module.translator."):]
            translator_state_dict[new_key] = value
        else:
            unknown_keys.append(key)

    print(f"       Student 参数: {len(student_state_dict)}")
    print(f"       Translator 参数: {len(translator_state_dict)}")
    if unknown_keys:
        print(f"       ⚠️  未识别的 key: {len(unknown_keys)}")
        for k in unknown_keys[:5]:
            print(f"          {k}")

    # 释放原始 state dict 节省内存
    del full_state_dict, state
    import gc
    gc.collect()

    # =========================================================================
    # 保存 Student（HuggingFace 格式）
    # =========================================================================
    print(f"\n[3/4] 保存 Student 模型到 HuggingFace 格式...")
    student_dir = output_dir / "student"
    student_dir.mkdir(parents=True, exist_ok=True)

    # 加载原始模型结构（不加载权重，只要 config）
    from transformers import AutoProcessor, AutoConfig
    try:
        from transformers import Qwen3VLForConditionalGeneration
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # 用 config 初始化空模型，然后加载权重
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    except Exception:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

    # 加载训练后的权重
    missing, unexpected = model.load_state_dict(student_state_dict, strict=False)
    if missing:
        print(f"       ⚠️  Student 缺失 key: {len(missing)}")
        for k in missing[:5]:
            print(f"          {k}")
    if unexpected:
        print(f"       ⚠️  Student 多余 key: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"          {k}")

    # 保存为 safetensors 格式
    model.save_pretrained(str(student_dir), safe_serialization=True)
    print(f"       ✅ Student 模型已保存到: {student_dir}")

    # 保存 processor/tokenizer
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.save_pretrained(str(student_dir))
    print(f"       ✅ Processor 已保存到: {student_dir}")

    # 释放 Student 内存
    del model, student_state_dict
    gc.collect()

    # =========================================================================
    # 保存 Translator（state_dict 格式）
    # =========================================================================
    print(f"\n[4/4] 保存 Translator state_dict...")
    translator_dir = output_dir / "translator"
    translator_dir.mkdir(parents=True, exist_ok=True)

    translator_path = translator_dir / "translator_state_dict.pt"
    torch.save(translator_state_dict, str(translator_path))
    print(f"       ✅ Translator state_dict 已保存到: {translator_path}")
    print(f"       文件大小: {translator_path.stat().st_size / 1024**3:.2f} GB")

    print(f"\n{'='*60}")
    print(f"转换完成！输出目录: {output_dir}")
    print(f"  student/     → HuggingFace 格式（可直接 from_pretrained）")
    print(f"  translator/  → translator_state_dict.pt")
    print(f"{'='*60}")
    print(f"\n推理命令：")
    print(f"  python scripts/infer_stage1_sft_v5.py \\")
    print(f"    --checkpoint_dir {output_dir} \\")
    print(f"    --video_path <视频路径> \\")
    print(f"    --question \"你的问题\"")


if __name__ == "__main__":
    main()
