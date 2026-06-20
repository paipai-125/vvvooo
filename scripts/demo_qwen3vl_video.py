"""
Qwen3-VL 视频问答 / 推理最小 Demo
==================================

目的:
    在不涉及任何训练、潜空间、collator、teacher forcing 等额外逻辑的前提下,
    用最干净、最官方的 API 跑通 Qwen3-VL 的视频理解推理。

参考:
    transformers/models/qwen3_vl/modeling_qwen3_vl.py 的 docstring 官方样例
    (Qwen3VLForConditionalGeneration.forward 的 Example 部分)
    我们仅把官方样例里的 {"type": "image", "image": URL} 改为
    {"type": "video", "video": "/path/to/video.mp4"}, 其它一字不改。

用法:
    # 单卡推理（仅 0 号卡）
    CUDA_VISIBLE_DEVICES=0 python -m scripts.demo_qwen3vl_video \
        --video /path/to/video.mp4 \
        --question "Describe this video." \
        --max_new_tokens 256

    # 不指定 --video, 默认从 charades_sta 拿一个
"""
import argparse
import os
import sys
from pathlib import Path

import torch

# 允许 `python -m scripts.demo_qwen3vl_video` 与脚本直接运行两种方式
_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from configs import paths as cfg_paths  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("Qwen3-VL Video Demo (official-style)")
    p.add_argument("--model_path", default=str(cfg_paths.QWEN3_VL_4B_PATH),
                   help="Qwen3-VL 模型路径")
    p.add_argument("--video", default=None,
                   help="本地视频路径; 不指定则自动从 charades_sta 选一个")
    p.add_argument("--question", default="Please describe what happens in this video, step by step.",
                   help="提问")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_frames", type=int, default=32,
                   help="视频最多采样多少帧（控制显存）")
    p.add_argument("--fps", type=float, default=1.0,
                   help="视频采样 fps")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return p.parse_args()


def get_default_video() -> str:
    """从 charades_sta 数据集随便挑一个视频。"""
    cand_dir = cfg_paths.CHARADES_STA_PATH / "videos"
    if not cand_dir.exists():
        raise FileNotFoundError(
            f"默认视频目录不存在: {cand_dir}\n"
            f"请用 --video /path/to/your.mp4 指定一个本地视频"
        )
    for v in sorted(cand_dir.glob("*.mp4")):
        return str(v)
    raise FileNotFoundError(f"{cand_dir} 下找不到 .mp4")


def main():
    args = parse_args()

    # ===== 0. 选 dtype / device =====
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    if not torch.cuda.is_available():
        raise RuntimeError("需要 GPU")
    device = torch.device("cuda:0")  # demo 一律走 0 号卡

    # ===== 1. 决定视频路径 =====
    video_path = args.video or get_default_video()
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频不存在: {video_path}")
    print(f"[Demo] 模型路径: {args.model_path}")
    print(f"[Demo] 视频路径: {video_path}")
    print(f"[Demo] 问题:    {args.question}")
    print(f"[Demo] dtype:   {dtype}, device: {device}")

    # ===== 2. 加载 processor 与模型（官方 API）=====
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelCls

    # ===== 关键 monkey-patch =====
    # transformers 5.10.0.dev0 的 Qwen3VLProcessor.replace_video_token 有 bug:
    # 它把 <|video_pad|> 在文本里展开为 "<|vision_start|><|placeholder|>*N<|vision_end|>",
    # 但 <|placeholder|> 不在 tokenizer 词表中, 也没有任何后续代码把它换回 <|video_pad|>;
    # 结果 tokenize 后 input_ids 里 <|video_pad|>(151656) 数量为 0, 而 video features 仍是 N,
    # 触发 "Video features and video tokens do not match, tokens: 0, features: N".
    # 详见 utils/qwen3vl_patch.py
    from utils.qwen3vl_patch import apply_qwen3vl_patches
    apply_qwen3vl_patches()
    print("[Demo] 已 monkey-patch Qwen3VLProcessor.replace_video_token "
          "(<|placeholder|> -> <|video_pad|>)")

    print("[Demo] 加载 processor ...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # 控制视频采样（避免 OOM）
    # 注意：Qwen3-VL 的 sample_frames 内部要求 num_frames 与 fps 互斥（二选一）。
    # 我们采用 fps + max_frames 的组合：
    #   - fps：每秒采几帧（默认 2，太多会 OOM；这里降到 args.fps）
    #   - max_frames：基于 fps 计算出的 num_frames 之上的硬上限
    # 绝不要设置 num_frames，否则与 fps 冲突。
    if hasattr(processor, "video_processor"):
        vp = processor.video_processor
        vp.fps = args.fps
        # 强制 num_frames=None，避免和 fps 冲突
        if hasattr(vp, "num_frames"):
            try:
                vp.num_frames = None
            except Exception:
                pass
        if hasattr(vp, "max_frames"):
            vp.max_frames = args.max_frames
        print(f"[Demo] video_processor: fps={args.fps}, max_frames={args.max_frames}, num_frames=None")

    print("[Demo] 加载模型 ...")
    model = ModelCls.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()

    # ===== 3. 构造 messages（严格按官方 docstring）=====
    # 官方样例（image 版本）:
    #   messages = [{"role": "user", "content": [
    #       {"type": "image", "image": URL},
    #       {"type": "text", "text": "Describe the image."},
    #   ]}]
    # 视频版本只把 image -> video, image_url -> 本地路径
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": args.question},
            ],
        }
    ]

    # ===== 4. apply_chat_template（一行 API，官方推荐）=====
    print("[Demo] processor.apply_chat_template(...) ...")
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # 调试: 打印关键字段形状以及 video_token 数量是否匹配 features
    video_token_id = getattr(model.config, "video_token_id", None)
    if video_token_id is None and hasattr(processor, "video_token_id"):
        video_token_id = processor.video_token_id
    print(f"[Demo] inputs keys: {list(inputs.keys())}")
    print(f"[Demo] input_ids shape: {tuple(inputs['input_ids'].shape)}")
    if "pixel_values_videos" in inputs:
        pv = inputs["pixel_values_videos"]
        print(f"[Demo] pixel_values_videos shape: {tuple(pv.shape)} dtype={pv.dtype}")
    if "video_grid_thw" in inputs:
        thw = inputs["video_grid_thw"]
        print(f"[Demo] video_grid_thw: {thw.tolist()}  (合计 features = "
              f"{int((thw[:, 0] * thw[:, 1] * thw[:, 2]).sum().item())})")
    if "mm_token_type_ids" in inputs:
        mm = inputs["mm_token_type_ids"]
        print(f"[Demo] mm_token_type_ids shape: {tuple(mm.shape)}, "
              f"video positions(=2) = {(mm == 2).sum().item()}")
    if video_token_id is not None:
        n_video_tokens = (inputs["input_ids"] == video_token_id).sum().item()
        print(f"[Demo] video_token_id={video_token_id}, "
              f"出现次数(input_ids==video_token_id) = {n_video_tokens}")

    # 把所有张量挪到 device，并把视觉 tensor 转为模型 dtype
    moved = {}
    for k, v in inputs.items():
        if not torch.is_tensor(v):
            moved[k] = v
            continue
        if v.dtype.is_floating_point:
            moved[k] = v.to(device=device, dtype=dtype)
        else:
            moved[k] = v.to(device=device)
    inputs = moved

    # ===== 5. generate（官方 API）=====
    print("[Demo] model.generate(...) ...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    # 截掉输入部分
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("\n" + "=" * 80)
    print("[Demo] Qwen3-VL 输出:")
    print("=" * 80)
    print(output_text)
    print("=" * 80)


if __name__ == "__main__":
    main()
