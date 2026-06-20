"""Stage1-SFT v5 推理脚本：从 checkpoint 加载 Student + Translator，输出两种结果。

输出模式：
  1. 不翻译：只展示 Student 的 latent 步数 + 最终 answer（推理时 Translator 不参与）
  2. 翻译：用 Translator 把每个 latent hidden 还原为文字，展示完整思维链

用法：
  python scripts/infer_stage1_sft_v5.py \
    --checkpoint_dir <ckpt_path> \
    --video_path <video> \
    --question "What is the dog doing?"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from configs import paths as cfg_paths


def parse_args():
    p = argparse.ArgumentParser("Stage1-SFT v5 推理")
    p.add_argument("--checkpoint_dir", required=True, help="checkpoint 目录（含 student/ 和 translator/ 子目录）")
    p.add_argument("--video_path", required=True, help="视频文件路径")
    p.add_argument("--question", required=True, help="问题文本")
    p.add_argument("--model_path", default=str(cfg_paths.QWEN3_VL_4B_PATH), help="基础模型路径（用于 Translator 结构）")
    p.add_argument("--max_latent_steps", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--video_max_pixels", type=int, default=0,
                   help="视频 3D 体积上限(T×H×W)，0=使用默认值")
    p.add_argument("--show_translated", action="store_true", default=True, help="是否展示 Translator 翻译结果")
    return p.parse_args()


def load_student(ckpt_dir: str, dtype: torch.dtype):
    """从 checkpoint 加载 Student 模型。"""
    from utils.qwen3vl_patch import apply_qwen3vl_patches
    apply_qwen3vl_patches()
    from transformers import AutoProcessor

    student_dir = os.path.join(ckpt_dir, "student")
    if not os.path.exists(student_dir):
        # fallback: 如果 checkpoint_dir 本身就有 config.json，直接用它
        if os.path.exists(os.path.join(ckpt_dir, "config.json")):
            student_dir = ckpt_dir
        else:
            raise FileNotFoundError(
                f"Student checkpoint 不存在: {student_dir}\n"
                f"提示: checkpoint 可能被 save_total_limit 清理了。\n"
                f"请用含 student/ 子目录的 checkpoint，或重新训练。"
            )

    processor = AutoProcessor.from_pretrained(student_dir, trust_remote_code=True)
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            student_dir, torch_dtype=dtype, trust_remote_code=True)
    except Exception:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            student_dir, torch_dtype=dtype, trust_remote_code=True)

    model.eval()
    return processor, model


def load_translator(ckpt_dir: str, model_path: str, dtype: torch.dtype):
    """从 checkpoint 加载 Translator（Translator 参与训练，必须加载 checkpoint）。"""
    from training.translator_v5 import Translator

    translator_dir = os.path.join(ckpt_dir, "translator")
    state_dict_path = os.path.join(translator_dir, "translator_state_dict.pt")

    translator = Translator(model_path, dtype=dtype)

    if os.path.exists(state_dict_path):
        state_dict = torch.load(state_dict_path, map_location="cpu")
        translator.load_state_dict(state_dict)
        print(f"  Translator 从 checkpoint 加载: {state_dict_path}")
    else:
        print(f"  [警告] Translator checkpoint 不存在，使用原始预训练权重（翻译效果可能不佳）")

    translator.eval()
    return translator


def translate_hidden(translator, hidden: torch.Tensor, tokenizer, max_len: int = 256,
                     prefix_hidden=None) -> str:
    """用 Translator 自回归解码一个 latent hidden 为文字。

    Translator 已解冻参与训练，接收 Student 的 raw hidden state + 可选的 prefix 上下文。
    利用 KV cache 做高效自回归生成。

    Args:
        translator: Translator 模型（包装了完整 Qwen3-VL 模型，删除了视觉塔）
        hidden: [1, H] latent hidden state
        tokenizer: tokenizer
        max_len: 最大生成长度
        prefix_hidden: [1, L_prefix, H] 可选的前缀上下文（来自 Student Phase 1）

    Returns:
        decoded_text: 还原的文字
    """
    device = hidden.device
    dtype = hidden.dtype

    embed_tokens = translator._get_embed_tokens()

    # 第一步：[prefix_hidden (可选)] + latent hidden 当 input embedding
    if prefix_hidden is not None:
        # 拼接 prefix_hidden + latent_hidden
        input_embeds = torch.cat([prefix_hidden, hidden.unsqueeze(1)], dim=1)  # [1, L_prefix+1, H]
        seq_len = input_embeds.shape[1]
    else:
        input_embeds = hidden.unsqueeze(1)  # [1, 1, H]
        seq_len = 1
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

    outputs = translator.model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]  # [1, vocab_size]
    next_id = logits.argmax(dim=-1).item()

    generated_ids = []

    # EOS token ids: <|im_end|>=151645, <|endoftext|>=151643, pad=0
    eos_ids = {151645, 151643, 0}

    if next_id in eos_ids:
        return ""

    generated_ids.append(next_id)

    # 后续步骤：用 KV cache 逐 token 自回归
    for step in range(max_len - 1):
        # 新 token 的 embedding
        new_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        new_embed = embed_tokens(new_token)  # [1, 1, H]

        # 扩展 attention mask
        attention_mask = torch.cat([
            attention_mask,
            torch.ones(1, 1, dtype=torch.long, device=device)
        ], dim=1)

        # Forward with KV cache（只传新 token 的 embedding）
        outputs = translator.model(
            inputs_embeds=new_embed,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]  # [1, vocab_size]
        next_id = logits.argmax(dim=-1).item()

        if next_id in eos_ids:
            break

        generated_ids.append(next_id)

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print("=" * 60)
    print("Stage1-SFT v5 推理")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Video: {args.video_path}")
    print(f"  Question: {args.question}")
    print()

    # 加载模型
    print("[1/3] 加载 Student 模型...")
    processor, student_model = load_student(args.checkpoint_dir, dtype)
    student_model = student_model.to(device)
    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = args.fps
        processor.video_processor.max_frames = args.max_frames
        # 通过 video_max_pixels 控制视频 token 预算
        if args.video_max_pixels > 0:
            processor.video_processor.size = {
                "shortest_edge": min(args.video_max_pixels, processor.video_processor.size.get("shortest_edge", 131072)),
                "longest_edge": args.video_max_pixels,
            }
        # 强制 num_frames=None，避免和 fps 冲突（参考 demo_qwen3vl_video.py）
        if hasattr(processor.video_processor, "num_frames"):
            try:
                processor.video_processor.num_frames = None
            except Exception:
                pass

    print("[2/3] 加载 Translator 模型...")
    translator = load_translator(args.checkpoint_dir, args.model_path, dtype)
    translator = translator.to(device)

    # 加载 exit_head
    from training.latent_forward import LatentForwardEngine, LatentExitHead, THINK_END_ID
    exit_head = None
    exit_head_path = os.path.join(args.checkpoint_dir, "exit_head.pt")
    if os.path.exists(exit_head_path):
        # 从 student model config 获取 hidden_size（Qwen3VL 的 hidden_size 在 text_config 中）
        hidden_size = student_model.config.text_config.hidden_size
        exit_head = LatentExitHead(hidden_size)
        exit_head.load_state_dict(torch.load(exit_head_path, map_location="cpu"))
        exit_head = exit_head.to(device=device, dtype=dtype).eval()
        print(f"  exit_head 已加载: {exit_head_path}")
    else:
        print(f"  [警告] exit_head 不存在，将使用 lm_head 判断退出（旧模式）")

    print("[3/3] 构造输入...")
    from training.latent_sft_helpers import load_system_prompt
    sys_prompt = load_system_prompt("student")

    msgs = []
    if sys_prompt:
        msgs.append({"role": "system", "content": [{"type": "text", "text": sys_prompt}]})
    msgs.append({"role": "user", "content": [
        {"type": "video", "video": args.video_path},
        {"type": "text", "text": args.question},
    ]})
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": "<think>"}]})

    inputs = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device) if "attention_mask" in inputs else torch.ones_like(input_ids)
    pixel_values_videos = inputs.get("pixel_values_videos")
    video_grid_thw = inputs.get("video_grid_thw")
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.to(device=device, dtype=dtype)
    if video_grid_thw is not None:
        video_grid_thw = video_grid_thw.to(device)
    if mm_token_type_ids is not None:
        mm_token_type_ids = mm_token_type_ids.to(device)

    # 推理
    print("\n[推理] 开始计时...")
    engine = LatentForwardEngine(student_model, exit_head=exit_head)
    # CUDA 同步确保计时准确
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_total_start = time.time()

    with torch.no_grad():
        # ===================== Phase 1: 视觉编码 + prefix forward =====================
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_phase1_start = time.time()

        h_0, past_key_values = engine.phase1_forward_efficient(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_phase1_end = time.time()
        phase1_ms = (t_phase1_end - t_phase1_start) * 1000
        print(f"  [Phase 1] 视觉编码 + prefix forward: {phase1_ms:.1f} ms")

        # ===================== Phase 2: 潜空间推理（串行 latent） =====================
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_phase2_start = time.time()
        latent_step_times_ms = []  # 记录每步耗时

        latent_hiddens = []
        prev_hidden = h_0
        for step in range(args.max_latent_steps):
            t_step_start = time.time()

            new_hidden, past_key_values = engine.latent_step(
                prev_hidden=prev_hidden,
                past_key_values=past_key_values,
            )

            latent_hiddens.append(new_hidden)

            # 检查是否退出 latent mode（优先使用 exit_head）
            should_exit = False
            if exit_head is not None:
                exit_logit = exit_head(new_hidden)
                should_exit = exit_logit.item() > 0.0
            else:
                logits = engine.lm_head(new_hidden)
                next_token_id = logits.argmax(dim=-1).item()
                should_exit = (next_token_id == THINK_END_ID)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_step_end = time.time()
            latent_step_times_ms.append((t_step_end - t_step_start) * 1000)

            if should_exit:
                break

            prev_hidden = new_hidden

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_phase2_end = time.time()
        phase2_ms = (t_phase2_end - t_phase2_start) * 1000

        K = len(latent_hiddens)
        avg_step_ms = sum(latent_step_times_ms) / max(1, len(latent_step_times_ms))
        print(f"  [Phase 2] 潜空间推理: {phase2_ms:.1f} ms | {K} 步 | 平均 {avg_step_ms:.1f} ms/步")

        # ===================== Phase 3: 自回归输出 answer =====================
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_phase3_start = time.time()

        generated_ids = [THINK_END_ID]  # </think>
        current_id = torch.tensor([[THINK_END_ID]], dtype=torch.long, device=device)

        for _ in range(args.max_new_tokens):
            token_embed = engine.embed_tokens(current_id)
            backbone_out = engine.backbone(
                inputs_embeds=token_embed,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = backbone_out.past_key_values
            hidden = backbone_out.last_hidden_state[:, -1, :]
            logits = engine.lm_head(hidden)
            next_id = logits.argmax(dim=-1).item()
            generated_ids.append(next_id)
            current_id = torch.tensor([[next_id]], dtype=torch.long, device=device)

            if next_id in (151645, 151643):
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_phase3_end = time.time()
        phase3_ms = (t_phase3_end - t_phase3_start) * 1000
        num_answer_tokens = len(generated_ids) - 1  # 减去初始的 </think>
        avg_answer_ms = phase3_ms / max(1, num_answer_tokens)
        print(f"  [Phase 3] Answer 自回归: {phase3_ms:.1f} ms | {num_answer_tokens} tokens | 平均 {avg_answer_ms:.1f} ms/token")

    t_total_end = time.time()
    total_ms = (t_total_end - t_total_start) * 1000
    answer_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)

    # =========================================================================
    # 输出结果 1：不翻译（纯 latent + answer）
    # =========================================================================
    print("\n" + "=" * 60)
    print("【结果 1】不翻译（推理时 Translator 不参与）")
    print("=" * 60)
    print(f"  输出: {answer_text}")
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │              ⏱️  推理耗时分解                        │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  Phase 1 (视觉+prefix):  {phase1_ms:>8.1f} ms               │")
    print(f"  │  Phase 2 (潜空间推理):   {phase2_ms:>8.1f} ms  ← {K} 步     │")
    print(f"  │  Phase 3 (Answer生成):   {phase3_ms:>8.1f} ms  ← {num_answer_tokens} tokens │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  总耗时:                 {total_ms:>8.1f} ms               │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print("  📊 关键指标:")
    print(f"     潜空间每步耗时:  {avg_step_ms:.1f} ms/step")
    print(f"     Answer每token:   {avg_answer_ms:.1f} ms/token")
    print(f"     潜空间 vs 文本:  1 latent step ≈ {avg_step_ms/max(0.01, avg_answer_ms):.1f}x 单token")
    if K > 0:
        # 假设每段 GT 平均 50 token，潜空间用 1 步替代
        equiv_tokens_saved = K * 50  # 粗略估计
        equiv_text_time_ms = equiv_tokens_saved * avg_answer_ms
        print(f"     潜空间节省估算:  {K} 步替代 ~{equiv_tokens_saved} tokens 文本推理")
        print(f"     预估节省时间:    ~{equiv_text_time_ms:.0f} ms → 实际只用 {phase2_ms:.0f} ms")
        speedup = equiv_text_time_ms / max(0.01, phase2_ms)
        print(f"     ⚡ 潜空间加速比:  {speedup:.1f}x")

    # =========================================================================
    # 输出结果 2：翻译（用 Translator 还原每个 latent）
    # =========================================================================
    if args.show_translated:
        print("\n" + "=" * 60)
        print("【结果 2】翻译（Translator 还原潜空间特征）")
        print("=" * 60)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_translate_start = time.time()

        with torch.no_grad():
            for i, h_i in enumerate(latent_hiddens):
                t_seg_start = time.time()
                translated = translate_hidden(
                    translator, h_i, processor.tokenizer, max_len=256)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_seg_end = time.time()
                seg_ms = (t_seg_end - t_seg_start) * 1000
                print(f"  [latent_{i+1}] ({seg_ms:.0f}ms) {translated}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_translate_end = time.time()
        translate_total_ms = (t_translate_end - t_translate_start) * 1000

        print(f"\n  → </think>")
        print(f"  → {answer_text}")
        print(f"\n  📝 Translator 翻译总耗时: {translate_total_ms:.0f} ms（仅用于可视化，推理时不需要）")

    print("\n" + "=" * 60)
    print("推理完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
