"""Student 串行 latent forward + KV cache 管理。

实现 Coconut/VaLR 式串行 latent 生成：
- 每步取 last hidden state 直接当下一步的 input embedding（不走 lm_head 解码为 text token）
- KV cache 保证效率：每步只 forward 1 个位置
- 训练时：串行 K 步得到 h_1~h_K，每个对应一段 GT 文字
- 推理时：串行直到 exit_head 判断退出
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# Qwen3-VL-4B-Instruct 原生 token ID
THINK_START_ID = 151667  # <think>
THINK_END_ID = 151668    # </think>


class LatentExitHead(nn.Module):
    """独立的退出判断头：判断当前 latent hidden 是否应该退出潜空间。

    使用独立的线性层（不共享 lm_head），避免退出信号和语言建模目标互相干扰。
    输出 1 维 logit：>0 表示应该退出（输出 </think>），<0 表示继续。

    这解决了 lm_head 同时被 loss_think_end（鼓励输出 </think>）和
    loss_aux（压低 </think>）驱动导致的矛盾问题。
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        # 初始化 bias 为负值，让模型初始倾向于        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, -2.0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, H] latent hidden state

        Returns:
            exit_logit: [B, 1] 退出 logit（>0 退出，<0 继续）
        """
        return self.linear(hidden)

    def should_exit(self, hidden: torch.Tensor) -> bool:
        """推理时判断是否退出（单样本）。

        Args:
            hidden: [1, H]

        Returns:
            True if should exit
        """
        logit = self.forward(hidden)  # [1, 1]
        return logit.item() > 0.0


class LatentForwardEngine:
    """管理 Student 模型的串行 latent forward。

    核心流程（训练时）：
      Phase 1: forward [system, user, video, <think>] → 得到 KV cache + h_0
      Phase 2: 串行 K 步，每步 input_embed = prev_hidden → forward → new_hidden
      Phase 3: forward [</think>, <answer>, ...] 拼接 Phase 2 的 KV cache

    核心流程（推理时）：
      Phase 1: 同上
      Phase 2: 循环直到 exit_head 判断退出
      Phase 3: lm_head 自回归输出 answer
    """

    def __init__(self, student_model: nn.Module, exit_head: Optional[LatentExitHead] = None):
        """
        Args:
            student_model: Qwen3VLForConditionalGeneration 完整模型
            exit_head: 独立的退出判断头（可选，不提供时退回 lm_head 判断）
        """
        self.model = student_model
        # 获取模型内部的 transformer 主干和相关组件
        if hasattr(student_model, 'model'):
            self.backbone = student_model.model  # Qwen3VLModel
        else:
            raise AttributeError("Student 模型缺少 .model 属性")

        if hasattr(student_model, 'lm_head'):
            self.lm_head = student_model.lm_head
        else:
            raise AttributeError("Student 模型缺少 .lm_head 属性")

        # embed_tokens: 通过顶层模型的 get_input_embeddings() 获取
        self.embed_tokens = student_model.get_input_embeddings()

        # 独立的退出判断头
        self.exit_head = exit_head

    def _is_gradient_checkpointing_enabled(self) -> bool:
        """检测 gradient checkpointing 是否开启。

        transformers 不同版本存储 GC 状态的方式不同：
        - 新版本：model.is_gradient_checkpointing 属性
        - 旧版本：model.gradient_checkpointing 属性
        - 某些版本：通过 _gradient_checkpointing_func 是否存在判断
        需要多种方式检测，确保兼容。
        """
        # 方式 1：检查顶层模型的 is_gradient_checkpointing 属性
        if hasattr(self.model, 'is_gradient_checkpointing'):
            return self.model.is_gradient_checkpointing
        # 方式 2：检查 backbone 的 gradient_checkpointing 属性
        if getattr(self.backbone, 'gradient_checkpointing', False):
            return True
        # 方式 3：检查 backbone 的 _gradient_checkpointing_func
        if getattr(self.backbone, '_gradient_checkpointing_func', None) is not None:
            return True
        # 方式 4：检查顶层模型的 gradient_checkpointing 属性
        if getattr(self.model, 'gradient_checkpointing', False):
            return True
        return False

    def phase1_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, "DynamicCache"]:
        """Phase 1: forward 到 <think> 位置，返回 h_0 和 KV cache。

        Args:
            input_ids: [B, L] 包含 [system, user, video, <think>] 的 token ids
            attention_mask: [B, L]
            pixel_values_videos: 视频像素值
            video_grid_thw: 视频网格信息
            mm_token_type_ids: [B, L] 多模态 token 类型 ID（用于 M-RoPE）

        Returns:
            h_0: [B, H] <think> 位置的 last hidden state（进入 latent mode 的种子）
            past_key_values: KV cache
        """
        fwd_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": True,
            "return_dict": True,
            "output_hidden_states": False,
        }
        if pixel_values_videos is not None:
            fwd_kwargs["pixel_values_videos"] = pixel_values_videos
        if video_grid_thw is not None:
            fwd_kwargs["video_grid_thw"] = video_grid_thw
        if mm_token_type_ids is not None:
            fwd_kwargs["mm_token_type_ids"] = mm_token_type_ids

        outputs = self.model(**fwd_kwargs)

        past_key_values = outputs.past_key_values

        backbone_out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            return_dict=True,
        )
        last_hidden = backbone_out.last_hidden_state
        h_0 = last_hidden[:, -1, :]  # [B, H]

        return h_0, past_key_values

    def phase1_forward_efficient(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, object]:
        """Phase 1: 关闭 GC，一次 forward 同时拿到 h_0 和 KV cache（都有梯度）。

        关闭 gradient checkpointing + use_cache=True，一次 forward 同时产出：
        - h_0：<think> 位置的 last hidden state（有梯度）
        - KV cache：完整的 key/value cache（有梯度）

        梯度链（完整）：
          L_trans → Translator → h_i → latent_step attention → KV cache → backbone 全部参数
          L_ans  → Phase 3 attention → KV cache → backbone 全部参数

        显存控制：通过 --video_max_pixels 限制视频分辨率，从而控制序列长度。
        部署方式：Student 独占一张卡（96GB），序列长度控制在合理范围内不会 OOM。

        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            pixel_values_videos: 视频像素值
            video_grid_thw: 视频网格信息
            mm_token_type_ids: [B, L] 多模态 token 类型 ID

        Returns:
            h_0: [B, H] last hidden state（有梯度）
            past_key_values: KV cache（有梯度）
        """
        gc_was_enabled = self._is_gradient_checkpointing_enabled()
        if gc_was_enabled:
            self.model.gradient_checkpointing_disable()

        fwd_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": True,
            "return_dict": True,
        }
        if pixel_values_videos is not None:
            fwd_kwargs["pixel_values_videos"] = pixel_values_videos
        if video_grid_thw is not None:
            fwd_kwargs["video_grid_thw"] = video_grid_thw
        if mm_token_type_ids is not None:
            fwd_kwargs["mm_token_type_ids"] = mm_token_type_ids

        backbone_out = self.backbone(**fwd_kwargs)
        last_hidden = backbone_out.last_hidden_state  # [B, L, H]
        h_0 = last_hidden[:, -1, :]  # [B, H]，有梯度
        past_key_values = backbone_out.past_key_values  # 有梯度！

        if gc_was_enabled:
            self.model.gradient_checkpointing_enable()

        assert past_key_values is not None, \
            "Phase 1 的 past_key_values 为 None！"

        return h_0, past_key_values

    def latent_step(
        self,
        prev_hidden: torch.Tensor,
        past_key_values: object,
    ) -> Tuple[torch.Tensor, object]:
        """串行 latent forward 一步：prev_hidden 直接当 input embedding。

        注意：不传 attention_mask，让模型根据 KV cache 长度自动推断 position_ids
        和 causal mask。这避免了视觉 token 展开导致的长度不匹配问题。

        Args:
            prev_hidden: [B, H] 上一步的 hidden state
            past_key_values: 当前 KV cache

        Returns:
            new_hidden: [B, H] 本步的 hidden state
            new_past_key_values: 更新后的 KV cache
        """
        # prev_hidden 直接当 input embedding，shape: [B, 1, H]
        inputs_embeds = prev_hidden.unsqueeze(1)

        backbone_out = self.backbone(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        # new_hidden = 本步输出的 hidden state
        new_hidden = backbone_out.last_hidden_state[:, -1, :]  # [B, H]
        new_past_key_values = backbone_out.past_key_values

        return new_hidden, new_past_key_values

    def phase2_serial_forward(
        self,
        h_0: torch.Tensor,
        past_key_values: object,
        num_latent_steps: int,
    ) -> Tuple[List[torch.Tensor], object]:
        """Phase 2: 串行 K 步 latent forward。

        不传 attention_mask，让模型根据 KV cache 自动推断。

        # 临时关闭 gradient checkpointing，确保 KV cache 的梯度链完整。
        # 这样 Phase 3 的 L_ans 梯度能通过 attention → KV cache → latent hiddens 回传。
        # Phase 2 每步只 forward 1 个 token，显存开销极小，不需要 gradient checkpointing。

        Args:
            h_0: [B, H] Phase 1 输出的种子 hidden
            past_key_values: Phase 1 的 KV cache
            num_latent_steps: K，要生成的 latent 数量

        Returns:
            latent_hiddens: [h_1, h_2, ..., h_K]，每个 shape [B, H]
            past_key_values: 更新后的 KV cache（包含所有 latent 步的 KV）
        """
        # 临时关闭 gradient checkpointing，确保 KV cache 梯度链完整
        gc_was_enabled = self._is_gradient_checkpointing_enabled()
        if gc_was_enabled:
            self.model.gradient_checkpointing_disable()
        latent_hiddens: List[torch.Tensor] = []
        prev_hidden = h_0

        for step in range(num_latent_steps):
            new_hidden, past_key_values = self.latent_step(
                prev_hidden=prev_hidden,
                past_key_values=past_key_values,
            )

            latent_hiddens.append(new_hidden)
            prev_hidden = new_hidden

        # 恢复 gradient checkpointing（Phase 1 下次调用时仍需要）
        if gc_was_enabled:
            self.model.gradient_checkpointing_enable()

        return latent_hiddens, past_key_values

    def phase3_forward(
        self,
        answer_ids: torch.LongTensor,
        past_key_values: object,
    ) -> torch.Tensor:
        """Phase 3: 回到 text mode，forward answer 部分，返回 logits。

        不传 attention_mask，让模型根据 KV cache 自动推断 position 和 causal mask。

        重要：临时关闭 gradient checkpointing，确保 L_ans 梯度能完整回传到 KV cache。
        answer 部分通常只有几十个 token，显存开销极小。

        Args:
            answer_ids: [B, L_ans] 包含 [</think>, <answer>, ..., </answer>, <|im_end|>]
            past_key_values: Phase 2 结束后的 KV cache

        Returns:
            logits: [B, L_ans, vocab_size] answer 部分的 logits
        """
        # 临时关闭 gradient checkpointing，确保梯度能回传到 KV cache 中的 latent hiddens
        gc_was_enabled = self._is_gradient_checkpointing_enabled()
        if gc_was_enabled:
            self.model.gradient_checkpointing_disable()

        # 获取 answer token 的 embeddings
        answer_embeds = self.embed_tokens(answer_ids)

        backbone_out = self.backbone(
            inputs_embeds=answer_embeds,
            past_key_values=past_key_values,
            use_cache=False,
            return_dict=True,
        )

        hidden_states = backbone_out.last_hidden_state  # [B, L_ans, H]
        logits = self.lm_head(hidden_states)  # [B, L_ans, vocab_size]

        # 恢复 gradient checkpointing
        if gc_was_enabled:
            self.model.gradient_checkpointing_enable()

        return logits

    @torch.no_grad()
    def inference_loop(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        max_latent_steps: int = 32,
        max_new_tokens: int = 512,
    ) -> torch.LongTensor:
        """推理时完整流程：latent mode → 检测退出 → 自回归输出 answer。

        Args:
            input_ids: [1, L] 包含 [system, user, video, <think>]
            attention_mask: [1, L]
            pixel_values_videos: 视频像素值
            video_grid_thw: 视频网格信息
            max_latent_steps: latent 步数上限
            max_new_tokens: answer 部分最大 token 数

        Returns:
            output_ids: [1, N] 生成的 token ids（从 </think> 开始）
        """
        device = input_ids.device

        # Phase 1
        h_0, past_key_values = self.phase1_forward_efficient(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

        # Phase 2: latent mode，直到 exit_head 判断退出
        prev_hidden = h_0
        for step in range(max_latent_steps):
            new_hidden, past_key_values = self.latent_step(
                prev_hidden=prev_hidden,
                past_key_values=past_key_values,
            )

            # 检查是否应该退出 latent mode
            if self.exit_head is not None:
                if self.exit_head.should_exit(new_hidden):
                    break
            else:
                # 兼容旧模式：用 lm_head 判断
                logits = self.lm_head(new_hidden)
                next_token_id = logits.argmax(dim=-1).item()
                if next_token_id == THINK_END_ID:
                    break

            prev_hidden = new_hidden

        # Phase 3: 自回归输出 answer
        # 先把 </think> token 喂入
        generated_ids = [THINK_END_ID]
        current_id = torch.tensor([[THINK_END_ID]], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            token_embed = self.embed_tokens(current_id)
            backbone_out = self.backbone(
                inputs_embeds=token_embed,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = backbone_out.past_key_values
            hidden = backbone_out.last_hidden_state[:, -1, :]
            logits = self.lm_head(hidden)
            next_id = logits.argmax(dim=-1).item()
            generated_ids.append(next_id)
            current_id = torch.tensor([[next_id]], dtype=torch.long, device=device)

            # 遇到 EOS 或 <|im_end|> 停止
            if next_id in (151645, 151643):  # <|im_end|> 或 <|endoftext|>
                break

        return torch.tensor([generated_ids], dtype=torch.long, device=device)
