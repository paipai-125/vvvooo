"""Translator 模块（v5 设计）。

Translator = Qwen3-VL-4B 完整模型（删除视觉塔以省显存），解冻参与训练。
任务：接收 1 个 latent hidden → teacher-forcing 还原整段 GT 文字。

梯度链：L_trans → Translator forward → 梯度穿过 h_i → Student 参数更新。
         同时 Translator 自身参数也更新，学会从 hidden state 解码文字。
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Translator(nn.Module):
    """Translator: 从 Student latent hidden 还原 GT 段文字。

    直接使用 Qwen3-VL-4B 完整模型（删除视觉塔），解冻全参数参与训练。
    Translator 学会从 Student 的 latent hidden state 解码还原 GT 文字。

    Forward 逻辑（teacher-forcing）：
      输入 embedding 序列：
        位置 0: h_i（来自 Student，直接当 input embedding，不经过 embed 层）
        位置 1~n-1: embed(GT_segment_tokens[0:n-1])
      目标：
        位置 0 → 预测 GT_segment_tokens[0]
        位置 1 → 预测 GT_segment_tokens[1]
        ...
        位置 n-1 → 预测 GT_segment_tokens[n-1]
    """

    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16):
        """加载完整模型，删除视觉塔。

        Args:
            model_path: Qwen3-VL-4B-Instruct 模型路径
            dtype: 模型精度
        """
        super().__init__()
        self.dtype = dtype
        self._load_model(model_path, dtype)

    def _load_model(self, model_path: str, dtype: torch.dtype):
        """加载完整 Qwen3-VL 模型，删除视觉塔释放显存。"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Translator 模型路径不存在: {model_path}")

        # 加载完整模型
        try:
            from transformers import Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=dtype, trust_remote_code=True)
        except Exception:
            from transformers import AutoModelForVision2Seq
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path, torch_dtype=dtype, trust_remote_code=True)

        # 删除视觉塔释放显存（Translator 永远不接收视觉输入）
        if hasattr(self.model, 'visual'):
            del self.model.visual
            torch.cuda.empty_cache()

        # 所有参数可训练
        for param in self.model.parameters():
            param.requires_grad_(True)

    def _get_embed_tokens(self):
        """获取 embed_tokens 层（兼容不同模型结构）。"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            return self.model.model.embed_tokens
        elif hasattr(self.model, 'get_input_embeddings'):
            return self.model.get_input_embeddings()
        else:
            raise AttributeError("无法获取 embed_tokens 层")

    def _get_lm_head(self):
        """获取 lm_head 层。"""
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head
        elif hasattr(self.model, 'get_output_embeddings'):
            return self.model.get_output_embeddings()
        else:
            raise AttributeError("无法获取 lm_head 层")

    def forward(
        self,
        latent_hidden: torch.Tensor,
        gt_token_ids: torch.LongTensor,
        gt_attention_mask: Optional[torch.LongTensor] = None,
        prefix_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forcing forward，返回 CE loss。

        Args:
            latent_hidden: [B, H] 来自 Student 的 latent hidden state
            gt_token_ids: [B, N] GT 段文字的 token ids（完整段，包含要预测的所有 token）
            gt_attention_mask: [B, N] 可选的 attention mask
            prefix_hidden: [B, L_prefix, H] 可选的前缀上下文（来自 Student Phase 1 的
                           last_hidden_state，有梯度）。提供时 Translator 能 attend 到
                           视频/问题的完整编码，L_trans 梯度能回传到 Student 全部参数。

        Returns:
            loss: scalar，CE loss（梯度穿过 latent_hidden 和 prefix_hidden 回传到 Student）
        """
        B, N = gt_token_ids.shape

        embed_tokens = self._get_embed_tokens()

        # 构造 input embeddings:
        #   [prefix_hidden (可选)] + [latent_hidden] + [embed(gt_token_ids[0:N-1])]
        gt_embeds = embed_tokens(gt_token_ids[:, :-1])  # [B, N-1, H]
        latent_embed = latent_hidden.unsqueeze(1)  # [B, 1, H]

        if prefix_hidden is not None:
            # prefix_hidden: [B, L_prefix, H]（有梯度，来自 Student backbone）
            # 拼接顺序：[prefix_hidden, latent_hidden, gt_embeds]
            L_prefix = prefix_hidden.shape[1]
            input_embeds = torch.cat([prefix_hidden, latent_embed, gt_embeds], dim=1)  # [B, L_prefix+N, H]
            total_len = L_prefix + N

            # attention mask：prefix 部分全 1 + 原始部分
            prefix_mask = torch.ones(B, L_prefix, dtype=torch.long, device=input_embeds.device)
            if gt_attention_mask is None:
                gen_mask = torch.ones(B, N, dtype=torch.long, device=input_embeds.device)
            else:
                gen_mask = gt_attention_mask
            attention_mask = torch.cat([prefix_mask, gen_mask], dim=1)  # [B, L_prefix+N]
        else:
            # 无 prefix：原始行为
            input_embeds = torch.cat([latent_embed, gt_embeds], dim=1)  # [B, N, H]
            total_len = N
            L_prefix = 0
            if gt_attention_mask is None:
                attention_mask = torch.ones(B, N, dtype=torch.long, device=input_embeds.device)
            else:
                attention_mask = gt_attention_mask

        # 用模型自身的 forward（传 inputs_embeds，不传 input_ids）
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        # 获取 logits
        logits = outputs.logits  # [B, total_len, vocab_size]

        # 只取 prefix 之后的部分计算 loss（prefix 部分不需要预测）
        if L_prefix > 0:
            logits = logits[:, L_prefix:, :]  # [B, N, vocab_size]

        # 目标: 位置 i 预测 gt_token_ids[i]
        targets = gt_token_ids  # [B, N]

        # 计算 CE loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

        return loss

    def forward_batch(
        self,
        latent_hiddens: list,
        gt_segments: list,
        tokenizer,
        max_seg_len: int = 256,
        prefix_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """批量处理多个 latent hidden 和对应的 GT 段。

        Args:
            latent_hiddens: [h_1, h_2, ..., h_K]，每个 shape [B, H]（这里 B=1 for simplicity）
            gt_segments: [seg_1_ids, seg_2_ids, ..., seg_K_ids]，每个是 token id list
            tokenizer: 未使用，保留接口兼容
            max_seg_len: 单段最大 token 数
            prefix_hidden: [B, L_prefix, H] 可选的前缀上下文（有梯度）

        Returns:
            avg_loss: 所有段的平均 CE loss
        """
        if not latent_hiddens or not gt_segments:
            return torch.tensor(0.0, requires_grad=True)

        assert len(latent_hiddens) == len(gt_segments), \
            f"latent 数量 ({len(latent_hiddens)}) != GT 段数量 ({len(gt_segments)})"

        device = latent_hiddens[0].device
        K = len(latent_hiddens)
        total_loss = torch.tensor(0.0, device=device)

        for i in range(K):
            h_i = latent_hiddens[i]  # [B, H] 或 [H]
            if h_i.dim() == 1:
                h_i = h_i.unsqueeze(0)  # [1, H]

            seg_ids = gt_segments[i]  # list of int 或 tensor
            if isinstance(seg_ids, list):
                seg_ids = torch.tensor(seg_ids, dtype=torch.long, device=device)
            if seg_ids.dim() == 1:
                seg_ids = seg_ids.unsqueeze(0)  # [1, N]

            # 截断到 max_seg_len
            if seg_ids.shape[1] > max_seg_len:
                raise RuntimeError(
                    f"段 {i} 的 token 数 ({seg_ids.shape[1]}) 超过上限 ({max_seg_len})，"
                    f"请检查数据！")

            loss_i = self.forward(h_i, seg_ids)
            total_loss = total_loss + loss_i

        avg_loss = total_loss / K
        return avg_loss
