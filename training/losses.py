"""Loss 计算模块（v5.2 设计）。

五个 loss：
  L_trans:     Translator 还原 GT 段文字的 CE loss（梯度穿过 h_i → Student）
  L_ans:       Phase 3 中 answer 部分的标准 next-token CE loss
  L_aux:       中间 latent 位置压低退出概率的 BCE loss（使用独立 exit_head）
  L_think_end: 最后一个 latent h_K 应退出的正向 BCE loss（使用独立 exit_head）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Qwen3-VL-4B-Instruct 原生 token ID
THINK_END_ID = 151668  # </think>


def compute_think_end_loss(
    latent_hiddens: list,
    exit_head: nn.Module,
) -> torch.Tensor:
    """计算 h_K 应退出的正向 BCE loss。

    对最后一个 latent hidden h_K：鼓励 exit_head(h_K) > 0（应该退出）。
    使用 BCE with logits，目标为 1。

    Args:
        latent_hiddens: [h_1, h_2, ..., h_K]，每个 shape [B, H]
        exit_head: 独立的退出判断头（LatentExitHead）

    Returns:
        think_end_loss: scalar BCE loss
    """
    if not latent_hiddens:
        return torch.tensor(0.0, requires_grad=True)

    # h_K = 最后一个 latent
    h_K = latent_hiddens[-1]  # [B, H]
    exit_logit = exit_head(h_K)  # [B, 1]

    # 目标：h_K 应该退出（target=1）
    target = torch.ones_like(exit_logit)
    loss = F.binary_cross_entropy_with_logits(exit_logit, target)
    return loss


def compute_aux_loss(
    latent_hiddens: list,
    exit_head: nn.Module,
) -> torch.Tensor:
    """计算中间 latent 不应退出的 BCE loss（使用独立 exit_head）。

    对 h_1 到 h_{K-1}（中间 latent）：压低 exit_head(h_i) 的退出概率。
    h_K 的正向约束由 compute_think_end_loss 单独处理。

    使用 BCE with logits，目标为 0（不应该退出）。

    Args:
        latent_hiddens: [h_1, h_2, ..., h_K]，每个 shape [B, H]
        exit_head: 独立的退出判断头（LatentExitHead）

    Returns:
        aux_loss: scalar
    """
    K = len(latent_hiddens)
    if K <= 1:
        # 只有 1 个 latent（即 h_K），中间 latent 为空，不需要辅助 loss
        return torch.tensor(0.0, device=latent_hiddens[0].device, requires_grad=True)

    device = latent_hiddens[0].device
    total_loss = torch.tensor(0.0, device=device)

    # 对 h_1 到 h_{K-1}：压低退出概率
    for i in range(K - 1):
        h_i = latent_hiddens[i]  # [B, H]
        exit_logit = exit_head(h_i)  # [B, 1]
        # BCE with logits: 目标为 0（不应该退出）
        target = torch.zeros_like(exit_logit)
        loss_i = F.binary_cross_entropy_with_logits(exit_logit, target)
        total_loss = total_loss + loss_i

    avg_loss = total_loss / (K - 1)
    return avg_loss


def compute_answer_loss(
    answer_logits: torch.Tensor,
    answer_labels: torch.LongTensor,
) -> torch.Tensor:
    """计算 answer 部分的标准 next-token CE loss。

    Args:
        answer_logits: [B, L_ans, vocab_size] Phase 3 输出的 logits
        answer_labels: [B, L_ans] GT answer token ids（-100 表示忽略）

    Returns:
        ans_loss: scalar
    """
    # shift: logits[:-1] 预测 labels[1:]
    shift_logits = answer_logits[:, :-1, :].contiguous()
    shift_labels = answer_labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def compute_total_loss(
    l_trans: torch.Tensor,
    l_ans: torch.Tensor,
    l_aux: torch.Tensor,
    l_think_end: torch.Tensor,
    lambda_trans: float = 1.0,
    lambda_ans: float = 1.0,
    lambda_aux: float = 1.0,
    lambda_think_end: float = 1.0,
) -> torch.Tensor:
    """计算总 loss = λ_trans * L_trans + λ_ans * L_ans + λ_aux * L_aux + λ_think_end * L_think_end。

    Args:
        l_trans: Translator 还原 loss
        l_ans: Answer CE loss
        l_aux: 中间 latent 不退出 BCE loss
        l_think_end: h_K 退出 BCE loss
        lambda_trans: L_trans 权重（默认 1.0）
        lambda_ans: L_ans 权重（默认 1.0）
        lambda_aux: L_aux 权重（默认 1.0）
        lambda_think_end: L_think_end 权重（默认 1.0）

    Returns:
        total_loss: scalar
    """
    total = (lambda_trans * l_trans + lambda_ans * l_ans
             + lambda_aux * l_aux + lambda_think_end * l_think_end)
    return total
