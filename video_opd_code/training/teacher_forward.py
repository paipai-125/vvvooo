"""Teacher Forward 模块：封装 Teacher_R / Teacher_P 的 teacher-forcing forward。

OPD 训练中，教师模型冻结，接收学生 Translator 解码出的文本（on-policy），
teacher-forcing 得到 logits，用于 KL 蒸馏。

Teacher_R（推理教师）：纯文本输入，接收问题 + 前序所有段文本（含 <result>）
Teacher_P（感知教师）：视频 + 感知问题输入，接收 pipeline 处理后的聚焦视觉
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.latent_sft_helpers import load_system_prompt


class TeacherRForward:
    """Teacher_R teacher-forcing 封装。

    输入：问题 + 前序上下文（含 <result> 内容）+ 当前段文本
    输出：当前段对应位置的 teacher logits
    """

    def __init__(self, model, tokenizer, device: torch.device, max_length: int = 8192):
        """
        Args:
            model: 已加载的 Teacher_R 模型（冻结）
            tokenizer: 对应的 tokenizer
            device: 模型所在设备
            max_length: 最大序列长度
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.sys_prompt = load_system_prompt("teacher_r")

    @torch.no_grad()
    def get_segment_logits(
        self,
        question: str,
        context: str,
        segment_text: str,
    ) -> Optional[torch.Tensor]:
        """Teacher-forcing forward，返回 segment 部分的 logits。

        构造输入：[system] + [user: question] + [assistant: context + segment_text]
        只返回 segment_text 对应位置的 logits。

        Args:
            question: 原始问题文本
            context: 前序所有段的累积文本（含 <result> 内容）
            segment_text: 当前段文本（Translator 解码出的 on-policy 文本）

        Returns:
            logits: [1, seg_len, vocab_size] 或 None（如果序列过长）
        """
        # 构造完整文本
        full_assistant = context + segment_text

        # 构造 prefix（不含 segment_text）
        prefix_assistant = context

        # 编码完整序列
        msgs_full = []
        if self.sys_prompt:
            msgs_full.append({"role": "system", "content": self.sys_prompt})
        msgs_full.append({"role": "user", "content": question})
        msgs_full.append({"role": "assistant", "content": full_assistant})

        # 用 chat template 格式化
        full_text = self._apply_chat_template(msgs_full)
        full_enc = self.tokenizer(
            full_text, return_tensors="pt", truncation=True,
            max_length=self.max_length
        )

        # 编码 prefix 序列（用于确定 segment 起始位置）
        msgs_prefix = []
        if self.sys_prompt:
            msgs_prefix.append({"role": "system", "content": self.sys_prompt})
        msgs_prefix.append({"role": "user", "content": question})
        msgs_prefix.append({"role": "assistant", "content": prefix_assistant})

        prefix_text = self._apply_chat_template(msgs_prefix)
        prefix_enc = self.tokenizer(
            prefix_text, return_tensors="pt", truncation=True,
            max_length=self.max_length
        )

        prefix_len = prefix_enc["input_ids"].shape[1]
        full_len = full_enc["input_ids"].shape[1]
        seg_len = full_len - prefix_len

        if seg_len <= 0:
            return None

        # Forward
        input_ids = full_enc["input_ids"].to(self.device)
        attention_mask = full_enc["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        # 提取 segment 部分的 logits
        # logits[i] 预测 token[i+1]，所以 segment 的预测 logits 是 [prefix_len-1 : full_len-1]
        logits = outputs.logits[:, prefix_len - 1: full_len - 1, :]  # [1, seg_len, V]

        return logits

    def _apply_chat_template(self, msgs: list) -> str:
        """简单的 chat template 格式化（纯文本，不含视觉）。"""
        parts = []
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        return "\n".join(parts)


class TeacherPForward:
    """Teacher_P teacher-forcing 封装。

    输入：pipeline 处理后的聚焦视频 + 感知问题 + result 文本
    输出：result 文本对应位置的 teacher logits
    """

    def __init__(self, model, processor, device: torch.device, max_length: int = 32768,
                 fps: float = 1.0):
        """
        Args:
            model: 已加载的 Teacher_P 模型（冻结）
            processor: 对应的 processor（含 tokenizer + video_processor）
            device: 模型所在设备
            max_length: 最大序列长度
            fps: 视频采样帧率
        """
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.device = device
        self.max_length = max_length
        self.fps = fps
        self.sys_prompt = load_system_prompt("teacher_p")

    @torch.no_grad()
    def get_segment_logits(
        self,
        video_path: str,
        perception_question: str,
        result_text: str,
    ) -> Optional[torch.Tensor]:
        """Teacher-forcing forward，返回 result_text 部分的 logits。

        Args:
            video_path: pipeline 处理后的聚焦视频路径
            perception_question: 感知问题
            result_text: <result>...</result> 内容（Translator 解码出的 on-policy 文本）

        Returns:
            logits: [1, result_len, vocab_size] 或 None（如果处理失败）
        """
        if not os.path.exists(video_path):
            return None

        # 构造完整消息（含 assistant 回复）
        msgs_full = []
        if self.sys_prompt:
            msgs_full.append({"role": "system",
                              "content": [{"type": "text", "text": self.sys_prompt}]})
        msgs_full.append({"role": "user", "content": [
            {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": self.fps},
            {"type": "text", "text": perception_question},
        ]})
        msgs_full.append({"role": "assistant",
                          "content": [{"type": "text", "text": result_text}]})

        # 构造 prefix 消息（不含 assistant 回复）
        msgs_prefix = msgs_full[:-1]

        try:
            # 编码完整序列
            full_inp = self.processor.apply_chat_template(
                msgs_full, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_tensors="pt"
            )
            # 编码 prefix
            prefix_inp = self.processor.apply_chat_template(
                msgs_prefix, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
        except Exception:
            return None

        prefix_len = prefix_inp["input_ids"].shape[1]
        full_len = full_inp["input_ids"].shape[1]
        seg_len = full_len - prefix_len

        if seg_len <= 0:
            return None

        # 移到设备
        fwd_kwargs = {}
        for k, v in full_inp.items():
            if isinstance(v, torch.Tensor):
                fwd_kwargs[k] = v.to(self.device)
            else:
                fwd_kwargs[k] = v

        fwd_kwargs["use_cache"] = False
        fwd_kwargs["return_dict"] = True

        outputs = self.model(**fwd_kwargs)

        # 提取 result 部分的 logits
        logits = outputs.logits[:, prefix_len - 1: full_len - 1, :]  # [1, seg_len, V]

        return logits


def compute_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """计算 forward KL 散度 loss（GKD 风格）。

    L_kl = KL(teacher || student) * τ²
         = Σ p_teacher * (log p_teacher - log p_student) * τ²

    使用 F.kl_div 时注意：F.kl_div(log_q, p) = KL(p || q)
    所以传入 (log_student, teacher_softmax) = KL(teacher || student)

    Args:
        student_logits: [B, L, V] 学生 logits（Translator 输出）
        teacher_logits: [B, L, V] 教师 logits（detached）
        temperature: 温度参数 τ

    Returns:
        kl_loss: scalar
    """
    # 确保长度对齐（取最短）
    min_len = min(student_logits.shape[1], teacher_logits.shape[1])
    s_logits = student_logits[:, :min_len, :]
    t_logits = teacher_logits[:, :min_len, :]

    # 温度缩放
    s_log_probs = F.log_softmax(s_logits / temperature, dim=-1)
    t_probs = F.softmax(t_logits / temperature, dim=-1)

    # Forward KL: KL(teacher || student)
    kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean")

    # 乘以 τ² 补偿温度缩放
    kl_loss = kl * (temperature ** 2)

    return kl_loss
