"""
Qwen3-VL 在 transformers 5.10.0.dev0 下的 monkey-patch
========================================================

已知 bug:
    Qwen3VLProcessor.replace_video_token 把 <|video_pad|> 在文本里展开为
        "<|vision_start|><|placeholder|>*N<|vision_end|>"
    但 <|placeholder|> 是字面量字符串, 既不在 tokenizer 词表中,
    也没有任何后续代码把它换回 <|video_pad|>;
    结果 tokenize 后 input_ids 里 <|video_pad|>(151656) 数量为 0,
    而 video features 仍是 N, 触发:
        ValueError: Video features and video tokens do not match,
                    tokens: 0, features: N
    (同包 Qwen2.5-VL 写的是 `return self.video_token * N`, 没有这个 bug)

修复:
    覆盖 Qwen3VLProcessor.replace_video_token, 把字面量 <|placeholder|> 改为
    self.video_token (即 <|video_pad|>), 其它逻辑(时间戳/vision_start_end)保持一致。

用法:
    from utils.qwen3vl_patch import apply_qwen3vl_patches
    apply_qwen3vl_patches()  # 在 AutoProcessor.from_pretrained 之前调用一次即可
"""
from __future__ import annotations

_PATCHED = False


def apply_qwen3vl_patches() -> None:
    """对 Qwen3VLProcessor 做一次性 monkey-patch (重复调用幂等)。"""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from transformers.models.qwen3_vl.processing_qwen3_vl import (
            Qwen3VLProcessor,
        )
    except ImportError as e:
        raise ImportError(
            "无法导入 transformers.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor，"
            "请确认 transformers 版本支持 Qwen3-VL。"
        ) from e

    def _fixed_replace_video_token(
        self, video_inputs: dict, video_idx: int
    ) -> str:
        merge_length = self.video_processor.merge_size ** 2
        num_frames = video_inputs["video_grid_thw"][video_idx][0]
        frame_seqlen = (
            video_inputs["video_grid_thw"][video_idx][1:].prod() // merge_length
        )
        metadata = video_inputs["video_metadata"][video_idx]

        if metadata.fps is None:
            metadata.fps = 24

        # 与官方实现保持一致的时间戳计算
        curr_timestamp = self._calculate_timestamps(
            metadata.frames_indices,
            metadata.fps,
            self.video_processor.temporal_patch_size,
        )

        video_placeholder = ""
        for frame_idx in range(int(num_frames)):
            curr_time = curr_timestamp[frame_idx]
            video_placeholder += f"<{curr_time:.1f} seconds>"
            # ↓↓↓ 唯一的修复: 用真正的 video_token (<|video_pad|>) 而非 <|placeholder|>
            video_placeholder += (
                self.vision_start_token
                + self.video_token * int(frame_seqlen)
                + self.vision_end_token
            )
        return video_placeholder

    Qwen3VLProcessor.replace_video_token = _fixed_replace_video_token
    _PATCHED = True
