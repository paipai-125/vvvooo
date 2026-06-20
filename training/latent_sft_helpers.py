"""Stage1-SFT 潜空间训练辅助模块（0529 设计 · 独立实现）。

- 从 prompts/<role>.txt 加载 4 份 system prompt
- 在 GT trajectory 字符串上抽取奇/偶 latent 块的 char-range
- 借助 tokenizer 的 offset_mapping 把 char-range 映射回 input_ids 的 token 区间
- 任一块 GT token 数 > MAX_BLOCK_TOKENS(默认 128) 时返回 ok=False，调用方应弃样本
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_CODE_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = _CODE_ROOT / "prompts"

_PROMPT_CACHE: Dict[str, str] = {}

#: 任意单个潜空间块（奇/偶）的 GT 文本经 tokenize 后允许的最大 token 数。
MAX_BLOCK_TOKENS = 128


def load_system_prompt(role: str, verbose: bool = False) -> str:
    """从 prompts/<role>.txt 加载 system prompt（缺失则返回空串）。"""
    if role in _PROMPT_CACHE:
        return _PROMPT_CACHE[role]
    fp = PROMPTS_DIR / f"{role}.txt"
    if not fp.exists():
        if verbose and role != "decoder":
            print(f"[Prompt] 警告：{fp} 不存在", flush=True)
        _PROMPT_CACHE[role] = ""
        return ""
    text = fp.read_text(encoding="utf-8").strip()
    _PROMPT_CACHE[role] = text
    if verbose:
        head = text.replace("\n", " ")[:120]
        print(f"[Prompt] 加载 {role}: {len(text)} chars | head: {head!r}", flush=True)
    return text


def load_all_prompts(verbose: bool = True) -> Dict[str, str]:
    """启动时把 4 份 prompt 都加载到缓存。"""
    out = {}
    for role in ("student", "teacher_r", "teacher_p", "decoder"):
        out[role] = load_system_prompt(role, verbose=verbose)
    return out


# =============================================================================
# trajectory 文本上的奇/偶块解析
# =============================================================================

_RE_BLOCK_HEAD = re.compile(r"\[(Analyze|Reason|Conclude)\]")
# 真正的 <result> 开标签，排除自闭合占位符 <result/>
_RE_RESULT_OPEN = re.compile(r"<result>(?!\s*/)")
_RE_RESULT_CLOSE = re.compile(r"</result>")


def parse_trajectory_blocks(trajectory: str) -> List[Tuple[int, int, str]]:
    """在 trajectory 字符串上识别奇/偶块，返回 [(start, end, kind), ...]。

    奇数块 (kind="odd")：以 [Analyze]/[Reason]/[Conclude] 起头，到下一个
                        <result> 开标签前为止；连续多个 [...] 头属于同一块。
    偶数块 (kind="even")：<result>...</result> 之间的纯内容。

    总块数必须为奇数且严格 odd-even-odd-... 交替；否则返回空列表。
    """
    blocks: List[Tuple[int, int, str]] = []
    if not trajectory:
        return blocks

    think_open = trajectory.find("<think>")
    think_close = trajectory.find("</think>")
    if think_open < 0 or think_close < 0 or think_close <= think_open:
        return blocks
    region_start = think_open + len("<think>")
    region_end = think_close
    region = trajectory[region_start:region_end]

    odd_starts = [m.start() for m in _RE_BLOCK_HEAD.finditer(region)]
    even_opens = [m.start() for m in _RE_RESULT_OPEN.finditer(region)]
    even_closes = [m.start() for m in _RE_RESULT_CLOSE.finditer(region)]
    if not odd_starts:
        return blocks

    cursor = 0
    odd_idx = 0
    n_odd = len(odd_starts)

    while True:
        while odd_idx < n_odd and odd_starts[odd_idx] < cursor:
            odd_idx += 1
        if odd_idx >= n_odd:
            break

        odd_start_local = odd_starts[odd_idx]
        next_even_open_local: Optional[int] = None
        for o in even_opens:
            if o > odd_start_local:
                next_even_open_local = o
                break

        if next_even_open_local is None:
            odd_text = region[odd_start_local:].rstrip()
            odd_end_local = odd_start_local + len(odd_text)
            blocks.append((region_start + odd_start_local,
                           region_start + odd_end_local, "odd"))
            break

        odd_text = region[odd_start_local:next_even_open_local].rstrip()
        odd_end_local = odd_start_local + len(odd_text)
        blocks.append((region_start + odd_start_local,
                       region_start + odd_end_local, "odd"))

        even_start_local = next_even_open_local + len("<result>")
        next_close_local: Optional[int] = None
        for c in even_closes:
            if c >= even_start_local:
                next_close_local = c
                break
        if next_close_local is None:
            return []
        even_end_local = next_close_local
        blocks.append((region_start + even_start_local,
                       region_start + even_end_local, "even"))

        cursor = next_close_local + len("</result>")
        while odd_idx < n_odd and odd_starts[odd_idx] < cursor:
            odd_idx += 1

    if not blocks:
        return blocks
    if len(blocks) % 2 == 0:
        return []
    for i, (_, _, k) in enumerate(blocks):
        expected = "odd" if i % 2 == 0 else "even"
        if k != expected:
            return []
    return blocks


def _char_to_token_span(
    offsets: List[Tuple[int, int]],
    abs_char_start: int,
    abs_char_end: int,
) -> Tuple[int, int]:
    """根据 tokenizer 的 offset_mapping 把字符区间映射到 token 区间。"""
    tok_s = -1
    tok_e = -1
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue  # 跳过 special / 零宽 token
        if e <= abs_char_start:
            continue
        if s >= abs_char_end:
            break
        if tok_s == -1:
            tok_s = i
        tok_e = i + 1
    if tok_s < 0 or tok_e <= tok_s:
        return (-1, -1)
    return (tok_s, tok_e)


def compute_latent_odd_even_mask(
    full_text: str,
    trajectory: str,
    offsets: List[Tuple[int, int]],
    seq_len: int,
    max_block_tokens: int = MAX_BLOCK_TOKENS,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """生成 (L,) 的 odd_mask / even_mask 与有效性 ok。

    Args:
        full_text: chat template 完整文本（含 system / user / assistant）。
        trajectory: 学生 assistant 部分的原始文本（用来在 full_text 里反查起始）。
        offsets: tokenizer(return_offsets_mapping=True) 的 [(s,e), ...]。
        seq_len: input_ids 长度。
        max_block_tokens: 单块上限（默认 128，超过则丢样本）。

    Returns:
        odd_mask  : (L,) bool，奇数块（推理段）位置
        even_mask : (L,) bool，偶数块（感知段）位置
        ok        : True 表示样本有效；False 表示 trajectory 在 full_text 中找不到，
                    或某 latent 块 GT token 数 > max_block_tokens。
    """
    odd_mask = torch.zeros(seq_len, dtype=torch.bool)
    even_mask = torch.zeros(seq_len, dtype=torch.bool)

    base_char = full_text.rfind(trajectory)
    if base_char < 0:
        return odd_mask, even_mask, False

    blocks = parse_trajectory_blocks(trajectory)
    if not blocks:
        # trajectory 格式异常 → 整条样本走全 student-head CE 兜底（ok=True 但 mask 全 False）
        return odd_mask, even_mask, True

    for (cs, ce, kind) in blocks:
        ts, te = _char_to_token_span(offsets, base_char + cs, base_char + ce)
        if ts < 0 or te > seq_len:
            continue
        if (te - ts) > max_block_tokens:
            return odd_mask, even_mask, False
        if kind == "odd":
            odd_mask[ts:te] = True
        else:
            even_mask[ts:te] = True
    return odd_mask, even_mask, True


def debug_print_blocks(trajectory: str) -> None:
    blocks = parse_trajectory_blocks(trajectory)
    print(f"[debug] traj_len={len(trajectory)} blocks={len(blocks)}")
    for i, (s, e, k) in enumerate(blocks):
        snip = trajectory[s:e].replace("\n", "\\n")
        if len(snip) > 100:
            snip = snip[:97] + "..."
        print(f"  #{i+1:>2} {k:>4} [{s:>4},{e:>4}) len={e-s:>4} text={snip!r}")


if __name__ == "__main__":
    sample_a = (
        '<think>\n[Analyze] What did the person do with the food?\n'
        '[Reason] Based on the video content, I can determine the answer.\n'
        '</think>\n<answer>(A) Put down.</answer>'
    )
    sample_b = (
        '<think>\n[Analyze] Need to localize when "x" happens.\n'
        '<observe type="temporal_locate" target="x"/>\n'
        '<result>It happens from 0.0s to 13.6s.</result>\n'
        '[Conclude] Time span confirmed.\n'
        '</think>\n<answer>0.0s-13.6s</answer>'
    )
    print("=== sample A (0 observe) ==="); debug_print_blocks(sample_a)
    print("=== sample B (1 observe) ==="); debug_print_blocks(sample_b)