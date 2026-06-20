"""
XML标签解析工具。
解析模型输出中的 <observe>, <result>, <answer> 标签。
"""
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ObserveQuery:
    """解析后的observe请求"""
    type: str
    target: str
    time: Optional[str] = None
    frame: Optional[str] = None
    bbox: Optional[str] = None
    objects: Optional[str] = None
    raw_text: str = ""


# 正则模式
_OBSERVE_PATTERN = re.compile(
    r'<observe\s+([^>]*)/?>', re.DOTALL
)
_RESULT_PATTERN = re.compile(
    r'<result>(.*?)</result>', re.DOTALL
)
_ANSWER_PATTERN = re.compile(
    r'<answer>(.*?)</answer>', re.DOTALL
)
_ATTR_PATTERN = re.compile(
    r'(\w+)\s*=\s*"([^"]*)"'
)


def parse_observe(text: str) -> list[ObserveQuery]:
    """从文本中解析所有<observe .../>标签"""
    results = []
    for match in _OBSERVE_PATTERN.finditer(text):
        attrs_str = match.group(1)
        attrs = dict(_ATTR_PATTERN.findall(attrs_str))

        if "type" not in attrs:
            raise ValueError(f"<observe>标签缺少type属性: {match.group(0)}")

        results.append(ObserveQuery(
            type=attrs["type"],
            target=attrs.get("target", ""),
            time=attrs.get("time"),
            frame=attrs.get("frame"),
            bbox=attrs.get("bbox"),
            objects=attrs.get("objects"),
            raw_text=match.group(0),
        ))
    return results


def parse_result(text: str) -> list[str]:
    """从文本中解析所有<result>...</result>"""
    return [m.group(1).strip() for m in _RESULT_PATTERN.finditer(text)]


def parse_answer(text: str) -> Optional[str]:
    """从文本中解析<answer>...</answer>"""
    m = _ANSWER_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    return None


def split_segments(text: str) -> list[dict]:
    """
    将完整轨迹文本按<observe>和</result>切分为段列表。
    
    Returns:
        list of {"type": "reasoning"|"perception", "text": str, "start": int, "end": int}
    """
    segments = []
    pos = 0
    full_text = text

    # 找所有observe和result的位置
    observe_matches = list(_OBSERVE_PATTERN.finditer(full_text))
    result_matches = list(_RESULT_PATTERN.finditer(full_text))

    if len(observe_matches) != len(result_matches):
        raise ValueError(
            f"<observe>数量({len(observe_matches)})与<result>数量({len(result_matches)})不匹配"
        )

    for obs_m, res_m in zip(observe_matches, result_matches):
        # observe之前的文本 = 推理段
        if obs_m.start() > pos:
            segments.append({
                "type": "reasoning",
                "text": full_text[pos:obs_m.end()],  # 包含<observe>标签本身
                "start": pos,
                "end": obs_m.end(),
            })

        # observe之后到result结束 = 感知段
        segments.append({
            "type": "perception",
            "text": full_text[obs_m.end():res_m.end()],
            "start": obs_m.end(),
            "end": res_m.end(),
        })
        pos = res_m.end()

    # 最后剩余的文本 = 推理段（含结论和answer）
    if pos < len(full_text):
        segments.append({
            "type": "reasoning",
            "text": full_text[pos:],
            "start": pos,
            "end": len(full_text),
        })

    return segments
