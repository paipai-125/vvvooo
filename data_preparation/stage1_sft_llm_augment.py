"""Stage 1-SFT 来源C: Qwen3-32B 基于已知答案补写推理文字。"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.paths import (  # noqa: E402
    CLEVRER_PATH, NEXTQA_PATH, QWEN3_32B_PATH,
    SFT_DATA_PATH, STAR_PATH, ensure_dirs,
)
from utils.parser import (  # noqa: E402
    parse_answer, parse_observe, parse_result, split_segments,
)

_LLM = None
_TOKENIZER = None
_VALID_TYPES = {
    "temporal_locate", "temporal_clip", "spatial_detect", "spatial_crop",
    "depth_overlay", "tracking_overlay", "ocr_zoom", "raw",
}


def _load_llm(dtype: str = "bfloat16"):
    """lazy loading: 第一次调用时加载 Qwen3-32B"""
    global _LLM, _TOKENIZER
    if _LLM is not None:
        return _LLM, _TOKENIZER
    if not QWEN3_32B_PATH.exists():
        raise FileNotFoundError(
            f"Qwen3-32B 路径不存在: {QWEN3_32B_PATH}\n"
            "请下载到该路径，或在 configs/paths.yaml 中调整 model_root / overrides。"
        )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    print(f"[LLM] 加载 Qwen3-32B ({dtype}) ...", flush=True)
    _TOKENIZER = AutoTokenizer.from_pretrained(str(QWEN3_32B_PATH), trust_remote_code=True)
    _LLM = AutoModelForCausalLM.from_pretrained(
        str(QWEN3_32B_PATH),
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    _LLM.eval()
    return _LLM, _TOKENIZER


def llm_generate(prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
    import torch
    model, tok = _load_llm()
    messages = [
        {"role": "system",
         "content": "你是严谨的视频QA数据标注员，必须严格按照给定的格式输出。"},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


PROMPT_C1 = """你是视频QA数据标注员。给定问题和正确答案，写一个**单步**推理轨迹。

问题: {question}
选项: {choices}
正确答案: {gt_answer}

请严格按以下格式输出（只输出格式内容，不要其他文字）:
<think>
[分析]（一句话说需要什么感知信息，不超过50字）
<observe type="TYPE" 参数.../>
<result>（直接写正确答案对应的"观察到的事实"）</result>
[结论]（一句话推出答案，不超过50字）
</think>
<answer>{gt_answer}</answer>

规则:
- TYPE 仅可从 [temporal_locate, temporal_clip, spatial_detect, spatial_crop, depth_overlay, tracking_overlay, raw] 中选
- 只允许出现一次 <observe> 和一次 <result>
- <result> 写"观察到的事实"，不是答案本身
- 必须包含 [分析]、[结论] 两个标记
"""

PROMPT_C2 = """你是视频QA数据标注员（空间关系任务）。给定问题和正确答案，写单步推理轨迹。

问题: {question}
正确答案: {gt_answer}

请严格按以下格式输出:
<think>
[分析]（说明需要观察哪些物体的空间位置，不超过50字）
<observe type="depth_overlay" frame="0.0" objects="物体A,物体B" target="空间位置关系"/>
<result>（写出两个物体的位置bbox/方位事实）</result>
[结论]（一句话给出方位结论）
</think>
<answer>{gt_answer}</answer>

只输出上述内容，不要其他文字。
"""

PROMPT_C3 = """你是视频QA数据标注员（时间关系任务）。给定问题和正确答案，写单步推理轨迹。

问题: {question}
正确答案: {gt_answer}

请严格按以下格式输出:
<think>
[分析]（说明需要观察哪个时间段，不超过50字）
<observe type="temporal_clip" time="0.0-5.0" target="..."/>
<result>（写出该时间段内发生的事实）</result>
[结论]（一句话给出结论）
</think>
<answer>{gt_answer}</answer>

只输出上述内容，不要其他文字。
"""


def validate_trajectory(traj: str, gt_answer: str) -> Optional[str]:
    """校验轨迹格式。返回 None 表示通过；否则返回错误说明。"""
    if "<think>" not in traj or "</think>" not in traj:
        return "缺少<think>标签"
    if "<answer>" not in traj or "</answer>" not in traj:
        return "缺少<answer>标签"
    obs = parse_observe(traj)
    if len(obs) != 1:
        return f"<observe>数量={len(obs)}，应为1"
    if obs[0].type not in _VALID_TYPES:
        return f"非法type: {obs[0].type}"
    res = parse_result(traj)
    if len(res) != 1:
        return f"<result>数量={len(res)}，应为1"
    ans = parse_answer(traj)
    if not ans:
        return "<answer>为空"
    if ans.strip() != gt_answer.strip():
        return f"<answer>={ans!r}与gt={gt_answer!r}不一致"
    if "[分析]" not in traj or "[结论]" not in traj:
        return "缺少[分析]或[结论]标记"
    try:
        split_segments(traj)
    except Exception as e:
        return f"split_segments失败: {e}"
    return None


def _extract_trajectory(raw: str) -> str:
    m_start = raw.find("<think>")
    m_end = raw.rfind("</answer>")
    if m_start < 0 or m_end < 0:
        return raw
    return raw[m_start: m_end + len("</answer>")]


def generate_with_retry(prompt: str, gt_answer: str, max_retry: int = 3) -> str:
    last_err = None
    for attempt in range(max_retry):
        temperature = 0.7 if attempt == 0 else 0.9
        raw = llm_generate(prompt, temperature=temperature)
        traj = _extract_trajectory(raw)
        err = validate_trajectory(traj, gt_answer)
        if err is None:
            return traj
        last_err = err
        print(f"[WARN] 生成不合规(尝试{attempt+1}/{max_retry}): {err}", file=sys.stderr)
    raise RuntimeError(
        f"生成失败 {max_retry} 次仍不合规: 最后错误={last_err}\nprompt={prompt[:200]}"
    )


def iter_nextqa(split: str = "train", subset: str = "causal_temporal") -> Iterator[dict]:
    """NExT-QA: JSONL 简化格式，每行 {video, question, choices, gt_answer, type, qid}"""
    ann_file = NEXTQA_PATH / f"{split}.jsonl"
    if not ann_file.exists():
        raise FileNotFoundError(f"NExT-QA标注未找到: {ann_file}")
    video_dir = NEXTQA_PATH / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"NExT-QA视频目录未找到: {video_dir}")
    with open(ann_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            for k in ("video", "question", "choices", "gt_answer"):
                if k not in item:
                    raise KeyError(f"NExT-QA字段缺失{k}: {item}")
            qtype = item.get("type", "")
            if subset == "causal_temporal" and qtype not in ("causal", "temporal", "C", "T"):
                continue
            video_path = (video_dir / item["video"]
                          if not os.path.isabs(item["video"]) else Path(item["video"]))
            if not video_path.exists():
                raise FileNotFoundError(f"NExT-QA视频不存在: {video_path}")
            yield {
                "video": str(video_path),
                "question": item["question"],
                "choices": item["choices"],
                "gt_answer": str(item["gt_answer"]),
                "type": qtype,
                "qid": item.get("qid", ""),
            }


def iter_star(split: str = "train", subset: str = "interaction_sequence") -> Iterator[dict]:
    """STAR: JSONL 简化格式"""
    ann_file = STAR_PATH / f"{split}.jsonl"
    if not ann_file.exists():
        raise FileNotFoundError(f"STAR标注未找到: {ann_file}")
    video_dir = STAR_PATH / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"STAR视频目录未找到: {video_dir}")
    valid = {
        "interaction_sequence": {"interaction", "sequence", "Interaction", "Sequence"},
        "temporal": {"temporal", "Temporal"},
        "all": None,
    }
    flt = valid.get(subset)
    with open(ann_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            for k in ("video", "question", "choices", "gt_answer"):
                if k not in item:
                    raise KeyError(f"STAR字段缺失{k}: {item}")
            qtype = item.get("type", "")
            if flt is not None and qtype not in flt:
                continue
            video_path = (video_dir / item["video"]
                          if not os.path.isabs(item["video"]) else Path(item["video"]))
            if not video_path.exists():
                raise FileNotFoundError(f"STAR视频不存在: {video_path}")
            yield {
                "video": str(video_path),
                "question": item["question"],
                "choices": item["choices"],
                "gt_answer": str(item["gt_answer"]),
                "type": qtype,
            }


def iter_clevrer(split: str = "train", subset: str = "spatial_relation") -> Iterator[dict]:
    """CLEVRER: JSONL 简化格式"""
    ann_file = CLEVRER_PATH / f"{split}.jsonl"
    if not ann_file.exists():
        raise FileNotFoundError(f"CLEVRER标注未找到: {ann_file}")
    video_dir = CLEVRER_PATH / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"CLEVRER视频目录未找到: {video_dir}")
    with open(ann_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            for k in ("video", "question", "choices", "gt_answer"):
                if k not in item:
                    raise KeyError(f"CLEVRER字段缺失{k}: {item}")
            qtype = item.get("type", "")
            if subset == "spatial_relation" and qtype not in (
                "spatial", "relation", "Spatial", "Relation"
            ):
                continue
            video_path = (video_dir / item["video"]
                          if not os.path.isabs(item["video"]) else Path(item["video"]))
            if not video_path.exists():
                raise FileNotFoundError(f"CLEVRER视频不存在: {video_path}")
            yield {
                "video": str(video_path),
                "question": item["question"],
                "choices": item["choices"],
                "gt_answer": str(item["gt_answer"]),
                "type": qtype,
            }


def _format_choices(choices) -> str:
    if isinstance(choices, dict):
        return ", ".join(f"{k}={v}" for k, v in choices.items())
    if isinstance(choices, list):
        return ", ".join(f"{chr(65+i)}={c}" for i, c in enumerate(choices))
    return str(choices)


def _load_done_ids(out_path: Path) -> set:
    """断点续跑：已写入的 sample_id"""
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "sample_id" in obj:
                    done.add(obj["sample_id"])
            except json.JSONDecodeError:
                continue
    return done


def _make_sample_id(source: str, idx: int, item: dict) -> str:
    base = item.get("qid") or f"{idx}"
    return f"{source}::{base}"


def _gen_one(it: dict, prompt_tpl: str, source: str, subset_tag: str,
             out_f, sid: str) -> None:
    if prompt_tpl is PROMPT_C1:
        prompt = prompt_tpl.format(
            question=it["question"],
            choices=_format_choices(it["choices"]),
            gt_answer=it["gt_answer"],
        )
    else:
        prompt = prompt_tpl.format(question=it["question"], gt_answer=it["gt_answer"])
    traj = generate_with_retry(prompt, it["gt_answer"])
    sample = {
        "sample_id": sid,
        "video": it["video"],
        "question": it["question"],
        "choices": it.get("choices", []),
        "trajectory": traj,
        "gt_answer": it["gt_answer"],
        "verifiable": True,
        "type": parse_observe(traj)[0].type,
        "source": source,
        "subset": subset_tag,
    }
    out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    out_f.flush()


def gen_c1(n_nextqa: int, n_star: int, output_path: Path):
    done = _load_done_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        cnt = 0
        for idx, it in enumerate(tqdm(iter_nextqa("train", "causal_temporal"),
                                      desc="C-1 NExT-QA", total=n_nextqa)):
            if cnt >= n_nextqa:
                break
            sid = _make_sample_id("nextqa", idx, it)
            if sid in done:
                cnt += 1
                continue
            _gen_one(it, PROMPT_C1, "nextqa", "C-1", f, sid)
            cnt += 1
        cnt2 = 0
        for idx, it in enumerate(tqdm(iter_star("train", "interaction_sequence"),
                                      desc="C-1 STAR", total=n_star)):
            if cnt2 >= n_star:
                break
            sid = _make_sample_id("star", idx, it)
            if sid in done:
                cnt2 += 1
                continue
            _gen_one(it, PROMPT_C1, "star", "C-1", f, sid)
            cnt2 += 1


def gen_c2(n: int, output_path: Path):
    done = _load_done_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        cnt = 0
        for idx, it in enumerate(tqdm(iter_clevrer("train", "spatial_relation"),
                                      desc="C-2 CLEVRER", total=n)):
            if cnt >= n:
                break
            sid = _make_sample_id("clevrer", idx, it)
            if sid in done:
                cnt += 1
                continue
            _gen_one(it, PROMPT_C2, "clevrer", "C-2", f, sid)
            cnt += 1


def gen_c3(n: int, output_path: Path):
    done = _load_done_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        cnt = 0
        for idx, it in enumerate(tqdm(iter_star("train", "temporal"),
                                      desc="C-3 STAR temporal", total=n)):
            if cnt >= n:
                break
            sid = _make_sample_id("star_temporal", idx, it)
            if sid in done:
                cnt += 1
                continue
            _gen_one(it, PROMPT_C3, "star", "C-3", f, sid)
            cnt += 1


def main():
    parser = argparse.ArgumentParser(description="Stage 1-SFT 来源C: LLM补写推理文字")
    parser.add_argument("--subset", choices=["all", "c1", "c2", "c3"], default="all")
    parser.add_argument("--n_c1_nextqa", type=int, default=3000)
    parser.add_argument("--n_c1_star", type=int, default=2000)
    parser.add_argument("--n_c2", type=int, default=3000)
    parser.add_argument("--n_c3", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    ensure_dirs()
    out_dir = Path(args.output_dir) if args.output_dir else SFT_DATA_PATH
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.subset in ("all", "c1"):
        gen_c1(args.n_c1_nextqa, args.n_c1_star,
               out_dir / "stage1_sft_c1_causal_temporal.jsonl")
    if args.subset in ("all", "c2"):
        gen_c2(args.n_c2, out_dir / "stage1_sft_c2_spatial_relation.jsonl")
    if args.subset in ("all", "c3"):
        gen_c3(args.n_c3, out_dir / "stage1_sft_c3_temporal_relation.jsonl")
    print("[DONE] LLM补写完成")


if __name__ == "__main__":
    main()
