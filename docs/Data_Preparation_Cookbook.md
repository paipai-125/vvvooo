# 数据准备 Cookbook

> 两阶段（每阶段含SFT+OPD）的完整数据准备指南。

---

# 总览

```
Stage 1 感知预训练
  ├── Stage 1-SFT  ~30K   (教格式+基础感知)
  └── Stage 1-OPD  ~80K   (双教师OPD精炼)

Stage 2 推理训练
  ├── Stage 2-SFT  ~30K   (教多步推理格式)
  └── Stage 2-OPD  ~80K   (双教师OPD精炼)
```

---

# 一、标签格式

直接用XML，不注册新token：

```
<think>
[分析] ...
<observe type="TYPE" time="..." frame="..." bbox="..." objects="..." target="..."/>
<result>感知结果</result>
[推理/结论] ...
</think>
<answer>最终答案</answer>
```

解析用正则，检测`</observe>`切到感知段，`</result>`切回推理段。

---

# 二、8个Type（按预处理pipeline划分）

| Type | 必需参数 | 预处理（给Teacher_P看） | 工具 |
|------|---------|----------------------|------|
| `temporal_locate` | target | 完整视频+时间轴候选标记 | OpenCV |
| `temporal_clip` | time, target | ffmpeg裁切[start-1s, end+1s]高帧率 | ffmpeg |
| `spatial_detect` | frame, target | 帧高分辨率+Grounding-DINO全物体高亮 | Grounding-DINO |
| `spatial_crop` | frame, bbox | bbox裁切+zoom+20%边距 | PIL |
| `depth_overlay` | frame, objects | 物体bbox+Depth Anything深度图叠加 | Grounding-DINO+Depth Anything V2 |
| `tracking_overlay` | time, target | SAM3 tracking可视化叠加 | SAM3 |
| `ocr_zoom` | frame, bbox | bbox区域裁切+放大（PIL lanczos） | PIL |
| `raw` | target | 不做特殊预处理 | - |

每个type对应唯一确定的预处理流程，无内部分支。

复杂感知通过组合多个type实现（如"X在Y之前还是之后"=两次temporal_locate+推理判断）。

---

# 三、Stage 1-SFT 数据准备

## 3.1 目标

让学生学会格式 + 基础感知能力。**每条数据只有一次observe**（单步感知任务）。

## 3.2 数据来源A：规则模板生成（~20K，零LLM成本）

从视频数据集出发，用Python脚本直接生成完整轨迹。

### A-1: temporal_locate（时间定位）— 10K

**数据集**: Charades-STA (16K) + ActivityNet Captions (100K)，各取5K

```python
def wrap_temporal_locate(video_path, query, gt_start, gt_end):
    return {
        "video": video_path,
        "question": f""{query}"发生在视频什么时间？",
        "trajectory": f"""<think>
[分析]需要定位"{query}"的时间段。
<observe type="temporal_locate" target="{query}"/>
<result>从{gt_start:.1f}s到{gt_end:.1f}s。</result>
[结论]时间段已确定。
</think>
<answer>{gt_start:.1f}s-{gt_end:.1f}s</answer>""",
        "gt_answer": f"{gt_start:.1f}s-{gt_end:.1f}s",
        "verifiable": True,
        "type": "temporal_locate"
    }
```

### A-2: temporal_clip（时间段描述）— 3K

**数据集**: ActivityNet Captions（每段有dense caption标注）

```python
def wrap_temporal_describe(video_path, start, end, gt_caption):
    return {
        "video": video_path,
        "question": f"描述视频中{start:.1f}s到{end:.1f}s之间发生了什么。",
        "trajectory": f"""<think>
[分析]需要观察指定时间段内容。
<observe type="temporal_clip" time="{start:.1f}-{end:.1f}" target="该时间段内的活动"/>
<result>{gt_caption}</result>
[结论]描述完成。
</think>
<answer>{gt_caption}</answer>""",
        "gt_answer": gt_caption,
        "verifiable": False,
        "type": "temporal_clip"
    }
```

### A-3: spatial_detect（空间定位）— 3K

**数据集**: VidSTG（含video+frame+bbox+referring expression）

```python
def wrap_spatial_detect(video_path, frame_time, gt_bbox, referring_text):
    b = gt_bbox
    return {
        "video": video_path,
        "question": f"在{frame_time:.1f}s时，{referring_text}在画面哪里？",
        "trajectory": f"""<think>
[分析]需要在指定帧中定位"{referring_text}"。
<observe type="spatial_detect" frame="{frame_time:.1f}" target="{referring_text}"/>
<result>{referring_text}位于[{b[0]},{b[1]},{b[2]},{b[3]}]。</result>
[结论]位置已确定。
</think>
<answer>[{b[0]},{b[1]},{b[2]},{b[3]}]</answer>""",
        "gt_answer": gt_bbox,
        "verifiable": True,
        "type": "spatial_detect"
    }
```

### A-4: spatial_crop（区域描述）— 2K

**数据集**: VidSTG（反向：给定bbox，要求描述）

```python
def wrap_spatial_describe(video_path, frame_time, bbox, gt_description):
    b = bbox
    return {
        "video": video_path,
        "question": f"在{frame_time:.1f}s时，[{b[0]},{b[1]},{b[2]},{b[3]}]区域是什么？",
        "trajectory": f"""<think>
[分析]需要观察指定区域内容。
<observe type="spatial_crop" frame="{frame_time:.1f}" bbox="[{b[0]},{b[1]},{b[2]},{b[3]}]" target="该区域物体"/>
<result>{gt_description}</result>
[结论]描述完成。
</think>
<answer>{gt_description}</answer>""",
        "gt_answer": gt_description,
        "verifiable": False,
        "type": "spatial_crop"
    }
```

### A-5: tracking_overlay（时空追踪）— 2K

**数据集**: HC-STVG（含video+时间段+逐帧bbox序列+description）

```python
def wrap_tracking(video_path, start, end, bbox_start, bbox_end, desc):
    return {
        "video": video_path,
        "question": f""{desc}"在视频中出现的时间和位置？",
        "trajectory": f"""<think>
[分析]需要定位并追踪"{desc}"。
<observe type="tracking_overlay" time="{start:.1f}-{end:.1f}" target="{desc}"/>
<result>从{start:.1f}s到{end:.1f}s出现，起始[{bbox_start}]，结束[{bbox_end}]。</result>
[结论]时空信息已确定。
</think>
<answer>{start:.1f}s-{end:.1f}s, [{bbox_start}]至[{bbox_end}]</answer>""",
        "gt_answer": {"time": [start, end], "bbox_start": bbox_start, "bbox_end": bbox_end},
        "verifiable": True,
        "type": "tracking_overlay"
    }
```

## 3.3 数据来源C：文本LLM基于答案构造推理过程（~10K）

**适用场景**: 任务略需一步推理（因果/关系/时序），模板太死板搞不定。

**做法**: 用**开源文本LLM（Qwen3-32B）**，输入问题+正确答案，让它补充a1和a3的推理文字。感知段(a2)的`<result>`直接填入GT答案。**LLM不做推理，只做文字编排。**

### C-1: 简单因果/时序推理 — 5K

**数据集**: NExT-QA causal/temporal选择题（3K）+ STAR interaction/sequence（2K）

```python
def build_stage1_trajectory_with_llm(question, choices, gt_answer, llm="Qwen3-32B"):
    prompt = f"""你是视频QA数据标注员。给定问题和正确答案，写一个单步推理轨迹。

问题: {question}
选项: {choices}
正确答案: {gt_answer}

格式（严格遵循）:
<think>
[分析]（一句话说需要什么感知信息）
<observe type="TYPE" 参数.../>
<result>（直接写正确答案对应的感知结果）</result>
[结论]（一句话推出答案）
</think>
<answer>{gt_answer}</answer>

规则:
- TYPE从: temporal_locate / temporal_clip / spatial_detect / spatial_crop / depth_overlay / tracking_overlay / raw 中选
- 只写一次observe
- 每段不超过50字
- <result>内容应该是"观察到的事实"而非答案本身"""
    
    trajectory = llm.generate(prompt)
    return trajectory if validate_format(trajectory) else None
```

### C-2: 空间关系 — 3K

**数据集**: CLEVRER spatial/relation选择题

```python
# 例: "红球在蓝球的哪个方向？" gt_answer="左边"
prompt = f"""问题: {question}
正确答案: {gt_answer}

<think>
[分析]需要确定红球和蓝球的空间关系。
<observe type="depth_overlay" frame="0.0" objects="红球,蓝球" target="空间位置关系"/>
<result>红球在画面左侧[120,200,180,260]，蓝球在右侧[400,200,460,260]。</result>
[结论]红球在蓝球的左边。
</think>
<answer>左边</answer>"""
```

### C-3: 时间关系推理 — 2K

**数据集**: STAR temporal子集

```python
# 例: "人打开笔记本前做了什么？" gt_answer="拿起杯子"
prompt = f"""问题: {question}
正确答案: {gt_answer}

<think>
[分析]需要先知道"打开笔记本"的时间，然后看之前的动作。
<observe type="temporal_clip" time="0.0-5.0" target="人打开笔记本之前的动作"/>
<result>人在3.2s拿起了杯子。</result>
[结论]打开笔记本前，人拿起了杯子。
</think>
<answer>拿起杯子</answer>"""
```

## 3.4 配比总结

```
Stage 1-SFT ~30K:

来源A 规则模板（零LLM成本）: ~20K
  ├── A-1 temporal_locate:    10K  (Charades-STA 5K + ActivityNet 5K)
  ├── A-2 temporal_clip:       3K  (ActivityNet 描述)
  ├── A-3 spatial_detect:      3K  (VidSTG)
  ├── A-4 spatial_crop:        2K  (VidSTG 反向)
  └── A-5 tracking_overlay:    2K  (HC-STVG)

来源C 文本LLM补写（Qwen3-32B本地）: ~10K
  ├── C-1 因果/时序:           5K  (NExT-QA + STAR)
  ├── C-2 空间关系:            3K  (CLEVRER)
  └── C-3 时间关系:            2K  (STAR)
```

## 3.5 数据集表

| 数据集 | 视频格式 | 标注内容 | 用于type | 生成方式 | 是否可校验 |
|--------|---------|---------|---------|---------|-----------|
| Charades-STA | 视频clips | query+时间段 | temporal_locate | 模板 | ✅ IoU |
| ActivityNet Captions | 长视频 | 时间段+描述 | temporal_locate, temporal_clip | 模板 | ✅/❌ |
| VidSTG | 视频+帧 | referring+bbox | spatial_detect, spatial_crop | 模板 | ✅ IoU / ❌ |
| HC-STVG | 视频 | 时间+bbox序列+描述 | tracking_overlay | 模板 | ✅ 时空IoU |
| NExT-QA | 视频 | 选择题+答案 | raw, temporal_clip | LLM | ✅ 选项 |
| STAR | 视频 | 选择题+答案 | raw, temporal_clip | LLM | ✅ 选项 |
| CLEVRER | 合成视频 | 选择题+答案 | depth_overlay, spatial_detect | LLM | ✅ 选项 |

**全部是视频数据集。不用图像数据集。**

---

# 四、Stage 1-OPD 数据准备

## 4.1 目标

准备OPD用的~80K筛选后样本。OPD数据只需 `(video, question, gt_answer, type, 参数)`，不需要标注轨迹——学生自己rollout。

## 4.2 数据筛选：学生失败 ∩ 教师成功

```python
def prepare_stage1_opd(raw_dataset):
    student = load_qwen3_vl()   # base模型
    teacher_p = load_qwen3_vl() # 同模型
    
    kept = []
    for sample in raw_dataset:
        # 学生直答
        student_pred = student.answer(sample.video, sample.question)
        if matches(student_pred, sample.gt_answer):
            continue  # 学生已会，跳过
        
        # 教师+视觉聚焦
        focused = build_focused_input(sample)
        teacher_pred = teacher_p.answer(focused["video"], focused["question"])
        
        if sample.verifiable:
            if matches(teacher_pred, sample.gt_answer):
                kept.append(sample)  # Type A: 保留
        else:
            kept.append(sample)  # Type B: 描述类，保留
    
    return kept  # ~80K
```

## 4.3 OPD数据来源（候选池~200K）

| 数据集 | 主要Type | 校验方式 | 规模 |
|--------|---------|---------|------|
| Charades-STA | temporal_locate | IoU>0.5 | 16K |
| ActivityNet Captions | temporal_locate, temporal_clip | IoU / 不校验 | 50K(子采样) |
| VidSTG | spatial_detect, spatial_crop | IoU>0.5 / 不校验 | 50K(子采样) |
| HC-STVG | tracking_overlay | 时空IoU>0.3 | 16K |
| NExT-QA (causal/temporal) | raw, temporal_clip | 选项匹配 | 30K |
| STAR (interaction/sequence) | raw, temporal_clip | 选项匹配 | 30K |
| CLEVRER | depth_overlay, spatial_detect | 选项匹配 | 21K |

## 4.4 视觉聚焦实现

```python
PIPELINES = {
    "temporal_locate": lambda v, **kw: temporal_locate_pipeline(v, kw["target"]),
    "temporal_clip":   lambda v, **kw: temporal_clip_pipeline(v, kw["time"], kw["target"]),
    "spatial_detect":  lambda v, **kw: spatial_detect_pipeline(v, kw["frame"], kw["target"]),
    "spatial_crop":    lambda v, **kw: spatial_crop_pipeline(v, kw["frame"], kw["bbox"]),
    "depth_overlay":   lambda v, **kw: depth_overlay_pipeline(v, kw["frame"], kw["objects"]),
    "tracking_overlay":lambda v, **kw: tracking_overlay_pipeline(v, kw["time"], kw["target"]),
    "ocr_zoom":        lambda v, **kw: ocr_zoom_pipeline(v, kw["frame"], kw["bbox"]),
    "raw":             lambda v, **kw: raw_pipeline(v, kw["target"]),
}

def build_focused_input(sample):
    return PIPELINES[sample.type](sample.video, **sample.params)
```

---

# 五、Stage 2-SFT 数据准备

## 5.1 目标

让学生学会多步推理格式（1-4次observe）。

## 5.2 做法：Gemini-2.5-Pro生成轨迹 + Teacher_P填充result

```python
prompt = f"""You are a video reasoning annotator.
Given the question and correct answer, generate a multi-step reasoning trajectory.

Question: {question}
Correct answer: {gt_answer}

Format:
<think>
[分析] ...
<observe type="TYPE" 参数.../>
<result>PLACEHOLDER</result>
[推理] ...
（可重复1-4次observe/result）
[结论] ...
</think>
<answer>{gt_answer}</answer>

Rules:
- 每段不超过80字
- TYPE从8种中选
- <result>写PLACEHOLDER，稍后填充
"""

# 生成轨迹 → 解析observe → Teacher_P填充result → 校验answer
```

## 5.3 数据来源

| 来源 | 规模 | 题型 |
|------|------|------|
| NExT-QA (causal/temporal) | ~10K | 选择题 |
| STAR | ~10K | 选择题 |
| Video-R1 (选择题部分) | ~10K | 选择题 |
| LongVideo-Reason (选择题) | ~5K | 选择题 |
| VideoEspresso | ~5K | 选择题 |

筛选后~30K（answer匹配GT才保留）。

## 5.4 课程学习切分

```python
easy   = [s for s in data if count_observe(s) == 1]  # ~40%
medium = [s for s in data if count_observe(s) == 2]  # ~30%
hard   = [s for s in data if count_observe(s) >= 3]  # ~30%
```

---

# 六、Stage 2-OPD 数据准备

同Stage 1-OPD思路：`(video, question, gt_answer)` 做学生失败∩教师成功的双过滤。

数据源（候选~200K）：NExT-QA + STAR + CLEVRER reasoning + Video-R1选择题 + LongVideo-Reason选择题。

筛选后~80K。按observe次数做课程学习采样。

---

# 七、工具表

| 工具 | 用途 | 部署 |
|------|------|------|
| Qwen3-VL-7B | 学生+教师+Decoder | HuggingFace |
| **Grounding-DINO** | spatial_detect | 推理服务 |
| **SAM3** | tracking_overlay | 推理服务 |
| **Depth Anything V2** | depth_overlay | 推理服务 |
| ffmpeg | temporal_clip | CLI |
| OpenCV | temporal_locate标记绘制 | Python库 |
| PIL | spatial_crop, ocr_zoom | Python库 |
| **Qwen3-32B** | Stage 1来源C轨迹生成（纯文本） | 本地/API |
| **Gemini-2.5-Pro** | Stage 2轨迹生成 | 网络API |

---

# 八、预实验清单

## 预实验1：视觉聚焦有效性 ⭐最关键

- Charades-STA验证集500条
- 对比：学生直答 vs Teacher_P+聚焦
- 期望：Teacher_P IoU > 学生 + 5%
- 1天，1张H100

## 预实验2：SFT格式适配

- 1万Stage 1-SFT样本训练
- 验证格式合规率 > 90%
- 1天，2张H100

## 预实验3：OPD pipeline跑通

- 100条样本跑20步
- loss下降合理
- 1-2天，4张H100

## 预实验4：Gemini轨迹生成质量

- 200条Stage 2轨迹
- 人工抽查50条
- 1天

---

# 九、测试集

| 测试集 | 对应训练 |
|--------|---------|
| LVReason | Stage 2长视频推理 |
| Video-Holmes | Stage 2线索追踪 |
| MLVU | 综合 |
| NExT-QA | Stage 1/2因果时序 |
| STAR | Stage 1/2组合时空 |
| VideoMME | 通用 |
| MVBench | 通用 |
| LVBench | 长视频 |
