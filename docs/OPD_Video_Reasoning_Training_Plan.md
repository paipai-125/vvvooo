# 潜空间视频推理 + 双教师 On-Policy Distillation

## 核心一句话

学生(Qwen3-VL-7B)在潜空间rollout思考过程→Decoder还原为logits→对rollout出的token序列，按`</observe>`和`</result>`切分为推理段/感知段→推理段送Teacher_R做on-policy前向，感知段送Teacher_P做on-policy前向→逐token KL对齐→只更新学生。

---

# 一、角色

| 角色 | 模型 | 输入 | 状态 |
|------|------|------|------|
| **学生** | Qwen3-VL-7B | 完整长视频 + 问题 | 可训练 |
| **Decoder** | Qwen3-VL-7B独立副本 | 学生潜空间hidden | 冻结 |
| **推理教师 Teacher_R** | Qwen3-VL-7B | 纯文本：问题+前序已rollout的解码文本 | 冻结 |
| **感知教师 Teacher_P** | Qwen3-VL-7B | 感知问题 + 视觉聚焦预处理后的视频 | 冻结 |

训练时GPU部署：4卡H100（Student+Decoder占2卡，Teacher_R占1卡，Teacher_P占1卡）。

---

# 二、标签格式（XML，不注册新token）

```
<think>
[分析] ...
<observe type="TYPE" time="..." frame="..." bbox="..." objects="..." target="..."/>
<result>感知结果</result>
[推理] ...
（可重复observe/result多次）
[结论] ...
</think>
<answer>最终答案</answer>
```

段切换：`</observe>` → 切到感知段，`</result>` → 切回推理段。

---

# 三、8个Type（每个对应唯一预处理pipeline）

| Type | 必需参数 | 预处理（给Teacher_P看） | 工具 |
|------|---------|----------------------|------|
| `temporal_locate` | target | 完整视频+时间轴候选标记 | OpenCV |
| `temporal_clip` | time, target | ffmpeg裁切[start-1s, end+1s] | ffmpeg |
| `spatial_detect` | frame, target | 帧+Grounding-DINO全物体高亮 | Grounding-DINO |
| `spatial_crop` | frame, bbox | bbox裁切+zoom+20%边距 | PIL |
| `depth_overlay` | frame, objects | 物体bbox+Depth Anything深度图 | Grounding-DINO+Depth Anything V2 |
| `tracking_overlay` | time, target | SAM3 tracking可视化 | SAM3 |
| `ocr_zoom` | frame, bbox | bbox裁切+放大(lanczos) | PIL |
| `raw` | target | 原视频不处理 | - |

一个type对应一个pipeline，无内部分支。复杂感知通过组合多个type实现。

---

# 四、On-Policy Distillation 流程

```
每个训练step:

1. 学生前向: video + question → 潜空间rollout → hidden states
2. Decoder前向: hidden states → student_logits → token序列
3. 按XML标签切段: 遇</observe>切到感知段，遇</result>切回推理段
4. 对每个段调用对应教师（冻结）做on-policy前向:
   - 推理段: Teacher_R(问题+前序解码文本) → teacher_logits
   - 感知段: 解析<observe>参数 → 构造视觉聚焦输入 → Teacher_P前向 → teacher_logits
5. Loss = Σ KL(teacher_logits || student_logits) 逐token
6. 反向传播只更新学生
```

---

# 五、两阶段训练

## Stage 1: 感知预训练

| 子阶段 | 数据 | 方法 | 目标 |
|--------|------|------|------|
| Stage 1-SFT | ~30K | Cross-entropy | 学会格式+基础感知 |
| Stage 1-OPD | ~80K(筛选后) | On-policy KL | 精炼感知精度 |

## Stage 2: 推理训练

| 子阶段 | 数据 | 方法 | 目标 |
|--------|------|------|------|
| Stage 2-SFT | ~30K | Cross-entropy | 学会多步推理格式 |
| Stage 2-OPD | ~80K(筛选后) | On-policy KL + 课程学习 | 精炼推理质量 |

---

# 六、数据筛选（OPD阶段）

```
可校验题: 学生失败 ∩ 教师成功 → 保留
描述类题: 不校验答案 → 保留（信任教师）
学生成功: 丢弃
教师失败(可校验): 丢弃
```

全部基于GT的字符串/IoU比较，零LLM judge。

---

# 七、课程学习（Stage 2-OPD）

数据按GT中observe次数分类，训练时采样比例渐进：

```
Phase 2.1 (前1/3): Easy(1次) 70% / Medium(2次) 25% / Hard(3-4次) 5%
Phase 2.2 (中1/3): Easy 40% / Medium 40% / Hard 20%
Phase 2.3 (后1/3): Easy 30% / Medium 35% / Hard 35%
```

模型从问题特征→轨迹复杂度的相关性中自动学会"何时多观察何时少观察"。

---

# 八、推理时

学生在潜空间连续生成，不显式decode中间过程，直接输出明文`<answer>`。无需任何外部工具。

---

# 九、测试集

主战场：LVReason, Video-Holmes, MLVU, NExT-QA, STAR, LVBench, VideoMME, MVBench

不上：VSIBench（缺3D数据）, VideoReasonBench（需状态追踪）

---

# 十、与已有工作区分

| 工作 | 区别 |
|------|------|
| Coconut | 我们用于视频MLLM+双教师OPD |
| Vision-OPD | 单教师regional crop，我们双教师+视频推理 |
| Video-OPD | 仅temporal grounding，我们覆盖完整时空+推理 |
| AoTD | 显式CoT蒸馏，我们是潜空间+on-policy |
| Weaver | 推理时调外部工具，我们推理时纯模型 |
