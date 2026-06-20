# 最终方案 v4 — 潜空间视频推理 + 双教师 OPD

> 本文档是**唯一权威方案**，写代码以本文档为准。
> 与 `OPD_Video_Reasoning_Training_Plan.md` / `Latent_Reasoning_Method_Analysis.md` / `Data_Preparation_Cookbook.md` / `Action_Plan_StepByStep.md` 中任何冲突描述，**以本文档为准**。
>
> Date: 2026-05-31
> Reviewer 角色：独立 AI，对前序方案做了正式审稿后落版。

---

## 目录

- [0. 旧方案错误清单（先看这里）](#0-旧方案错误清单先看这里)
- [1. 范式总览](#1-范式总览)
- [2. 四个角色](#2-四个角色)
- [3. 潜空间表示约定（核心）](#3-潜空间表示约定核心)
- [4. 段切换机制（核心）](#4-段切换机制核心)
- [5. 标签格式](#5-标签格式)
- [6. 八种 observe type](#6-八种-observe-type)
- [7. 训练阶段一：Stage 1（感知）](#7-训练阶段一stage-1感知)
- [8. 训练阶段二：Stage 2（推理）](#8-训练阶段二stage-2推理)
- [9. OPD 训练 step 完整伪代码](#9-opd-训练-step-完整伪代码)
- [10. 推理时流程](#10-推理时流程)
- [11. 损失函数完整定义](#11-损失函数完整定义)
- [12. 数据准备](#12-数据准备)
- [13. 工程实现细节](#13-工程实现细节)
- [14. 评测](#14-评测)
- [15. 时间线](#15-时间线)

---

## 0. 旧方案错误清单（先看这里）

下面列出旧方案中**必须修正**的设计点。代码实现时如发现旧文档与本文档冲突，**一律以本文档为准**。

| # | 旧方案错误 | 在哪份文档 | 正确做法 | 原因 |
|---|----------|----------|---------|------|
| **E1** | "N latent 对应一段话，N = ceil(len_segment/8)，8:1 压缩" | `Latent_Reasoning_Method_Analysis.md` §五/Q3 | **改为 1 latent ↔ 1 段话**（M:1 高压缩，M 由段长决定，无固定比） | 1:1/N:M 易引发"Student 用 hidden 伪装 token embedding"的短路解；1 latent ↔ 1 段话天然带信息瓶颈，避免短路 |
| **E2** | OPD step 4 描述："推理段送 Teacher_R 做 on-policy 前向 → 逐 token KL" | `OPD_Video_Reasoning_Training_Plan.md` §四 | **明确写**：Teacher 用 **teacher-forcing 模式 forward**，输入 = Student 解码出的 text token id 序列，取每个位置的 teacher logits 与 student logits 做 KL | 旧措辞有歧义，会被误读为"teacher 自由 generate vs student rollout 直接对齐"，长度对不上 |
| **E3** | "推理时学生在潜空间连续生成，不显式 decode 中间过程" — 但段切换信号没说推理时怎么来 | `OPD_Video_Reasoning_Training_Plan.md` §八 | **每生成一个 hidden，用 Student 自带 LM head 做一次 `hidden @ W_vocab`，取 argmax**：若 = `</think>` 则切到 answer 文本解码模式；否则当 latent 继续走 | 推理时不能调 Decoder（开销 200ms/步），LM head 复用 forward 输出，开销 < 0.5ms/步 |
| **E4** | 推理时段切换信号缺失对应的训练监督 | 旧方案没写 | 训练时加 **`</think>` 二元辅助 loss**：要求潜空间位置 LM head 在 `</think>` 上的概率 ≈ 0，段末位置 ≈ 1 | 否则推理时 LM head argmax 可能误触 `</think>`，导致段切换错乱 |
| **E5** | "奇数段=推理段，偶数段=感知段" 的奇偶规则 | `MEMORY.md` 旧版 | **删除奇偶规则**，段切换完全由 XML 标签（`</observe>` / `</result>` / `</think>`）字符串触发 | 奇偶规则脆弱，一旦 Decoder 解码漏一段就全错位 |
| **E6** | Stage 1-SFT 数据来源中包含 RefCOCO 图像数据集 | 旧版 cookbook（已被用户纠正过，但 MEMORY.md 残留） | **彻底删除图像数据集**，只用视频 | 学生是视频模型，图像数据会污染分布 |
| **E7** | 课程学习按 "GT observe 次数" 划分 Easy/Medium/Hard | `OPD_Video_Reasoning_Training_Plan.md` §七 | **改为按 "Student SFT 后失败率" 划分**（失败率 < 30% = Easy，30-60% = Med，> 60% = Hard） | GT observe 次数 ≠ 真实难度（教师可能啰嗦） |
| **E8** | 数据筛选 "教师失败 → 丢弃" | `OPD_Video_Reasoning_Training_Plan.md` §六 | **教师失败的可校验样本保留 20%**，走 GT 直接 SFT（不走 OPD） | 全扔会丢失所有长尾 hard cases |
| **E9** | Stage 1 → Stage 2 没有 gate 条件 | 旧方案没写 | **加 gate**：格式正确率 ≥ 95% 且单段感知 acc ≥ Qwen3-VL zero-shot baseline + 5pp，否则不进 Stage 2 | 防止 Stage 1 没训好就开 Stage 2 浪费算力 |

---

## 1. 范式总览

**一句话**：Student 在潜空间生成一串 hidden，每个 hidden 对应一段自然语言（推理段或感知段），Decoder 在训练时还原文字给双教师做 on-policy KL，推理时 Student 自带 LM head 监控 `</think>` 切换到 answer 文本解码。

```
训练时:
  video + question
        ↓
  Student forward (autoregressive in hidden space)
        ↓
  [h_1, h_2, ..., h_K]   ← 每个 h_i 对应一段话
        ↓
  Decoder (frozen, teacher-forcing) → text_segment_i for each h_i
        ↓
  按 text_segment_i 类型分发:
      推理段/answer段 → Teacher_R (text-only, teacher-forcing)
      感知段          → Teacher_P (focused video, teacher-forcing)
        ↓
  逐 token KL(teacher || student_via_decoder) + 辅助 loss(</think>判别)
        ↓
  反向传播 → 只更新 Student

推理时:
  video + question
        ↓
  Student forward 出 h_i
        ↓
  LM_head(h_i) 看 argmax 是否 = </think>?
      否 → h_i 是潜空间 latent，继续下一步 (h_i 进入 KV cache)
      是 → 切到 answer 模式，LM_head 自回归输出明文 answer
        ↓
  最终输出: <answer>...</answer> 明文
  全程不调 Decoder、不调任何外部工具
```

---

## 2. 四个角色

| 角色 | 模型 | 输入 | 输出 | 训练状态 |
|------|------|------|------|---------|
| **Student** | Qwen3-VL-7B | 完整长视频 + 原始问题 | hidden 序列 [h_1, ..., h_K] + 段末 answer 明文 | ✅ 可训练 |
| **Decoder** | Qwen3-VL-7B（Student 的冻结副本） | 单个 h_i 作为第一个输入 embedding | text segment（自回归解码到 EOS_segment） | ❌ 冻结 |
| **Teacher_R** | Qwen3-VL-7B | 纯文本：问题 + 前序所有解码出的段文本 + (若有) 感知段 result | text logits | ❌ 冻结 |
| **Teacher_P** | Qwen3-VL-7B | 感知问题 + 视觉聚焦预处理后的视频 | text logits | ❌ 冻结 |

**GPU 部署（4× H100）**：
- GPU 0-1: Student（fp16）+ Decoder（fp16 + activation checkpointing）
- GPU 2: Teacher_R（int8 推理，纯文本快）
- GPU 3: Teacher_P（fp16 推理）

---

## 3. 潜空间表示约定（核心）

### 3.1 一个 hidden 对应一段话（修正 E1）

- Student 在潜空间模式下，每输出 **1 个 hidden 向量 h_i**（shape: `[hidden_dim]`），就代表 **1 段完整自然语言文本**
- "1 段" 的定义：XML 中相邻两个边界标签之间的连续文本，例如：
  - 一个 `[分析] ...` 段
  - 一个 `<observe type=.../>` 段（含标签本身）
  - 一个 `<result>...</result>` 段
  - 一个 `[推理] ...` 段
  - 一个 `[结论] ...` 段
- 段长度不固定（10~300 token 都可能），压缩比 = `段长 : 1`，由段决定

### 3.2 Decoder 如何把 1 个 h_i 还原成一段文本

```python
# Decoder 输入序列
decoder_input_embeds = concat([
    instruction_prompt_embeds,    # 固定文本: "Decode this latent to text segment:"
    h_i.unsqueeze(0),             # 1 个 latent embedding（shape: [1, hidden_dim]）
    BOS_segment_embed,            # 段开始 token
])

# 自回归解码
text_tokens = []
for step in range(MAX_SEG_LEN):  # MAX_SEG_LEN = 512
    logits = decoder.forward(decoder_input_embeds + embed(text_tokens))
    next_token = argmax(logits[-1])
    if next_token == EOS_segment:
        break
    text_tokens.append(next_token)

decoded_segment = tokenizer.decode(text_tokens)
```

**关键**：h_i 一直在 Decoder 的 KV cache 里被 attention 到，Decoder 通过 cross-attend h_i 把信息展开成完整文字。

### 3.3 段类型判定（在 Decoder 解码出文本后）

```python
def classify_segment(decoded_text: str) -> str:
    s = decoded_text.strip()
    if s.startswith("<observe") and s.endswith("/>"):
        return "OBSERVE"   # 感知段，下一段由 Teacher_P 监督 result
    if s.startswith("<result>") and s.endswith("</result>"):
        return "RESULT"    # 感知 result，下一段切回推理
    if s.startswith("<answer>") and s.endswith("</answer>"):
        return "ANSWER"
    return "REASON"        # 纯推理文字
```

---

## 4. 段切换机制（核心，修正 E3/E4/E5）

### 4.1 训练时段切换

完全由 **Decoder 解码出的 XML 字符串** 决定（详见 §3.3），不用奇偶规则。

### 4.2 推理时段切换（基于 Student 自带 LM head）

**核心 invariant**：

> Student 在潜空间模式下输出的每个 hidden `h_i`，经 `LM_head(h_i)` 后 argmax 出来的 token，**绝对不能是 `</think>`**。
> 只有当 Student 真的要结束 think、进入 answer 时，才允许 argmax = `</think>`。

```python
# 推理时主循环
hidden_sequence = []
mode = "LATENT"  # 初始进入潜空间模式

while True:
    h = student.forward_one_step(prev_inputs)  # 生成下一个 hidden

    if mode == "LATENT":
        # 用 Student 自带 LM head 监控 </think>
        logits = student.lm_head(h)            # shape: [vocab_size]
        if argmax(logits) == THINK_END_TOKEN_ID:
            mode = "ANSWER"
            # 把当前 h 作为 </think> token 喂回 Student，开始 answer 文本生成
            prev_inputs = append_token(prev_inputs, THINK_END_TOKEN_ID)
            continue
        else:
            # 继续潜空间：h 作为 latent 加入 KV cache
            hidden_sequence.append(h)
            prev_inputs = append_latent(prev_inputs, h)

    elif mode == "ANSWER":
        # 标准文本自回归
        logits = student.lm_head(h)
        next_token = argmax(logits)
        if next_token == ANSWER_END_TOKEN_ID:
            break
        prev_inputs = append_token(prev_inputs, next_token)
```

### 4.3 训练时 `</think>` 二元辅助 loss（核心新增）

为保证 4.2 的 invariant，训练时对每个 h_i 加一个**只针对 `</think>` 这一个 token 的二元 BCE loss**：

```python
THINK_END_ID = tokenizer.encode("</think>")[0]

# 对每个潜空间位置 h_i
p_think = softmax(student.lm_head(h_i))[THINK_END_ID]

if i == K - 1:  # 段序列的最后一个 latent（GT 标注是 </think> 位置）
    aux_loss_i = -log(p_think + ε)             # 拉高 </think> 概率
else:
    aux_loss_i = -log(1 - p_think + ε)         # 压低 </think> 概率
```

**关键**：这个 loss **只约束 `</think>` 这一个 token 的概率**，其他 vocab token 的 logit 完全自由，不污染 LM head 在其他 token 上的分布。

---

## 5. 标签格式

XML，**不注册新 token**，全部用现有词表：

```
<think>
[分析] 我需要先确定狗在视频中出现的时间段。
<observe type="temporal_locate" target="dog"/>
<result>狗出现在 12.5s-18.3s。</result>
[推理] 接下来需要看狗在这段时间内的动作。
<observe type="temporal_clip" time="12.5-18.3" target="dog's action"/>
<result>狗在追一只球。</result>
[结论] 综上，狗在追球。
</think>
<answer>追球</answer>
```

**每个 `[xxx]` 块、每个 `<observe .../>`、每个 `<result>...</result>` 都是独立的一段**，对应 1 个 latent h_i。

上面例子总共 **8 段**，所以 Student 会输出 8 个 hidden + 最后的 `</think>` 触发 answer 模式。

---

## 6. 八种 observe type

每个 type 对应唯一的预处理 pipeline，**无内部分支**。

| Type | 必需参数 | 预处理（给 Teacher_P 看） | 工具 |
|------|---------|--------------------------|------|
| `temporal_locate` | target | 完整视频 + 时间轴候选标记 | OpenCV |
| `temporal_clip` | time, target | ffmpeg 裁切 [start-1s, end+1s]，高帧率 | ffmpeg |
| `spatial_detect` | frame, target | 高分辨率单帧 + Grounding-DINO 全物体高亮 | Grounding-DINO |
| `spatial_crop` | frame, bbox | bbox 裁切 + zoom + 20% 边距 | PIL |
| `depth_overlay` | frame, objects | 物体 bbox + Depth Anything V2 深度图叠加 | Grounding-DINO + Depth Anything V2 |
| `tracking_overlay` | time, target | SAM3 tracking 可视化叠加 | SAM3 |
| `ocr_zoom` | frame, bbox | bbox 区域裁切 + lanczos 放大 | PIL |
| `raw` | target | 原视频不处理 | - |

---

## 7. 训练阶段一：Stage 1（感知）

### 7.1 Stage 1-SFT（~30K，cross-entropy）

**目标**：学生学会格式 + 基础单步感知。每条数据**只有 1 次 observe**。

**数据**（删除所有图像数据集，修正 E6）：

| 来源 | 规模 | 数据集 | 生成方式 |
|------|------|--------|---------|
| A-1 temporal_locate | 10K | Charades-STA 5K + ActivityNet 5K | Python 模板 |
| A-2 temporal_clip | 3K | ActivityNet Captions | Python 模板 |
| A-3 spatial_detect | 3K | VidSTG | Python 模板 |
| A-4 spatial_crop | 2K | VidSTG 反向 | Python 模板 |
| A-5 tracking_overlay | 2K | HC-STVG | Python 模板 |
| C-1 因果/时序 | 5K | NExT-QA + STAR | Qwen3-72B 文本补写 |
| C-2 空间关系 | 3K | CLEVRER | Qwen3-72B 文本补写 |
| C-3 时间关系 | 2K | STAR | Qwen3-72B 文本补写 |

**训练**：
```yaml
base: Qwen3-VL-7B-Instruct
data: stage1_sft.jsonl (30K)
loss: CE on Decoder output + aux_loss(</think> BCE)
lr: 1e-5
epochs: 2
gpus: 4× H100
time: ~8h
```

**Gate（修正 E9）**：50 条 holdout 上格式合规率 > 95%，单段感知 acc ≥ Qwen3-VL zero-shot + 5pp，否则不进 7.2。

### 7.2 Stage 1-OPD（~80K，on-policy KL）

**目标**：用双教师精炼感知精度。

**数据筛选**（修正 E8）：
```python
for sample in candidate_pool_200K:
    student_pred = student.answer(sample.video, sample.question)
    if matches(student_pred, sample.gt_answer):
        continue  # 学生已会，扔
    teacher_pred = teacher_p.answer(focused(sample))
    if sample.verifiable:
        if matches(teacher_pred, sample.gt_answer):
            kept_opd.append(sample)        # 主流：OPD 训练
        else:
            if random.random() < 0.2:
                kept_self_sft.append(sample)  # 20% 保留为 GT-SFT 样本
    else:
        kept_opd.append(sample)            # 描述类直接保留
```

**训练**：
```yaml
base: student_stage1_sft.pt
data: stage1_opd.jsonl (~80K) + stage1_self_sft.jsonl (~10K)
loss:
  - OPD 样本: on-policy KL(双教师) + aux_loss(</think>)
  - SELF_SFT 样本: 标准 CE + aux_loss(</think>)
lr: 5e-6
epochs: 2
gpus: 4× H100
time: ~3 天
```

**Gate**：感知 IoU > Stage 1-SFT 模型 + 5pp。

---

## 8. 训练阶段二：Stage 2（推理）

### 8.1 Stage 2-SFT（~30K）

**目标**：学生学会多步推理格式（1-4 次 observe）。

**数据生成**：Gemini-2.5-Pro 生成多步轨迹 → Teacher_P 填充 `<result>` → answer 必须匹配 GT。

数据源：NExT-QA + STAR + Video-R1 + LongVideo-Reason + VideoEspresso，筛选后 ~30K。

### 8.2 Stage 2-OPD（~80K，课程学习）

**课程学习难度划分**（修正 E7）：

```python
# 先用 stage2_sft 模型在每条样本上 rollout 一次，按失败率分桶
for sample in stage2_opd_pool:
    successes = sum(student_sft.answer(sample) == sample.gt for _ in range(5))
    fail_rate = 1 - successes / 5
    if fail_rate < 0.3:
        sample.difficulty = "EASY"
    elif fail_rate < 0.6:
        sample.difficulty = "MEDIUM"
    else:
        sample.difficulty = "HARD"
```

**采样比例**：
```
Phase 2.1 (前1/3): Easy 70% / Medium 25% / Hard 5%
Phase 2.2 (中1/3): Easy 40% / Medium 40% / Hard 20%
Phase 2.3 (后1/3): Easy 30% / Medium 35% / Hard 35%
```

---

## 9. OPD 训练 step 完整伪代码（修正 E2）

```python
def opd_training_step(batch):
    video, question, gt_answer = batch

    # ===== 1. Student rollout in latent space =====
    student_hiddens = []       # [h_1, h_2, ..., h_K]
    student_logits_per_seg = []  # for KL later
    with torch.no_grad():
        # 用 SFT 阶段的 Student 在潜空间生成 hidden 序列
        # 停止条件: LM_head argmax = </think>，或 K > MAX_LATENT_STEPS
        student_hiddens, lm_head_logits = student.rollout_latent(
            video, question, max_steps=MAX_LATENT_STEPS
        )

    # ===== 2. Decoder 还原每个 h_i 为文本段 =====
    decoded_segments = []      # list of str
    for h_i in student_hiddens:
        seg_text = decoder.decode_latent(h_i)  # frozen, no_grad
        decoded_segments.append(seg_text)

    # ===== 3. 按段类型分发到 Teacher_R 或 Teacher_P =====
    prev_text_context = build_initial_context(question)
    kl_losses = []

    for i, seg_text in enumerate(decoded_segments):
        seg_type = classify_segment(seg_text)

        if seg_type in ("REASON", "OBSERVE", "ANSWER"):
            teacher = teacher_r
            teacher_input = prev_text_context  # 纯文本

        elif seg_type == "RESULT":
            teacher = teacher_p
            prev_observe = parse_prev_observe(decoded_segments[:i])
            focused = PIPELINES[prev_observe.type](video, **prev_observe.params)
            teacher_input = (focused, build_perception_question(prev_observe))

        # ===== 3.5 Teacher-forcing forward（核心，修正 E2）=====
        # 把 student 解码出的 seg_text 的 token id 序列喂给 teacher
        seg_token_ids = tokenizer.encode(seg_text)
        teacher_logits = teacher.forward_teacher_forcing(
            context=teacher_input,
            target_token_ids=seg_token_ids,
        )  # shape: [len(seg_token_ids), vocab_size]

        # ===== 3.6 Student 在同一 token 序列上的 logits =====
        # 把 h_i 通过 Decoder 投影到 vocab 空间（teacher-forcing）
        # 注意: 梯度要穿过 Decoder → h_i → Student
        student_logits_seg = decoder.forward_with_latent_teacher_forcing(
            latent=h_i,
            target_token_ids=seg_token_ids,
        )  # shape: [len(seg_token_ids), vocab_size]

        # ===== 3.7 逐 token KL =====
        kl_i = F.kl_div(
            F.log_softmax(student_logits_seg, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )
        kl_losses.append(kl_i)

        # 把当前段文本拼到上下文里
        prev_text_context = prev_text_context + seg_text

    # ===== 4. </think> 辅助 loss =====
    aux_losses = []
    THINK_END_ID = tokenizer.convert_tokens_to_ids("</think>")
    for i, h_i in enumerate(student_hiddens):
        p_think = F.softmax(student.lm_head(h_i), dim=-1)[THINK_END_ID]
        if i == len(student_hiddens) - 1:
            aux_losses.append(-torch.log(p_think + 1e-8))
        else:
            aux_losses.append(-torch.log(1 - p_think + 1e-8))

    # ===== 5. 总 loss =====
    total_loss = sum(kl_losses) / len(kl_losses) + LAMBDA_AUX * sum(aux_losses) / len(aux_losses)
    # LAMBDA_AUX = 0.1 起步

    total_loss.backward()
    # 只 Student 参数 requires_grad=True，Decoder/Teacher_R/Teacher_P 都冻结
    optimizer.step()
    optimizer.zero_grad()
```

**关键点强调**：
1. **Teacher 必须是 teacher-forcing 模式 forward**，输入 = Student 解码出的 token id（不是 teacher 自由 generate）
2. **Student 端的 logits 也是 teacher-forcing 出来的**：Decoder 用 latent h_i 做条件，对每个 GT 位置 i 输出 logits → 和 teacher logits 对齐
3. 梯度链：`KL → student_logits_seg → Decoder forward → h_i → Student params`。Decoder 冻结但梯度照样穿过。

---

## 10. 推理时流程

详见 §4.2。补充：

- **不调 Decoder**
- **不调任何外部工具**（SAM3/Grounding-DINO 等只在训练时给 Teacher_P 用）
- 单次推理总开销 ≈ Student forward + 每步 ~0.5ms 的 `</think>` 监测
- 输出：直接是明文 `<answer>...</answer>`

---

## 11. 损失函数完整定义

### Stage 1-SFT / Stage 2-SFT

```
L_sft = CE(Decoder(student_hidden) → GT_segment_text)  [对每个段]
      + λ_aux · L_think_aux

L_think_aux = Σ_i BCE(LM_head(h_i)[</think>], gt_is_think_end_i)
```

### Stage 1-OPD / Stage 2-OPD

```
L_opd = (1/K) · Σ_i KL(Teacher_i(seg_i) || Decoder(h_i, seg_i))   [teacher-forcing 对齐]
      + λ_aux · L_think_aux
```

`λ_aux = 0.1` 起步，若 `</think>` 误触率 > 1% 则升到 0.3。

---

## 12. 数据准备

详见 `Data_Preparation_Cookbook.md`，本文档对其的修订：

- 删除一切图像数据集
- C 来源用 **Qwen3-72B 文本 LLM**（不调用网络 API，本地部署），不用 GPT-4
- Stage 2 SFT 用 Gemini-2.5-Pro 生成多步轨迹（这是 cookbook 已确定的）
- 数据筛选按 §7.2 修订（保留 20% 教师失败样本走 SELF_SFT）

---

## 13. 工程实现细节

### 13.1 目录结构

```
video_opd_code/
├── configs/
│   ├── stage1_sft.yaml
│   ├── stage1_opd.yaml
│   ├── stage2_sft.yaml
│   └── stage2_opd.yaml
├── data/
│   ├── builders/         # 来源A模板脚本
│   ├── llm_writers/      # 来源C Qwen3-72B 调用
│   └── filters/          # OPD 数据筛选
├── pipelines/            # 8 个 observe 预处理
│   ├── temporal_locate.py
│   ├── temporal_clip.py
│   ├── spatial_detect.py
│   ├── spatial_crop.py
│   ├── depth_overlay.py
│   ├── tracking_overlay.py
│   ├── ocr_zoom.py
│   └── raw.py
├── models/
│   ├── student.py        # Qwen3-VL-7B + latent rollout
│   ├── decoder.py        # 冻结副本 + decode_latent
│   ├── teacher_r.py      # 纯文本 teacher-forcing forward
│   └── teacher_p.py      # 带视觉 teacher-forcing forward
├── training/
│   ├── sft_trainer.py
│   ├── opd_trainer.py    # 实现 §9 伪代码
│   └── losses.py         # KL + L_think_aux
├── inference/
│   └── latent_infer.py   # 实现 §4.2 推理逻辑
└── eval/
    └── benchmarks.py     # LVReason / Video-Holmes / MLVU / ...
```

### 13.2 关键超参

| 参数 | 值 | 说明 |
|------|---|------|
| `MAX_LATENT_STEPS` | 32 | 单次 rollout 最多 32 个 latent（即最多 32 段） |
| `MAX_SEG_LEN` | 512 | Decoder 单段解码上限 |
| `λ_aux` | 0.1 | `</think>` 辅助 loss 权重 |
| Student lr | 1e-5 (SFT) / 5e-6 (OPD) | |
| batch_size | 4 video / GPU × 4 GPU = 16 全局 | |
| grad accumulation | 8 | |
| 混合精度 | bf16 | Qwen3-VL 推荐 |
| Decoder activation ckpt | ✅ | 省显存 |

### 13.3 显存预估

- Student fp16: ~14 GB
- Decoder fp16 + ckpt: ~10 GB
- Teacher_R int8: ~7 GB
- Teacher_P fp16: ~14 GB
- 中间激活: ~15 GB（4 video × 32 latent × MAX_SEG_LEN）

→ 单卡 H100 80GB 足够，4 卡并行用 DDP。

---

## 14. 评测

**主战场**：LVReason, Video-Holmes, MLVU, NExT-QA, STAR, LVBench, VideoMME, MVBench

**主 baseline**：
- Qwen3-VL-7B-Instruct (zero-shot)
- Qwen3-VL-7B + 显式 CoT prompt
- AoTD (CVPR 2025)
- Weaver
- **Video-o3 (ICML 2026, MCG-NJU)** ← 核心对比对象

**消融**：
| 消融 | 验证什么 |
|------|---------|
| w/o Stage 1-OPD | 感知 OPD 是否必要 |
| w/o Stage 2-OPD | 推理 OPD 是否必要 |
| w/o 视觉聚焦 | Teacher_P 聚焦是否有效 |
| 单教师 | 双教师设计是否有效 |
| 显式 CoT 输出 | 潜空间是否优于显式（直接输出 think 文本） |
| w/o `</think>` aux loss | 辅助 loss 是否必要 |
| w/o 课程学习 | 课程是否必要 |
| 不同 `λ_aux` (0.05/0.1/0.3) | 辅助 loss 权重敏感性 |

---

## 15. 时间线（8 周）

```
Week 1:   环境部署 + 8 个 pipeline + 预实验1（视觉聚焦有效性，gate: Teacher_P IoU > 学生+5%）
Week 2:   Stage 1-SFT 数据准备（30K）+ 训练（8h × 4卡）+ gate 验证
Week 3-4: Stage 1-OPD 数据筛选（~80K）+ OPD framework 实现 + 训练（3 天）
Week 5-6: Stage 2-SFT 数据准备（Gemini-2.5-Pro）+ 训练
Week 7:   Stage 2-OPD 数据筛选 + 课程学习训练（5 天）
Week 8:   评估 + 8 项消融
```

**Gates（不达标停下来排查）**：
- W1 末: Teacher_P 聚焦 IoU > 学生 +5%
- W2 末: Stage 1-SFT 格式合规 > 95%
- W4 末: Stage 1-OPD 感知 IoU > base +5pp
- W6 末: Stage 2-SFT 多步格式 > 90%
- W8: LVReason > base +3pp（最低目标）

---

## 附录 A：与 Video-o3 的关系

- **数据**：直接复用 Seeker-173K 作为 Stage 1/2 SFT 的冷启动种子（schema 映射后），节省 2-3 周
- **方法对比**：Video-o3 = 显式 CoT + 显式工具调用；我们 = 潜空间 + 推理时无工具，核心区分点
- **基线表必须有 Video-o3 行**

## 附录 B：与已有潜空间工作的区分

| 工作 | 区分点 |
|------|-------|
| Coconut | 我们是视频多模态 + 双教师 OPD + 段切换由 LM head 监控 |
| CCoT | 我们用 KL 而非 MSE；段长由语义决定而非固定；多模态 |
| Heima | 我们要求 Decoder 还原可读文字（Heima 不要求）；推理/感知分离 |
| Mirage | 我们的 latent 是"压缩文字思维"而非"视觉想象" |
| LVR | 我们在 LLM hidden 空间，不在 vision embedding 空间 |
| Vision-OPD | 我们是潜空间且双教师，它是显式文字且单教师 |
| Video-OPD | 我们覆盖完整时空感知 + 推理，它仅 temporal grounding |

---

_本方案已通过独立 AI 审稿（2026-05-29 + 2026-05-31 两轮），列出的所有 E1-E9 错误均已在本文档中修正，可直接交付代码实现。_
