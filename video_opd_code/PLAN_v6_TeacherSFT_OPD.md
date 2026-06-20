# v6 方案：教师 SFT + OPD 实现

> Date: 2026-06-02 | 范围: Teacher_R SFT + Teacher_P SFT + Stage1-OPD
> 前置依赖: v5 学生 + Translator SFT 已完成（checkpoint-7500）
> 设计优先级: 用户口头指令 > v5 > v6 > v4（冲突时以高优先级为准）

---

## 1. 总体目标

在 v5 学生 SFT 基础上，完成三件事：

1. **Teacher_R SFT**：训练推理教师（纯文本，不看视频），学会推理框架
2. **Teacher_P SFT**：训练感知教师（看视觉聚焦输入），学会感知回答
3. **Stage1-OPD**：四模型协同蒸馏，用双教师的 logits 指导学生潜空间表示

---

## 2. 四个角色（OPD 阶段全部参与）

| 角色 | 模型 | GPU 分配 | 训练状态 | 输入 |
|------|------|---------|---------|------|
| **Student** | Qwen3-VL-4B（v5 SFT ckpt） | GPU 0 | ✅ 可训练 | 完整视频 + 问题 |
| **Translator** | Qwen3-VL-4B LM 主干（冻结） | GPU 1 | ❌ 冻结 | Student 的 h_i |
| **Teacher_R** | Qwen3-VL-4B（Teacher_R SFT ckpt） | GPU 2 | ❌ 冻结 | 纯文本：问题 + 前序上下文 |
| **Teacher_P** | Qwen3-VL-4B（Teacher_P SFT ckpt） | GPU 3 | ❌ 冻结 | 视觉聚焦输入 + 感知问题 |

**单机八卡 = 两组并行训练**（GPU 0-3 一组，GPU 4-7 一组），每组独立处理不同样本。

---

## 3. Teacher_R SFT

### 3.1 角色定义

推理教师：**不看视频**，只看问题 + 前序推理上下文，输出推理段。

- 学会推理框架：`[Analyze]...` → `<observe .../>` → 接收 `<result/>` → `[Conclude]...`
- 感知结果 `<result>...</result>` 在 SFT 阶段用占位符 `<result/>`
- OPD 阶段接收 Teacher_P 的真实 result，可以多步 observe

### 3.2 训练数据

从学生 SFT 数据提取（`prepare_teacher_sft_data.py --role teacher_r`）：

```json
{
  "role": "teacher_r",
  "question": "What is the dog doing between 12s and 18s?",
  "trajectory": "<think>\n[Analyze] I need to...\n<observe type=\"temporal_locate\" target=\"dog\"/>\n<result/>\n[Conclude] The dog is chasing a ball.\n</think>\n<answer>Chasing a ball</answer>",
  "has_video_input": false
}
```

### 3.3 训练方式

标准 SFT（CE loss），**不涉及潜空间**，不涉及视频：

```python
# 输入格式（Qwen3 chat template）:
messages = [
    {"role": "system", "content": TEACHER_R_SYSTEM_PROMPT},
    {"role": "user", "content": question},
    {"role": "assistant", "content": trajectory}  # 监督目标
]
# 标准 next-token prediction，只对 assistant 部分计算 loss
```

### 3.4 OPD 阶段提示词（多步 observe）

```
System: 你是一个视频推理专家。你可以通过 <observe> 标签请求视觉感知，
感知结果会以 <result> 形式返回。你可以多次 observe 获取不同信息。
在 <think>...</think> 中推理，在 <answer>...</answer> 中给出答案。
```

### 3.5 超参

```yaml
model: Qwen3-VL-4B-Instruct（纯文本，不加载视觉塔）
training: 全参数微调
gpus: 8 × H100 96GB, DDP
precision: bf16
lr: 1e-5, cosine + 3% warmup
batch: 1/GPU × 8GPU, grad_acc=2, effective=16
epochs: 1
max_length: 8192（纯文本，不需要长序列）
deepspeed: ZeRO-2
```

---

## 4. Teacher_P SFT

### 4.1 角色定义

感知教师：看**视觉聚焦输入**（pipeline 预处理后的视频/帧），回答感知问题。

- 输入不是原始完整视频，而是 pipeline 产物（裁切片段、放大区域等）
- SFT 阶段用原始视频 + 帧时间近似（pipeline 预处理需要 GPU 模型，SFT 时不方便跑）
- OPD 阶段用真实 pipeline 预处理输出

### 4.2 训练数据

从学生 SFT 数据提取（`prepare_teacher_sft_data.py --role teacher_p`）：

```json
{
  "role": "teacher_p",
  "video": "/path/to/video.mp4",
  "perception_question": "请定位视频中\"dog\"发生的时间区间",
  "result_text": "Dog appears from 12.5s to 18.3s.",
  "observe_type": "temporal_locate",
  "has_video_input": true
}
```

### 4.3 训练方式

标准 SFT（CE loss），看视频，回答感知问题：

```python
messages = [
    {"role": "system", "content": TEACHER_P_SYSTEM_PROMPT},
    {"role": "user", "content": [
        {"type": "video", "video": video_path},
        {"type": "text", "text": perception_question}
    ]},
    {"role": "assistant", "content": result_text}  # 监督目标
]
```

### 4.4 超参

```yaml
model: Qwen3-VL-4B-Instruct（完整模型含视觉塔）
training: 全参数微调
gpus: 8 × H100 96GB, DDP
precision: bf16
lr: 1e-5, cosine + 3% warmup
batch: 1/GPU × 8GPU, grad_acc=2, effective=16
epochs: 1
max_length: 32768（视频 token 较多）
max_frames: 64, fps: 自适应
deepspeed: ZeRO-2
```

---

## 5. Stage1-OPD 核心设计

### 5.1 OPD 训练流程（单个样本）

```
Step 1: Student 潜空间 forward → 得到 [h_1, ..., h_K]
Step 2: Translator 还原每个 h_i → decoded_segments（文本）
Step 3: 按段类型分发到 Teacher_R 或 Teacher_P
Step 4: Teacher teacher-forcing forward → teacher_logits
Step 5: Translator teacher-forcing forward → student_logits（梯度穿过 h_i）
Step 6: 逐 token KL(teacher || student) + aux losses
Step 7: 反向传播，只更新 Student
```

### 5.2 关键设计决策（vs v4 的差异）

| 项 | v4 设计 | v6 实际做法 | 原因 |
|----|---------|-----------|------|
| Decoder | 冻结的 Student 副本，自回归解码 | **Translator**（冻结 LM 主干） | v5 已验证 Translator 有效 |
| Student logits | Decoder forward with latent | **Translator** teacher-forcing（h_i 作为首 token） | 复用 v5 的 Translator 架构 |
| 段类型判定 | Decoder 解码后判定 | **Translator 解码后判定**（相同逻辑） | 换了名字，逻辑不变 |
| 模型大小 | 7B | **4B** | 实际硬件约束 |
| GPU 分配 | Student+Decoder 占 2 卡 | **每模型 1 卡** | 4B 模型单卡放得下 |

### 5.3 OPD 伪代码（完整版）

```python
def opd_training_step(video, question, gt_segments, gt_answer):
    """
    四模型协同：Student(GPU0) + Translator(GPU1) + Teacher_R(GPU2) + Teacher_P(GPU3)
    """
    # ===== Step 1: Student 潜空间 forward =====
    # 复用 v5 的 LatentForwardEngine
    h_0, kv_cache = student_engine.phase1_forward_efficient(
        input_ids, attention_mask, pixel_values_videos, video_grid_thw
    )
    latent_hiddens = student_engine.phase2_serial_forward(h_0, kv_cache, K=len(gt_segments))
    # latent_hiddens: [h_1, ..., h_K]，每个 shape [H]，有梯度

    # ===== Step 2: Translator 还原文本（用于段类型判定 + KL 对齐） =====
    decoded_segments = []
    student_logits_per_seg = []
    
    for i, h_i in enumerate(latent_hiddens):
        # Translator teacher-forcing：用 GT 段文本做 teacher-forcing
        # 同时拿到 student 侧的 logits（梯度穿过 h_i → Student）
        seg_token_ids = tokenize(gt_segments[i])
        student_logits_i = translator.forward_get_logits(
            h_i.to(translator_device),  # h_i 从 GPU0 → GPU1
            seg_token_ids
        )  # shape: [seg_len, vocab_size]
        student_logits_per_seg.append(student_logits_i)
        decoded_segments.append(gt_segments[i])  # 训练时直接用 GT

    # ===== Step 3: 按段类型分发到教师 =====
    kl_losses = []
    prev_context = question  # 累积文本上下文

    for i, seg_text in enumerate(decoded_segments):
        seg_type = classify_segment(seg_text)

        if seg_type in ("REASON", "OBSERVE"):
            # 推理段 → Teacher_R（纯文本，GPU2）
            teacher_logits_i = teacher_r.forward_teacher_forcing(
                context=prev_context,
                target_token_ids=tokenize(seg_text)
            )  # shape: [seg_len, vocab_size]

        elif seg_type == "RESULT":
            # 感知段 → Teacher_P（视觉聚焦，GPU3）
            prev_observe = parse_prev_observe(decoded_segments[:i])
            perception_q = build_perception_question(prev_observe)
            teacher_logits_i = teacher_p.forward_teacher_forcing(
                video=video,
                question=perception_q,
                target_token_ids=tokenize(seg_text)
            )  # shape: [seg_len, vocab_size]

        # ===== Step 4: 逐 token KL =====
        kl_i = F.kl_div(
            F.log_softmax(student_logits_per_seg[i].to(device0) / τ, dim=-1),
            F.softmax(teacher_logits_i.to(device0) / τ, dim=-1),
            reduction="batchmean",
        ) * (τ ** 2)
        kl_losses.append(kl_i)

        prev_context += seg_text  # 累积上下文

    # ===== Step 5: 辅助 losses =====
    l_aux = compute_aux_loss(latent_hiddens[:-1], student_engine.exit_head)
    l_think_end = compute_think_end_loss(latent_hiddens, student_engine.exit_head)
    l_ans = compute_answer_loss(...)  # Phase 3 answer CE

    # ===== Step 6: 总 loss =====
    l_kl = sum(kl_losses) / len(kl_losses)
    total_loss = (
        1.0 * l_kl +          # 教师 KL 蒸馏
        1.0 * l_ans +          # answer CE
        0.1 * l_aux +          # 中间 latent 压低 </think>
        1.0 * l_think_end      # 最后 latent 拉高 </think>
    )

    total_loss.backward()
    # 只有 Student 参数有 grad → optimizer.step() 只更新 Student
```

### 5.4 段类型判定规则

```python
def classify_segment(text: str) -> str:
    s = text.strip()
    if s.startswith("<result>") and s.endswith("</result>"):
        return "RESULT"    # 感知段 → Teacher_P
    if "<observe" in s and s.rstrip().endswith("/>"):
        return "OBSERVE"   # observe 指令 → Teacher_R
    return "REASON"        # 推理/结论段 → Teacher_R
```

### 5.5 Teacher_R 在 OPD 中的 teacher-forcing

```python
class TeacherR:
    """纯文本推理教师，不看视频。"""
    
    def forward_teacher_forcing(self, context: str, target_token_ids: List[int]):
        """
        context: 问题 + 前序所有段文本（含 <result> 内容）
        target_token_ids: 当前段的 token ids
        返回: [seg_len, vocab_size] 的 teacher logits
        """
        # 构造输入：system + context 作为 prefix，target 做 teacher-forcing
        prefix_ids = tokenize(TEACHER_R_SYSTEM + context)
        # 拼接 prefix + target[:-1]，forward 取 target 位置的 logits
        all_ids = concat(prefix_ids, target_token_ids[:-1])
        outputs = self.model(input_ids=all_ids, use_cache=False)
        teacher_logits = outputs.logits[-len(target_token_ids):]
        return teacher_logits.detach()
```

### 5.6 Teacher_P 在 OPD 中的 teacher-forcing

```python
class TeacherP:
    """视觉感知教师，看聚焦视觉输入。"""
    
    def forward_teacher_forcing(self, video, question: str, target_token_ids: List[int]):
        """
        video: 原始视频路径（或 pipeline 预处理后的视觉输入）
        question: 感知问题（由 observe 属性构造）
        target_token_ids: <result>...</result> 的 token ids
        返回: [seg_len, vocab_size] 的 teacher logits
        """
        # 构造多模态输入
        prefix_ids, pixel_values = process_video_input(video, question)
        all_ids = concat(prefix_ids, target_token_ids[:-1])
        outputs = self.model(
            input_ids=all_ids,
            pixel_values_videos=pixel_values,
            use_cache=False
        )
        teacher_logits = outputs.logits[-len(target_token_ids):]
        return teacher_logits.detach()
```

---

## 6. 跨 GPU 通信设计

### 6.1 数据流

```
GPU 0 (Student):
  video + question → Phase 1 → h_0 → Phase 2 → [h_1, ..., h_K]
  h_i ──────────────────────────────────────────────→ GPU 1 (Translator)
  
GPU 1 (Translator):
  h_i + gt_seg_tokens → teacher-forcing → student_logits_i
  student_logits_i ──────────────────────────────────→ GPU 0 (汇总)

GPU 2 (Teacher_R):
  context + seg_tokens → teacher-forcing → teacher_r_logits
  teacher_r_logits ──────────────────────────────────→ GPU 0 (汇总)

GPU 3 (Teacher_P):
  video + perception_q + seg_tokens → teacher-forcing → teacher_p_logits
  teacher_p_logits ──────────────────────────────────→ GPU 0 (汇总)

GPU 0 (汇总):
  KL(student_logits, teacher_logits) → loss → backward → 更新 Student
```

### 6.2 梯度回传路径

```
KL loss (GPU 0)
  → student_logits_i (GPU 1, Translator forward)
    → h_i (GPU 0→1 的 tensor，保留梯度)
      → Student Phase 2 latent_step
        → KV cache (Phase 1 产出，有梯度)
          → Student backbone 全部参数 ✅
```

**关键**：h_i 从 GPU0 传到 GPU1 时用 `.to(device1)` 保留计算图，梯度能回传。

---

## 7. Loss 设计

### 7.1 OPD 总 loss

```python
L_total = λ_kl * L_kl + λ_ans * L_ans + λ_aux * L_aux + λ_think_end * L_think_end

# 默认权重
λ_kl = 1.0          # 教师 KL 蒸馏（核心）
λ_ans = 1.0          # answer CE（保持答案质量）
λ_aux = 0.1          # 中间 latent 压低 </think>
λ_think_end = 1.0    # 最后 latent 拉高 </think>
```

### 7.2 L_kl 的计算

```python
# 对每个段 i:
#   student_logits_i: Translator(h_i, gt_seg_tokens) 的输出 logits
#   teacher_logits_i: Teacher_R 或 Teacher_P 的输出 logits
#   两者 shape 相同: [seg_len, vocab_size]

L_kl = (1/K) * Σ_i KL(
    log_softmax(student_logits_i / τ),
    softmax(teacher_logits_i / τ)
) * τ²

# τ = 2.0（温度，让分布更平滑，KL 信号更丰富）
```

### 7.3 vs v5 SFT 的 loss 对比

| Loss | v5 SFT | v6 OPD | 说明 |
|------|--------|--------|------|
| L_trans | ✅ CE(Translator, GT) | ❌ 移除 | OPD 用 KL 替代 CE |
| L_kl | ❌ 无 | ✅ KL(teacher, student) | OPD 核心 |
| L_ans | ✅ | ✅ | 保持 answer 质量 |
| L_aux | ✅ | ✅ | 不变 |
| L_think_end | ✅ | ✅ | 不变 |

---

## 8. 训练超参

### 8.1 Teacher SFT 阶段

```yaml
# Teacher_R（纯文本，快）
model: Qwen3-VL-4B-Instruct
gpus: 8 × H100, DDP
lr: 1e-5
batch: 1/GPU, grad_acc=2
epochs: 1
max_length: 8192
time: ~2h

# Teacher_P（看视频，慢）
model: Qwen3-VL-4B-Instruct
gpus: 8 × H100, DDP
lr: 1e-5
batch: 1/GPU, grad_acc=2
epochs: 1
max_length: 32768
max_frames: 64
time: ~8h
```

### 8.2 OPD 阶段

```yaml
student_base: v5 SFT checkpoint-7500
translator: v5 Translator（冻结）
teacher_r: Teacher_R SFT checkpoint
teacher_p: Teacher_P SFT checkpoint

gpus: 8 × H100（2 组 × 4 卡/组）
student_lr: 5e-6（比 SFT 低，避免遗忘）
batch: 1/组, grad_acc=4, effective=8
epochs: 2
max_length: 32768
τ (KL temperature): 2.0
λ_kl / λ_ans / λ_aux / λ_think_end: 1.0 / 1.0 / 0.1 / 1.0
```

### 8.3 显存估算（单组 4 卡）

| GPU | 模型 | 参数 bf16 | 优化器 | 激活 | 总计 |
|-----|------|----------|--------|------|------|
| 0 | Student（可训练） | 8.4 GB | 33.6 GB | ~15 GB | ~57 GB |
| 1 | Translator（冻结） | 7.0 GB | 0 | ~5 GB | ~12 GB |
| 2 | Teacher_R（冻结，纯文本） | 8.4 GB | 0 | ~3 GB | ~11 GB |
| 3 | Teacher_P（冻结，视频） | 8.4 GB | 0 | ~10 GB | ~18 GB |

Student 单卡 57 GB < 96 GB ✅（不需要 ZeRO，单卡放得下）

---

## 9. 文件清单

```
# 教师 SFT（新增）
training/teacher_sft.py              # 教师 SFT 训练入口（标准 CE，支持 teacher_r/teacher_p）
scripts/run_teacher_r_sft.sh         # Teacher_R SFT 启动脚本
scripts/run_teacher_p_sft.sh         # Teacher_P SFT 启动脚本

# OPD（新增）
training/stage1_opd.py               # OPD 训练入口（四模型协同）
training/teacher_forward.py          # Teacher_R/P 的 teacher-forcing forward 封装
scripts/run_stage1_opd.sh            # OPD 启动脚本

# 复用 v5
training/latent_forward.py           # Student 串行 latent forward（不变）
training/translator_v5.py            # Translator（冻结，不变）
training/losses.py                   # L_aux + L_think_end + L_ans（不变）
data_preparation/prepare_teacher_sft_data.py  # 教师数据准备（已有）
```

---

## 10. 执行顺序

```
Phase A: 教师 SFT（可并行）
  A1. 生成教师训练数据:
      python -m data_preparation.prepare_teacher_sft_data --role all
  A2. Teacher_R SFT（~2h）:
      bash scripts/run_teacher_r_sft.sh
  A3. Teacher_P SFT（~8h）:
      bash scripts/run_teacher_p_sft.sh
  （A2 和 A3 可以分别用 4 卡并行跑，或者串行用 8 卡跑）

Phase B: OPD
  B1. 准备 OPD 数据（筛选学生做错的样本）:
      python -m data_preparation.stage1_opd_filter
  B2. 运行 OPD 训练:
      bash scripts/run_stage1_opd.sh
```

---

## 11. 与 v4 的关键差异总结

| 项 | v4 | v6 | 原因 |
|----|----|----|------|
| 模型大小 | 7B | 4B | 实际硬件 |
| Decoder | 冻结 Student 副本 | Translator（独立 LM 主干） | v5 验证有效 |
| Student rollout | no_grad 推理 | **有梯度的 latent forward** | v5 验证 KV cache 有梯度很重要 |
| KL 对齐方式 | Decoder(h_i) vs Teacher | **Translator(h_i) vs Teacher** | 复用 v5 架构 |
| GPU 分配 | Student+Decoder 2卡 | 每模型 1 卡 | 4B 单卡放得下 |
| Teacher_R 提示 | 无多步说明 | **明确告知可多步 observe** | 用户要求 |
| 训练时 Student | no_grad rollout → KL | **有梯度 forward → KL** | 梯度必须回传 |

---

## 12. 注意事项

1. **Teacher_R 不看视频**：它是纯文本模型，OPD 时只接收文本上下文
2. **Teacher_P 看聚焦视觉**：SFT 阶段用原始视频近似，OPD 阶段可用 pipeline 预处理
3. **OPD 时 Student 有梯度**：不是 v4 的 no_grad rollout，而是完整的有梯度 forward
4. **Translator 在 OPD 中的双重角色**：
   - 角色 1：还原 h_i 为文本（用于段类型判定）
   - 角色 2：提供 student 侧 logits（用于 KL 对齐）
5. **跨 GPU 梯度**：h_i 从 GPU0 → GPU1 时保留计算图，KL loss 的梯度能回传到 Student
6. **温度 τ=2.0**：让教师分布更平滑，KL 信号更丰富（避免 one-hot 导致 KL 退化为 CE）
