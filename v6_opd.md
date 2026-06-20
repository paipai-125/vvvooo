# v6-C: Stage1-OPD（四模型协同蒸馏）

> 前置: Student SFT(v5) + Teacher_R SFT + Teacher_P SFT 均已完成
> 预计耗时: ~3天（8卡，2组并行）
> 核心参考: GKD (Google DeepMind, ICLR 2024) — On-Policy Distillation of Language Models

---

## 架构总览

```
GPU 0: Student（可训练）    GPU 4: Student（可训练）
GPU 1: Translator（冻结）   GPU 5: Translator（冻结）
GPU 2: Teacher_R（冻结）    GPU 6: Teacher_R（冻结）
GPU 3: Teacher_P（冻结）    GPU 7: Teacher_P（冻结）
└──── 第 1 组 ────┘        └──── 第 2 组 ────┘
```

两组独立处理不同样本，等效 batch_size × 2。

---

## OPD 核心思想（来自 GKD）

```
标准 OPD = 学生生成轨迹 → 教师在学生轨迹上 teacher-forcing → KL 对齐
```

我们的特殊性：
1. 学生在**潜空间**生成（hidden 序列），需要 Translator 解码为文本
2. **双教师**：推理段用 Teacher_R，感知段用 Teacher_P
3. Teacher_P 的输入**依赖 observe 指令的解析和工具调用**
4. observe 格式错误 → 本条样本学习失败，跳过

---

## 训练流程（单样本，详细版）

```
Step 1: Student 潜空间 forward → [h_1, ..., h_K]（有梯度）

Step 2: 逐段处理（核心循环）
  for i = 1 to K:
    a) Translator 解码 h_i → decoded_text_i
    b) 判断段类型：
       - REASON/OBSERVE → 走 Teacher_R 分支
       - RESULT → 走 Teacher_P 分支（需要工具调用！）
    
    c) 如果是 OBSERVE 段（含 <observe .../>）：
       - 解析 observe 指令（parse_observe）
       - 如果解析失败 → 标记本样本为 FAILED，跳出循环
       - 调用 pipeline 工具处理视频 → 得到聚焦视觉输入
       - 保存给下一个 RESULT 段的 Teacher_P 使用
    
    d) 如果是 RESULT 段：
       - 用上一步 observe 解析得到的聚焦视觉输入
       - Teacher_P teacher-forcing → teacher_logits
    
    e) 如果是 REASON 段（不含 observe）：
       - Teacher_R teacher-forcing（输入=问题+前序所有段文本）→ teacher_logits
       - 注意：Teacher_R 的输入要包含之前的 <result> 内容！
    
    f) Translator teacher-forcing(h_i, GT段) → student_logits
    g) KL(student_logits, teacher_logits)

Step 3: 如果 FAILED → 跳过本样本（不 backward）
Step 4: L_total = L_kl + L_ans + 0.1*L_aux + L_think_end
Step 5: backward → 只更新 Student
```

---

## 关键设计要点

### 1. Teacher_P 输入依赖 observe 解析

```python
# 当 Translator 解码出包含 <observe .../> 的段时：
observe_queries = parse_observe(decoded_text)
if not observe_queries:
    # observe 格式无法解析 → 本样本失败
    return FAILED

# 调用 pipeline 工具处理视频
for obs_q in observe_queries:
    pipeline_result = run_pipeline(obs_q, video_path)
    # pipeline_result = {"video": 处理后路径, "perception_question": str}

# 下一个 RESULT 段时，Teacher_P 用 pipeline_result 作为输入
```

### 2. Teacher_R 输入包含 result 文本

```python
# Teacher_R 的上下文累积：
context = question
for i, seg in enumerate(decoded_segments):
    if classify_segment(seg) == "RESULT":
        context += seg  # 包含 <result>...</result> 内容！
    elif classify_segment(seg) in ("REASON", "OBSERVE"):
        # Teacher_R teacher-forcing 时，输入 = context + 当前段
        teacher_r_logits = teacher_r.forward_tf(context, seg_ids)
        context += seg
```

### 3. 失败处理

```python
# observe 格式无法解析的情况：
# - <observe> 标签格式错误（缺少 type 属性等）
# - observe type 不在支持列表中
# - pipeline 工具调用失败（视频不存在、参数错误等）
# 
# 处理方式：标记为 FAILED，跳过本样本，不计算 loss，不 backward
# 记录失败原因到 wandb，方便后续分析
```

### 4. 训练数据

使用现有 SFT 数据（`stage1_sft_template_all.jsonl`），因为：
- GT trajectory 中的 observe 格式是正确的（模板生成）
- 但学生的 Translator 解码结果可能格式错误 → 自然产生失败样本
- 这正是 on-policy 的意义：学生从自己的错误中学习

---

## 段类型判定

```python
def classify_segment(text: str) -> str:
    s = text.strip()
    if s.startswith("<result>"):   return "RESULT"   # → Teacher_P
    if "<observe" in s:            return "OBSERVE"  # → Teacher_R（含工具调用触发）
    return "REASON"                                  # → Teacher_R
```

---

## Loss 设计

```
L_total = 1.0 * L_kl + 1.0 * L_ans + 0.1 * L_aux + 1.0 * L_think_end
```

| Loss | 含义 | 来源 |
|------|------|------|
| L_kl | KL(Translator(h_i) ‖ Teacher) × τ², τ=2.0 | 每段平均 |
| L_ans | Phase 3 answer CE | 标准 next-token |
| L_aux | 中间 latent 压低 `</think>` | BCE |
| L_think_end | 最后 latent 拉高 `</think>` | CE |

KL 散度选择 **forward KL**（GKD 论文推荐，对 LLM 蒸馏效果最好）：
```python
L_kl = F.kl_div(
    F.log_softmax(student_logits / τ, dim=-1),  # student (Translator output)
    F.softmax(teacher_logits / τ, dim=-1),       # teacher (detached)
    reduction="batchmean"
) * τ²
```

---

## 梯度回传路径

```
L_kl → student_logits (Translator forward, GPU1)
     → h_i (GPU0→1, 保留计算图)
       → Phase 2 latent_step (GPU0)
         → KV cache (Phase 1, 有梯度)
           → Student backbone 全部参数 ✅
```

**关键**：h_i 跨 GPU 时用 `.to(device)` 不 detach，梯度能回传。

---

## 超参

| 参数 | 值 |
|------|---|
| Student base | v5 SFT checkpoint-7500 |
| Translator | v5 冻结 |
| Teacher_R | Teacher_R SFT ckpt，冻结 |
| Teacher_P | Teacher_P SFT ckpt，冻结 |
| GPU | 8卡 = 2组×4卡 |
| student_lr | 5e-6（低于 SFT，防遗忘） |
| batch | 1/组, grad_acc=4, effective=8 |
| epochs | 2 |
| max_length | 32768 |
| τ | 2.0 |
| λ_kl / λ_ans / λ_aux / λ_think_end | 1.0 / 1.0 / 0.1 / 1.0 |

---

## 显存估算（单组 4 卡）

| GPU | 角色 | 估算 |
|-----|------|------|
| 0 | Student（可训练，含优化器） | ~57 GB |
| 1 | Translator（冻结） | ~12 GB |
| 2 | Teacher_R（冻结，纯文本） | ~11 GB |
| 3 | Teacher_P（冻结，视频） | ~18 GB |

全部 < 96 GB ✅

---

## 新增文件

```
training/stage1_opd.py          # OPD 训练主入口
training/teacher_forward.py     # Teacher_R/P teacher-forcing 封装
scripts/run_stage1_opd.sh       # 启动脚本
```

---

## 执行命令

```bash
# 默认配置（使用已有 checkpoint）
WANDB_PROJECT=video-opd-opd WANDB_RUN_NAME=opd-v1 \
bash scripts/run_stage1_opd.sh

# 过拟合测试（少量样本）
MAX_SAMPLES=10 NUM_TRAIN_EPOCHS=10 GRAD_ACCUM=1 \
WANDB_PROJECT=video-opd-opd WANDB_RUN_NAME=opd-overfit \
bash scripts/run_stage1_opd.sh
```
