# 0531 Stage1-SFT 实现方案（v5 最终版）

> Date: 2026-05-31 | 范围: Stage1-SFT only
> 基于论文: VaLR (ICML 2026, Qwen2.5-VL) + Coconut (ICLR 2025) + LT-Tuning (2026.02)
> 不自创设计，严格参考已发表工作

---

## 1. 核心做法：Coconut/VaLR 式串行 latent

**一句话**：Student 在 `<think>...</think>` 区间内，**每步取 last hidden state 直接当下一步的 input embedding**（不走 lm_head 解码为 text token）。每个 hidden state 代表一整段推理/感知内容。Translator 在训练时把 hidden 还原为文字，用于计算 CE loss。

**推理时**：Student 每步取 hidden → lm_head argmax 看是否 `</think>` → 不是则 hidden 作为下一步 input 继续 → 是则切到文本模式输出 answer。

---

## 2. 五个角色（SFT 阶段只用前三个）

| 角色 | 是什么 | 训练状态 | SFT 用？ |
|------|--------|---------|---------|
| **Student** | Qwen3-VL-4B-Instruct 完整模型 | ✅ 全参数微调 | ✅ |
| **Translator** | Qwen3-VL-4B 的 LM 主干（不含视觉塔） | ✅ 全参数微调 | ✅ |
| **Decoder** | Student 自带 lm_head（不是独立模型） | 随 Student 一起训 | ✅ (辅助 loss) |
| Teacher_P | Qwen3-VL-4B-Instruct | ❌ 冻结 | ❌ |
| Teacher_R | Qwen3-VL-4B-Instruct | ❌ 冻结 | ❌ |

**注意**：Student 用 **全参数微调**（不用 LoRA）。理由：VaLR/LT-Tuning/CODI/Heima 在 2025-2026 年全部用全参数。LoRA 只有 2024 年的 CCoT 用过，效果不如全参数。

---

## 3. Token 约定

以下 token 全部是 Qwen3-VL-4B-Instruct **原生词表自带**的，不自定义任何新 token：

| Token | ID | 用途 |
|-------|-----|------|
| `<think>` | 151667 | 思考开始标记（进入 latent mode） |
| `</think>` | 151668 | 思考结束标记（退出 latent mode） |
| `<|im_start|>` | 151644 | 对话轮次开始 |
| `<|im_end|>` | 151645 | 对话轮次结束 |

**不再需要 `<|fim_pad|>` 或任何槽位 token**。latent 位置的 input 是 hidden state（动态计算的），不是 token。

---

## 4. 训练时 latent 生成方式（Coconut/VaLR 原味）

### 核心机制

```python
# 在 <think> 和 </think> 之间，模型进入 latent mode
# 每步：取 last hidden state → 直接当 next input embedding

# 具体实现（训练时）：
for step in range(K):  # K = 这条样本的段数（GT决定）
    if step == 0:
        # 第一步：输入 = [system, user, video, <think>] 的完整 KV cache
        hidden = student.forward_last_hidden(kv_cache)  # shape: [hidden_dim]
    else:
        # 后续步：输入 = 上一步的 hidden（直接当 embedding，不经过 embed 层）
        hidden = student.forward_one_step(prev_hidden, kv_cache)
    
    latent_hiddens.append(hidden)
    prev_hidden = hidden  # 串行传递！
```

**关键点**（来自 Coconut 论文原文）：
> "In latent mode, it directly utilizes the last hidden state as the next input embedding."

- `hidden` 已经过 final LayerNorm，magnitude 不会太大
- 不需要额外 MLP 变换（Coconut 原版不用，VaLR 也不用）
- KV cache 保证效率：每步只 forward 1 个位置

### 训练效率

- K 步串行 forward = 多生成 K 个 token 的开销
- 一般 K = 3~8（Stage 1 数据 K ≤ 5）
- 有 KV cache 时，单步 ≈ 2-5ms (4B 模型 on H100)
- 总额外开销：10-40ms/sample，**比标准 SFT 约慢 1.5x**，完全可接受

---

## 5. 段切分（推理/感知交错）

GT trajectory `<think>...</think>` 之间的内容，按**推理段**和**感知段**交错切分：

```
示例 GT:
<think>
[Analyze] I need to determine when the dog appears in the video.
<observe type="temporal_locate" target="dog"/>
<result>Dog appears from 12.5s to 18.3s.</result>
[Reason] Next I need to see what the dog does during this period.
<observe type="temporal_clip" time="12.5-18.3" target="dog's action"/>
<result>The dog is chasing a ball.</result>
[Conclude] Therefore, the dog is chasing a ball.
</think>
<answer>Chasing a ball</answer>
```

切分结果：

| 段号 | 类型 | 内容 |
|------|------|------|
| latent_1 | 推理段 | `[Analyze] I need to determine...\n<observe type="temporal_locate" target="dog"/>` |
| latent_2 | 感知段 | `<result>Dog appears from 12.5s to 18.3s.</result>` |
| latent_3 | 推理段 | `[Reason] Next I need to see...\n<observe type="temporal_clip" .../>` |
| latent_4 | 感知段 | `<result>The dog is chasing a ball.</result>` |
| latent_5 | 推理段 | `[Conclude] Therefore, the dog is chasing a ball.` |

**切分规则**：
- `<result>...</result>` 独立成一段 = 感知段
- 两个 `<result>` 之间的所有内容合并为一段 = 推理段
- 每段 token 化后 **≤ 256 token**，超出则**报错中断**（不跳过，人工检查）
- 每条样本 1 ≤ K ≤ 32

---

## 6. Student 输入序列（训练时）

**不再有静态槽位 `<|fim_pad|>`！** 改为动态串行 forward：

```
Phase 1（标准 forward 到 <think>）：
  input_ids = [system_tokens, user_tokens, video_tokens, <think>]
  → Student forward → 得到完整 KV cache + 最后位置的 hidden h_0
  ⚠️ h_0 = <think> token 位置的 last hidden state，是"进入 latent mode 的种子"
     h_0 不送 Translator，不算 latent 内容（与 Coconut 的 <bot> token 一致）

Phase 2（latent mode，串行 K 步）：
  step 1: input_embed = h_0           → forward → h_1  ← 第一个真正的 latent
  step 2: input_embed = h_1           → forward → h_2
  ...
  step K: input_embed = h_{K-1}       → forward → h_K
  真正的 latent = h_1 到 h_K（每个对应一段 GT 文字，送 Translator 还原）

Phase 3（回到 text mode）：
  input_ids = [</think>, <answer>, gt_answer_tokens, </answer>, <|im_end|>]
  → Student forward（拼接 Phase 2 的 KV cache）→ 标准 next-token prediction
```

**核心区别 vs 之前的槽位方案**：
- ❌ 不再把 `<|fim_pad|>` 放到 input_ids 里
- ✅ latent 位置的 input 是**前一步的 hidden state**（动态计算的，有信息的）
- ✅ 每个 latent 能 attend 到前面所有 latent 的 KV（串行依赖保留）

---

## 7. 训练流程（详细版）

### 步骤 1：Student latent forward（得到 K 个 hidden）

见 §4 + §6，串行 forward K 步，得到 `h_1, h_2, ..., h_K`。

### 步骤 2：Translator 还原文字（计算 L_trans）

对每个 `h_i`，用 Translator 做 teacher-forcing：

```python
# Translator 输入：
#   位置 0: h_i（来自 Student，直接当 input embedding）
#   位置 1~n-1: embed(GT_segment_tokens[0:n-1])
# Translator 目标：
#   位置 0 → 预测 GT_segment_tokens[0]
#   位置 1 → 预测 GT_segment_tokens[1]
#   ...

L_trans = (1/K) * Σ_i CE(Translator(h_i, GT_seg_i), GT_seg_i)
```

**梯度链**：L_trans → Translator 参数更新 + 梯度穿过 h_i → Student 参数更新。

### 步骤 3：`</think>` 辅助 loss（计算 L_aux）

```python
# 对 h_1 到 h_{K-1}（中间 latent）：
#   压低 Student.lm_head(h_i) 在 </think> 上的概率
# 对 h_K（最后一个 latent）：
#   不需要额外 loss（Phase 3 的标准 CE 已覆盖 </think> 预测）

for i in range(K - 1):
    p_think = softmax(student.lm_head(h_i))[THINK_END_ID]
    L_aux += -log(1 - p_think)
L_aux /= max(K - 1, 1)
```

### 步骤 4：Answer 区标准 CE（计算 L_ans）

Phase 3 中 `</think>` 之后的部分，标准 next-token prediction：

```python
L_ans = CE(student_logits_in_phase3, gt_answer_tokens)
```

### 步骤 5：总 loss

```python
L_total = 1.0 * L_trans + 1.0 * L_ans + 0.1 * L_aux
```

反向传播：
- Student 全部参数：收到 L_trans + L_ans + L_aux 三路梯度
- Translator 全部参数：只收到 L_trans 梯度
- Student.lm_head：收到 L_ans + L_aux 梯度（作为 Student 一部分）

---

## 8. Translator 实现

```python
class Translator(nn.Module):
    """
    Qwen3-VL-4B 的 LM 主干（不含视觉塔），全参数可训练。
    任务：接收 1 个 hidden → teacher-forcing 还原整段文字。
    """
    def __init__(self, model_path):
        self.backbone = load_qwen3_lm_only(model_path)
        self.lm_head = load_qwen3_lm_head(model_path)
        self.embed = load_qwen3_embed_tokens(model_path)
        # 全部 requires_grad = True

    def forward(self, h_i, gt_token_ids):
        """
        h_i: [hidden_dim] 来自 Student latent hidden
        gt_token_ids: [n] GT 段文字 token ids
        """
        gt_embeds = self.embed(gt_token_ids[:-1])        # [n-1, H]
        input_embeds = torch.cat([h_i.unsqueeze(0), gt_embeds], dim=0)  # [n, H]
        
        hidden_states = self.backbone(inputs_embeds=input_embeds)
        logits = self.lm_head(hidden_states)             # [n, vocab_size]
        targets = gt_token_ids                            # [n]
        
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

---

## 9. Decoder（= Student.lm_head）的角色

**不是独立模型**，就是 Student 自带的 lm_head 层。

- **训练时**：对每个中间 latent hidden 算 `</think>` 辅助 loss（§7 步骤 3）
- **推理时**：`argmax(Student.lm_head(h_i))` 判断是否 = `</think>`
  - 不是 → hidden 作为下一步 input embedding，继续 latent forward
  - 是 → 切到 answer 模式，lm_head 自回归输出明文

---

## 10. 推理时流程

```python
# Phase 1: 标准 forward 到 <think>
kv_cache = student.forward([system, user, video, <think>])
hidden = kv_cache.last_hidden

# Phase 2: latent mode
while True:
    logits = student.lm_head(hidden)
    if argmax(logits) == THINK_END_ID:  # </think>
        break
    # hidden 作为下一步 input embedding，继续 latent forward
    hidden = student.forward_one_step(input_embed=hidden, kv_cache=kv_cache)

# Phase 3: text mode（自回归输出 answer）
output_tokens = student.generate(kv_cache, start_token=THINK_END_ID)
# 输出: </think><answer>Chasing a ball</answer>
```

**推理时不用 Translator，不用任何教师。**

---

## 11. 训练超参

```yaml
model: Qwen3-VL-4B-Instruct
training: 全参数微调（不用 LoRA）
gpus: 8 × H100 96GB
distributed: DeepSpeed ZeRO-2（优化器状态分片到 8 卡）
precision: bf16
student_lr: 2e-5, cosine + 3% warmup
translator_lr: 2e-5, cosine + 3% warmup
batch: 2/GPU × 8GPU = 16, grad_acc=2, effective=32
epochs: 2
max_length: 32768
max_seg_len: 256（单段 GT 上限，超出报错）
max_latent_steps: 32（K 上限）
λ_trans / λ_ans / λ_aux: 1.0 / 1.0 / 0.1
gradient_checkpointing: True（Student + Translator 都开）
```

### 显存估算

| 组件 | 参数量 | bf16 参数 | AdamW 优化器 (fp32 m+v) | 合计 |
|------|--------|----------|------------------------|------|
| Student (Qwen3-VL-4B 完整) | ~4.2B | 8.4 GB | 33.6 GB | 42 GB |
| Translator (LM 主干无视觉塔) | ~3.5B | 7.0 GB | 28.0 GB | 35 GB |
| **总可训参数** | ~7.7B | 15.4 GB | 61.6 GB | **77 GB** |

ZeRO-2 下每卡：参数 15.4 + 优化器 61.6/8≈7.7 + 激活 ~8 ≈ **~31 GB/卡** ✅

### DeepSpeed 配置

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "none"},
    "allgather_partitions": true,
    "reduce_scatter": true
  },
  "gradient_clipping": 1.0,
  "bf16": {"enabled": true},
  "gradient_checkpointing": true
}
```

**为什么选 ZeRO-2 而不是 ZeRO-3**：ZeRO-3 参数也分片，串行 latent forward 时每步都要 allgather 参数，通信开销大。ZeRO-2 只分片优化器状态，参数和梯度不分片，串行 forward 时不卡通信。

---

## 12. vs 之前方案的改动总结

| 项 | 之前（v4 修订版） | 现在（v5 最终版） | 改动理由 |
|----|-----------------|-----------------|---------|
| latent 生成 | 静态 `<|fim_pad|>` 槽位，并行 | **串行 hidden → next input** | Coconut/VaLR 验证的唯一可行路线；并行 slot 间无信息传递 |
| Student 训练 | LoRA rank=64 | **全参数微调** | 所有 2025-2026 论文都用全参数（VaLR/LT-Tuning/CODI/Heima） |
| 段间依赖 | ❌ 无（各 slot 独立） | ✅ 有（每步 attend 前面所有 latent 的 KV） | 串行自然保留依赖 |
| 槽位 token | `<|fim_pad|>` (151662) | **不需要任何槽位 token** | latent 位置的 input 是 hidden state，不是 token |
| 训练效率 | 1 次 forward | K 步串行 + KV cache（约慢 1.5x） | KV cache 下额外开销可控 |
| Translator | 全参数（保留） | 全参数（保留，不变） | 需要还原文字做 OPD |
| `</think>` aux loss | 对中间 slot 压低（保留） | 对中间 latent 压低（不变） | 逻辑相同 |

---

## 13. 文件清单

```
training/stage1_sft_v5.py            # 主训练入口（串行 latent forward）
training/latent_forward.py           # Student 串行 latent forward + KV cache
training/translator_v5.py            # Translator 类
training/losses.py                   # L_trans + L_ans + L_aux
scripts/run_stage1_sft_v5.sh         # 启动脚本
```

---

## 14. 参考论文对应关系

| 本方案的设计 | 参考来源 |
|------------|---------|
| last hidden → next input embedding | **Coconut** (Meta, ICLR 2025) 原版做法 |
| 串行 K 步 + KV cache | **VaLR** (ICML 2026, Qwen2.5-VL) 的 latent mode |
| 全参数微调 | **VaLR** + **LT-Tuning** + **CODI** + **Heima** 共同验证 |
| Translator 还原文字 | **Heima** (ICML 2026) 的 Interpreter（独立 LLM 还原 thinking token） |
| `</think>` aux loss | 自有设计（FINAL_PLAN_v4），逻辑与 Coconut 的 `<eot>` 检测类似 |
| 两阶段训练 (Stage 1 SFT → Stage 2 OPD) | **VaLR** 的两阶段 curriculum |
