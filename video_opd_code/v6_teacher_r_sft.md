# v6-A: Teacher_R SFT（推理教师）

> 前置: v5 学生 SFT 数据已就绪 | 预计耗时: ~2h（8卡）

---

## 角色定义

- **纯文本推理教师**，不看视频，英文 prompt
- 输入：问题 + 完整 trajectory（**含 `<result>...</result>` 内容**）
- 输出：推理段（`[Analyze]...` / `<observe .../>` / `[Conclude]...`）
- **Loss mask**：只对推理段计算 loss，`<result>...</result>` 区间不计算 loss
- 这样教师能看到感知结果并基于此推理，但不学习预测感知内容

---

## 关键设计变更（vs 旧方案）

| 旧方案 | 新方案 |
|--------|--------|
| `<result>` 替换为 `<result/>` 占位符 | **保留完整 `<result>` 内容** |
| 教师看不到感知结果 | 教师能看到并基于结果推理 |
| prompt 中文 | **prompt 英文** |
| 无 observe type 举例 | **每个 type 都有具体例子** |

---

## System Prompt（英文，含 8 种 observe type 举例）

见 `prompts/teacher_r.txt`，关键内容：
- 列举 8 种 observe type：`temporal_locate`, `temporal_clip`, `spatial_detect`, `spatial_crop`, `tracking_overlay`, `depth_overlay`, `ocr_zoom`, `raw`
- 每种都有 Example + Result 示例
- 明确说明会接收 `<result>` 内容

---

## 数据格式

```json
{
  "question": "At what time does 'a person sneezes' happen?",
  "trajectory": "<think>\n[Analyze] Need to localize...\n<observe type=\"temporal_locate\" target=\"a person sneezes\"/>\n<result>It happens from 0.0s to 13.6s.</result>\n[Conclude] Time span confirmed.\n</think>\n<answer>0.0s-13.6s</answer>",
  "no_loss_spans": [{"start": 95, "end": 140}],
  "has_video_input": false
}
```

`no_loss_spans`：标记 trajectory 中 `<result>...</result>` 的字符位置，训练时这些 token 的 label 设为 -100。

---

## 超参

| 参数 | 值 |
|------|---|
| 模型 | Qwen3-VL-4B-Instruct |
| 训练方式 | 全参数微调 |
| GPU | 8 × H100 96GB, DDP |
| 精度 | bf16 |
| lr | 1e-5, cosine + 3% warmup |
| batch | 1/GPU, grad_acc=2, effective=16 |
| epochs | 1 |
| max_length | 8192（纯文本） |
| gradient_checkpointing | True |

---

## 执行命令

```bash
cd /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/video_opd_code

# 完整流程（数据准备 + 训练 + 推理检查）
bash scripts/run_teacher_sft.sh teacher_r

# 或者先用少量数据过拟合测试
MAX_SAMPLES=10 EPOCHS=5 bash scripts/run_teacher_sft.sh teacher_r
```

---

## 代码文件

| 文件 | 作用 |
|------|------|
| `prompts/teacher_r.txt` | System prompt（英文，含 observe type 举例） |
| `data_preparation/prepare_teacher_sft_data.py` | 数据生成（保留 result，标记 no_loss_spans） |
| `training/teacher_sft.py` | 训练代码（支持 loss mask） |
| `scripts/run_teacher_sft.sh` | 一体化脚本 |

---

## 产出

- checkpoint: `outputs/checkpoints/stage1_sft_teacher_r/final/`
- 用途：OPD 阶段作为冻结推理教师（GPU 2）
