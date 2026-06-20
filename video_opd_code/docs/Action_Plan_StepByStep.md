# 行动清单

---

# 第1步：环境部署

部署所有模型和工具：

| 工具 | 用途 |
|------|------|
| Qwen3-VL-7B | 学生+教师+Decoder |
| Grounding-DINO | spatial_detect |
| SAM3 | tracking_overlay |
| Depth Anything V2 | depth_overlay |
| ffmpeg | temporal_clip |
| OpenCV | temporal_locate |
| PIL | spatial_crop, ocr_zoom |
| Qwen3-32B（文本） | Stage 1来源C轨迹生成 |
| Gemini-2.5-Pro API | Stage 2轨迹生成 |

**产出**：每个工具有test脚本验证可用。

---

# 第2步：实现8个预处理pipeline

```
pipelines/
├── temporal_locate.py    # 完整视频+时间轴标记
├── temporal_clip.py      # ffmpeg裁切
├── spatial_detect.py     # Grounding-DINO全物体高亮
├── spatial_crop.py       # PIL裁切+zoom
├── depth_overlay.py      # Grounding-DINO+Depth Anything
├── tracking_overlay.py   # SAM3 tracking可视化
├── ocr_zoom.py          # PIL裁切+放大
└── raw.py               # 不处理
```

每个pipeline接口统一：`def pipeline(video, target, time, frame, bbox, objects) → (processed_video, perception_question)`

**产出**：8个pipeline + 调度器。

---

# 第3步：预实验1 — 视觉聚焦有效性 ⭐必过

- Charades-STA验证集500条
- 对比：学生直答 vs Teacher_P+聚焦
- **必须**：Teacher_P IoU > 学生 + 5%
- 不通过 → 停下重新设计pipeline

---

# 第4步：Stage 1-SFT 数据准备（~30K）

```
来源A 规则模板 (~20K):
  A-1 temporal_locate:    10K  (Charades-STA 5K + ActivityNet 5K)
  A-2 temporal_clip:       3K  (ActivityNet 描述)
  A-3 spatial_detect:      3K  (VidSTG)
  A-4 spatial_crop:        2K  (VidSTG 反向)
  A-5 tracking_overlay:    2K  (HC-STVG)

来源C Qwen3-32B补写 (~10K):
  C-1 因果/时序:           5K  (NExT-QA + STAR)
  C-2 空间关系:            3K  (CLEVRER)
  C-3 时间关系:            2K  (STAR)
```

- 来源A：纯Python模板脚本，零LLM成本
- 来源C：Qwen3-32B基于已知答案补写a1/a3推理文字（不做推理，只做文字编排）
- 不用图像数据集，不用人工标注，不用Gemini

**产出**：`stage1_sft.jsonl`

---

# 第5步：Stage 1-SFT 训练

```yaml
base: Qwen3-VL-7B-Instruct
data: stage1_sft.jsonl (30K)
loss: cross-entropy
lr: 1e-5, epochs: 2
gpus: 4× H100, ~8小时
```

**检查**：50条holdout上格式合规率 > 90%

---

# 第6步：Stage 1-OPD 数据筛选（~80K）

候选池~200K（Charades-STA + ActivityNet + VidSTG + HC-STVG + NExT-QA + STAR + CLEVRER）

```
对每条:
  学生直答 → 答对则丢弃
  教师+聚焦答 → 答错(可校验)则丢弃 / 描述类不校验直接保留
```

**产出**：`stage1_opd.jsonl`（~80K）

---

# 第7步：OPD训练框架实现

核心逻辑：
1. 学生rollout → Decoder → student_logits + token序列
2. 按XML标签切段
3. 推理段→Teacher_R前向，感知段→解析observe→pipeline→Teacher_P前向
4. 逐段KL loss → 反向传播只更新学生

**预实验**：100条样本跑20步，loss下降合理。

---

# 第8步：Stage 1-OPD 全量训练

```yaml
base: student_stage1_sft.pt
data: stage1_opd.jsonl (80K)
loss: on-policy KL
lr: 5e-6, epochs: 2
gpus: 4× H100, ~3天
```

**检查**：感知IoU > base + 5%

---

# 第9步：Stage 2-SFT 数据准备（~30K）

- 用Gemini-2.5-Pro生成多步轨迹（1-4次observe）
- 解析每个`<observe>`用Teacher_P填`<result>`占位符
- 最终`<answer>`必须匹配GT
- 数据源：NExT-QA + STAR + Video-R1选择题 + LongVideo-Reason + VideoEspresso

**产出**：`stage2_sft.jsonl`

---

# 第10步：Stage 2-SFT 训练

```yaml
base: student_stage1_opd.pt
data: stage2_sft.jsonl (30K)
loss: cross-entropy
lr: 1e-5, epochs: 2
```

**检查**：多步格式合规 > 90%

---

# 第11步：Stage 2-OPD 数据筛选（~80K）

同Stage 1思路。数据源：NExT-QA + STAR + CLEVRER + Video-R1 + LongVideo-Reason选择题部分。

按observe次数分类：Easy/Medium/Hard。

---

# 第12步：Stage 2-OPD 训练（课程学习）

```yaml
base: student_stage2_sft.pt
data: stage2_opd.jsonl (80K)
sampling:
  Phase 2.1: Easy 70% / Med 25% / Hard 5%
  Phase 2.2: Easy 40% / Med 40% / Hard 20%
  Phase 2.3: Easy 30% / Med 35% / Hard 35%
lr: 5e-6
gpus: 4× H100, ~5天
```

**产出**：`student_final.pt`

---

# 第13步：评估

测试集：LVReason, Video-Holmes, MLVU, NExT-QA, STAR, LVBench, VideoMME, MVBench

---

# 第14步：消融实验

| 消融 | 验证 |
|------|------|
| w/o Stage 1-OPD | 感知OPD是否必要 |
| w/o Stage 2-OPD | 推理OPD是否必要 |
| w/o 视觉聚焦 | Teacher_P聚焦是否有效 |
| 单教师 vs 双教师 | 双教师设计是否有效 |
| 显式CoT vs 潜空间 | 潜空间是否优于显式 |
| w/o 课程学习 | 课程学习是否必要 |

---

# 关键里程碑

| 时间 | 标准 | 不达标 |
|------|------|--------|
| 第3步 | Teacher_P IoU > 学生+5% | 重设计pipeline |
| 第5步 | 格式合规 > 90% | 加SFT数据 |
| 第8步 | 感知IoU > base+5% | 检查教师质量 |
| 第10步 | 多步格式 > 90% | 检查Gemini生成 |
| 第13步 | LVReason > base+3% | 加数据/调参 |

---

# 时间线（8周）

```
Week 1:   第1-3步（环境+pipeline+预实验1）
Week 2:   第4-5步（Stage 1-SFT）
Week 3-4: 第6-8步（Stage 1-OPD）
Week 5-6: 第9-10步（Stage 2-SFT）
Week 7:   第11-12步（Stage 2-OPD）
Week 8:   第13-14步（评估+消融）
```
