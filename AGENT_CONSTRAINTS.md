# AGENT_CONSTRAINTS — Agent 必读备忘录

> 用途：避免 Agent 在长对话中遗忘用户已多次申明的硬约束。
> 每次新一轮回答前先按本文档自检；与最新指令冲突时以最新指令为准并立即回写。

## 0. 项目身份
- 项目根目录：`/apdcephfs/aigc/group_2/user_sleepfeng/video_opd_code`
- 数据/模型根：`/mnt/gemininjceph3/.../user_sleepfeng/video_opd_data`（由 `configs/paths.py` 注入，**不让用户敲 ${VIDEO_OPD_DATA}**）
- conda 环境名：`video_opd`

## 1. 模型选型（已锁死）
- 学生 / 教师 / 解码器：**Qwen3-VL-4B-Instruct**
- 文本 LLM：**Qwen3-32B**（不是 32B-Instruct，不是 72B）
- 分割 / 跟踪：**SAM 3**（不是 SAM 2.1）；权重在 `/apdcephfs/aigc/group_2/user_sleepfeng/sam3`，已软链
- 检测：Grounding-DINO base
- 深度：Depth Anything V2 large
- 3D 物体朝向：**Cube R-CNN** (DLA34 backbone, 366MB) ✅ 已下载 → `models/cube-rcnn/cubercnn_DLA34_FPN.pth`
- 相机视角估计：**PerspectiveFields** (798MB) ✅ 已下载 → `models/perspective-fields/paramnet_360cities_edina_rpf.pth`

## 2. 训练范式
- 学生输出**潜空间特征**，由**冻结的 Qwen3-VL** 自回归解码成文本（自问自答式）。
- 学生包含 `<think>` 内 `[Analyze]` / `<observe>` / `<result>` / `[Conclude]` + `<answer>` 全部内容。
- SFT 与 OPD 都监督**文本输出**。
- Teacher_R 与 Teacher_P 的输入提示词与学生**不同**：
  - 学生：question → 自己生 observe/result/answer
  - Teacher_R：question + 工具结果 → 推理链
  - Teacher_P：**不输入原始问题**，只输入工具产物（裁剪/深度图/检测可视化）→ 描述

## 3. 数据硬约束（最常被遗忘）
### 语言
- **训练数据全英文**（question/trajectory/gt_answer/params）。论文投稿要求。
- 文档/注释/对话仍可中文（仅 jsonl 必须英文）。

### 验证性
- **不要软奖励**。
- OPD 阶段：要么 `verifiable=True`（可机械验证），要么不参与奖励，二选一。
- "学生已会 → 丢" 的过滤**只对 verifiable=True 的样本**生效。
- 7 个工具的 verifiable：
  - ✅ temporal_locate / spatial_detect / ocr_zoom / depth_overlay
  - ❌ temporal_clip / spatial_crop / tracking_overlay

### 物体指代方式（VIPSeg 空间任务）
- **统一用 bbox 坐标指代物体**：`"the object at [120,80,200,160]"`
- **不需要类别唯一性约束**：即使画面中有多个同类物体（如 3 个杯子），bbox 天然唯一
- **不使用位置消歧**（如 "the leftmost cup"）：bbox 更精确、更通用
- 问题模板示例：`"Which is closer to the camera, the object at [x1,y1,x2,y2] or the object at [x1,y1,x2,y2]?"`
- gt_answer 中可附带类别名辅助理解：`"The object at [120,80,200,160] (cup) is closer."`

### 数据来源
- A 类：必须**有视频 + 有标注**才下；只有标注没视频→跳过。
- B 类：NExT-QA / STAR / CLEVRER 等 raw VideoQA。
- C 类：Qwen3-32B 文本增强。
- **禁止从 YouTube 下载**（禁用 youtube-dl/yt-dlp）。
- A-1 用 Charades-STA + DiDeMo（替 ActivityNet）。
- A-2 temporal_clip 用 **Charades-STA + DiDeMo**（YouCook2 已彻底否定移除）。
- A-3/A-4 用 **VIPSeg**（主力，124类通用物体 bbox）+ HC-STVG v2（补充，人的跟踪+自然语言描述）。
- A-5 tracking_describe 用 **VIPSeg**（跨帧 instance_id 追踪）+ HC-STVG v2（人的轨迹描述）。
- A-6 depth_overlay 用 **VIPSeg**（124类全景分割，bbox 指代物体）。
- A-8 raw_videoqa 用 **NExT-QA + STAR + CLEVRER**（B类原始视频问答，多选题可验证）。

### 数据量
- 100 GB 以内随便下；100+ GB 找 caption 子集或公开子集。
- 困难数据集**不跳过**，标记"稍后处理"。

## 4. 七个感知工具
| 工具 | 输入 | 输出 | verifiable |
|---|---|---|---|
| temporal_locate | query | [t1,t2] | ✅ |
| temporal_clip | [t1,t2] | 段描述 | ❌ |
| spatial_detect | frame + referring | bbox | ✅ |
| spatial_crop | frame + bbox | 区域描述 | ❌ |
| tracking_overlay | [t1,t2] + target(+起止 bbox 可选) | 轨迹描述 | ❌ |
| depth_overlay | frame + objects[] | 谁更近/远 | ✅ |
| ocr_zoom | frame + bbox | 文本 | ✅ |
| raw_videoqa (A-8) | video + question + choices | 选择/描述 | ✅（多选题） |

A-5 `tracking_overlay` 设计：不监督正确性，pipeline 中 SAM3+DINO 给框/高亮作为 Teacher_P 视觉提示。

## 5. 工程纪律
1. **不擅自改用户代码**：用户只是问问题/讨论方案时，不调 edit_file。
2. **不主动创建文档**：除非用户明确要。本备忘录是允许的唯一例外。
3. **改 URL/镜像前先核实可达**（404/500 都不算可达）。
4. 路径统一走 `configs/paths.py` + `configs/paths.yaml`，不让用户 export 环境变量。
5. Stage1-SFT 7 个 wrap_* 函数的 question/trajectory/params 必须全英文。
6. OPD `filter_one()` 第一行先判 `verifiable`，False 直接保留不跑学生 forward。
7. SAM3 用 `Sam3VideoModel` + `Sam3VideoProcessor`，不要混用 SAM 2.1 API。
8. 批处理任务：每个数据集/文件一个独立 todo，做完一个划掉一个。

## 6. 数据集实际状态（2026-05-26 核实）

### ✅ 完整可用
| 数据集 | 视频数 | 标注 | 用途 |
|--------|--------|------|------|
| Charades-STA | 9848 | train.json + test.json | A-1 temporal_locate |
| DiDeMo | 9399 | train/val/test .json | A-1 temporal_locate |
| CLEVRER | 10000 | train.json + train.jsonl | B类 VideoQA（因果/反事实） |
| TextVR | 10596 | TextVR_train.json + OCR标注 | A-7 ocr_zoom |
| STAR | 复用Charades视频 | STAR_train/val/test.json | B类 VideoQA |
| NExT-QA | 1570（覆盖23085条QA） | train.jsonl + parquet | B类 VideoQA |
| VIPSeg | 2806 mp4（2fps合成） | panomasks + categories | A-3/A-4 spatial + A-6 depth_overlay |
| HC-STVG v2 | 8235 | train.jsonl (5032条) + videos/ | A-3/A-4/A-5 spatial+tracking |

### ⏳ 下载/解压中
（无）


## 7. 当前已知遗留
- [x] `scripts/download_datasets.sh` 的自动下载策略已弃用 → 改为手动 + `Video-OPD数据集手动下载指南.md`（2026-05-25）
- [x] HC-STVG v2 视频源：OneDrive → 阿里云盘/百度网盘（指南已修正）
- [x] NExT-QA：已完整（1570视频 + 23085条QA + train.jsonl）
- [x] DiDeMo：HF `lmms-lab/DiDeMo` 下载完成（9399视频）
- [x] TextVR：HF `WHB139426/TextVR` 下载完成（10596视频）
- [x] VIPSeg：已完成（2806视频 mp4 + panomasks + categories），需运行 `build_all_jsonl.py vipseg` 生成 train.jsonl
- [x] YouCook2：**已彻底否定**，从项目中移除
- [x] HC-STVG v2：已完成（8235视频 + 5032条 train.jsonl）
