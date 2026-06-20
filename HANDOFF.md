# CodeBuddy 交接说明

> 本文档是完整的项目交接。CodeBuddy请严格按此实现Stage 1全部代码。

---

# 一、项目背景

基于Qwen3-VL-7B的**潜空间视频推理 + 双教师On-Policy Distillation(OPD)**。

当前任务：**完成Stage 1（感知预训练）的完整代码实现**，包括：
1. 8个预处理pipeline
2. Stage 1-SFT数据生成脚本
3. Stage 1-OPD数据筛选脚本
4. Stage 1-SFT训练脚本
5. Stage 1-OPD训练脚本
6. 预实验脚本
7. 所有运行shell脚本
8. README

---

# 二、核心方法

## 2.1 角色

| 角色 | 模型 | 输入 | 状态 |
|------|------|------|------|
| 学生 | Qwen3-VL-7B | 完整长视频+问题 | 可训练 |
| Decoder | Qwen3-VL-7B独立副本 | 学生潜空间hidden | 冻结 |
| Teacher_R | Qwen3-VL-7B | 纯文本（问题+前序解码文本） | 冻结 |
| Teacher_P | Qwen3-VL-7B | 感知问题+视觉聚焦预处理后的视频 | 冻结 |

## 2.2 标签格式（XML，不注册新token）

```xml
<think>
[分析] 简短分析
<observe type="TYPE" time="..." frame="..." bbox="..." objects="..." target="..."/>
<result>感知结果</result>
[结论] 简短结论
</think>
<answer>最终答案</answer>
```

段切换：遇`</observe>`切到感知段，遇`</result>`切回推理段。

## 2.3 8个预处理Type

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

每个pipeline的Python接口统一为：
```python
def pipeline(video_path: str, target: str = "", time: str = None, 
             frame: str = None, bbox: str = None, objects: str = None) -> dict:
    """
    Returns:
        {"video": 处理后视频/帧路径, "perception_question": 给Teacher_P的问题字符串}
    """
```

## 2.4 On-Policy Distillation流程

```
每个训练step:
1. 学生前向: video+question → 潜空间rollout → hidden states
2. Decoder前向: hidden states → student_logits → token序列
3. 按XML标签切段（用utils/parser.py的split_segments）
4. 推理段: Teacher_R(问题+前序解码文本) → teacher_logits
   感知段: 解析<observe>→pipeline→Teacher_P前向 → teacher_logits
5. Loss = Σ KL(teacher_logits || student_logits)
6. 反向传播只更新学生
```

## 2.5 Stage 1-SFT 数据来源

**来源A (20K, 规则模板, 零LLM成本)**：
- A-1: temporal_locate 10K (Charades-STA 5K + ActivityNet 5K)
- A-2: temporal_clip 3K (ActivityNet描述)
- A-3: spatial_detect 3K (VidSTG)
- A-4: spatial_crop 2K (VidSTG反向)
- A-5: tracking_overlay 2K (HC-STVG)

**来源C (10K, Qwen3-32B文本LLM基于已知答案补写推理文字)**：
- C-1: 因果/时序 5K (NExT-QA + STAR选择题)
- C-2: 空间关系 3K (CLEVRER)
- C-3: 时间关系 2K (STAR)

**全部是视频数据集，不用图像数据集。**

## 2.6 Stage 1-OPD 数据筛选

```
对每条(video, question, gt_answer):
  学生直答 → 答对则丢弃
  教师+聚焦答 → 可校验题答错则丢弃; 描述类不校验保留
  保留: 学生失败 ∩ (教师成功 或 不可校验)
```

## 2.7 训练配置

- SFT: cross-entropy, lr=1e-5, epochs=2, batch_size=32
- OPD: on-policy KL, lr=5e-6, epochs=2, batch_size=4/device
- GPU: 4×H20 (每卡96GB), 默认单机八卡但兼容单卡
- 分布式: torchrun / accelerate

---

# 三、已完成的代码

以下文件已写完，**不要修改**：

- `configs/__init__.py`
- `configs/paths.py` — 路径配置（用环境变量，相对路径）
- `utils/__init__.py`
- `utils/parser.py` — XML标签解析（parse_observe, parse_result, split_segments）
- `utils/video_utils.py` — 视频加载/裁切/帧提取

---

# 四、需要实现的代码

## 4.1 pipelines/（8个文件）

每个pipeline一个文件，接口统一。关键要求：
- Grounding-DINO加载用 `configs.paths.GROUNDING_DINO_PATH`
- Depth Anything加载用 `configs.paths.DEPTH_ANYTHING_PATH`
- SAM3如果暂时没有可用版本，先用SAM2（注释标明后续替换）
- 所有外部模型做**lazy loading**（第一次调用时加载，避免import时就占显存）
- 出错直接raise，不做容错

## 4.2 data_preparation/

### stage1_sft_template.py
- 实现来源A的5个模板函数（wrap_temporal_locate等）
- 从各数据集读取标注，批量生成JSONL
- 支持命令行参数指定数据集路径和输出路径

### stage1_sft_llm_augment.py
- 实现来源C：调用Qwen3-32B生成推理文字
- 输入：数据集(NExT-QA/STAR/CLEVRER)的(question, choices, gt_answer)
- 输出：完整轨迹文本
- 格式验证：生成后用parser.py验证格式合规，不合规则重试(max 3次)

### stage1_opd_filter.py
- 实现双过滤pipeline
- 批量推理学生和教师
- 支持断点续跑（已处理的样本跳过）
- 支持多卡推理加速

## 4.3 training/

### stage1_sft_train.py
- 基于Qwen3-VL官方finetune代码修改
- 支持DeepSpeed ZeRO-2/3
- 数据加载：读JSONL，tokenize轨迹文本
- 进度条：tqdm，多卡只主进程打印

### stage1_opd_train.py
- 实现On-Policy Distillation
- 学生rollout → Decoder → 切段 → 双教师前向 → KL loss
- 4卡部署：student+decoder卡0-1，teacher_r卡2，teacher_p卡3
- 支持gradient accumulation

## 4.4 evaluation/

### pre_experiment_focus.py
- 预实验1：视觉聚焦有效性验证
- 在Charades-STA验证集上运行
- 对比：学生直答 vs Teacher_P+聚焦
- 输出：IoU对比表 + 统计摘要

## 4.5 scripts/

所有.sh运行脚本，torchrun启动，参数可配置。

## 4.6 其他

- `requirements.txt` — 完整依赖
- `setup_env.sh` — conda环境一键配置
- `README.md` — 傻瓜式操作指南

---

# 五、代码规范（严格遵守！）

1. **路径**：全部用相对路径或configs.paths中的变量，严禁硬编码绝对路径
2. **错误处理**：有错必须报错，严禁try-except吞异常、严禁容错跳过
3. **多卡**：默认单机八卡(torchrun --nproc_per_node=8)，兼容单卡
4. **进度条**：每个epoch有tqdm，多卡只rank0打印
5. **不用wandb**
6. **模型加载**：从configs.paths读取路径，路径不存在直接FileNotFoundError
7. **数据加载**：从configs.paths读取，不存在直接报错
8. **import**：使用相对导入或从项目根目录导入，确保`python -m xxx`能运行
9. **SAM3**：如果当前没有SAM3的pip包，先用SAM2实现，代码中注释标明
10. **Qwen3-VL**：使用transformers库的Qwen3VL相关类，参考官方示例

---

# 六、服务器环境

- 路径: `/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/`
- Python: 需新建conda环境 (Python 3.11)
- CUDA: 12.4
- GPU: H20 (96GB)
- 模型/数据集都需要下载，README中需写清楚下载命令

---

# 七、数据集下载参考

| 数据集 | 下载方式 |
|--------|---------|
| Charades-STA | https://github.com/jiyanggao/TALL (annotations) + Charades官网(视频) |
| ActivityNet Captions | http://activity-net.org/download.html |
| VidSTG | https://github.com/Guaranteer/VidSTG-Dataset |
| HC-STVG | https://github.com/tzhhhh123/HC-STVG |
| NExT-QA | https://github.com/doc-doc/NExT-QA |
| STAR | https://bobbywu.com/STAR/ |
| CLEVRER | http://clevrer.csail.mit.edu/ |

README中需包含每个数据集的下载命令和期望目录结构。

---

# 八、验收标准

1. 所有代码能`python -c "import video_opd_code"`无报错
2. `pre_experiment_focus.py`能在Charades-STA验证集上跑通
3. `stage1_sft_template.py`能生成格式合规的JSONL
4. `stage1_sft_train.py`能用`torchrun --nproc_per_node=8`启动训练
5. `stage1_opd_train.py`能在100条样本上loss下降
6. README中的每一步操作都是copy-paste可执行的
