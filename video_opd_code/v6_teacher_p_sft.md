# v6-B: Teacher_P SFT（感知教师）

> 前置: v5 学生 SFT 数据已就绪 | 预计耗时: ~8h（8卡）

---

## 角色定义

- **视觉感知教师**，看视觉聚焦输入（pipeline 产物）
- 输入：视频 + 感知问题（由 observe 属性构造）
- 输出：`<result>` 内容（定位结果、描述、坐标等）
- SFT 阶段用原始视频近似，OPD 阶段用真实 pipeline 预处理

---

## 数据格式

从学生 SFT 数据提取（`prepare_teacher_sft_data.py --role teacher_p`）：

```json
{
  "video": "/path/to/video.mp4",
  "perception_question": "请定位视频中\"dog\"发生的时间区间",
  "result_text": "Dog appears from 12.5s to 18.3s.",
  "observe_type": "temporal_locate",
  "observe_attrs": {"type": "temporal_locate", "target": "dog"},
  "has_video_input": true
}
```

每条学生数据的每个 observe-result 对 → 一条 Teacher_P 训练样本。

---

## 训练方式

标准 SFT（CE loss），多模态输入：

```python
messages = [
    {"role": "system", "content": "你是一个视觉感知专家..."},
    {"role": "user", "content": [
        {"type": "video", "video": video_path},
        {"type": "text", "text": perception_question}
    ]},
    {"role": "assistant", "content": result_text}  # 只对此部分计算 loss
]
```

**System prompt**:
```
你是一个视觉感知专家。根据给定的视频内容，精确回答感知问题。
回答要简洁准确，包含具体的时间/坐标/描述信息。
```

---

## 感知问题模板（按 observe type）

| type | 问题模板 |
|------|---------|
| temporal_locate | 请定位视频中"{target}"发生的时间区间 |
| temporal_clip | 这是 {time} 时间段的片段，请描述"{target}"相关内容 |
| spatial_detect | 在 {frame}s 画面中，请定位"{target}"的位置 |
| spatial_crop | 这是 {frame}s 帧 {bbox} 区域裁切，请描述内容 |
| tracking_overlay | 这是 {time} 内对"{target}"的追踪，请描述运动轨迹 |
| raw | 请观察并描述"{target}" |

---

## 超参

| 参数 | 值 |
|------|---|
| 模型 | Qwen3-VL-4B-Instruct（完整模型含视觉塔） |
| 训练方式 | 全参数微调 |
| GPU | 8 × H100 96GB, DDP |
| 精度 | bf16 |
| lr | 1e-5, cosine + 3% warmup |
| batch | 1/GPU, grad_acc=2, effective=16 |
| epochs | 1 |
| max_length | 32768 |
| max_frames | 64, fps: 自适应 |
| deepspeed | ZeRO-2 |

---

## 执行命令

```bash
# 1. 生成数据
python -m data_preparation.prepare_teacher_sft_data --role teacher_p

# 2. 训练
bash scripts/run_teacher_p_sft.sh
```

---

## 产出

- checkpoint: `outputs/checkpoints/stage1_sft_teacher_p/`
- 用途：OPD 阶段作为冻结感知教师（GPU 3）
