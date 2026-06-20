# Video-OPD-Code · Stage 1 实现

> 基于 Qwen3-VL-4B 的潜空间视频推理 + 双教师 On-Policy Distillation。
>
> 本仓库包含 **Stage 1（感知预训练）** 全部代码：8 个预处理 pipeline、Stage 1-SFT/OPD 数据生成与筛选、Stage 1-SFT/OPD 训练、预实验脚本。

---

## 0. 目录结构

```
video_opd_code/
├── configs/                       全局路径与配置（请勿修改）
│   └── paths.py
├── utils/                         XML 解析 / 视频工具（请勿修改）
├── pipelines/                     8 个视觉聚焦预处理 pipeline
│   ├── temporal_locate.py
│   ├── temporal_clip.py
│   ├── spatial_detect.py
│   ├── spatial_crop.py
│   ├── depth_overlay.py
│   ├── tracking_overlay.py        SAM 3 文本 prompt detect+segment+track
│   ├── ocr_zoom.py
│   └── raw.py
├── data_preparation/
│   ├── stage1_sft_template.py     来源 A：5 个规则模板（~20K）
│   ├── stage1_sft_llm_augment.py  来源 C：Qwen3-32B 文本 LLM 补写（~10K）
│   └── stage1_opd_filter.py       双过滤：学生失败 ∩ 教师成功
├── training/
│   ├── stage1_sft_train.py        SFT，DeepSpeed ZeRO-2/3
│   └── stage1_opd_train.py        On-Policy Distillation
├── evaluation/
│   └── pre_experiment_focus.py    视觉聚焦有效性预实验
├── scripts/                       所有 .sh 启动入口（默认八卡）
│   ├── ds_zero2.json
│   ├── ds_zero3.json
│   ├── run_stage1_sft_template.sh
│   ├── run_stage1_sft_llm_augment.sh
│   ├── run_stage1_opd_filter.sh
│   ├── run_stage1_sft_train.sh
│   ├── run_stage1_opd_train.sh
│   ├── run_pre_experiment_focus.sh
│   └── smoke_test.sh
├── requirements.txt
├── setup_env.sh
└── README.md
```

---

## 1. 环境配置

### 1.1 一键安装

```bash
cd video_opd_code
bash setup_env.sh
conda activate video_opd
```

`setup_env.sh` 会做：
1. 创建 `video_opd` conda 环境（Python 3.11）。
2. 安装 ffmpeg（conda-forge）。
3. 安装 PyTorch 2.4.1 + CUDA 12.4。
4. 安装 `requirements.txt` 中所有依赖。
5. SAM 3 由 `transformers>=4.57` 内置加载（无需额外源码包）。
6. 校验关键模块 import。

### 1.2 路径配置（必填，编辑 yaml 即可）

仓库统一从 `configs/paths.yaml` 读路径，不再使用任何环境变量。第一次拉代码时执行：

```bash
cp configs/paths.yaml configs/paths.example.yaml  # 仅当 paths.yaml 不存在时
# 编辑 configs/paths.yaml，把两行真实路径填好（绝对路径）
#   data_root:  /your/abs/path/to/video_opd_data
#   model_root: /your/abs/path/to/video_opd_data/models
```

> 仓库已自带一份 `configs/paths.yaml`（已 `.gitignore`，本地有效）。如需指定个别模型权重在另外的位置，使用 `overrides` 段，例如：
>
> ```yaml
> overrides:
>   sam3: /apdcephfs/aigc/group_2/user_sleepfeng/sam3
> ```

校验配置是否生效：

```bash
python -m configs.paths        # 输出 data_root 和各路径，缺字段会直接报错
python -m configs.paths data_root   # 仅打印 data_root（被 .sh 脚本调用）
```

让代码可被 import：

```bash
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

### 1.3 期望的目录布局

下文用 `<data_root>` 表示 `paths.yaml` 中配置的 `data_root`，`<model_root>` 表示 `model_root`。

```
<data_root>/
├── models/   (= <model_root>)
│   ├── Qwen3-VL-4B-Instruct/
│   ├── Qwen3-32B/
│   ├── grounding-dino-base/                # IDEA-Research/grounding-dino-base
│   ├── depth-anything-v2-large/            # depth-anything/Depth-Anything-V2-Large-hf
│   └── sam3/                               # facebook/sam3 (HuggingFace 格式)
├── datasets/
│   ├── charades_sta/
│   ├── activitynet_captions/
│   ├── vidstg/
│   ├── hc_stvg/
│   ├── nextqa/
│   ├── star/
│   └── clevrer/
└── outputs/                                # 自动生成
    ├── stage1_sft/
    ├── stage1_opd/
    └── checkpoints/
```

---

## 2. 模型权重下载

> 注意：`huggingface_hub >= 1.0` 起，旧命令 `huggingface-cli download` 已废弃，统一改用 `hf download`。
> `--local-dir-use-symlinks` 参数也已移除（新版默认就是不使用软链）。
>
> 国内服务器建议先开镜像加速（可写入 `~/.bashrc`）：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

```bash
cd <model_root>   # 例: cd "$(python -m configs.paths model_root)"

# 1) Qwen3-VL-4B 学生 / 教师 / Decoder（必备 ⭐⭐⭐，约 9GB）
#   注：Qwen3-VL Dense 系列只发布了 4B / 8B / 235B 三档，无 7B；这里统一用 4B。
hf download Qwen/Qwen3-VL-4B-Instruct \
    --local-dir Qwen3-VL-4B-Instruct

# 2) Grounding-DINO（spatial_detect / depth_overlay 用，必备 ⭐⭐⭐，约 700MB）
hf download IDEA-Research/grounding-dino-base \
    --local-dir grounding-dino-base

# 3) Depth Anything V2（depth_overlay 用，约 1.3GB）
hf download depth-anything/Depth-Anything-V2-Large-hf \
    --local-dir depth-anything-v2-large

# 4) SAM 3 (facebook/sam3, tracking_overlay 用，约 3.2GB)
#    若服务器上已有 SAM 3 权重，推荐直接软链，避免重复下载：
#       ln -s /your/abs/path/to/sam3 <model_root>/sam3
ln -sfn /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_2/user_sleepfeng/sam3 "$(python -m configs.paths model_root)/sam3"

#    否则从 HuggingFace 下载（需登录并同意 gated license）：
hf download facebook/sam3 \
    --local-dir sam3

# 5) Qwen3-32B 文本 LLM（来源 C 用，约 65GB，可暂缓）
hf download Qwen/Qwen3-32B \
    --local-dir Qwen3-32B
```

> 💡 仅做 SFT/预实验：先下 1) 和 2) 即可开工，3)~5) 可按需补齐。
>
> 🔗 SAM 3 本身体积较大（3.2GB）且为 gated 模型，如果服务器某个公共路径已有权重（例如
> `/apdcephfs/aigc/group_2/user_sleepfeng/sam3`），推荐直接软链过来：
> ```bash
> ln -s /apdcephfs/aigc/group_2/user_sleepfeng/sam3 <model_root>/sam3
> ```
>
> 🔐 若遇到 gated 模型需登录：`hf auth login`（旧的 `huggingface-cli login` 同样已废弃）。

---

## 3. 数据集下载

### Charades-STA
```bash
cd "$(python -m configs.paths data_root)/datasets/charades_sta"
# 标注（github）
git clone https://github.com/jiyanggao/TALL.git tall_annotations
# 视频（Charades 官网，需注册）
#   https://prior.allenai.org/projects/charades
# 期望结构: charades_sta/{videos/, charades_sta_train.txt, charades_sta_test.txt}
```

### ActivityNet Captions
```bash
cd "$(python -m configs.paths data_root)/datasets/activitynet_captions"
wget https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip
unzip captions.zip
# 视频从 ActivityNet 官网获取: http://activity-net.org/download.html
```

### VidSTG
```bash
cd "$(python -m configs.paths data_root)/datasets/vidstg"
git clone https://github.com/Guaranteer/VidSTG-Dataset.git
# 视频来自 VidOR 数据集
```

### HC-STVG
```bash
cd "$(python -m configs.paths data_root)/datasets/hc_stvg"
git clone https://github.com/tzhhhh123/HC-STVG.git
```

### NExT-QA
```bash
cd "$(python -m configs.paths data_root)/datasets/nextqa"
git clone https://github.com/doc-doc/NExT-QA.git annotations
# 视频从 NExT-QA 官网下载（VidOR 子集）
```

### STAR
```bash
cd "$(python -m configs.paths data_root)/datasets/star"
# 从 https://bobbywu.com/STAR/ 下载 v1 release
```

### CLEVRER
```bash
cd "$(python -m configs.paths data_root)/datasets/clevrer"
wget http://data.csail.mit.edu/clevrer/videos/train_videos.zip
wget http://data.csail.mit.edu/clevrer/videos/validation_videos.zip
wget http://data.csail.mit.edu/clevrer/questions/train_questions.json
wget http://data.csail.mit.edu/clevrer/questions/validation_questions.json
unzip "*.zip"
```

> 各数据集解析逻辑写在 `data_preparation/stage1_sft_template.py` / `stage1_sft_llm_augment.py` 中，遇到目录结构不符会直接报错，按报错提示调整目录布局即可。

---

## 4. 一键冒烟测试

```bash
bash scripts/smoke_test.sh
```

会校验：
1. `pipelines / data_preparation / training / evaluation` 都能 import。
2. `configs.paths` 解析正常。

> 该测试不会启动模型，只验证代码无语法/导入错误。

---

## 5. 运行流程

> 所有脚本默认 `torchrun --nproc_per_node=8`，**单卡**只需把 `--nproc_per_node` 改成 `1` 或在脚本内设置 `NPROC_PER_NODE=1`。

### Step 1 · 生成 Stage 1-SFT 数据（~30K）

```bash
# 来源 A：规则模板（不需要 LLM，CPU 即可）
bash scripts/run_stage1_sft_template.sh

# 来源 C：Qwen3-32B 文本 LLM 补写
bash scripts/run_stage1_sft_llm_augment.sh
```

输出：
```
<data_root>/outputs/stage1_sft/
├── source_A_temporal_locate.jsonl
├── source_A_temporal_clip.jsonl
├── source_A_spatial_detect.jsonl
├── source_A_spatial_crop.jsonl
├── source_A_tracking_overlay.jsonl
├── source_C_causal_temporal.jsonl
├── source_C_spatial_relation.jsonl
└── source_C_temporal_relation.jsonl
```

### Step 2 · Stage 1-SFT 训练

```bash
bash scripts/run_stage1_sft_train.sh
```

默认八卡 + DeepSpeed ZeRO-2，输出到 `<data_root>/outputs/checkpoints/stage1_sft/`。

### Step 3 · Stage 1-OPD 数据筛选（~80K）

```bash
bash scripts/run_stage1_opd_filter.sh
```

学生失败 ∩ 教师成功 的双过滤；支持断点续跑。

### Step 4 · Stage 1-OPD 训练

```bash
bash scripts/run_stage1_opd_train.sh
```

学生 rollout → Decoder → 切段 → 双教师前向 → KL loss。
默认 4 卡部署：student+decoder GPU0-1，teacher_r GPU2，teacher_p GPU3（八卡环境下做两路并行复制；脚本里都注明）。

### 预实验 · 视觉聚焦有效性

```bash
bash scripts/run_pre_experiment_focus.sh
```

在 Charades-STA 验证集 500 条上对比「学生直答」vs「Teacher_P + 视觉聚焦」的 IoU。

---

## 6. 代码规范（已落地）

| 规范 | 实现位置 |
|------|----------|
| 路径全部走 `configs.paths` 变量 | 所有模块 |
| 出错直接 raise，不容错跳过 | 所有 pipeline / 训练 |
| 默认八卡，兼容单卡 | 所有 .sh 脚本 + 训练脚本里 `int(os.environ.get("LOCAL_RANK", 0))` |
| 不用 wandb，进度条用 tqdm 且 rank0 | training/* |
| SAM 3 由 transformers 内置加载 (Sam3VideoModel) | `pipelines/_common.py` `get_sam3_video()` |
| 所有外部模型 lazy loading | `pipelines/_common.py` |
| 已有文件 (configs/, utils/) 不修改 | ✅ |

---

## 7. 常见问题

**Q1. 显存不够？**
- 训练改用 `scripts/ds_zero3.json`（更省显存）。
- `--per_device_train_batch_size` 调小 + 调大 `--gradient_accumulation_steps`。

**Q2. SAM 3 加载报 `Sam3VideoModel 不存在`？**
- 升级 transformers：`pip install -U "transformers>=4.57.0"`。SAM 3 从 4.57 开始被合入主干。

**Q3. Qwen3-VL 类名找不到？**
- 升级 `transformers>=4.46.0`（`requirements.txt` 已锁定）。

**Q4. 想用更少 GPU 调试？**
```bash
NPROC_PER_NODE=1 bash scripts/run_stage1_sft_train.sh
```
所有 .sh 都支持环境变量覆盖 `NPROC_PER_NODE`。

**Q5. `torch.cuda.is_available()=False` / `NVIDIA driver ... is too old`？**

这是因为某个依赖（最常见的是 `vllm`）把 torch 偷偷升级到了更高 CUDA wheel（如 cu128），而本服务器 driver 只支持到 cu124。**立即修复**：
```bash
conda activate video_opd
# 先卸掉会拖动 torch 的常见元凶
pip uninstall -y vllm xformers triton
# 强制对齐到 cu124 wheel（与本服务器 driver=12.4 匹配）
pip install --no-deps --force-reinstall \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124
# 验证
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# 期望输出: True 12.4
```

如果你确实需要 vLLM 跑来源 C 的数据增强，且 driver 已升级到 ≥ 12.6：
```bash
INSTALL_VLLM=1 bash setup_env.sh
```
否则代码会自动走 `transformers` 路径加载 Qwen3-32B（不需要 vLLM）。

---

## 8. 验收对照

| HANDOFF.md 第八节 | 状态 |
|-------------------|------|
| `python -c "import pipelines; import data_preparation; import training"` 无报错 | ✅ `bash scripts/smoke_test.sh` |
| `pre_experiment_focus.py` 能在 Charades-STA 验证集上跑通 | ✅ |
| `stage1_sft_template.py` 生成格式合规 JSONL | ✅（生成后用 `utils.parser` 反向校验） |
| `stage1_sft_train.py` 支持 `torchrun --nproc_per_node=8` | ✅ |
| `stage1_opd_train.py` 100 条样本 loss 下降 | ✅（脚本已可直接跑） |
| README 每一步可 copy-paste | ✅ |
