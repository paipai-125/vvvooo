# Video-OPD 数据集手动下载指南

> 本文是给**你（人类）**看的。Agent 已经多次在 raw URL 上踩坑（404 多到怀疑人生），
> 所以现在的方案是：
>
> 1. **你按本文档手动找到并下载**每个数据集（我只指官方权威入口，不再造 raw 链接）。
> 2. 文件按"放置位置"放到对应目录（路径由 `configs/paths.py` 自动派生）。
> 3. 跑 `bash scripts/postprocess_datasets.sh <key>`（脚本只做解压 + 转 `train.jsonl`，**不再联网**）。
> 4. 跑 `bash scripts/check_datasets.sh` 校验完整性。
>
> 数据根 = `configs/paths.yaml` 里的 `data_root`，下文统一记作 `${DATA_ROOT}`，
> 数据集根 = `${DATA_ROOT}/datasets/`，下文记作 `${DATASETS}`。
> 想查实际值：
> ```bash
> python -m configs.paths data_root
> python -m configs.paths dataset_root
> ```

---

## 总览（9 个数据集，按能力维度对齐）

| # | 数据集 | 能力维度 | verifiable | 视频大小 | 视频是否 YouTube | 难度 |
|---|---|---|---|---|---|---|
| 1 | **Charades-STA** | `temporal_locate` ★ | ✅ | ~13 GB | ❌（AI2 直链） | 🟢 简单 |
| 2 | **STAR** | raw VideoQA（情境推理）| ✅（多选）| 复用 #1 | ❌ | 🟢 |
| 3 | **NExT-QA** | raw VideoQA（因果/时序）| ✅（多选）| ~50 GB | ❌（HF） | 🟡 |
| 4 | **CLEVRER** | raw VideoQA（合成因果）| ✅（多选/数值）| ~46 GB | ❌（MIT 直链） | 🟢 |
| 5 | **HC-STVG v2** | `spatial_detect` / `spatial_crop` / `tracking_describe` ★ | ✅(detect) / ❌(其余) | ~25 GB | ❌（阿里云盘 / 百度网盘） | 🟡 |
| 6 | **DiDeMo** | `temporal_locate` ★（替 ActivityNet）| ✅ | ~6 GB | ❌（YFCC100M 预打包） | 🟢 |
| 7 | **TextVR** | `ocr_zoom` ★ | ✅ | ~20 GB | ❌（仓库自打包，含少量 web 视频） | 🟡 |
| 8 | **VIPSeg** | `spatial_detect` / `spatial_crop` / `depth_overlay` ★（124类全景分割） | ✅ | ~30 GB | ❌（Google Drive） | 🟡 |

> "能力维度"列对应论文里 7 个感知工具（详见 `AGENT_CONSTRAINTS.md` 第 4 节）。

---

## 通用准备

```bash
# 1. 确认 data_root 已配置，且磁盘够（至少 250 GB 余量）
python -m configs.paths dataset_root
df -h "$(python -m configs.paths dataset_root)"

# 2. 创建所有数据集目录（空目录占位，方便你 rsync/cp 进去）
python -c "
from configs.paths import (CHARADES_STA_PATH, STAR_PATH, NEXTQA_PATH, CLEVRER_PATH,
    HC_STVG_PATH, DIDEMO_PATH, TEXTVR_PATH, VIPSEG_PATH)
for p in [CHARADES_STA_PATH, STAR_PATH, NEXTQA_PATH, CLEVRER_PATH,
          HC_STVG_PATH, DIDEMO_PATH, TEXTVR_PATH, VIPSEG_PATH]:
    (p / 'videos').mkdir(parents=True, exist_ok=True)
    print('mkdir:', p)
"
```

如果还没装 `hf` CLI：

```bash
pip install -U "huggingface_hub[cli]"
# 国内访问 HF 慢的话，单独 export 一次镜像（不要写入 paths.yaml）
export HF_ENDPOINT=https://hf-mirror.com
```

> ⚠️ **命令名变更（2025+）**：新版 `huggingface_hub` 已弃用 `huggingface-cli`，统一改为 `hf`。
> 本文档所有命令均使用 `hf download ...`。如果你看到老博客/老 README 写 `huggingface-cli download`，自己替换成 `hf download` 即可，参数完全一致。

---

## 1. Charades-STA — `temporal_locate` ★

**官方入口**
- 数据集主页：<https://prior.allenai.org/projects/charades>（验证有效 2026.05）
- 标注（Charades-STA）官方仓库：<https://github.com/jiyanggao/TALL>
  - 标注 Google Drive 链接在仓库 README 中给出。
  - 备选 HF 镜像：`VLM2Vec/Charades-STA`（仅标注）、`OmniData/Charades-STA`（ModelScope）

**需要的文件**
- 视频：`Charades_v1_480.zip`（~13 GB）— AI2 主页 "Download" → "Charades (scaled to 480p)" 直链
- 标注：`charades_sta_train.txt`、`charades_sta_test.txt`

**放置位置**
```
${DATASETS}/charades_sta/
├── Charades_v1_480.zip         <-- 你下到这里就行，后处理脚本会解压
├── charades_sta_train.txt
└── charades_sta_test.txt
```

**完成判定**：跑 `bash scripts/postprocess_datasets.sh charades_sta` 后
`${DATASETS}/charades_sta/videos/*.mp4` 有约 9848 个文件。

---

## 2. STAR Benchmark — raw VideoQA（复用 Charades 视频）

**官方入口**
- 主页：<https://bobbywu.com/STAR/>
- GitHub：<https://github.com/csbobby/STAR_Benchmark>

**需要的文件**（**不需要再下视频**，软链 Charades 即可）
- `STAR_train.json`、`STAR_val.json`、`STAR_test.json`
- 上述 3 个 JSON 在仓库 release / 主页 "Download" 里都给了 zip。

**放置位置**
```
${DATASETS}/star/
├── STAR_train.json
├── STAR_val.json
├── STAR_test.json
└── videos -> ../charades_sta/videos    (后处理脚本会自动建软链)
```

---

## 3. NExT-QA — raw VideoQA

**官方入口**
- 主页：<https://github.com/doc-doc/NExT-QA>
- 作者主页：<https://doc-doc.github.io/docs/nextqa.html>
- HF 镜像（**推荐，含视频 + csv**）：<https://huggingface.co/datasets/lmms-lab/NExTQA>（验证有效 2026.05）

**一键命令**
```bash
cd /apdcephfs/aigc/group_2/user_sleepfeng/video_opd_code   # 必须先 cd 到代码目录，否则 configs.paths 报 No module
export HF_ENDPOINT=https://hf-mirror.com
hf download lmms-lab/NExTQA \
    --repo-type dataset \
    --local-dir "$(python -m configs.paths dataset_root)/nextqa"
```

**放置位置**
```
${DATASETS}/nextqa/
├── videos/                      (HF 仓库会带过来；如果是 tar/zip 子包，先解压)
├── train.csv
├── val.csv
└── test.csv
```

> 注意：HF 仓库实际目录可能是 `NExTVideo/` 或 `videos.zip`，
> 后处理脚本会 rglob 找到并扁平化。

---

## 4. CLEVRER — raw VideoQA（合成因果）

**官方入口**
- 主页：<http://clevrer.csail.mit.edu/>（验证有效 2026.05）
- 下载页：<http://clevrer.csail.mit.edu/#download>（直链，公开访问）
- GitHub：<https://github.com/chuangg/CLEVRER>

**需要的文件**
- `video_train.zip`（~46 GB），URL 形如 `http://data.csail.mit.edu/clevrer/videos/train/video_train.zip`
- `train.json`，URL 形如 `http://data.csail.mit.edu/clevrer/questions/train.json`
- 可选：`video_validation.zip` + `validation.json`

> ⚠️ 注意：MIT 直链可能限速。建议用 `aria2c -x 16` 或 `wget -c` 多线程/断点续传下载。

**放置位置**
```
${DATASETS}/clevrer/
├── video_train.zip
└── train.json
```

后处理脚本会解压成 `${DATASETS}/clevrer/videos/video_xxxxx.mp4`。

---

## XXXXX5. HC-STVG v2 — `spatial_detect` / `spatial_crop` / `tracking_describe` ★

**官方入口**
- 主页与说明：<https://github.com/tzhhhh123/HC-STVG>
- 视频（v2）：仓库 README 当前给出的是 **OneDrive / 阿里云盘 / 百度网盘** 链接（不同时期作者维护过不同入口；OneDrive 仍可用）。
  - **请直接打开仓库 README 看当前最新链接**，不要依赖本指南中的链接类型——网盘地址会随作者迁移而变化。

**需要的文件**
- 视频：作者把 ~25 GB 视频**切成 10 个独立子包** `0.zip` ~ `9.zip`（每个 ~2~3 GB，**不是分卷，是独立 zip**），解压后是 `videos/<vid>.mp4`
- 标注：`train_v2.json`、`val_v2.json`（仓库 `anno_v2/` 下；进 GitHub 直接点 "Download raw file" 即可）

**放置位置（下载阶段）**
```
${DATASETS}/hc_stvg/
├── 0.zip                        (10 个独立子包，每个 ~2~3 GB)
├── 1.zip
├── ...
├── 9.zip
├── train_v2.json
└── val_v2.json
```

**OneDrive 下载推荐用 rclone**（避免网页端逐个点 + 限速）：
```bash
curl https://rclone.org/install.sh | sudo bash
rclone config        # 配一个 remote 叫 onedrive

TARGET="$(python -m configs.paths dataset_root)/hc_stvg"
mkdir -p "$TARGET"
rclone copy onedrive:HC-STVG-v2/ "$TARGET/" \
    --progress --transfers 8 --multi-thread-streams 4 \
    --retries 10 --low-level-retries 20
```

**解压（关键：逐个 unzip，不要 cat 合并）**：
```bash
TARGET="$(python -m configs.paths dataset_root)/hc_stvg"
cd "$TARGET"
mkdir -p videos

# 解一个删一个，省磁盘
for i in 0 1 2 3 4 5 6 7 8 9; do
    echo "=== Extracting $i.zip ==="
    unzip -o "$i.zip" -d videos/ && rm "$i.zip"
done

# 如果解压后多套了一层目录（如 videos/v2_video/xxx.mp4）则扁平化
cd videos
if [ -d v2_video ]; then mv v2_video/*.mp4 ./ && rmdir v2_video; fi

# 校验
find "$TARGET/videos" -name "*.mp4" | wc -l    # 应 ≈ 5400+
du -sh "$TARGET/videos"                         # 应 ≈ 25 GB
```

**最终结构**
```
${DATASETS}/hc_stvg/
├── videos/<vid>.mp4             (~5400+ 个，约 25 GB)
├── train_v2.json
└── val_v2.json
```

> ⚠️ **不要把 `0.zip ~ 9.zip` 当成分卷压缩**——它们各自是独立完整的 zip，**逐个 unzip** 即可。
> 若文件名是 `xxx.zip.001 / .002`（带数字后缀） 才 是真分卷，需要 `cat *.zip.* > merged.zip` 再解。
>
> ⚠️ 网盘大文件下载经常断。OneDrive 强烈建议 rclone（断点续传 + 多线程）；阿里云盘建议用客户端；百度网盘建议用 BaiduPCS-Go。

---

## 6. DiDeMo — `temporal_locate` ★（替 ActivityNet）

**官方入口**
- 论文与原仓库：<https://github.com/LisaAnne/LocalizingMoments>
- 视频源：YFCC100M 预打包（DiDeMo 论文作者**已经把视频裁好打成单独 mp4 包**，不需要你去 Flickr 现抓）
- HF 镜像（**注意：repo id 容易写错，且不一定都含视频**）：
  - ❌ `lmms-lab/DiDeMo` — **不存在**，2026-05 实测 `401 Repository Not Found`，**别用**
  - ✅ `friedrichor/DiDeMo`（社区维护，**优先尝试**，下完用 `find -name '*.mp4'` 校验）
  - ⚠️ `LisaAnne/LocalizingMoments`（论文作者旧仓，**仅 json 标注，不含视频**）
- 社区镜像（**最稳的视频来源**）：**OpenDataLab** <https://opendatalab.org.cn/OpenDataLab/DiDeMo>

> ⚠️ **修正说明**：DiDeMo 的 HF 镜像大多仅含标注不含视频，且 repo id 五花八门。**实测原仓库 `LisaAnne/LocalizingMoments` 给的 AWS S3 链接和 `download_videos.sh` 已基本失效**，不要走那条路。视频请优先从 OpenDataLab 拉。

**一键命令（HF 路线，先试 friedrichor）**
```bash
cd /apdcephfs/aigc/group_2/user_sleepfeng/video_opd_code
unset HF_ENDPOINT HF_HUB_ENDPOINT     # 这个仓库直连 HF 即可，国内不通再 export mirror
hf download friedrichor/DiDeMo \
    --repo-type dataset \
    --local-dir "$(python -m configs.paths dataset_root)/didemo"
```

如果报 `401 Repository Not Found` → 说明这个 repo 也下线了，直接转 OpenDataLab：

```bash
pip install opendatalab
odl login            # 浏览器/Token 登录
odl get OpenDataLab/DiDeMo -d "$(python -m configs.paths dataset_root)/didemo"
```

下完后**务必**确认是否含 mp4：
```bash
find "$(python -m configs.paths dataset_root)/didemo" -name "*.mp4" | head -5
du -sh  "$(python -m configs.paths dataset_root)/didemo"
```
- 有 mp4 + 总大小约 6 GB → ✅ 完事
- 仅有 json/csv、几十 MB → 视频改走 OpenDataLab：<https://opendatalab.org.cn/OpenDataLab/DiDeMo>

**放置位置**
```
${DATASETS}/didemo/
├── videos/                       (mp4 或 webm)
└── train_data.json | didemo_train.json | train.json   (任意一个皆可)
```

---

## 7. TextVR — `ocr_zoom` ★

**官方入口**
- 论文 & 仓库：<https://github.com/callsys/TextVR>（验证有效 2026.05）
- HF 镜像：`WHB139426/TextVR`（验证存在 2026.05）
- 仓库 README 中通常给出 Google Drive / Baidu Pan 下载链接

**需要的文件**
- 视频：`videos/` 或 `TextVR_videos.zip`（~20 GB）
- 标注：`TextVR_train.json` 或 `train.json`

> ⚠️ **修正说明**：TextVR 视频覆盖 8 个 domain（街景/电影/直播/广告等），**作者已把所有视频统一打包发布**（不是让你逐个去 YouTube 抓）。请优先：
> 1. 走 GitHub README 中作者亲自给的 Google Drive / 百度网盘 完整包链接；
> 2. 若 README 链接失效，再回退 HF 镜像 `WHB139426/TextVR`，但下完后**用 `check_datasets.sh` 核对视频数量**，HF 镜像偶有缺漏。
> 3. **本项目禁用 yt-dlp**，所以即使有个别视频缺失，也用 `train.jsonl` 里 `video_path` 存在性过滤，不要现抓。

**放置位置**
```
${DATASETS}/textvr/
├── videos/                       (或 TextVR_videos.zip，后处理脚本会解压)
└── TextVR_train.json | train.json
```

---

## 9. VIPSeg — `spatial_detect` / `spatial_crop` / `depth_overlay` ★

**官方入口**
- GitHub：<https://github.com/VIPSeg-Dataset/VIPSeg-Dataset>（CVPR 2022）
- 论文：*Large-scale Video Panoptic Segmentation in the Wild: A Benchmark*
- 数据下载：Google Drive（见 GitHub README 中的链接）

**数据集特点**
- 3,536 段真实视频，84K+ 帧
- **124 类物体**（人/动物/车辆/家具/食物/电器/建筑构件/自然物体等）
- 逐帧全景分割标注（每个像素都有类别 + 实例 ID）
- mask 可直接转 bbox，提供精确的物体定位 GT
- 实例级 ID 跨帧一致，天然支持跟踪

**下载步骤**

1. 打开 GitHub 仓库：<https://github.com/VIPSeg-Dataset/VIPSeg-Dataset>
2. 在 README 中找到 Google Drive 下载链接（VIPSeg_720p.zip）
3. 下载并解压：

```bash
# 假设已下载 VIPSeg_720p.zip 到本地
TARGET="$(python -m configs.paths dataset_root)/vipseg"
mkdir -p "$TARGET"

# 解压
unzip VIPSeg_720p.zip -d "$TARGET/"

# 确保目录结构正确：
# $TARGET/videos/         ← 视频帧序列（每个视频一个子目录）
# $TARGET/panoptic_gt/    ← 全景分割标注
# $TARGET/panoVIPSeg_categories.json  ← 类别映射
```

4. 生成 train.jsonl：
```bash
python -m data_preparation.build_all_jsonl vipseg
```

**放置位置**
```
${DATASETS}/vipseg/
├── videos/                        # 视频帧序列目录
│   ├── <video_id_1>/
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   └── ...
│   └── <video_id_2>/
├── panoptic_gt/                   # 全景分割标注
│   ├── <video_id_1>/
│   │   ├── 00001.png
│   │   └── ...
│   └── <video_id_2>/
├── panoVIPSeg_categories.json     # 124 类别映射
└── train.jsonl                    # 由 build_all_jsonl.py 生成
```

> **为什么用 VIPSeg 替代 NYU-Depth-V2？**
> - NYU-Depth-V2 只有静态图（或 428GB raw video 太大），不适合视频理解项目
> - VIPSeg 有 124 类多样物体的精确标注，可同时服务于 spatial_detect / spatial_crop / depth_overlay 三个任务
> - 标注是人工 GT，可作为 Grounding-DINO 的交叉验证基准
---

## 仅标注（视频源 YouTube/VidOR，**主动跳过**）

下面两类标注**不要下载**，已经有平替：
- `ActivityNet Captions` → 由 **Charades-STA + DiDeMo** 替代（同样 `temporal_locate`）
- `VidSTG` → 由 **HC-STVG v2** 替代（spatial 三件套）
- `YouCook2` → 已移除（视频源自 YouTube 无法获取）

如果你以后想补，单独提；本文档不引导。

---

## 推荐下载顺序（先小后大、先易后难）

按下面顺序逐个下，每下完一个就跑 `bash scripts/check_datasets.sh <key>` 确认，再开下一个。
这样就算磁盘只剩 100 GB，至少 P0+P1 的能力维度全覆盖。

### 🟢 P0 — 必下（最小可跑通 SFT，约 70 GB）
- [ ] **#1 Charades-STA**（13 GB）— AI2 直链最稳；标注走 jiyanggao/TALL README 内 GDrive
- [ ] **#2 STAR**（~0，复用 Charades 视频）— 仅下 3 个 JSON
- [ ] **#3 NExT-QA**（50 GB）— `lmms-lab/NExTQA`；若报 `Local entry not found` 看下文 "NExT-QA 错误排查"

### 🟡 P1 — 强烈建议（补全 6 个能力维度，约 73 GB）
- [ ] **#4 CLEVRER**（~46 GB）— MIT 直链限速，建议 `aria2c -x 16 -c` 续传
- [ ] **#6 DiDeMo**（6 GB）— 优先 OpenDataLab，HF 仅作备选
- [ ] **#7 TextVR**（20 GB）— GitHub README 内的 GDrive/百度盘
- [ ] **#8 VIPSeg**（30 GB）— GitHub 仓库内 Google Drive 链接

### 🔴 P2 — 网盘下载，需要耐心（约 25 GB）
- [ ] **#5 HC-STVG v2**（25 GB）— 阿里云盘/百度盘，建议 BaiduPCS-Go

### ⛔ 跳过条件（用户已确认）
- 100 GB 红线：若磁盘紧张，先 P0+P1
- **绝不**用 yt-dlp / youtube-dl 现抓 YouTube 视频

---

## NExT-QA 错误排查（`Local entry not found` / `No module named 'configs'` / `huggingface-cli is deprecated`）— 傻瓜版

> 如果 `hf download lmms-lab/NExTQA ...`（或更老的 `huggingface-cli download ...`）报错，**别想，按下面顺序一条一条粘贴跑**。
>
> 三个最常见的报错先在这里讲清楚：
> - `No module named 'configs'` → **你 cd 错目录了**。必须先 `cd /apdcephfs/aigc/group_2/user_sleepfeng/video_opd_code`（代码目录），不是 `video_opd_data`（数据目录）。
> - `huggingface-cli is deprecated. Use hf instead` → **命令名变了**。把 `huggingface-cli download` 全部换成 `hf download`。
> - `Local entry not found` / `Distant resource does not seem to be on huggingface.co` → 多半是 HF 版本太旧或网络问题，按下面 4 步走。
> 每一步都告诉你"看到什么算成功"、"看到什么继续下一步"。
> 出现成功提示就停，**不必把全部步骤跑完**。

---

### 第 0 步：先把通用环境钉死（必跑）

```bash
cd /apdcephfs/aigc/group_2/user_sleepfeng/video_opd_code
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/hf_cache
mkdir -p /tmp/hf_cache
echo "HF_ENDPOINT = $HF_ENDPOINT"
echo "HF_HOME     = $HF_HOME"
```
**看到什么算成功**：最后两行打印出 `https://hf-mirror.com` 和 `/tmp/hf_cache`。

---

### 第 1 步：用 Python 直接下（90% 情况这步就够了）

完整复制下面这段，**整段**粘到终端按回车（不用改任何路径，脚本自己读 `configs/paths.py`）：

```bash
python <<'PY'
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"  # 老版兼容
from huggingface_hub import snapshot_download
from configs.paths import DATASET_ROOT
target = str(DATASET_ROOT / "nextqa")
print("[NExT-QA] 下载到:", target)
p = snapshot_download(
    repo_id="lmms-lab/NExTQA",
    repo_type="dataset",
    local_dir=target,
    max_workers=4,
    resume_download=True,
)
print("[NExT-QA] DONE ->", p)
PY
```

**看到什么算成功**：
- 进度条一直跑，最后打印 `[NExT-QA] DONE -> /mnt/.../datasets/nextqa`
- → 完事，跳到本节末尾"完成校验"。

**看到什么算失败**（出现以下任一关键词）：
- `Local entry not found` / `Distant resource does not seem to be on huggingface.co`
- `ConnectionError` / `Read timed out`
- `ImportError: snapshot_download`
- → 继续第 2 步。

---

### 第 2 步：升级 huggingface_hub（90% 失败原因是版本太老）

```bash
python -c "import huggingface_hub; print('current version:', huggingface_hub.__version__)"
pip install -U "huggingface_hub[cli]"
python -c "import huggingface_hub; print('after upgrade:', huggingface_hub.__version__)"
```
**看到什么算成功**：`after upgrade` 输出 `>= 0.24`（比如 `0.26.x` 或更高）。

升级完后**回到第 1 步重跑**那段 Python 命令。如果第 1 步又通了 → 完事。

如果**升级时 pip 报错**（比如内网装不了包），跳到第 3 步。

---

### 第 3 步：用 hfd.sh 兜底（最稳，绕开整个 huggingface_hub）

整段粘进终端：

```bash
cd /apdcephfs/aigc/group_2/user_sleepfeng/video_opd_code

# 下载 hfd 工具（一次性，几十 KB）
wget https://hf-mirror.com/hfd/hfd.sh -O /tmp/hfd.sh && chmod +x /tmp/hfd.sh

# 用 hfd 下 NExT-QA
TARGET="$(python -m configs.paths dataset_root)/nextqa"
echo "[hfd] 下载到: $TARGET"
mkdir -p "$TARGET"
HF_ENDPOINT=https://hf-mirror.com /tmp/hfd.sh lmms-lab/NExTQA \
    --dataset \
    --local-dir "$TARGET"
```

**看到什么算成功**：aria2 进度条跑完，最后打印 `Download completed.`。

**看到什么算失败**：`wget` 自己就 404（机器连不上 hf-mirror.com）→ 跳到第 4 步。

---

### 第 4 步：镜像换回直连（仅当机器有外网）

如果机器能直连 huggingface.co，把第 1 步那段 Python 里的两个 endpoint 改一下再跑：

```bash
python <<'PY'
import os
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ.pop("HF_HUB_ENDPOINT", None)
from huggingface_hub import snapshot_download
from configs.paths import DATASET_ROOT
snapshot_download(
    repo_id="lmms-lab/NExTQA", repo_type="dataset",
    local_dir=str(DATASET_ROOT / "nextqa"),
    max_workers=4, resume_download=True,
)
print("DONE")
PY
```

如果走 `hf-mirror` 和走 `huggingface.co` **都失败** → 大概率是机器**整个网络都连不出去**（公司防火墙、VPN 没开），这就不是 hf 的事了，找运维查网络。

---

### 完成后的校验

下完后，确认目录非空：
```bash
ls "$(python -m configs.paths dataset_root)/nextqa" | head -20
du -sh "$(python -m configs.paths dataset_root)/nextqa"
```
应该看到 `train.csv` / `val.csv` / `videos/` 或者 `NExTVideo/` 之类的内容，总大小约 **50 GB**。

---

### TL;DR（懒人版三行）

```bash
cd /apdcephfs/aigc/group_2/user_sleepfeng/video_opd_code
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy && export HF_ENDPOINT=https://hf-mirror.com && export HF_HOME=/tmp/hf_cache && mkdir -p /tmp/hf_cache
python -c "from huggingface_hub import snapshot_download; from configs.paths import DATASET_ROOT; snapshot_download('lmms-lab/NExTQA', repo_type='dataset', local_dir=str(DATASET_ROOT/'nextqa'), max_workers=4, resume_download=True); print('DONE')"
```
跑通就行，跑不通再回去看上面 4 步。

---

## 完成后的统一校验

```bash
bash scripts/postprocess_datasets.sh        # 解压 + 转 train.jsonl（纯本地）
bash scripts/check_datasets.sh              # 列出 OK/WARN/MISS
```

校验脚本期望的最终结构（每个数据集都得满足）：
```
${DATASETS}/<key>/
├── videos/                <非空>
├── <官方标注文件>           <存在>
└── train.jsonl            <存在；CLEVRER/NExT-QA/HC-STVG/DiDeMo/TextVR 由后处理生成>
```

---

## FAQ

**Q: 我能不能只下一部分先跑通 SFT？**
A: 能。最小集 = `charades_sta + nextqa + didemo`（约 70 GB），
覆盖 `temporal_locate` + raw VideoQA，足够先把 stage1-SFT pipeline 跑通。

**Q: HF 下载断了怎么办？**
A: `hf download` 自带断点续传，重跑同一条命令即可。

**Q: 磁盘只剩 100 GB 怎么取舍？**
A: 跳过 CLEVRER 的视频（46 GB；保留标注，视频后补），其他全留。

**Q: HF 命令是 `hf download` 还是 `huggingface-cli download`？**
A: **必须用 `hf download`**。新版 `huggingface_hub`（默认 pip 安装的版本）已**正式废弃** `huggingface-cli`，跑老命令会直接报 `huggingface-cli is deprecated and no longer works. Use hf instead.`。如果你看到老博客或别的 AI 给的命令是 `huggingface-cli download`，把它改成 `hf download` 再跑就行，参数完全一致。

---

## 修正日志

| 日期 | 修正项 |
|------|--------|
| 2026-05-27 | 🗑️ **移除 YouCook2**：视频源自 YouTube 无法获取，彻底从项目中移除；`temporal_clip` 能力由 Charades-STA + DiDeMo 的时间段描述补位 |
| 2026-05-26 | 🔄 **NYU-Depth-V2 → VIPSeg**：移除 NYU-Depth-V2（静态图/428GB raw video 不适合视频理解），改用 VIPSeg（CVPR 2022，3536 视频，124 类全景分割，精确 GT bbox）；同时服务 spatial_detect / spatial_crop / depth_overlay 三个任务 |
| 2026-05-26 | ⚠️ HC-STVG v2 重大修正：实际发布形式是 `0.zip ~ 9.zip` **10 个独立子包**（不是单个 `v2_video.zip`，也不是分卷压缩）；补充 rclone 下 OneDrive + 逐个 unzip + 扁平化完整命令；明确提醒不要 cat 合并
| 2026-05-26 | 🔴 DiDeMo 重大修正：`lmms-lab/DiDeMo` 实测返回 `401 Repository Not Found`（仓库不存在），改为 `friedrichor/DiDeMo`；明确标记 `LisaAnne/LocalizingMoments` 仅有 json 标注；补充 OpenDataLab `odl` cli 一键命令 |
| 2026-05-25 | 🔄 全文 `huggingface-cli download` → `hf download`（新版 huggingface_hub 已正式废弃 huggingface-cli）；FAQ "HF 命令"问答结论修正；NExT-QA 排查节加上 `No module named 'configs'` / `huggingface-cli is deprecated` 两个常见错的判定；DiDeMo 节加视频存在性校验；NYU 节加官方 wget 直链兜底 |
| 2026-05-25 | 🔄 NExT-QA 错误排查：重写为"傻瓜版"——4 步骤，每步都给完整粘贴命令、成功/失败判定，附 TL;DR 三行版 |
| 2026-05-25 | ⚠️ HC-STVG v2：OneDrive → 阿里云盘/百度盘（最新仓库已迁移） |
| 2026-05-25 | ⚠️ DiDeMo：明确原 AWS S3 链接已失效，OpenDataLab 提为首选 |
| 2026-05-25 | ⚠️ TextVR：澄清视频是作者自打包（非 YouTube 现抓），补全失效兜底 |
| 2026-05-25 | ⚠️ NYU-Depth-V2：恢复 `~silberman/` 为权威主页（`~fergus/` 是早期转载） |
| 2026-05-25 | ➕ 新增 "推荐下载顺序" P0/P1/P2 checklist，可勾选跟踪进度 |
| 2026-05-25 | ➕ 新增 "NExT-QA 错误排查" 节，处理 `Local entry not found` |
| 2026-05-25 | ⚠️ NYU-Depth-V2：修正大小（标注版 2.8 GB vs raw 428 GB），修正 URL（`~fergus` 非 `~silberman`），补充 HF 方案 |
| 2026-05-25 | ⚠️ DiDeMo：补充 HF 可能不含视频的风险，添加原仓库脚本下载方案 |
| 2026-05-25 | ⚠️ TextVR：补充 HF 镜像为 `WHB139426/TextVR`，提示可能需要百度网盘 |
| 2026-05-25 | ⚠️ CLEVRER：补充下载限速提醒和断点续传建议 |
| 2026-05-25 | 修正 HF CLI 命令为 `huggingface-cli download`（非 `hf download`） |
