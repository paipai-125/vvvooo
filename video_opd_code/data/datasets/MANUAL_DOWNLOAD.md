# Video-OPD 数据集手动下载指南（已迁移）

> ⚠️ **本文档已废弃**。请直接看项目根目录下的最新版：
>
> 👉 [`/apdcephfs/aigc/group_2/user_sleepfeng/video_opd_code/Video-OPD数据集手动下载指南.md`](../../Video-OPD数据集手动下载指南.md)
>
> 新版已修正以下不准确之处（2026-05-25）：
> - HC-STVG v2 视频源：OneDrive → 阿里云盘 / 百度网盘
> - DiDeMo 视频：原 AWS S3 直链已失效，OpenDataLab 为首选
> - TextVR 视频：作者自打包（非 YouTube 现抓）
 > - NYU-Depth-V2 已移除，改用 VIPSeg（124类全景分割）
> - 新增 P0/P1/P2 优先级 checklist
> - 新增 NExT-QA `Local entry not found` 错误排查节
>
> 本目录（`data/datasets/`）下放置数据集软链或临时下载文件，
> 实际数据落盘位置由 `configs/paths.py` 派生（参见 `configs/paths.yaml: data_root`），
> 通常在 `${VIDEO_OPD_DATA}/datasets/<key>/` 下。
