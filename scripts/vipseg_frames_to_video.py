#!/usr/bin/env python3
"""
VIPSeg 帧序列 → MP4 视频合成脚本

将 VIPSeg 数据集中的帧序列（jpg）合成为 2fps 的 mp4 视频。
这样 Qwen3-VL 可以直接以视频模式输入，并通过时间戳精确定位每一帧。

合成规则：
- fps = 2（每帧间隔 0.5 秒）
- 帧按文件名排序后顺序合成
- 第 N 帧（0-indexed）的时间戳 = N * 0.5 秒
- 输出为 H.264 编码的 mp4 文件

用法：
  cd /path/to/video_opd_code
  python scripts/vipseg_frames_to_video.py                    # 处理全部
  python scripts/vipseg_frames_to_video.py --split train      # 只处理 train split
  python scripts/vipseg_frames_to_video.py --workers 8        # 8 进程并行
  python scripts/vipseg_frames_to_video.py --dry-run          # 只打印命令不执行
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# 路径配置（从 configs.paths 动态读取，兼容不同环境）
# ============================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.paths import VIPSEG_PATH  # noqa: E402

VIPSEG_ROOT = VIPSEG_PATH
IMGS_DIR = VIPSEG_ROOT / "videos"          # 软链接 → VIPSeg/imgs
OUTPUT_DIR = VIPSEG_ROOT / "videos_mp4"    # 合成视频输出目录

# 合成参数
FPS = 2  # 合成帧率：2fps，每帧 0.5 秒


def get_split_videos(split: str) -> list:
    """读取指定 split 的视频 ID 列表"""
    split_file = VIPSEG_ROOT / f"{split}.txt"
    if not split_file.exists():
        print(f"[ERROR] split 文件不存在: {split_file}")
        sys.exit(1)
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


def convert_one_video(video_id: str, dry_run: bool = False) -> dict:
    """
    将单个视频目录的帧序列合成为 mp4。
    
    返回: {"video_id": str, "status": "ok"|"skip"|"error", "n_frames": int, "duration": float, "msg": str}
    """
    frames_dir = IMGS_DIR / video_id
    output_path = OUTPUT_DIR / f"{video_id}.mp4"
    
    # 跳过已存在的
    if output_path.exists():
        return {"video_id": video_id, "status": "skip", "n_frames": 0, "duration": 0, "msg": "已存在"}
    
    # 检查帧目录
    if not frames_dir.exists():
        return {"video_id": video_id, "status": "error", "n_frames": 0, "duration": 0, "msg": f"帧目录不存在: {frames_dir}"}
    
    # 获取排序后的帧文件列表
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.endswith('.jpg') or f.endswith('.png')
    ])
    
    if len(frame_files) < 2:
        return {"video_id": video_id, "status": "error", "n_frames": len(frame_files), "duration": 0, "msg": "帧数不足"}
    
    n_frames = len(frame_files)
    duration = n_frames / FPS  # 视频总时长（秒）
    
    if dry_run:
        return {"video_id": video_id, "status": "ok", "n_frames": n_frames, "duration": duration, "msg": f"[dry-run] {n_frames} 帧 → {duration:.1f}s"}
    
    # 使用 ffmpeg concat demuxer 方式合成（支持非连续帧名）
    # 创建临时文件列表
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, prefix='vipseg_') as tmp:
            for frame_file in frame_files:
                frame_path = frames_dir / frame_file
                # ffmpeg concat 格式：file 'path' + duration
                tmp.write(f"file '{frame_path}'\n")
                tmp.write(f"duration {1.0 / FPS}\n")  # 每帧持续 0.5 秒
            # 最后一帧需要重复写一次（ffmpeg concat demuxer 的要求）
            tmp.write(f"file '{frames_dir / frame_files[-1]}'\n")
            tmp_path = tmp.name
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ffmpeg 命令
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", tmp_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "23",
            "-r", str(FPS),  # 输出帧率
            "-an",  # 无音频
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='replace')[-200:]
            return {"video_id": video_id, "status": "error", "n_frames": n_frames, "duration": duration, "msg": f"ffmpeg 失败: {error_msg}"}
        
        return {"video_id": video_id, "status": "ok", "n_frames": n_frames, "duration": duration, "msg": ""}
    
    except subprocess.TimeoutExpired:
        return {"video_id": video_id, "status": "error", "n_frames": n_frames, "duration": duration, "msg": "ffmpeg 超时"}
    except Exception as e:
        return {"video_id": video_id, "status": "error", "n_frames": n_frames, "duration": duration, "msg": str(e)}
    finally:
        # 清理临时文件
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="VIPSeg 帧序列 → 2fps MP4 视频合成")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="只处理指定 split（默认处理全部）")
    parser.add_argument("--workers", type=int, default=16,
                        help="并行进程数（默认 16）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印统计信息，不实际合成")
    parser.add_argument("--force", action="store_true",
                        help="强制重新合成（覆盖已存在的文件）")
    args = parser.parse_args()
    
    # 确定要处理的视频列表
    if args.split:
        video_ids = get_split_videos(args.split)
        print(f"[VIPSeg→MP4] 处理 {args.split} split: {len(video_ids)} 个视频")
    else:
        # 处理所有视频目录
        if not IMGS_DIR.exists():
            print(f"[ERROR] 帧目录不存在: {IMGS_DIR}")
            print(f"  请先创建软链接: ln -sf VIPSeg/imgs videos")
            sys.exit(1)
        video_ids = sorted([
            d for d in os.listdir(IMGS_DIR)
            if os.path.isdir(IMGS_DIR / d)
        ])
        print(f"[VIPSeg→MP4] 处理全部: {len(video_ids)} 个视频")
    
    print(f"  合成帧率: {FPS} fps（每帧 {1.0/FPS:.2f} 秒）")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  并行进程: {args.workers}")
    if args.dry_run:
        print(f"  ⚠️  DRY-RUN 模式：不实际合成")
    print()
    
    # 如果 force 模式，删除已存在的文件
    if args.force and not args.dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 并行处理
    results = {"ok": 0, "skip": 0, "error": 0}
    errors = []
    total_frames = 0
    total_duration = 0.0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(convert_one_video, vid, args.dry_run): vid
            for vid in video_ids
        }
        
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            result = future.result()
            results[result["status"]] += 1
            total_frames += result["n_frames"]
            total_duration += result["duration"]
            
            if result["status"] == "error":
                errors.append(result)
            
            # 进度打印
            if done_count % 200 == 0 or done_count == len(video_ids):
                print(f"  进度: {done_count}/{len(video_ids)} "
                      f"(ok={results['ok']}, skip={results['skip']}, error={results['error']})")
    
    # 汇总
    print()
    print("=" * 60)
    print(f"✅ 完成！")
    print(f"  成功: {results['ok']}")
    print(f"  跳过（已存在）: {results['skip']}")
    print(f"  失败: {results['error']}")
    print(f"  总帧数: {total_frames}")
    print(f"  总时长: {total_duration:.1f} 秒 ({total_duration/3600:.2f} 小时)")
    print(f"  平均帧数: {total_frames/max(len(video_ids),1):.1f} 帧/视频")
    print(f"  平均时长: {total_duration/max(len(video_ids),1):.1f} 秒/视频")
    
    if errors:
        print(f"\n⚠️  失败详情（前 10 个）:")
        for e in errors[:10]:
            print(f"  {e['video_id']}: {e['msg']}")


if __name__ == "__main__":
    main()
