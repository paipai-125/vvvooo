#!/usr/bin/env python3
"""
检查 VIPSeg 数据集所有视频目录的帧间隔是否均匀。
验证：
1. 每个视频的帧编号间隔是否恒定为 3
2. 是否存在非均匀采样的视频
3. 统计每个视频的帧数
"""
import os
import re
from collections import Counter, defaultdict

IMGS_DIR = "/apdcephfs/aigc/group_2/user_sleepfeng/video_opd_data/datasets/vipseg/VIPSeg/imgs"

def analyze_video_dir(video_dir):
    """分析单个视频目录的帧间隔"""
    files = sorted(os.listdir(video_dir))
    # 提取帧编号
    frame_nums = []
    for f in files:
        match = re.match(r'(\d+)\.jpg', f)
        if match:
            frame_nums.append(int(match.group(1)))
    
    if len(frame_nums) < 2:
        return None, frame_nums, []
    
    frame_nums.sort()
    # 计算相邻帧间隔
    intervals = [frame_nums[i+1] - frame_nums[i] for i in range(len(frame_nums)-1)]
    return intervals, frame_nums, files

def main():
    video_dirs = sorted([
        d for d in os.listdir(IMGS_DIR)
        if os.path.isdir(os.path.join(IMGS_DIR, d))
    ])
    
    print(f"总视频目录数: {len(video_dirs)}")
    print("=" * 60)
    
    interval_counter = Counter()  # 统计不同间隔值出现的次数
    anomalies = []  # 非均匀采样的视频
    frame_count_stats = []  # 每个视频的帧数
    interval_type_counter = Counter()  # 统计每种间隔类型的视频数
    
    for vname in video_dirs:
        vpath = os.path.join(IMGS_DIR, vname)
        intervals, frame_nums, files = analyze_video_dir(vpath)
        
        if intervals is None:
            anomalies.append((vname, "帧数不足2", frame_nums))
            continue
        
        frame_count_stats.append(len(frame_nums))
        
        # 统计该视频的间隔
        unique_intervals = set(intervals)
        interval_counter.update(intervals)
        
        if len(unique_intervals) == 1:
            # 均匀采样
            interval_type_counter[list(unique_intervals)[0]] += 1
        else:
            # 非均匀采样！
            anomalies.append((vname, f"非均匀间隔: {sorted(unique_intervals)}", 
                            f"帧数={len(frame_nums)}, 首帧={frame_nums[0]}, 末帧={frame_nums[-1]}"))
    
    # 输出结果
    print("\n📊 间隔类型统计（按视频数）:")
    for interval, count in interval_type_counter.most_common():
        print(f"  间隔={interval}: {count} 个视频")
    
    print(f"\n📊 所有帧间隔值的全局统计:")
    for interval, count in interval_counter.most_common(10):
        print(f"  间隔={interval}: 出现 {count} 次")
    
    print(f"\n📊 帧数统计:")
    print(f"  总视频数: {len(frame_count_stats)}")
    print(f"  最小帧数: {min(frame_count_stats)}")
    print(f"  最大帧数: {max(frame_count_stats)}")
    print(f"  平均帧数: {sum(frame_count_stats)/len(frame_count_stats):.1f}")
    
    print(f"\n⚠️  异常视频数: {len(anomalies)}")
    if anomalies:
        print("异常详情（前20个）:")
        for vname, reason, detail in anomalies[:20]:
            print(f"  {vname}: {reason} | {detail}")
    
    # 结论
    print("\n" + "=" * 60)
    if len(anomalies) == 0 and len(interval_type_counter) == 1 and 3 in interval_type_counter:
        print("✅ 结论: 所有视频均为均匀间隔=3的采样（从原始视频每3帧取1帧）")
        print("   如果原始视频为 30fps → 标注帧率 = 10fps")
        print("   如果原始视频为 24fps → 标注帧率 = 8fps")
        print("   如果原始视频为 15fps → 标注帧率 = 5fps")
    else:
        print("⚠️  结论: 存在非均匀采样或不同间隔的视频，需要逐个处理！")

if __name__ == "__main__":
    main()
