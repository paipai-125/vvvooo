"""
video_opd_code: 基于 Qwen3-VL-4B 的潜空间视频推理 + 双教师 On-Policy Distillation。

子模块:
    configs/         路径与全局配置
    utils/           解析、视频处理等工具
    pipelines/       8 个视觉聚焦预处理 pipeline
    data_preparation/  Stage 1 数据生成与筛选
    training/        Stage 1 SFT / OPD 训练入口
    evaluation/      预实验脚本
"""

__version__ = "0.1.0"
