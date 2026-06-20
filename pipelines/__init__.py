"""
8个视觉聚焦预处理pipeline，每个对应一种 <observe type="..."/>。

统一接口：
    pipeline(video_path, target="", time=None, frame=None,
             bbox=None, objects=None) -> dict
返回：
    {"video": 处理后的视频/帧路径, "perception_question": 给Teacher_P的问题字符串}

所有外部模型均为lazy loading（首次调用才加载），出错直接raise，禁止容错跳过。
"""
from .temporal_locate import temporal_locate_pipeline
from .temporal_clip import temporal_clip_pipeline
from .spatial_detect import spatial_detect_pipeline
from .spatial_crop import spatial_crop_pipeline
from .depth_overlay import depth_overlay_pipeline
from .tracking_overlay import tracking_overlay_pipeline
from .ocr_zoom import ocr_zoom_pipeline
from .raw import raw_pipeline


# 统一映射：type -> pipeline 函数
PIPELINES = {
    "temporal_locate": temporal_locate_pipeline,
    "temporal_clip": temporal_clip_pipeline,
    "spatial_detect": spatial_detect_pipeline,
    "spatial_crop": spatial_crop_pipeline,
    "depth_overlay": depth_overlay_pipeline,
    "tracking_overlay": tracking_overlay_pipeline,
    "ocr_zoom": ocr_zoom_pipeline,
    "raw": raw_pipeline,
}


def run_pipeline(observe_query, video_path: str) -> dict:
    """
    根据 utils.parser.ObserveQuery 调度对应pipeline。

    Args:
        observe_query: ObserveQuery 实例
        video_path: 原始视频路径
    Returns:
        {"video": 处理后视频/帧路径, "perception_question": str}
    """
    fn = PIPELINES.get(observe_query.type)
    if fn is None:
        raise ValueError(f"未知 observe type: {observe_query.type}")
    return fn(
        video_path=video_path,
        target=observe_query.target,
        time=observe_query.time,
        frame=observe_query.frame,
        bbox=observe_query.bbox,
        objects=observe_query.objects,
    )


__all__ = [
    "temporal_locate_pipeline",
    "temporal_clip_pipeline",
    "spatial_detect_pipeline",
    "spatial_crop_pipeline",
    "depth_overlay_pipeline",
    "tracking_overlay_pipeline",
    "ocr_zoom_pipeline",
    "raw_pipeline",
    "PIPELINES",
    "run_pipeline",
]
