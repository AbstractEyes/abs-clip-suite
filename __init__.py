from .abs_nodes.abs_clip_splitter import AbsClipSplitter
from .abs_nodes.clip_joiner import AbsClipJoiner

NODE_CLASS_MAPPINGS = {
    "AbsClipSplitter": AbsClipSplitter,
    "AbsClipJoiner": AbsClipJoiner,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AbsClipSplitter": "🖼️ Abs Clip Splitter",
    "AbsClipJoiner": "🖼️ Abs Clip Joiner",
}


NODE_CLASS_CUSTOM_TYPES = {
    "Clip_L": ("clip/custom_pipeline", "#00BFFF"),
    "Clip_G": ("clip/custom_pipeline", "#9370DB"),
    "Clip_H": ("clip/custom_pipeline", "#BA55D3"),
    "Clip_VISION": ("clip/custom_pipeline", "#3CB371"),
    "LLM": ("clip/custom_pipeline", "#FF69B4"),
    "T5": ("clip/custom_pipeline", "#FFD700"),
    "T5L": ("clip/custom_pipeline", "#FFA500"),
    "T5XL": ("clip/custom_pipeline", "#FF8C00"),
    "T5XXL": ("clip/custom_pipeline", "#FF4500"),
    "FlanT5": ("clip/custom_pipeline", "#8B0000"),
}



__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]




