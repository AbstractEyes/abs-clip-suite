import torch
from ..utils.clip_wrapper import WrappedCLIP
from ..utils.clip_types import TYPE_MAP
class AbsClipSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),  # <<< field name must match method parameter name
            }
        }

    RETURN_TYPES = (
        "Clip_L",
        "Clip_G",
        "Clip_VISION",
        "Clip_H",
        "T5",
        "LLM",
        "UNKNOWN_SD",
    )
    RETURN_NAMES = (
        "clip_l",
        "clip_g",
        "clip_vision",
        "clip_h",
        "t5",
        "llm",
        "unknown",
    )

    FUNCTION = "split"
    CATEGORY = "clip/custom_pipeline"

    def split(self, clip):  # <<< argument name must match INPUT_TYPES
        # Wrap properly
        wrapped_clip = WrappedCLIP(target=clip)

        # Perform logical splits
        splits = wrapped_clip.split_components()

        return (
            splits.get("clip_l", None),
            splits.get("clip_g", None),
            splits.get("vision_model", None),
            splits.get("clip_h", None),
            splits.get("t5", None),
            splits.get("llm", None),
            splits.get("unknown", None),
        )

# ─── Node Mappings ──────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "AbsClipSplitter": AbsClipSplitter,
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

NODE_DISPLAY_NAME_MAPPINGS = {
    "AbsClipTranslator": "Abs Clip Translator",
}
