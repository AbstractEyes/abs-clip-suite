import torch
from ..utils.clip_wrapper import WrappedCLIP
from ..utils.clip_types import TYPE_MAP
from ..utils.clip_joiner_util import guess_clip_base_model, assign_to_cond_stage_model

class AbsClipJoiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},  # ← no required inputs
            "optional": {
                "clip_l": ("Clip_L", {}),
                "clip_g": ("Clip_G", {}),
                "clip_vision": ("Clip_VISION", {}),
                "clip_h": ("Clip_H", {}),
                "t5": ("T5", {}),
                "llm": ("LLM", {}),
                "unknown": ("UNKNOWN_SD", {}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("joined_clip",)

    FUNCTION = "join"
    CATEGORY = "clip/custom_pipeline"

    def join(self, clip_l=None, clip_g=None, clip_vision=None, clip_h=None, t5=None, llm=None, unknown=None):
        # Collect active clips
        available = {
            "clip_l": clip_l,
            "clip_g": clip_g,
            "clip_vision": clip_vision,
            "clip_h": clip_h,
            "t5": t5,
            "llm": llm,
            "unknown": unknown,
        }
        active_clips = {k: v for k, v in available.items() if v is not None}

        if not active_clips:
            raise ValueError("[AbsClipJoiner] No clips provided for joining.")

        # Guess base model
        core_model_class = guess_clip_base_model(active_clips)

        if callable(core_model_class):
            cond_stage_model = core_model_class()
        else:
            cond_stage_model = core_model_class()

        # Assign parts into reconstructed model
        for name, module in active_clips.items():
            assign_to_cond_stage_model(cond_stage_model, name, module)

        # Wrap properly
        joined_clip = WrappedCLIP(no_init=True)
        joined_clip.cond_stage_model = cond_stage_model
        joined_clip.patcher = None
        joined_clip.tokenizer = None
        joined_clip.clip_type = TYPE_MAP.get(type(cond_stage_model), "UNKNOWN")

        return (joined_clip,)

# ─── Node Mappings ──────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "AbsClipJoiner": AbsClipJoiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AbsClipJoiner": "Abs Clip Joiner",
}
