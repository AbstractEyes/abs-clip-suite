# clip_joiner_util.py

import torch.nn as nn

def guess_clip_base_model(active_clips):
    """
    Guess the appropriate CLIP model based on the available fragments.
    """
    if "clip_l" in active_clips and "clip_g" in active_clips:
        from comfy.sdxl_clip import SDXLClipModel
        return SDXLClipModel
    elif "clip_g" in active_clips:
        from comfy.sdxl_clip import SDXLRefinerClipModel
        return SDXLRefinerClipModel
    elif "clip_h" in active_clips:
        from comfy.text_encoders.sd2_clip import SD2ClipModel
        return SD2ClipModel
    elif "t5" in active_clips:
        from comfy.text_encoders.sd3_clip import sd3_clip
        return sd3_clip  # NOTE: function, not a class
    else:
        from comfy.sd1_clip import SD1ClipModel
        return SD1ClipModel

def assign_to_cond_stage_model(model, name, module_state_dict):
    """
    Reconstruct a module from a state_dict before attaching to cond_stage_model.
    PyTorch expects registered children to be nn.Module or None.
    """

    if module_state_dict is None:
        setattr(model, name, None)
        return

    dummy_module = nn.Module()
    dummy_module.load_state_dict(module_state_dict, strict=False)
    setattr(model, name, dummy_module)
