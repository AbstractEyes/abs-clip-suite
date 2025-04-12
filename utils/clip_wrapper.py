import torch
import torch.nn as nn
from comfy.sd import CLIP as ComfyCLIP

class WrappedCLIP:
    def __init__(self, target=None, tokenizer_data=None, model_options=None, parameters=0, no_init=False):
        """
        WrappedCLIP handles both:
         - Full ComfyUI CLIP objects
         - Split-out nn.Module clip fragments
        """
        self.raw_clip = None
        self.tokenizer = None
        self.cond_stage_model = None
        self.patcher = None
        self.layer_idx = None
        self.parameters = parameters
        self.model_options = model_options or {}
        self.tokenizer_data = tokenizer_data or {}
        self._target = target
        self.is_fragment = False  # <<< Fragment detector

        if no_init:
            return

        # Full Comfy CLIP case
        if isinstance(target, ComfyCLIP):
            self.raw_clip = target
            self.tokenizer = target.tokenizer
            self.cond_stage_model = target.cond_stage_model
            self.patcher = target.patcher
        # Fragment case (like clip_l, clip_g)
        elif isinstance(target, nn.Module):
            self.cond_stage_model = target
            self.is_fragment = True
        # Raw comfy clip_target case
        else:
            self._initialize_from_target(target)

    def _initialize_from_target(self, target):
        """For initializing from comfy clip_target class structure."""
        params = target.params.copy()
        clip_class = target.clip
        tokenizer_class = target.tokenizer

        from comfy import model_management
        load_device = self.model_options.get("load_device", model_management.text_encoder_device())
        offload_device = self.model_options.get("offload_device", model_management.text_encoder_offload_device())
        dtype = self.model_options.get("dtype", model_management.text_encoder_dtype(load_device))

        params['dtype'] = dtype
        params['device'] = self.model_options.get(
            "initial_device",
            model_management.text_encoder_initial_device(
                load_device,
                offload_device,
                self.parameters * model_management.dtype_size(dtype)
            )
        )
        params['model_options'] = self.model_options

        self.cond_stage_model = clip_class(**params)
        self.tokenizer = tokenizer_class(tokenizer_data=self.tokenizer_data)

        if params['device'] != load_device:
            self.cond_stage_model.to(load_device)

        import comfy.model_patcher
        self.patcher = comfy.model_patcher.ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)
        self.patcher.is_clip = True

    def load_model(self):
        from comfy import model_management
        model_management.load_model_gpu(self.patcher)
        return self.patcher

    def clone(self):
        n = WrappedCLIP(no_init=True)
        n.raw_clip = self.raw_clip
        n.tokenizer = self.tokenizer
        n.cond_stage_model = self.cond_stage_model
        n.patcher = self.patcher.clone() if self.patcher else None
        n.parameters = self.parameters
        n.model_options = self.model_options
        n.tokenizer_data = self.tokenizer_data
        n.layer_idx = self.layer_idx
        n.is_fragment = self.is_fragment
        return n

    def export_tokenizer(self):
        if self.is_fragment:
            raise AttributeError("WrappedCLIP (fragment): No tokenizer available.")
        return self.tokenizer.state_dict()

    def export_model(self):
        return self.cond_stage_model.state_dict()

    def load_sd(self, sd, full_model=False):
        if full_model:
            return self.cond_stage_model.load_state_dict(sd, strict=False)
        else:
            return self.cond_stage_model.load_sd(sd)

    def get_key_patches(self):
        if self.patcher:
            return self.patcher.get_key_patches()
        else:
            return None

    def inspect_state_dict(self):
        """Analyze model content."""
        sd = self.cond_stage_model.state_dict()
        clips_found = {
            "clip_l": any(k.startswith("clip_l.") for k in sd),
            "clip_g": any(k.startswith("clip_g.") for k in sd),
            "clip_h": any(k.startswith("clip_h.") for k in sd),
            "vision_model": any(k.startswith("vision_model.") or k.startswith("clip_vision.") for k in sd),
            "llm": any(k.startswith("hunyuan_clip.") or k.startswith("llama.") or k.startswith("llm.") for k in sd),
            "t5": any(k.startswith("t5.") and "small" in k for k in sd),
            "t5l": any(k.startswith("t5l.") for k in sd),
            "t5xl": any(k.startswith("t5xl.") for k in sd),
            "t5xxl": any(k.startswith("t5xxl.") or k.startswith("flan_t5xxl.") for k in sd),
            "flant5": any(k.startswith("flan_t5.") and "xxl" not in k for k in sd),
        }
        return clips_found

    def split_components(self):
        """Split model components into dictionary."""
        sd = self.cond_stage_model.state_dict()
        splits = {key: {} for key in self.inspect_state_dict().keys()}
        splits["unknown"] = {}

        for k, v in sd.items():
            for clip_type in splits.keys():
                if clip_type != "unknown" and (k.startswith(f"{clip_type}.") or (clip_type == "vision_model" and k.startswith("clip_vision."))):
                    splits[clip_type][k] = v
                    break
            else:
                splits["unknown"][k] = v

        return splits

    def get_component(self, component_name):
        """Retrieve individual nn.Module fragment."""
        splits = self.split_components()
        if component_name not in splits or not splits[component_name]:
            return None

        dummy_model = nn.Module()
        dummy_model.load_state_dict(splits[component_name], strict=False)
        return dummy_model

    def encode_text(self, text):
        if self.is_fragment:
            raise AttributeError("Cannot encode text on fragment-only WrappedCLIP.")
        tokens = self.tokenizer.tokenize_with_weights(text)
        return self.cond_stage_model.encode_token_weights(tokens)

    def encode_tokens(self, tokens):
        if self.is_fragment:
            raise AttributeError("Cannot encode tokens on fragment-only WrappedCLIP.")
        return self.cond_stage_model.encode_token_weights(tokens)

    def tokenize(self, text, return_word_ids=False):
        if self.is_fragment:
            raise AttributeError("Cannot tokenize on fragment-only WrappedCLIP.")
        if hasattr(self.tokenizer, "tokenize_with_weights"):
            return self.tokenizer.tokenize_with_weights(text, return_word_ids)
        elif hasattr(self.tokenizer, "tokenize"):
            return self.tokenizer.tokenize(text)
        else:
            raise AttributeError("WrappedCLIP: Tokenizer missing tokenize method.")

    def get_cond_stage_model(self):
        return self.cond_stage_model

    def get_patcher(self):
        return self.patcher

    def get_tokenizer(self):
        if self.is_fragment:
            raise AttributeError("Fragment has no tokenizer.")
        return self.tokenizer
