import torch
import torch.nn as nn
import comfy.clip_model

# ─── Core Base Class ─────────────────────────────────────────────────────────

class CustomClipBase:
    def __init__(self, model, precision="fp32", architecture=None, device="cpu"):
        self.model = model
        self.precision = precision
        self.architecture = architecture
        self.device = device

    def to_device(self, device="cuda"):
        self.device = device
        self.model = self.model.to(device)
        return self

    def runtime_quantize_tensor(self, tensor, num_bits=8):
        qmin, qmax = 0, 2 ** num_bits - 1
        min_val, max_val = tensor.min(), tensor.max()
        scale = max((max_val - min_val) / (qmax - qmin), 1e-8)
        tensor_quantized = ((tensor - min_val) / scale).round().clamp(qmin, qmax)
        tensor_dequantized = (tensor_quantized * scale) + min_val
        return tensor_dequantized

    def change_precision(self, new_precision="fp32"):
        if new_precision == "fp32":
            self.model = self.model.to(dtype=torch.float32)
        elif new_precision == "fp16":
            self.model = self.model.to(dtype=torch.float16)
        elif new_precision == "bf16":
            self.model = self.model.to(dtype=torch.bfloat16)
        elif new_precision == "fp8":
            with torch.no_grad():
                for param in self.model.parameters():
                    param.data = self.runtime_quantize_tensor(param.data, num_bits=8)
        elif new_precision in {"int8", "int4", "int2", "q8_0", "q8_1", "q6_0", "q5_0", "q5_1", "q4_0", "q4_1", "q3_0"}:
            bits = 8 if "8" in new_precision else 4 if "4" in new_precision else 2
            with torch.no_grad():
                for param in self.model.parameters():
                    param.data = self.runtime_quantize_tensor(param.data, num_bits=bits)
        else:
            raise ValueError(f"Unsupported precision type: {new_precision}")
        self.precision = new_precision
        return self

# ─── Specific Clip Wrapper Classes ──────────────────────────────────────────

class ClipLType(CustomClipBase): pass
class ClipGType(CustomClipBase): pass
class ClipHType(CustomClipBase): pass
class ClipVisionType(CustomClipBase): pass
class LLMType(CustomClipBase): pass
class T5Type(CustomClipBase): pass
class T5LType(CustomClipBase): pass
class T5XLType(CustomClipBase): pass
class T5XXLType(CustomClipBase): pass
class FlanT5Type(CustomClipBase): pass

# ─── Dynamic Type Map ────────────────────────────────────────────────────────

TYPE_MAP = {
    "Clip_L": ClipLType,
    "Clip_G": ClipGType,
    "Clip_H": ClipHType,
    "Clip_VISION": ClipVisionType,
    "LLM": LLMType,
    "T5": T5Type,
    "T5L": T5LType,
    "T5XL": T5XLType,
    "T5XXL": T5XXLType,
    "FlanT5": FlanT5Type,
}
