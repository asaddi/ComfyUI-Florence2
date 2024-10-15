from unittest.mock import patch

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports

from .models import *
from .nodes import fixed_get_imports

import comfy.model_management as mm


MODELS = ModelManager(allowed_files=['*.json', '*.safetensors', '*.py', 'pytorch_model.bin'])
MODELS.load()


class NewFlorence2ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        MODELS.refresh()
        return {"required": {
            "model": (MODELS.CHOICES,),
            "precision": ([ 'fp16','bf16','fp32'],
                    {
                    "default": 'fp16'
                    }),
            "attention": (
                    [ 'flash_attention_2', 'sdpa', 'eager'],
                    {
                    "default": 'sdpa'
                    }),
            },
            "optional": {
                "lora": ("PEFTLORA",),
            }
        }

    @classmethod
    def IS_CHANGED(s, model, precision, attention, lora=None):
        MODELS.refresh()
        return MODELS._mtime

    RETURN_TYPES = ("FL2MODEL",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, model, precision, attention, lora=None):
        model_path = MODELS.download(model)

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        print(f"using {attention} for attention")
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
            model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation=attention, device_map=device, torch_dtype=dtype,trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        if lora is not None:
            from peft import PeftModel
            adapter_name = lora
            model = PeftModel.from_pretrained(model, adapter_name, trust_remote_code=True)

        florence2_model = {
            'model': model,
            'processor': processor,
            'dtype': dtype
            }

        return (florence2_model,)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadFlorence2Model": NewFlorence2ModelLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadFlorence2Model": "Florence2 Model Loader (New)",
}
