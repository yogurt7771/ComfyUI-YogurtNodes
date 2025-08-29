import json

import torch

from ..utils import ANY_TYPE


class PreviewAnyBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (ANY_TYPE, {"tooltip": "The data to be returned."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE, "STRING")
    RETURN_NAMES = ("data", "text")
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "Preview Any Bridge"
    DESCRIPTION = "Preview Any Bridge"
    CATEGORY = "YogurtNodes/IO"

    def execute(self, data):
        value = 'None'
        if isinstance(data, str):
            value = data
        elif isinstance(data, (int, float, bool)):
            value = str(data)
        elif isinstance(data, torch.Tensor):
            value = f"Tensor: {data.shape}\n{data}"
        elif data is not None:
            try:
                value = json.dumps(data, ensure_ascii=False, indent=4)
            except Exception:
                try:
                    value = str(data)
                except Exception:
                    value = 'source exists, but could not be serialized.'

        return {"ui": {"text": (value,)}, "result": (data, value)}


class PreviewAnyBridgeOutput(PreviewAnyBridge):
    OUTPUT_NODE = True
    _NODE_NAME = "Preview Any Bridge (Output)"
