import random

import folder_paths

from .save_mask_bridge_ex import SaveMaskBridgeEx


class PreviewMaskBridge(SaveMaskBridgeEx):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("MASK",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("MASK", "INT", "INT", "INT")
    RETURN_NAMES = ("masks", "width", "height", "batch")
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "Preview Mask Bridge"
    CATEGORY = "YogurtNodes/IO"
    DESCRIPTION = "Preview the input masks."

    def execute(self, masks, prompt=None, extra_pnginfo=None):
        result = super().execute(masks, prompt=prompt, extra_pnginfo=extra_pnginfo)
        result["result"] = (*result["result"][:-1],)
        return result
