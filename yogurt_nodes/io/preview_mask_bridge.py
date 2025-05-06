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

    _NODE_NAME = "Preview Mask Bridge"
    CATEGORY = "YogurtNodes/IO"
    DESCRIPTION = "Preview the input masks."
