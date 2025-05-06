import random

import folder_paths

from .save_image_bridge_ex import SaveImageBridgeEx


class PreviewImageBridge(SaveImageBridgeEx):
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
                "images": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    _NODE_NAME = "Preview Image Bridge"
    CATEGORY = "YogurtNodes/IO"
    DESCRIPTION = "Preview the input images."
