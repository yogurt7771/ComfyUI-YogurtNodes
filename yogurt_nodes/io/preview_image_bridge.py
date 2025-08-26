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
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("images", "width", "height", "batch")
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "Preview Image Bridge"
    CATEGORY = "YogurtNodes/IO"
    DESCRIPTION = "Preview the input images."

    def execute(self, images, prompt=None, extra_pnginfo=None):
        result = super().execute(images, prompt=prompt, extra_pnginfo=extra_pnginfo)
        result["result"] = (*result["result"][:-1],)
        return result
