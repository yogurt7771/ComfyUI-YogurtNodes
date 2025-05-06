from yogurt_nodes.io.save_image_bridge_ex import SaveImageBridgeEx


class SaveImageBridgeSimple(SaveImageBridgeEx):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "output_dir": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The directory to save the images to, leave blank to save to the ComfyUI output directory.",
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    },
                ),
                "overwrite": (
                    ["true", "false"],
                    {"default": "false", "tooltip": "Overwrite existing files."},
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    _NODE_NAME = "Save Image Bridge Simple"
    CATEGORY = "YogurtNodes/IO"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."
