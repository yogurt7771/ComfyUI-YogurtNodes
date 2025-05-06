from yogurt_nodes.io.save_mask_bridge_ex import SaveMaskBridgeEx


class SaveMaskBridge(SaveMaskBridgeEx):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK", {"tooltip": "The masks to save."}),
                "output_dir": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The directory to save the masks to, leave blank to save to the ComfyUI output directory.",
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Mask.width% to include values from nodes.",
                    },
                ),
                "disable_metadata": (
                    ["true", "false"],
                    {
                        "default": "false",
                        "tooltip": "Disable saving metadata to the PNG file.",
                    },
                ),
                "overwrite": (
                    ["true", "false"],
                    {"default": "false", "tooltip": "Overwrite existing files."},
                ),
                "suffix": (
                    [".png", ".jpg"],
                    {
                        "default": ".png",
                        "tooltip": "The file extension to save the masks as.",
                    },
                ),
                "png_compression": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 9,
                        "tooltip": "The level of compression to use for PNG masks.",
                    },
                ),
                "jpeg_quality": (
                    "INT",
                    {
                        "default": 100,
                        "min": 0,
                        "max": 100,
                        "tooltip": "The quality of the JPEG mask.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    _NODE_NAME = "Save Mask Bridge"
    CATEGORY = "YogurtNodes/IO"
    DESCRIPTION = "Saves the input masks to your ComfyUI output directory."
