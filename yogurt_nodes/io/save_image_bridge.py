from .save_image_bridge_ex import SaveImageBridgeEx


class SaveImageBridge(SaveImageBridgeEx):
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
                        "tooltip": "The file extension to save the images as.",
                    },
                ),
                "png_compression": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 9,
                        "tooltip": "The level of compression to use for PNG images.",
                    },
                ),
                "jpeg_quality": (
                    "INT",
                    {
                        "default": 100,
                        "min": 0,
                        "max": 100,
                        "tooltip": "The quality of the JPEG image.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    _NODE_NAME = "Save Image Bridge"
    CATEGORY = "YogurtNodes/IO"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."
