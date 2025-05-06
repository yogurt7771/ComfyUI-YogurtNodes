from yogurt_nodes.io.save_image_bridge_ex import SaveImageBridgeEx


class SaveMaskBridgeEx(SaveImageBridgeEx):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("MASK", {"tooltip": "The masks to save."}),
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
                    [".png", ".jpg", "Custom"],
                    {
                        "default": ".png",
                        "tooltip": "The file extension to save the images as.",
                    },
                ),
                "custom_suffix": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The file extension to save the images as. If this is not empty, the suffix option will be ignored. Otherwise, the suffix option will override above suffix option.",
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

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "execute"

    _NODE_NAME = "Save Mask Bridge"
    CATEGORY = "YogurtNodes/IO"
    DESCRIPTION = "Saves the input masks to your ComfyUI output directory."

    def execute(
        self,
        masks,
        output_dir="",
        filename_prefix="ComfyUI",
        disable_metadata="false",
        overwrite="false",
        suffix=".png",
        custom_suffix="",
        png_compression=4,
        jpeg_quality=100,
        prompt=None,
        extra_pnginfo=None,
    ):
        preview_masks = (
            masks.reshape((-1, 1, masks.shape[-2], masks.shape[-1]))
            .movedim(1, -1)
            .expand(-1, -1, -1, 3)
        )
        result = super().execute(
            preview_masks,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
            disable_metadata=disable_metadata,
            overwrite=overwrite,
            suffix=suffix,
            custom_suffix=custom_suffix,
            png_compression=png_compression,
            jpeg_quality=jpeg_quality,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
        result["result"] = (masks,)
        return result
