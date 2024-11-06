import json
import os
import random

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np

import folder_paths


class SaveImageBridge:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "output_dir": ("STRING", {"default": "", "tooltip": "The directory to save the images to, default is the ComfyUI output directory."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "disable_metadata": ("BOOL", {"default": False, "tooltip": "Disable saving metadata to the PNG file."}),
                "overwrite": ("BOOL", {"default": False, "tooltip": "Overwrite existing files."}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    _NODE_NAME = "Save Image Bridge"
    CATEGORY = "YogurtNodes/IO"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, images, output_dir="", filename_prefix="ComfyUI", disable_metadata=False, overwrite=False, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        if os.path.isabs(output_dir):
            full_output_folder = output_dir
        else:
            full_output_folder = os.path.join(self.output_dir, output_dir)
        _, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            if overwrite:
                file = f"{filename_with_batch_num}.png"
            else:
                file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results }, "result": (images,) }


class PreviewImageBridge(SaveImageBridge):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)
        )
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    _NODE_NAME = "Preview Image Bridge"
