import json
import os
import random

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import shutil

import folder_paths


class SaveImageBridge:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4
        self.temp_prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "output_dir": ("STRING", {"default": "", "tooltip": "The directory to save the images to, default is the ComfyUI output directory."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "disable_metadata": (["true", "false"], {"default": "false", "tooltip": "Disable saving metadata to the PNG file."}),
                "overwrite": (["true", "false"], {"default": "false", "tooltip": "Overwrite existing files."}),
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

    def save_images(self, images, output_dir="", filename_prefix="ComfyUI", disable_metadata="false", overwrite="false", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        if os.path.isabs(output_dir):
            output_folder = output_dir
        else:
            output_folder = os.path.join(self.output_dir, output_dir)
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix,
                output_folder,
                images[0].shape[1],
                images[0].shape[0],
            )
        )
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not (disable_metadata == "true"):
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            if overwrite == "true":
                file = f"{filename_with_batch_num}.png"
            else:
                file = f"{filename_with_batch_num}_{counter:05}_.png"
            os.makedirs(full_output_folder, exist_ok=True)
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            if output_dir != "" and os.path.isabs(output_dir):
                temp_filename_prefix = (
                    filename_with_batch_num + self.temp_prefix_append
                )
                (
                    temp_output_folder,
                    temp_filename,
                    temp_counter,
                    temp_subfolder,
                    temp_filename_prefix,
                ) = folder_paths.get_save_image_path(
                    temp_filename_prefix,
                    self.temp_dir,
                    images[0].shape[1],
                    images[0].shape[0],
                )
                temp_file = f"{temp_filename_prefix}.png"
                shutil.copyfile(
                    os.path.join(full_output_folder, file),
                    os.path.join(temp_output_folder, temp_file),
                )

                results.append(
                    {"filename": temp_file, "subfolder": temp_subfolder, "type": "temp"}
                )
            else:
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
