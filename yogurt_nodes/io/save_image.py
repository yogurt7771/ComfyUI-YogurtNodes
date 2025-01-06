import json
import os
import random
import shutil
import time

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths


def get_save_image_path(
    filename_prefix: str, output_dir: str, image_width=0, image_height=0
) -> tuple[str, str, int, str, str]:
    def map_filename(filename: str) -> tuple[int, str]:
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[: prefix_len + 1]
        try:
            digits = int(filename[prefix_len + 1 :].split("_")[0])
        except:
            digits = 0
        return digits, prefix

    def compute_vars(input: str, image_width: int, image_height: int) -> str:
        input = input.replace("%width%", str(image_width))
        input = input.replace("%height%", str(image_height))
        now = time.localtime()
        input = input.replace("%year%", str(now.tm_year))
        input = input.replace("%month%", str(now.tm_mon).zfill(2))
        input = input.replace("%day%", str(now.tm_mday).zfill(2))
        input = input.replace("%hour%", str(now.tm_hour).zfill(2))
        input = input.replace("%minute%", str(now.tm_min).zfill(2))
        input = input.replace("%second%", str(now.tm_sec).zfill(2))
        return input

    if "%" in filename_prefix:
        filename_prefix = compute_vars(filename_prefix, image_width, image_height)

    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename = os.path.basename(os.path.normpath(filename_prefix))

    full_output_folder = os.path.join(output_dir, subfolder)

    try:
        counter = (
            max(
                filter(
                    lambda a: os.path.normcase(a[1][:-1]) == os.path.normcase(filename)
                    and a[1][-1] == "_",
                    map(map_filename, os.listdir(full_output_folder)),
                )
            )[0]
            + 1
        )
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return full_output_folder, filename, counter, subfolder, filename_prefix


def save_image(
    image,
    path,
    jpeg_quality=95,
    png_compression_level=6,
    metadata: PngInfo = None,
):
    """
    Save an image to a file with optional metadata for JPEG and PNG formats.

    Parameters:
        image (PIL.Image.Image): The image to save.
        path (str): The output file path (must end with .jpg, .jpeg, or .png).
        jpeg_quality (int): The quality of the JPEG file (1-100).
        png_compression_level (int): The compression level for PNG (0-9).
        metadata (PngImagePlugin.PngInfo): A PngInfo object for PNG metadata.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    # Determine the file format based on the extension
    ext = path.lower().split(".")[-1]

    if ext not in ("jpg", "jpeg", "png"):
        raise ValueError("Unsupported file format. Use .jpg, .jpeg, or .png.")

    if ext in ("jpg", "jpeg"):
        # Convert to RGB for JPEG (no alpha channel)
        rgb_image = image.convert("RGB")

        # Save as JPEG with EXIF metadata
        rgb_image.save(path, "JPEG", quality=jpeg_quality)

    elif ext == "png":
        # Save as PNG with metadata if provided
        if metadata is None:
            metadata = PngInfo()
        image.save(path, "PNG", compress_level=png_compression_level, pnginfo=metadata)

    print(f"Image saved to {path} with metadata: {metadata}")


class SaveImageBridge:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.type = "output"
        self.prefix_append = ""
        self.temp_prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)
        )

    @classmethod
    def INPUT_TYPES(s):
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

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    _NODE_NAME = "Save Image Bridge"
    CATEGORY = "YogurtNodes/IO"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(
        self,
        images,
        output_dir="",
        filename_prefix="ComfyUI",
        disable_metadata="false",
        overwrite="false",
        suffix=".png",
        png_compression=4,
        jpeg_quality=100,
        prompt=None,
        extra_pnginfo=None,
    ):
        filename_prefix += self.prefix_append
        if os.path.isabs(output_dir):
            output_folder = output_dir
        else:
            output_folder = os.path.join(self.output_dir, output_dir)
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            get_save_image_path(
                filename_prefix,
                output_folder,
                images[0].shape[1],
                images[0].shape[0],
            )
        )
        results = list()
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
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
                file = f"{filename_with_batch_num}{suffix}"
            else:
                file = f"{filename_with_batch_num}_{counter:05}{suffix}"
            os.makedirs(full_output_folder, exist_ok=True)
            # img.save(os.path.join(full_output_folder, file), pnginfo=metadata, quality=jpeg_quality, compress_level=png_compression)
            save_image(
                img,
                os.path.join(full_output_folder, file),
                jpeg_quality=jpeg_quality,
                png_compression_level=png_compression,
                metadata=metadata,
            )
            if output_dir != "" and os.path.isabs(output_dir):
                temp_filename_prefix = filename_with_batch_num + self.temp_prefix_append
                (
                    temp_output_folder,
                    temp_filename,
                    temp_counter,
                    temp_subfolder,
                    temp_filename_prefix,
                ) = get_save_image_path(
                    temp_filename_prefix,
                    self.temp_dir,
                    images[0].shape[1],
                    images[0].shape[0],
                )
                temp_file = f"{temp_filename}{suffix}"
                shutil.copyfile(
                    os.path.join(full_output_folder, file),
                    os.path.join(temp_output_folder, temp_file),
                )

                results.append(
                    {"filename": temp_file, "subfolder": temp_subfolder, "type": "temp"}
                )
            else:
                results.append(
                    {"filename": file, "subfolder": subfolder, "type": self.type}
                )
            counter += 1

        return {"ui": {"images": results}, "result": (images,)}


class PreviewImageBridge(SaveImageBridge):
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

    FUNCTION = "preview_images"
    _NODE_NAME = "Preview Image Bridge"

    def preview_images(self, images, prompt=None, extra_pnginfo=None):
        return self.save_images(
            images,
            suffix=".png",
            png_compression=9,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
