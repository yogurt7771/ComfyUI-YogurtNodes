import os
import random
import time
from pathlib import Path
import folder_paths


def get_save_text_path(filename_prefix: str, filename_suffix: str, output_dir: str):
    def map_filename(filename: str) -> tuple[int, str]:
        prefix, digits = Path(filename).stem.rsplit("_", maxsplit=2)
        if digits.isdigit():
            return int(digits), prefix
        else:
            return 0, filename

    def compute_vars(input: str) -> str:
        now = time.localtime()
        input = input.replace("%year%", str(now.tm_year))
        input = input.replace("%month%", str(now.tm_mon).zfill(2))
        input = input.replace("%day%", str(now.tm_mday).zfill(2))
        input = input.replace("%hour%", str(now.tm_hour).zfill(2))
        input = input.replace("%minute%", str(now.tm_min).zfill(2))
        input = input.replace("%second%", str(now.tm_sec).zfill(2))
        return input

    if "%" in filename_prefix:
        filename_prefix = compute_vars(filename_prefix)

    subfolder = str(Path(filename_prefix).parent)
    filename = str(Path(filename_prefix).name)

    full_output_folder = os.path.join(output_dir, subfolder)

    try:
        exists_files = list(
            Path(full_output_folder).glob(f"{filename}*{filename_suffix}")
        )
        counter = max(
            map(
                lambda x: map_filename(x.name)[0],
                exists_files,
            ),
            default=0,
        ) + 1
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return full_output_folder, filename, counter, subfolder, filename_prefix


def save_text(text, path):
    """
    Save text to a file.

    Parameters:
        text (str): The text to save.
        path (str): The output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Text saved to {path}")


class SaveTextBridge:
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
                "text": ("STRING", {"multiline": True, "tooltip": "The text to save."}),
                "output_dir": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The directory to save the text to, leave blank to save to the ComfyUI output directory.",
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd%.",
                    },
                ),
                "overwrite": (
                    ["true", "false"],
                    {"default": "false", "tooltip": "Overwrite existing files."},
                ),
                "suffix": (
                    [".txt", ".json", ".md", ".csv", "Custom"],
                    {
                        "default": ".txt",
                        "tooltip": "The file extension to save the text as.",
                    },
                ),
                "custom_suffix": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The file extension to save the text as. If this is not empty, the suffix option will be ignored. Otherwise, the suffix option will override above suffix option.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "save_text_file"

    OUTPUT_NODE = True

    _NODE_NAME = "Save Text Bridge"
    CATEGORY = "YogurtNodes/IO"
    DESCRIPTION = "Saves the input text to your ComfyUI output directory."

    def save_text_file(
        self,
        text,
        output_dir="",
        filename_prefix="ComfyUI",
        overwrite="false",
        suffix=".txt",
        custom_suffix="",
    ):
        filename_prefix += self.prefix_append

        if suffix == "Custom":
            suffix = custom_suffix

        if os.path.isabs(output_dir):
            output_folder = output_dir
        else:
            output_folder = os.path.join(self.output_dir, output_dir)

        full_output_folder, filename, counter, subfolder, filename_prefix = (
            get_save_text_path(filename_prefix, suffix, output_folder)
        )

        if overwrite == "true":
            file = f"{filename}{suffix}"
        else:
            file = f"{filename}_{counter:05}{suffix}"

        os.makedirs(full_output_folder, exist_ok=True)
        file_path = os.path.join(full_output_folder, file)

        save_text(text, file_path)

        result = {"filename": file, "subfolder": subfolder, "type": self.type}

        return {"ui": {"text": [result]}, "result": (text,)}
