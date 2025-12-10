import os
import random
import time
from pathlib import Path
import folder_paths


def get_save_bytes_path(filename_prefix: str, filename_suffix: str, output_dir: str):
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
        counter = (
            max(
                (map_filename(x.name)[0] for x in exists_files),
                default=0,
            )
            + 1
        )
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return full_output_folder, filename, counter, subfolder, filename_prefix


def save_bytes(bytes_data, path):
    """
    Save bytes data to a file.

    Parameters:
        bytes_data (bytes): The bytes data to save.
        path (str): The output file path.
    """
    with open(path, "wb") as f:
        f.write(bytes_data)

    print(f"Bytes data saved to {path}")


class SaveBytesBridge:
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
                "bytes_data": ("BYTES", {"tooltip": "The bytes data to save."}),
                "output_dir": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The directory to save the bytes data to, leave blank to save to the ComfyUI output directory.",
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
                    [".pkl", ".dat", ".bin", "Custom"],
                    {
                        "default": ".pkl",
                        "tooltip": "The file extension to save the bytes data as.",
                    },
                ),
                "custom_suffix": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The file extension to save the bytes data as. If this is not empty, the suffix option will be ignored. Otherwise, the suffix option will override above suffix option.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("BYTES", "STRING")
    RETURN_NAMES = ("bytes_data", "path")
    FUNCTION = "save_bytes_file"

    OUTPUT_NODE = True

    _NODE_NAME = "Save Bytes Bridge"
    CATEGORY = "YogurtNodes/IO"
    DESCRIPTION = "Saves the input bytes data to your ComfyUI output directory."

    def save_bytes_file(
        self,
        bytes_data,
        output_dir="",
        filename_prefix="ComfyUI",
        overwrite="false",
        suffix=".pkl",
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
            get_save_bytes_path(filename_prefix, suffix, output_folder)
        )

        if overwrite == "true":
            file = f"{filename}{suffix}"
        else:
            file = f"{filename}_{counter:05}{suffix}"

        os.makedirs(full_output_folder, exist_ok=True)
        file_path = os.path.join(full_output_folder, file)

        save_bytes(bytes_data, file_path)

        return {
            "ui": {"text": (f"Saved to: {file_path}",)},
            "result": (bytes_data, str(Path(file_path))),
        }


class LoadBytes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": (
                    "STRING",
                    {"tooltip": "The path to the file containing bytes data to load."},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("BYTES",)
    RETURN_NAMES = ("bytes_data",)
    FUNCTION = "load_bytes_file"
    OUTPUT_NODE = False

    _NODE_NAME = "Load Bytes"
    DESCRIPTION = "Load bytes data from a file"
    CATEGORY = "YogurtNodes/IO"

    def load_bytes_file(self, file_path):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(file_path, "rb") as f:
                bytes_data = f.read()

            print(f"Loaded bytes data from {file_path}")
            return (bytes_data,)
        except Exception as e:
            raise ValueError(f"Failed to load bytes data from {file_path}: {str(e)}")


class SaveBytesBridgeNonOutput(SaveBytesBridge):
    OUTPUT_NODE = False
    _NODE_NAME = "Save Bytes Bridge (Non Output)"
