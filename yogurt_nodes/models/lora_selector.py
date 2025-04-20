from pathlib import Path

from folder_paths import get_filename_list


class LoraSelector:
    """
    Select Lora from a list of LoRAs
    """

    initial_list = get_filename_list("loras")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": (
                    ["None"] + cls.initial_list,
                    {"default": "None", "tooltip": "Lora model to use."},
                ),
                "model_strength": (
                    "FLOAT",
                    {"default": 1.0, "step": 0.1},
                    {"tooltip": "Strength of the Lora model."},
                ),
                "clip_strength": (
                    "FLOAT",
                    {"default": 1.0, "step": 0.1},
                    {"tooltip": "Strength of the CLIP model."},
                ),
                "trigger_word": (
                    "STRING",
                    {"default": "", "tooltip": "Trigger word to use."},
                ),
            }
        }

    RETURN_TYPES = (
        initial_list,
        "STRING",
        "STRING",
        "FLOAT",
        "FLOAT",
        "STRING",
    )
    RETURN_NAMES = (
        "lora",
        "name",
        "stem",
        "model_strength",
        "clip_strength",
        "trigger_word",
    )
    OUTPUT_NODE = False

    FUNCTION = "lora_selector"

    _NODE_NAME = "Lora Selector"
    DESCRIPTION = "Select Lora"
    CATEGORY = "YogurtNodes/Models"

    def lora_selector(
        self, lora: str, model_strength: float, clip_strength: float, trigger_word: str
    ):
        lora_path = Path(lora)
        name = lora_path.name
        stem = lora_path.stem
        return (lora, str(name), str(stem), model_strength, clip_strength, trigger_word)
