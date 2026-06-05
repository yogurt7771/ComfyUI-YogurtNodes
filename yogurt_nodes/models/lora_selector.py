from pathlib import Path

from folder_paths import get_filename_list
import folder_paths


class LoraSelector:
    """Lora Selector node.

    Select Lora
    """

    initial_list = get_filename_list("loras")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": (
                    ["None"] + folder_paths.get_filename_list("loras"),
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
        "STRING",
    )
    RETURN_NAMES = (
        "lora",
        "name",
        "stem",
        "model_strength",
        "clip_strength",
        "trigger_word",
        "absolute_path",
    )
    OUTPUT_NODE = False

    FUNCTION = "lora_selector"

    _NODE_NAME = "Lora Selector"
    DESCRIPTION = "Select Lora"

    @staticmethod
    def _resolve_absolute_path(lora: str) -> str:
        if not lora or lora == "None":
            return ""

        get_full_path = getattr(folder_paths, "get_full_path", None)
        if callable(get_full_path):
            resolved = get_full_path("loras", lora)
            if resolved:
                return str(Path(resolved))

        get_full_path_or_raise = getattr(folder_paths, "get_full_path_or_raise", None)
        if callable(get_full_path_or_raise):
            try:
                return str(Path(get_full_path_or_raise("loras", lora)))
            except Exception:
                return ""

        return ""

    def lora_selector(
        self, lora: str, model_strength: float, clip_strength: float, trigger_word: str
    ):
        lora_path = Path(lora)
        name = lora_path.name
        stem = lora_path.stem
        absolute_path = self._resolve_absolute_path(lora)
        return (
            lora,
            str(name),
            str(stem),
            model_strength,
            clip_strength,
            trigger_word,
            absolute_path,
        )
