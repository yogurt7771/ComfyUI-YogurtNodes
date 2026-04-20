from pathlib import Path

import folder_paths
from folder_paths import get_filename_list


class DiffusionModelSelector:
    """
    Select Diffusion Model from a list of Diffusion Models
    """

    initial_list = get_filename_list("diffusion_models")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusion_model": (
                    ["None"] + get_filename_list("diffusion_models"),
                    {"default": "None", "tooltip": "Diffusion Model to use."},
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
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "diffusion_model",
        "name",
        "stem",
        "trigger_word",
        "absolute_path",
    )
    OUTPUT_NODE = False

    FUNCTION = "diffusion_model_selector"

    _NODE_NAME = "Diffusion Model Selector"
    DESCRIPTION = "Select Diffusion Model"
    CATEGORY = "YogurtNodes/Models"

    @staticmethod
    def _resolve_absolute_path(diffusion_model: str) -> str:
        if not diffusion_model or diffusion_model == "None":
            return ""

        get_full_path = getattr(folder_paths, "get_full_path", None)
        if callable(get_full_path):
            resolved = get_full_path("diffusion_models", diffusion_model)
            if resolved:
                return str(Path(resolved))

        get_full_path_or_raise = getattr(folder_paths, "get_full_path_or_raise", None)
        if callable(get_full_path_or_raise):
            try:
                return str(Path(get_full_path_or_raise("diffusion_models", diffusion_model)))
            except Exception:
                return ""

        return ""

    def diffusion_model_selector(self, diffusion_model: str, trigger_word: str):
        diffusion_model_path = Path(diffusion_model)
        name = diffusion_model_path.name
        stem = diffusion_model_path.stem
        absolute_path = self._resolve_absolute_path(diffusion_model)
        return (diffusion_model, str(name), str(stem), trigger_word, absolute_path)
