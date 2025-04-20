from pathlib import Path

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
                    ["None"] + cls.initial_list,
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
    )
    RETURN_NAMES = (
        "diffusion_model",
        "name",
        "stem",
        "trigger_word",
    )
    OUTPUT_NODE = False

    FUNCTION = "diffusion_model_selector"

    _NODE_NAME = "Diffusion Model Selector"
    DESCRIPTION = "Select Diffusion Model"
    CATEGORY = "YogurtNodes/Models"

    def diffusion_model_selector(self, diffusion_model: str, trigger_word: str):
        diffusion_model_path = Path(diffusion_model)
        name = diffusion_model_path.name
        stem = diffusion_model_path.stem
        return (diffusion_model, str(name), str(stem), trigger_word)
