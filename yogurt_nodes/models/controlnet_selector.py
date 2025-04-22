from pathlib import Path

from folder_paths import get_filename_list


class ControlNetSelector:
    """
    Select ControlNet from a list of ControlNets
    """

    initial_list = get_filename_list("controlnet")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet": (
                    ["None"] + get_filename_list("controlnet"),
                    {"default": "None", "tooltip": "ControlNet to use."},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "step": 0.1},
                    {"tooltip": "Strength of the ControlNet."},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1},
                    {"tooltip": "Start percent of the ControlNet."},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1},
                    {"tooltip": "End percent of the ControlNet."},
                ),
            }
        }

    RETURN_TYPES = (
        initial_list,
        "STRING",
        "STRING",
        "FLOAT",
        "FLOAT",
        "FLOAT",
    )
    RETURN_NAMES = (
        "controlnet",
        "name",
        "stem",
        "strength",
        "start_percent",
        "end_percent",
    )
    OUTPUT_NODE = False

    FUNCTION = "controlnet_selector"

    _NODE_NAME = "ControlNet Selector"
    DESCRIPTION = "Select ControlNet"
    CATEGORY = "YogurtNodes/Models"

    def controlnet_selector(
        self, controlnet: str, strength: float, start_percent: float, end_percent: float
    ):
        controlnet_path = Path(controlnet)
        name = controlnet_path.name
        stem = controlnet_path.stem
        return (controlnet, str(name), str(stem), strength, start_percent, end_percent)
