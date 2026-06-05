from pathlib import Path

import folder_paths
from folder_paths import get_filename_list


class ControlNetSelector:
    """ControlNet Selector node.

    Select ControlNet
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
        "STRING",
    )
    RETURN_NAMES = (
        "controlnet",
        "name",
        "stem",
        "strength",
        "start_percent",
        "end_percent",
        "absolute_path",
    )
    OUTPUT_NODE = False

    FUNCTION = "controlnet_selector"

    _NODE_NAME = "ControlNet Selector"
    DESCRIPTION = "Select ControlNet"

    @staticmethod
    def _resolve_absolute_path(controlnet: str) -> str:
        if not controlnet or controlnet == "None":
            return ""

        get_full_path = getattr(folder_paths, "get_full_path", None)
        if callable(get_full_path):
            resolved = get_full_path("controlnet", controlnet)
            if resolved:
                return str(Path(resolved))

        get_full_path_or_raise = getattr(folder_paths, "get_full_path_or_raise", None)
        if callable(get_full_path_or_raise):
            try:
                return str(Path(get_full_path_or_raise("controlnet", controlnet)))
            except Exception:
                return ""

        return ""

    def controlnet_selector(
        self, controlnet: str, strength: float, start_percent: float, end_percent: float
    ):
        controlnet_path = Path(controlnet)
        name = controlnet_path.name
        stem = controlnet_path.stem
        absolute_path = self._resolve_absolute_path(controlnet)
        return (
            controlnet,
            str(name),
            str(stem),
            strength,
            start_percent,
            end_percent,
            absolute_path,
        )
