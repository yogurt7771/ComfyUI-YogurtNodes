from pathlib import Path
from typing import Tuple

import folder_paths
from folder_paths import get_filename_list


class CheckpointSelector:
    """Checkpoint Selector node.

    Select Checkpoint
    """

    initial_list = get_filename_list("checkpoints")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (
                    ["None"] + get_filename_list("checkpoints"),
                    {"default": "None", "tooltip": "Checkpoint to use."},
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
        "checkpoint",
        "name",
        "stem",
        "trigger_word",
        "absolute_path",
    )
    OUTPUT_NODE = False

    FUNCTION = "checkpoint_selector"

    _NODE_NAME = "Checkpoint Selector"
    DESCRIPTION = "Select Checkpoint"

    @staticmethod
    def _resolve_absolute_path(checkpoint: str) -> str:
        if not checkpoint or checkpoint == "None":
            return ""

        get_full_path = getattr(folder_paths, "get_full_path", None)
        if callable(get_full_path):
            resolved = get_full_path("checkpoints", checkpoint)
            if resolved:
                return str(Path(resolved))

        get_full_path_or_raise = getattr(folder_paths, "get_full_path_or_raise", None)
        if callable(get_full_path_or_raise):
            try:
                return str(Path(get_full_path_or_raise("checkpoints", checkpoint)))
            except Exception:
                return ""

        return ""

    def checkpoint_selector(self, checkpoint: str, trigger_word: str) -> Tuple[str, str, str, str, str]:
        checkpoint_path = Path(checkpoint)
        name = checkpoint_path.name
        stem = checkpoint_path.stem
        absolute_path = self._resolve_absolute_path(checkpoint)
        return (checkpoint, str(name), str(stem), trigger_word, absolute_path)
