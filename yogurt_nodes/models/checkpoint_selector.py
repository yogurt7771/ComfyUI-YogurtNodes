from pathlib import Path

from folder_paths import get_filename_list


class CheckpointSelector:
    """
    Select Checkpoint from a list of Checkpoints
    """

    initial_list = get_filename_list("checkpoints")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (
                    ["None"] + cls.initial_list,
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
    )
    RETURN_NAMES = (
        "checkpoint",
        "name",
        "stem",
        "trigger_word",
    )
    OUTPUT_NODE = False

    FUNCTION = "checkpoint_selector"

    _NODE_NAME = "Checkpoint Selector"
    DESCRIPTION = "Select Checkpoint"
    CATEGORY = "YogurtNodes/Models"

    def checkpoint_selector(self, checkpoint: str, trigger_word: str):
        checkpoint_path = Path(checkpoint)
        name = checkpoint_path.name
        stem = checkpoint_path.stem
        return (checkpoint, str(name), str(stem), trigger_word)
