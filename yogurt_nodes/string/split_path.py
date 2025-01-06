from pathlib import Path


class SplitPath:
    """
    Split path to parts
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = ("path", "parent", "name", "stem", "suffix", "no_dot_suffix")
    OUTPUT_NODE = False

    FUNCTION = "split_path"

    _NODE_NAME = "Split Path"
    DESCRIPTION = "Split path to parts"
    CATEGORY = "YogurtNodes/String"

    def split_path(self, path: str):
        path = Path(path)
        parent = path.parent
        name = path.name
        stem = path.stem
        suffix = path.suffix
        no_dot_suffix = path.suffix.strip(".")
        return (str(path), str(parent), name, stem, suffix, no_dot_suffix)
