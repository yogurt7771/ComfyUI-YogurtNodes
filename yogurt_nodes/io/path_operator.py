import os
from pathlib import Path


class PathOperator:
    """
    Apply basic binary path operations to two inputs.
    """

    OPERATORS = [
        "path_join",
        "relative",
        "relative_to",
        "common_path",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operator": (
                    cls.OPERATORS,
                    {
                        "default": "path_join",
                        "tooltip": "Select the operation to apply to the two input paths.",
                    },
                ),
                "path_a": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Primary path input.",
                    },
                ),
                "path_b": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Secondary path input used by the selected operator.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    OUTPUT_NODE = False

    FUNCTION = "operate"

    _NODE_NAME = "Path Operator"
    DESCRIPTION = "Execute join, relative, or common path operations."
    CATEGORY = "YogurtNodes/IO"

    def operate(self, operator: str, path_a: str, path_b: str):
        if not path_a:
            raise ValueError("path_a cannot be empty.")
        if not path_b:
            raise ValueError("path_b cannot be empty.")

        path_a_obj = Path(path_a)
        path_b_obj = Path(path_b)

        if operator == "path_join":
            result = path_a_obj.joinpath(path_b)
        elif operator == "relative":
            result = os.path.relpath(str(path_a_obj), start=str(path_b_obj))
        elif operator == "relative_to":
            try:
                result = path_a_obj.relative_to(path_b_obj)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot compute relative_to for '{path_a}' and '{path_b}': {exc}"
                ) from exc
        elif operator == "common_path":
            try:
                result = os.path.commonpath([str(path_a_obj), str(path_b_obj)])
            except ValueError as exc:
                raise ValueError(
                    f"Cannot compute common_path for '{path_a}' and '{path_b}': {exc}"
                ) from exc
        else:
            raise ValueError(f"Unsupported operator: {operator}")

        return (str(result),)
