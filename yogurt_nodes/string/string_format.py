from typing import Any, Tuple

from ..utils import ANY_TYPE


INPUT_COUNT = 10


class StringFormat:
    """String Format node.

    Format strings
    """

    @classmethod
    def INPUT_TYPES(cls):
        texts = {f"input{i}": (ANY_TYPE, {"tooltip": f"The input to be formatted. {i}"}) for i in range(INPUT_COUNT)}
        return {
            "required": {
                "format": ("STRING", {"multiline": True}),
            },
            "optional": {
                **texts,
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "formatted",
    )
    OUTPUT_NODE = False

    FUNCTION = "main"

    _NODE_NAME = "String Format"
    DESCRIPTION = "Format strings"
    CATEGORY = "YogurtNodes/String"

    def main(
        self,
        **kwargs: Any,
    ) -> Tuple[str]:
        format_str = kwargs["format"]
        inputs = {k: v for k, v in kwargs.items() if k.startswith("input")}
        formatted_str = format_str.format(**inputs)
        return (formatted_str,)
