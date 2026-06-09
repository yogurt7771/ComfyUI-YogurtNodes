from typing import Any, Tuple

from ..utils import ANY_TYPE, DYNAMIC_INPUT_COUNT, make_dynamic_inputs


INPUT_COUNT = DYNAMIC_INPUT_COUNT


class StringFormat:
    """String Format node.

    Format strings
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "format": ("STRING", {"multiline": True}),
            },
            "optional": {
                **make_dynamic_inputs(
                    "input",
                    ANY_TYPE,
                    count=INPUT_COUNT,
                    start_index=0,
                    tooltip="The input to be formatted. {index}",
                ),
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

    def main(
        self,
        **kwargs: Any,
    ) -> Tuple[str]:
        format_str = kwargs["format"]
        inputs = {k: v for k, v in kwargs.items() if k.startswith("input")}
        formatted_str = format_str.format(**inputs)
        return (formatted_str,)
