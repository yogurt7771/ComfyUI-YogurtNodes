import re

from ..utils import ANY_TYPE


SWITCH_CASE_NUM = 8


class Switch:
    """Switch node."""
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        cases = {
            f"case{i}": (
                "STRING",
                {
                    "default": "",
                    "placeholder": "regex expression",
                    "tooltip": f"the regex to match for case{i}.",
                },
            )
            for i in range(1, SWITCH_CASE_NUM + 1)
        }
        options = {
            f"option{i}": (
                ANY_TYPE,
                {"lazy": True, "tooltip": f"The data to return if case{i} matches."},
            )
            for i in range(1, SWITCH_CASE_NUM + 1)
        }
        return {
            "required": {
                "condition": (
                    ANY_TYPE,
                    {
                        "placeholder": "text",
                        "tooltip": "Input condition text.",
                    },
                ),
                **cases,
            },
            "optional": {
                **options,
                "default": (
                    ANY_TYPE,
                    {"lazy": True, "tooltip": "The default option to switch to."},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("data",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "Switch"
    DESCRIPTION = "Switch"
    CATEGORY = "YogurtNodes/Logic"

    def check_lazy_status(
        self,
        condition,
        **kwargs,
    ):
        cases = [item[1] for item in kwargs.items() if item[0].startswith("case")]
        str_condition = str(condition)
        for i, case in enumerate(cases):
            if re.fullmatch(case, str_condition):
                return [f"option{i+1}"]
        return ["default"]

    def execute(
        self,
        condition,
        **kwargs,
    ):
        options = [item[1] for item in kwargs.items() if item[0].startswith("option")]
        option = self.check_lazy_status(
            condition,
            **kwargs,
        )[0]
        if option == "default":
            return (kwargs["default"],)
        return (options[int(option.removeprefix("option")) - 1],)
