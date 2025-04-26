import re

from ..utils import ANY_TYPE


SWITCH_CASE_NUM = 8


class Switch:
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
                        "default": "",
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
        case1,
        case2,
        case3,
        case4,
        case5,
        case6,
        case7,
        case8,
        option1=None,
        option2=None,
        option3=None,
        option4=None,
        option5=None,
        option6=None,
        option7=None,
        option8=None,
        default=None,
    ):
        cases = [case1, case2, case3, case4, case5, case6, case7, case8]
        options = [
            option1,
            option2,
            option3,
            option4,
            option5,
            option6,
            option7,
            option8,
        ]
        str_condition = str(condition)
        for i, (case, _option) in enumerate(zip(cases, options)):
            if re.fullmatch(case, str_condition):
                return f"option{i+1}"
        if default is not None:
            return "default"
        return None

    def execute(
        self,
        condition,
        case1,
        case2,
        case3,
        case4,
        case5,
        case6,
        case7,
        case8,
        option1=None,
        option2=None,
        option3=None,
        option4=None,
        option5=None,
        option6=None,
        option7=None,
        option8=None,
        default=None,
    ):
        option = self.check_lazy_status(
            condition,
            case1,
            case2,
            case3,
            case4,
            case5,
            case6,
            case7,
            case8,
            option1,
            option2,
            option3,
            option4,
            option5,
            option6,
            option7,
            option8,
            default,
        )
        if option is None:
            return (None,)
        return (eval(option),)
