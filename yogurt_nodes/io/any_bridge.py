class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


class AnyBridge:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("*", {}),
            },
            "optional": {
                "blackhole1": (
                    "*",
                    {"help": "the data will not be returned."},
                ),
                "blackhole2": (
                    "*",
                    {"help": "the data will not be returned."},
                ),
                "blackhole3": (
                    "*",
                    {"help": "the data will not be returned."},
                ),
                "blackhole4": (
                    "*",
                    {"help": "the data will not be returned."},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("data",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    _NODE_NAME = "Any Bridge"
    DESCRIPTION = "Any Bridge"
    CATEGORY = "YogurtNodes/IO"

    def execute(self, data, *blackholes):
        return (data,)
