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
                "blackhole": ("*", {"help": "If true, the data will not be returned."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("data",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    _NODE_NAME = "Any Bridge"
    DESCRIPTION = "Any Bridge"
    CATEGORY = "YogurtNodes/IO"

    def execute(self, data, blackhole=None):
        return (data,)
