from ..utils import ANY_TYPE


BLACKHOLE_NUM = 8


class AnyBridge:
    @classmethod
    def INPUT_TYPES(cls):
        blackholes = {
            f"blackhole{i}": (
                ANY_TYPE,
                {"tooltip": f"the data will not be returned. {i}"},
            )
            for i in range(1, BLACKHOLE_NUM + 1)
        }
        return {
            "required": {
                "data": (ANY_TYPE, {"tooltip": "The data to be returned."}),
            },
            "optional": {
                **blackholes,
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("data",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "Any Bridge"
    DESCRIPTION = "Any Bridge"
    CATEGORY = "YogurtNodes/IO"

    def execute(self, data, **blackholes):
        return (data,)
