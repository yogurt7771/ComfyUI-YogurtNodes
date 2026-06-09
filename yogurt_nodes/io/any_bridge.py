from ..utils import ANY_TYPE, DYNAMIC_INPUT_COUNT, make_dynamic_inputs


BLACKHOLE_NUM = DYNAMIC_INPUT_COUNT


class AnyBridge:
    """Any Bridge node."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (ANY_TYPE, {"tooltip": "The data to be returned."}),
            },
            "optional": {
                **make_dynamic_inputs(
                    "blackhole",
                    ANY_TYPE,
                    count=BLACKHOLE_NUM,
                    tooltip="The data will not be returned. {index}",
                ),
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

    def execute(self, data, **blackholes):
        show_data = str(data)
        return {"ui": {"text": show_data}, "result": (data,)}
