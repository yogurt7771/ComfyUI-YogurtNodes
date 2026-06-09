from ..utils import ANY_TYPE, DYNAMIC_INPUT_COUNT, make_dynamic_inputs


PACK_NUM = DYNAMIC_INPUT_COUNT


class PackAny:

    """PackAny node.

    Pack any
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": make_dynamic_inputs(
                "item",
                ANY_TYPE,
                count=PACK_NUM,
                tooltip="The {index}th item to pack.",
            ),
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("PACK_TUPLE",)
    RETURN_NAMES = ("pack",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "PackAny"
    DESCRIPTION = "Pack any"

    def execute(
        self,
        **items,
    ):
        result = []
        for i in range(1, PACK_NUM + 1):
            result.append(items.get(f"item{i}", None))
        return (tuple(result),)


class UnpackAny:

    """UnpackAny node.

    Unpack any
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pack": ("PACK_TUPLE",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,) * PACK_NUM
    RETURN_NAMES = tuple(f"item{i}" for i in range(1, PACK_NUM + 1))
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "UnpackAny"
    DESCRIPTION = "Unpack any"

    def execute(self, pack):
        return (*pack,)
