from ..utils import ANY_TYPE


PACK_NUM = 8


class PackAny:

    @classmethod
    def INPUT_TYPES(s):
        items = {
            f"item{i}": (
                ANY_TYPE,
                {"tooltip": f"The {i}th item to pack."},
            )
            for i in range(1, PACK_NUM + 1)
        }
        return {
            "optional": {**items},
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = ("PACK_TUPLE",)
    RETURN_NAMES = ("pack",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "PackAny"
    DESCRIPTION = "Pack any"
    CATEGORY = "YogurtNodes/Logic"

    def execute(
        self,
        **items,
    ):
        result = []
        for i in range(1, PACK_NUM + 1):
            result.append(items.get(f"item{i}", None))
        return (tuple(result),)


class UnpackAny:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pack": ("PACK_TUPLE",),
            },
        }

    RETURN_TYPES = (ANY_TYPE,) * PACK_NUM
    RETURN_NAMES = tuple(f"item{i}" for i in range(1, PACK_NUM + 1))
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "UnpackAny"
    DESCRIPTION = "Unpack any"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, pack):
        return (*pack,)
