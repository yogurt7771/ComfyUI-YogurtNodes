from ..utils import ANY_TYPE


class NoneNode:

    @classmethod
    def INPUT_TYPES(s):
        return {}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("none",)
    FUNCTION = "execute"

    OUTPUT_NODE = False

    _NODE_NAME = "None"
    CATEGORY = "YogurtNodes/Logic"
    DESCRIPTION = "Return None."

    def execute(self):
        return (None,)
