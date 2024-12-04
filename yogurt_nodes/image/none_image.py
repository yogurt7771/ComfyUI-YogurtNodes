
class NoneImage:
    @classmethod
    def INPUT_TYPES(s):
        return {}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "none_image"

    OUTPUT_NODE = False

    _NODE_NAME = "None Image"
    CATEGORY = "YogurtNodes/Image"
    DESCRIPTION = "Return None."

    def none_image(self):
        return (None,)
