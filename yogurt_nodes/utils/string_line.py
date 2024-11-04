

class StringLinesSwitch:
    """
    Get line from multiline string by index
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "index": ("INT", {"default": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_line"
    DESCRIPTION = "Get line from multiline string by index"
    CATEGORY = "YogurtNodes/Utils"
    NODE_NAME = "String Lines Switch"

    def get_line(self, text: str, index: int):
        lines = text.splitlines()
        try:
            return lines[index]
        except IndexError:
            return ""
