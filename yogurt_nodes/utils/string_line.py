

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

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("line", "count")
    FUNCTION = "get_line"
    DESCRIPTION = "Get line from multiline string by index"
    CATEGORY = "YogurtNodes/Utils"
    NODE_NAME = "String Lines Switch"
    OUTPUT_NODE = True

    def get_line(self, text: str, index: int):
        lines = str(text).splitlines()
        count = len(lines)
        try:
            result = lines[index]
        except IndexError:
            result = ""
        return {"ui": {"text": result}, "result": (result, count)}
