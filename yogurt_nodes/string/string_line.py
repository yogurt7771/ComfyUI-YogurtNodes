
class StringLinesSwitch:
    """
    Get line from multiline string by index
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "index": ("INT", {"default": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("text", "count")
    OUTPUT_NODE = True

    FUNCTION = "get_line"

    _NODE_NAME = "String Lines Switch"
    DESCRIPTION = "Get line from multiline string by index"
    CATEGORY = "YogurtNodes/String"

    def get_line(self, text: str, index: int):
        lines = str(text).splitlines()
        count = len(lines)
        try:
            result = lines[index]
        except IndexError:
            result = ""
        return {"result": (result, count), "ui": {"text": result}}
