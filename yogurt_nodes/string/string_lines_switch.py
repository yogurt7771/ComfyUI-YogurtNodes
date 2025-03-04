
class StringLinesSwitch:
    """
    Get line from multiline string by index
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "no_strip": ("BOOLEAN", {"default": False}),
                "keep_empty_lines": ("BOOLEAN", {"default": False}),
                "index": ("INT", {"default": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "INT",)
    RETURN_NAMES = ("string", "int", "float", "count")
    OUTPUT_NODE = True

    FUNCTION = "get_line"

    _NODE_NAME = "String Lines Switch"
    DESCRIPTION = "Get line from multiline string by index"
    CATEGORY = "YogurtNodes/String"

    def get_line(self, text: str = "", no_strip: bool = False, keep_empty_lines: bool = False, index: int = 0):
        lines = str(text).split("\n")
        if not no_strip:
            lines = [line.strip() for line in lines]
        if not keep_empty_lines:
            lines = [line for line in lines if line]
        count = len(lines)
        try:
            result = lines[index]
        except IndexError:
            result = ""
        try:
            int_result = int(result.strip())
        except ValueError:
            int_result = 0
        try:
            float_result = float(result.strip())
        except ValueError:
            float_result = 0.0
        return {"ui": {"text": [result]}, "result": (result, int_result, float_result, count)}
