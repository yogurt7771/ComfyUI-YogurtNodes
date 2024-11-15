
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

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "INT",)
    RETURN_NAMES = ("string", "int", "float", "count")
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
        try:
            int_result = int(result.strip())
        except ValueError:
            int_result = 0
        try:
            float_result = float(result.strip())
        except ValueError:
            float_result = 0.0
        return {"ui": {"text": [result]}, "result": (result, int_result, float_result, count)}
