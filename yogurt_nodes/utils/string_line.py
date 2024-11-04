

class StringLinesSwitch:
    """
    Get line from multiline string by index
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "tooltip": "Input multiline text"}),
                "index": ("INTEGER", {"tooltip": "Line index, starting from 0"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_line"
    DESCRIPTION = "Get line from multiline string by index"
    CATEGORY = "YogurtNodes/Utils"
    NODE_NAME = "String Lines Switch"

    def get_line(self, text: str, index: int):
        lines = text.splitlines()
        if index < 0 or index >= len(lines):
            return ""
        return lines[index]
