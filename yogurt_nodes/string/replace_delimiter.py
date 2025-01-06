import re


class ReplaceDelimiter:
    """
    Replace Delimiter
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "from": ("STRING", {"default": "/", "multiline": False}),
                "to": ("STRING", {"default": "_", "multiline": False}),
            }
        }

    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "text",
    )
    OUTPUT_NODE = True

    FUNCTION = "replace_delimiter"

    _NODE_NAME = "Replace Delimiter"
    DESCRIPTION = "Replace delimiter in string. Support regex"
    CATEGORY = "YogurtNodes/String"

    def replace_delimiter(self, text: str, from_: str, to: str):
        new_text = re.sub(re.escape(from_), to, text)
        return (new_text,)
