class StringLinesCount:
    """
    Get the number of lines in a multiline string
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "no_strip": ("BOOLEAN", {"default": False}),
                "keep_empty_lines": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    OUTPUT_NODE = True

    FUNCTION = "get_count"

    _NODE_NAME = "String Lines Count"
    DESCRIPTION = "Get the number of lines in a multiline string"
    CATEGORY = "YogurtNodes/String"

    def get_count(
        self,
        text: str = "",
        no_strip: bool = False,
        keep_empty_lines: bool = False,
    ):
        lines = str(text).split("\n")
        if not no_strip:
            lines = [line.strip() for line in lines]
        if not keep_empty_lines:
            lines = [line for line in lines if line]
        count = len(lines)
        return (count,)
