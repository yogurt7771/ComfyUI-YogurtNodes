# from itertools import chain, product
from math import prod


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
                "concat_method": (["concat", "product"], {"default": "concat"}),
            },
            "optional": {
                "text1": ("STRING", {"default": "", "multiline": True}),
                "text2": ("STRING", {"default": "", "multiline": True}),
                "text3": ("STRING", {"default": "", "multiline": True}),
                "text4": ("STRING", {"default": "", "multiline": True}),
                "text5": ("STRING", {"default": "", "multiline": True}),
                "text6": ("STRING", {"default": "", "multiline": True}),
                "text7": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = (
        "INT",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "count",
        "text",
        "text1",
        "text2",
        "text3",
        "text4",
        "text5",
        "text6",
        "text7",
    )
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
        concat_method: str = "concat",
        text1: str = None,
        text2: str = None,
        text3: str = None,
        text4: str = None,
        text5: str = None,
        text6: str = None,
        text7: str = None,
    ):
        texts = [text, text1, text2, text3, text4, text5, text6, text7]
        texts = [t for t in texts if t is not None and len(t) > 0]
        lines_list = []
        for t in texts:
            text_lines = str(t).split("\n")
            if not no_strip:
                text_lines = [line.strip() for line in text_lines]
            if not keep_empty_lines:
                text_lines = [line for line in text_lines if line]
            if len(text_lines) > 0:
                lines_list.append(text_lines)
        if concat_method == "concat":
            # all_lines = list(chain(*lines_list))
            count = sum(len(lines) for lines in lines_list)
        elif concat_method == "product":
            # all_lines = list(product(*lines_list))
            count = prod(len(lines) for lines in lines_list)
        else:
            raise ValueError(f"Invalid concat_method value: {concat_method}")
        return (count, text, text1, text2, text3, text4, text5, text6, text7)
