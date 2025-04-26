# from itertools import product, chain
from math import prod


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
                "concat_method": (["concat", "product"], {"default": "concat"}),
                "product_concat_delimiter": ("STRING", {"default": ""}),
                "product_order": ("STRING", {"default": "0,1,2,3,4,5,6,7"}),
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
        "STRING",
        "INT",
        "FLOAT",
        "INT",
    )
    RETURN_NAMES = ("string", "int", "float", "count")
    OUTPUT_NODE = False

    FUNCTION = "get_line"

    _NODE_NAME = "String Lines Switch"
    DESCRIPTION = "Get line from multiline string by index"
    CATEGORY = "YogurtNodes/String"

    def get_line(
        self,
        text: str = "",
        no_strip: bool = False,
        keep_empty_lines: bool = False,
        index: int = 0,
        concat_method: str = "concat",
        product_concat_delimiter: str = "",
        product_order: str = "0,1,2,3,4,5,6,7",
        text1: str = None,
        text2: str = None,
        text3: str = None,
        text4: str = None,
        text5: str = None,
        text6: str = None,
        text7: str = None,
    ):
        texts = [text, text1, text2, text3, text4, text5, text6, text7]
        texts = [text for text in texts if text is not None and len(text) > 0]
        lines_list = []
        for text in texts:
            text_lines = str(text).split("\n")
            if not no_strip:
                text_lines = [line.strip() for line in text_lines]
            if not keep_empty_lines:
                text_lines = [line for line in text_lines if line]
            if len(text_lines) > 0:
                lines_list.append(text_lines)
        if concat_method == "concat":
            # all_lines = list(chain(*lines_list))
            count = sum(len(lines) for lines in lines_list)
            lines_index = 0
            t = index
            while t >= 0:
                t -= len(lines_list[lines_index])
                if t < 0:
                    break
                lines_index += 1
                index -= len(lines_list[lines_index])
            result = lines_list[lines_index][index]
        elif concat_method == "product":
            # all_lines = list(prod(*lines_list))
            count = prod(len(lines) for lines in lines_list)
            # 根据索引组合获取结果
            product_indexes = [int(i) for i in product_order.split(",")]
            product_indexes = product_indexes[: len(lines_list)]
            indices = [0] * len(lines_list)
            t = index
            for dim in reversed(product_indexes):
                size = len(lines_list[dim])
                indices[dim] = t % size
                t //= size
            parts = [lines_list[i][indices[i]] for i in range(len(lines_list))]
            result = product_concat_delimiter.join(parts)
        else:
            raise ValueError(f"Invalid text_concat value: {concat_method}")

        try:
            int_result = int(result.strip())
        except ValueError:
            int_result = 0
        try:
            float_result = float(result.strip())
        except ValueError:
            float_result = 0.0
        return (
            result,
            int_result,
            float_result,
            count,
        )
