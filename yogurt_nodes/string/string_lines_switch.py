from math import prod

TEXT_COUNT = 8


class StringLinesSwitch:
    """String Lines Switch node.

    Get line from multiline string by index
    """

    @classmethod
    def INPUT_TYPES(cls):
        texts = {f"text{i+1}": ("STRING", {"default": "", "multiline": True}) for i in range(TEXT_COUNT)}
        return {
            "required": {
                "no_strip": ("BOOLEAN", {"default": False}),
                "keep_empty_lines": ("BOOLEAN", {"default": False}),
                "index": ("INT", {"default": 0, "step": 1}),
                "concat_method": (["concat", "product"], {"default": "concat"}),
                "product_concat_delimiter": ("STRING", {"default": ""}),
                "product_order": ("STRING", {"default": ",".join(str(i) for i in range(TEXT_COUNT))}),
            },
            "optional": {
                **texts,
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
        no_strip: bool = False,
        keep_empty_lines: bool = False,
        index: int = 0,
        concat_method: str = "concat",
        product_concat_delimiter: str = "",
        product_order: str = ",".join(str(i) for i in range(TEXT_COUNT)),
        **kwargs
    ):
        texts = [kwargs[f"text{i+1}"] for i in range(TEXT_COUNT)]
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
