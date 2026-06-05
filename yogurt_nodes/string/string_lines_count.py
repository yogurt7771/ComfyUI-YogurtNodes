from math import prod

TEXT_COUNT = 8


class StringLinesCount:
    """String Lines Count node.

    Get the number of lines in a multiline string
    """

    @classmethod
    def INPUT_TYPES(cls):
        texts = {f"text{i+1}": ("STRING", {"default": "", "multiline": True}) for i in range(TEXT_COUNT)}
        return {
            "required": {
                "no_strip": ("BOOLEAN", {"default": False}),
                "keep_empty_lines": ("BOOLEAN", {"default": False}),
                "concat_method": (["concat", "product"], {"default": "concat"}),
            },
            "optional": {
                **texts,
            },
        }

    RETURN_TYPES = (
        "INT",
        *["STRING" for _ in range(TEXT_COUNT)],
    )
    RETURN_NAMES = (
        "count",
        *[f"text{i+1}" for i in range(TEXT_COUNT)],
    )
    OUTPUT_NODE = True

    FUNCTION = "get_count"

    _NODE_NAME = "String Lines Count"
    DESCRIPTION = "Get the number of lines in a multiline string"

    def get_count(
        self,
        no_strip: bool = False,
        keep_empty_lines: bool = False,
        concat_method: str = "concat",
        **kwargs
    ):
        all_texts = [kwargs[f"text{i+1}"] for i in range(TEXT_COUNT)]
        texts = [t for t in all_texts if t is not None and len(t) > 0]
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
        return (count, *all_texts)
