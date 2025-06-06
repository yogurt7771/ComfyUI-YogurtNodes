TEXT_COUNT = 8


class StringConcat:
    """
    拼接多个字符串，支持自定义分隔符和可变数量的输入
    """

    @classmethod
    def INPUT_TYPES(cls):
        texts = {f"text{i+1}": ("STRING", {"default": "", "multiline": True, "tooltip": f"第{i+1}个文本输入"}) for i in range(TEXT_COUNT)}
        return {
            "required": {
                "separator": ("STRING", {"default": "", "multiline": True, "tooltip": "连接符号"}),
            },
            "optional": {
                **texts,
            },
        }

    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "result",
    )
    OUTPUT_NODE = False

    FUNCTION = "concat_strings"

    _NODE_NAME = "String Concat"
    DESCRIPTION = "拼接多个字符串，支持自定义分隔符和可变数量的输入"
    CATEGORY = "YogurtNodes/String"

    def concat_strings(
        self,
        separator: str = "",
        **kwargs
    ):
        # 收集指定数量的文本输入
        texts = [kwargs[f"text{i+1}"] for i in range(TEXT_COUNT)]
        texts = [t for t in texts if t is not None and len(t) > 0]
        # 使用分隔符连接所有文本
        result = separator.join(texts)
        return (result,)
