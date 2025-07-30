from typing import Any, List

from ..utils import ANY_TYPE


INPUT_COUNT = 8


def append_any_as_str_(lists: List[str], data: Any):
    if data is None:
        return
    elif isinstance(data, str):
        lists.append(data)
    elif isinstance(data, (list, tuple, set)):
        for item in data:
            append_any_as_str_(lists, item)
    elif isinstance(data, dict):
        for value in data.values():
            append_any_as_str_(lists, value)
    else:
        lists.append(str(data))


class StringJoin:
    """
    String Join Node
    ================

    字符串连接节点，将多个字符串使用指定的连接符连接成一个字符串。

    输入:
        string1~string8: 8个字符串输入，可以为空
        separator: 连接符，用于连接各个字符串

    输出:
        result: 连接后的字符串
    """

    @classmethod
    def INPUT_TYPES(cls):
        items = {
            f"item{i}": (ANY_TYPE, {"tooltip": f"The {i}th item to join."})
            for i in range(1, INPUT_COUNT + 1)
        }
        return {
            "required": {
                "separator": (
                    "STRING",
                    {
                        "default": ",",
                        "multiline": True,
                        "tooltip": "用于连接字符串的连接符",
                    },
                ),
            },
            "optional": {
                **items,
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    OUTPUT_NODE = False

    FUNCTION = "execute"

    _NODE_NAME = "String Join"
    DESCRIPTION = "将多个字符串使用指定连接符连接"
    CATEGORY = "YogurtNodes/String"

    def execute(
        self,
        separator: str,
        **kwargs,
    ):
        """
        执行字符串连接操作，会自动将输入的任何类型转换为字符串，并连接起来。

        Args:
            separator: 连接符
            item1~item8: 8个字符串输入

        Returns:
            tuple: 包含连接结果的元组
        """
        # 收集所有非空字符串
        strings: List[str] = []
        for k, v in kwargs.items():
            if k.startswith("item"):
                append_any_as_str_(strings, v)
        result = separator.join(strings)

        return (result,)
