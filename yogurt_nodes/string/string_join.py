from typing import Any, List

from ..utils import ANY_TYPE, DYNAMIC_INPUT_COUNT, make_dynamic_inputs, ordered_dynamic_values


INPUT_COUNT = DYNAMIC_INPUT_COUNT


def append_any_as_str_(lists: List[str], data: Any):
    if data is None:
        return
    elif isinstance(data, str):
        if data.strip():
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
    """String Join node.

    将多个字符串使用指定连接符连接
    """

    @classmethod
    def INPUT_TYPES(cls):
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
                **make_dynamic_inputs(
                    "item",
                    ANY_TYPE,
                    count=INPUT_COUNT,
                    tooltip="The {index}th item to join.",
                ),
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

    def execute(
        self,
        separator: str,
        **kwargs,
    ):
        """
        执行字符串连接操作，会自动将输入的任何类型转换为字符串，并连接起来。

        Args:
            separator: 连接符
            item1~item32: dynamic item inputs

        Returns:
            tuple: 包含连接结果的元组
        """
        # 收集所有非空字符串
        strings: List[str] = []
        for item in ordered_dynamic_values(kwargs, "item", count=INPUT_COUNT):
            append_any_as_str_(strings, item)
        result = separator.join(strings)

        return (result,)
