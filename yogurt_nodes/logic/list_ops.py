from ..utils import ANY_TYPE


class ListLength:
    """获取任何列表类型对象的长度。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_data": (
                    ANY_TYPE,
                    {"tooltip": "任何支持len()函数的列表类型对象"},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("length",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "ListLength"
    DESCRIPTION = "获取任何列表类型对象的长度"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, list_data):
        return (len(list_data),)


class ListIndex:
    """通过索引从任何列表类型对象中获取元素。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_data": (
                    ANY_TYPE,
                    {"tooltip": "任何支持索引访问的列表类型对象"},
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "tooltip": "要访问的索引（支持负索引）",
                        "control_after_generate": True,
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("item",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "ListIndex"
    DESCRIPTION = "通过索引从任何列表类型对象中获取元素"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, list_data, index):
        return (list_data[index],)


class ListSlice:
    """从任何列表类型对象中获取切片。
    
    start: 起始索引（包含）。None表示从开头开始
    stop: 结束索引（不包含）。None表示到结尾
    step: 步长。默认为1
    use_none_start: 使用None作为起始位置（从开头切片）
    use_none_stop: 使用None作为结束位置（切片到结尾）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_data": (
                    ANY_TYPE,
                    {"tooltip": "任何支持切片操作的列表类型对象"},
                ),
            },
            "optional": {
                "start": (
                    "INT",
                    {
                        "default": 0,
                        "tooltip": "起始索引（包含）。None表示从开头开始",
                        "min": -2**31,
                        "max": 2**31 - 1,
                    },
                ),
                "stop": (
                    "INT",
                    {
                        "default": -1,
                        "min": -2**31,
                        "max": 2**31 - 1,
                        "tooltip": "结束索引（不包含）。None表示到结尾",
                    },
                ),
                "step": (
                    "INT",
                    {
                        "default": 1,
                        "min": -2**31,
                        "max": 2**31 - 1,
                        "tooltip": "步长。默认为1",
                    },
                ),
                "use_none_start": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "使用None作为起始位置（从开头切片）",
                    },
                ),
                "use_none_stop": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "使用None作为结束位置（切片到结尾）",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("slice",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "ListSlice"
    DESCRIPTION = "从任何列表类型对象中获取切片"
    CATEGORY = "YogurtNodes/Logic"

    def execute(
        self,
        list_data,
        start=0,
        stop=-1,
        step=1,
        use_none_start=False,
        use_none_stop=True,
    ):
        actual_start = None if use_none_start else start
        actual_stop = None if use_none_stop else stop
        return (list_data[actual_start:actual_stop:step],)
