from ..utils import ANY_TYPE


class ListLength:
    """Get the length of any list-like object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_data": (
                    ANY_TYPE,
                    {"tooltip": "Any list-like object that supports len()"},
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
    DESCRIPTION = "Get the length of any list-like object"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, list_data):
        return (len(list_data),)


class ListIndex:
    """Get an item by index from any list-like object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_data": (
                    ANY_TYPE,
                    {"tooltip": "Any list-like object that supports indexing"},
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "tooltip": "Index to access (supports negative indexing)",
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
    DESCRIPTION = "Get an item by index from any list-like object"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, list_data, index):
        return (list_data[index],)


class ListSlice:
    """Get a slice from any list-like object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_data": (
                    ANY_TYPE,
                    {"tooltip": "Any list-like object that supports slicing"},
                ),
            },
            "optional": {
                "start": (
                    "INT",
                    {
                        "default": 0,
                        "tooltip": "Start index (inclusive). None means from beginning",
                    },
                ),
                "stop": (
                    "INT",
                    {
                        "default": -1,
                        "tooltip": "Stop index (exclusive). None means to end",
                    },
                ),
                "step": (
                    "INT",
                    {
                        "default": 1,
                        "tooltip": "Step size. Default is 1",
                    },
                ),
                "use_none_start": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use None as start (slice from beginning)",
                    },
                ),
                "use_none_stop": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use None as stop (slice to end)",
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
    DESCRIPTION = "Get a slice from any list-like object"
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
