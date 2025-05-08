import torch


class Range:
    """
    Get value from string
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "from": ("FLOAT", {"default": 0.0, "tooltip": "start value"}),
                "to": ("FLOAT", {"default": 1.0, "tooltip": "end value"}),
                "step": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "tooltip": "step size, 0 means unlimited",
                    },
                ),
                "count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "tooltip": "number of steps, 0 means unlimited",
                    },
                ),
            }
        }

    RETURN_TYPES = ("RANGE", "INT", "FLOAT")
    RETURN_NAMES = ("range", "count", "step")
    OUTPUT_NODE = False

    FUNCTION = "execute"

    _NODE_NAME = "Range"
    DESCRIPTION = "get a number from a range"
    CATEGORY = "YogurtNodes/Number"

    def execute(self, **kwargs):
        from_value = kwargs["from"]
        to_value = kwargs["to"]
        step = kwargs["step"]
        count = kwargs["count"]
        if step == 0:
            if count == 0:
                raise ValueError("step and count cannot be both 0")
            step = (to_value - from_value) / max(1, (count - 1))
        data = torch.arange(from_value, to_value, step)
        if count > 0:
            data = data[:count]
        return (data, len(data), step)


class RangeItem:
    """
    Get a value from a range
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "range": ("LIST", {"default": [0, 1, 0.1]}),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "step": 1,
                        "tooltip": "index of the value to return, 0 means first value, negative means last value",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("string", "int", "float")
    OUTPUT_NODE = False

    FUNCTION = "execute"

    _NODE_NAME = "RangeItem"
    DESCRIPTION = "get a value from a range"
    CATEGORY = "YogurtNodes/Number"

    def execute(self, **kwargs):
        data = kwargs["range"]
        index = kwargs["index"]
        if index >= len(data) or index < -len(data):
            raise ValueError("index is out of range")
        item = data[index].item()
        return (str(item), int(item), float(item))
