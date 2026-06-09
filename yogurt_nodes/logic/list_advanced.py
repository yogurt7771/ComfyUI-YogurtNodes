import re
from ..utils import ANY_TYPE, DYNAMIC_INPUT_COUNT, make_dynamic_inputs, ordered_dynamic_values


LIST_CONCAT_INPUT_COUNT = DYNAMIC_INPUT_COUNT


class ListContains:
    """ListContains node.

    Check if a list contains a specific element
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_data": (
                    ANY_TYPE,
                    {"tooltip": "Any list-like object"},
                ),
                "element": (
                    ANY_TYPE,
                    {"tooltip": "Element to search for"},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("contains",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "ListContains"
    DESCRIPTION = "Check if a list contains a specific element"

    def execute(self, list_data, element):
        return (element in list_data,)


class ListFind:
    """ListFind node.

    Find the index of an element in a list
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_data": (
                    ANY_TYPE,
                    {"tooltip": "Any list-like object"},
                ),
                "element": (
                    ANY_TYPE,
                    {"tooltip": "Element to find"},
                ),
            },
            "optional": {
                "default_index": (
                    "INT",
                    {
                        "default": -1,
                        "tooltip": "Index to return if element not found",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("index",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "ListFind"
    DESCRIPTION = "Find the index of an element in a list"

    def execute(self, list_data, element, default_index=-1):
        try:
            return (list_data.index(element),)
        except ValueError:
            return (default_index,)


class ListConcat:
    """ListConcat node.

    Concatenate multiple lists
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list1": (
                    ANY_TYPE,
                    {"tooltip": "First list"},
                ),
                "list2": (
                    ANY_TYPE,
                    {"tooltip": "Second list"},
                ),
            },
            "optional": {
                **make_dynamic_inputs(
                    "list",
                    ANY_TYPE,
                    count=LIST_CONCAT_INPUT_COUNT - 2,
                    start_index=3,
                    tooltip="Additional list {index} (optional)",
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "ListConcat"
    DESCRIPTION = "Concatenate multiple lists"

    def execute(self, list1, list2, list3=None, list4=None, **kwargs):
        result = list(list1) + list(list2)
        if list3 is not None:
            result.extend(list3)
        if list4 is not None:
            result.extend(list4)
        for list_value in ordered_dynamic_values(
            kwargs,
            "list",
            count=LIST_CONCAT_INPUT_COUNT - 4,
            start_index=5,
        ):
            result.extend(list_value)
        return (result,)


class ListUnique:
    """ListUnique node.

    Remove duplicate elements from a list while preserving order
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_data": (
                    ANY_TYPE,
                    {"tooltip": "Any list-like object"},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("unique_list",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "ListUnique"
    DESCRIPTION = "Remove duplicate elements from a list while preserving order"

    def execute(self, list_data):
        seen = set()
        result = []
        for item in list_data:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return (result,)


class ListJoin:
    """ListJoin node.

    Join list elements into a string
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_data": (
                    ANY_TYPE,
                    {"tooltip": "List of elements to join"},
                ),
            },
            "optional": {
                "separator": (
                    "STRING",
                    {
                        "default": ", ",
                        "multiline": True,
                        "tooltip": "Separator to use between elements",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "ListJoin"
    DESCRIPTION = "Join list elements into a string"

    def execute(self, list_data, separator=", "):
        return (separator.join(str(item) for item in list_data),)


class ListFilter:
    """ListFilter node.

    Filter list elements based on regex pattern
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_data": (
                    ANY_TYPE,
                    {"tooltip": "List to filter"},
                ),
                "pattern": (
                    "STRING",
                    {
                        "default": ".*",
                        "multiline": True,
                        "tooltip": "Regex pattern to match",
                    },
                ),
            },
            "optional": {
                "invert": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Invert the filter (exclude matches)",
                    },
                ),
                "ignore_case": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Case sensitive matching",
                    },
                ),
                "multiline": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Multiline matching",
                    },
                ),
                "dotall": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Dotall matching",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("filtered_list",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "ListFilter"
    DESCRIPTION = "Filter list elements based on regex pattern"

    def execute(
        self,
        list_data,
        pattern=".*",
        invert=False,
        ignore_case=True,
        multiline=False,
        dotall=False,
    ):
        flags = 0
        if ignore_case:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL
        compiled_pattern = re.compile(pattern, flags)

        result = []
        for item in list_data:
            str_item = str(item)
            matches = compiled_pattern.search(str_item) is not None
            if matches != invert:  # XOR logic
                result.append(item)

        return (result,)
