from ..utils import ANY_TYPE


class DictContainsKey:
    """Check if a dictionary contains a specific key."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dict_data": (
                    ANY_TYPE,
                    {"tooltip": "Any dict-like object"},
                ),
                "key": (
                    "STRING",
                    {"tooltip": "Key to search for"},
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

    _NODE_NAME = "DictContainsKey"
    DESCRIPTION = "Check if a dictionary contains a specific key"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, dict_data, key):
        return (key in dict_data,)


class DictContainsValue:
    """Check if a dictionary contains a specific value."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dict_data": (
                    ANY_TYPE,
                    {"tooltip": "Any dict-like object"},
                ),
                "value": (
                    ANY_TYPE,
                    {"tooltip": "Value to search for"},
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

    _NODE_NAME = "DictContainsValue"
    DESCRIPTION = "Check if a dictionary contains a specific value"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, dict_data, value):
        return (value in dict_data.values(),)


class DictMerge:
    """Merge multiple dictionaries."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dict1": (
                    ANY_TYPE,
                    {"tooltip": "First dictionary"},
                ),
                "dict2": (
                    ANY_TYPE,
                    {"tooltip": "Second dictionary"},
                ),
            },
            "optional": {
                "dict3": (
                    ANY_TYPE,
                    {"tooltip": "Third dictionary (optional)"},
                ),
                "dict4": (
                    ANY_TYPE,
                    {"tooltip": "Fourth dictionary (optional)"},
                ),
                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Whether later dicts should overwrite earlier keys",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("merged_dict",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "DictMerge"
    DESCRIPTION = "Merge multiple dictionaries"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, dict1, dict2, dict3=None, dict4=None, overwrite=True):
        result = dict(dict1)
        
        dicts_to_merge = [dict2]
        if dict3 is not None:
            dicts_to_merge.append(dict3)
        if dict4 is not None:
            dicts_to_merge.append(dict4)
        
        for d in dicts_to_merge:
            if overwrite:
                result.update(d)
            else:
                for key, value in d.items():
                    if key not in result:
                        result[key] = value
        
        return (result,)


class DictInvert:
    """Invert a dictionary (swap keys and values)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dict_data": (
                    ANY_TYPE,
                    {"tooltip": "Dictionary to invert"},
                ),
            },
            "optional": {
                "handle_duplicates": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If True, duplicate values become lists of keys",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("inverted_dict",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "DictInvert"
    DESCRIPTION = "Invert a dictionary (swap keys and values)"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, dict_data, handle_duplicates=False):
        if not handle_duplicates:
            return ({v: k for k, v in dict_data.items()},)
        
        result = {}
        for key, value in dict_data.items():
            if value in result:
                if not isinstance(result[value], list):
                    result[value] = [result[value]]
                result[value].append(key)
            else:
                result[value] = key
        
        return (result,)


class DictFromLists:
    """Create a dictionary from a list of keys and a list of values."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keys": (
                    ANY_TYPE,
                    {"tooltip": "List of keys"},
                ),
                "values": (
                    ANY_TYPE,
                    {"tooltip": "List of values"},
                ),
            },
            "optional": {
                "fill_missing": (
                    ANY_TYPE,
                    {
                        "tooltip": "Value to use for missing values if lists are different lengths",
                    },
                ),
                "use_fill": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use fill value for missing entries",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("dict_result",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "DictFromLists"
    DESCRIPTION = "Create a dictionary from a list of keys and a list of values"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, keys, values, fill_missing=None, use_fill=False):
        if use_fill:
            # Extend the shorter list with fill values
            max_len = max(len(keys), len(values))
            keys_list = list(keys) + [f"key_{i}" for i in range(len(keys), max_len)]
            values_list = list(values) + [fill_missing] * (max_len - len(values))
            return (dict(zip(keys_list, values_list)),)
        else:
            # Use only the overlapping portion
            return (dict(zip(keys, values)),)


class DictFilter:
    """Filter dictionary entries based on key or value patterns."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dict_data": (
                    ANY_TYPE,
                    {"tooltip": "Dictionary to filter"},
                ),
                "pattern": (
                    "STRING",
                    {
                        "default": ".*",
                        "tooltip": "Regex pattern to match",
                    },
                ),
            },
            "optional": {
                "filter_by": (
                    ["key", "value", "both"],
                    {
                        "default": "key",
                        "tooltip": "What to filter by: key, value, or both",
                    },
                ),
                "invert": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Invert the filter (exclude matches)",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("filtered_dict",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "DictFilter"
    DESCRIPTION = "Filter dictionary entries based on key or value patterns"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, dict_data, pattern=".*", filter_by="key", invert=False):
        import re
        compiled_pattern = re.compile(pattern)
        
        result = {}
        for key, value in dict_data.items():
            matches = False
            
            if filter_by == "key":
                matches = compiled_pattern.search(str(key)) is not None
            elif filter_by == "value":
                matches = compiled_pattern.search(str(value)) is not None
            elif filter_by == "both":
                matches = (compiled_pattern.search(str(key)) is not None or
                           compiled_pattern.search(str(value)) is not None)
            
            if matches != invert:  # XOR logic
                result[key] = value
        
        return (result,)
