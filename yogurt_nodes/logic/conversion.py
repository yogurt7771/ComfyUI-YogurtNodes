from ..utils import ANY_TYPE


class ToList:
    """Convert any iterable to a list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (
                    ANY_TYPE,
                    {"tooltip": "Any iterable data to convert to list"},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("list_result",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "ToList"
    DESCRIPTION = "Convert any iterable to a list"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, data):
        return (list(data),)


class ToDict:
    """Convert pairs or mapping to a dictionary."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (
                    ANY_TYPE,
                    {"tooltip": "Pairs (list of tuples) or mapping to convert to dict"},
                ),
            },
            "optional": {
                "default_value": (
                    ANY_TYPE,
                    {
                        "tooltip": "Default value for keys without values",
                    },
                ),
                "use_default": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use default value for incomplete pairs",
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

    _NODE_NAME = "ToDict"
    DESCRIPTION = "Convert pairs or mapping to a dictionary"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, data, default_value=None, use_default=False):
        try:
            # Try direct dict conversion first
            return (dict(data),)
        except (ValueError, TypeError):
            # If that fails, try to handle as list of items
            result = {}
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    result[item[0]] = item[1]
                elif isinstance(item, (list, tuple)) and len(item) == 1 and use_default:
                    result[item[0]] = default_value
                elif use_default:
                    result[str(item)] = default_value
            return (result,)


class IsEmpty:
    """Check if a data structure is empty."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (
                    ANY_TYPE,
                    {"tooltip": "Data structure to check"},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("is_empty",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "IsEmpty"
    DESCRIPTION = "Check if a data structure is empty"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, data):
        try:
            return (len(data) == 0,)
        except TypeError:
            # If len() doesn't work, check for None or common empty values
            return (data is None or data == "" or data == 0,)


class DataSize:
    """Get the size/length of any data structure."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (
                    ANY_TYPE,
                    {"tooltip": "Data structure to measure"},
                ),
            },
            "optional": {
                "default_size": (
                    "INT",
                    {
                        "default": 0,
                        "tooltip": "Default size for non-measurable objects",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("size",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "DataSize"
    DESCRIPTION = "Get the size/length of any data structure"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, data, default_size=0):
        try:
            return (len(data),)
        except TypeError:
            return (default_size,)


class StringSplit:
    """Split a string into a list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {"tooltip": "Text to split"},
                ),
            },
            "optional": {
                "separator": (
                    "STRING",
                    {
                        "default": ",",
                        "tooltip": "Separator to split on (empty string splits every character)",
                    },
                ),
                "max_split": (
                    "INT",
                    {
                        "default": -1,
                        "tooltip": "Maximum number of splits (-1 for no limit)",
                    },
                ),
                "strip_items": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Strip whitespace from each item",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("list_result",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "StringSplit"
    DESCRIPTION = "Split a string into a list"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, text, separator=",", max_split=-1, strip_items=True):
        if separator == "":
            # Split every character
            result = list(text)
        else:
            if max_split == -1:
                result = text.split(separator)
            else:
                result = text.split(separator, max_split)
        
        if strip_items:
            result = [item.strip() for item in result]
        
        return (result,)
