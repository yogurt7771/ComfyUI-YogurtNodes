from ..utils import ANY_TYPE


class DictLength:
    """Get the length (number of keys) of any dict-like object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dict_data": (
                    ANY_TYPE,
                    {"tooltip": "Any dict-like object that supports len()"},
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

    _NODE_NAME = "DictLength"
    DESCRIPTION = "Get the length (number of keys) of any dict-like object"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, dict_data):
        return (len(dict_data),)


class DictGet:
    """Get a value by key from any dict-like object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dict_data": (
                    ANY_TYPE,
                    {"tooltip": "Any dict-like object that supports string indexing"},
                ),
                "key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Key to access",
                    },
                ),
            },
            "optional": {
                "default_value": (
                    ANY_TYPE,
                    {
                        "tooltip": "Default value to return if key not found (uses dict.get())",
                    },
                ),
                "use_default": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use default value instead of raising KeyError",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "DictGet"
    DESCRIPTION = "Get a value by key from any dict-like object"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, dict_data, key, default_value=None, use_default=False):
        if use_default:
            return (dict_data.get(key, default_value),)
        else:
            return (dict_data[key],)


class DictKeys:
    """Get all keys from any dict-like object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dict_data": (
                    ANY_TYPE,
                    {"tooltip": "Any dict-like object that supports .keys()"},
                ),
            },
            "optional": {
                "as_list": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Convert keys to list, otherwise return keys view",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("keys",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "DictKeys"
    DESCRIPTION = "Get all keys from any dict-like object"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, dict_data, as_list=True):
        keys = dict_data.keys()
        if as_list:
            keys = list(keys)
        return (keys,)


class DictValues:
    """Get all values from any dict-like object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dict_data": (
                    ANY_TYPE,
                    {"tooltip": "Any dict-like object that supports .values()"},
                ),
            },
            "optional": {
                "as_list": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Convert values to list, otherwise return values view",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("values",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "DictValues"
    DESCRIPTION = "Get all values from any dict-like object"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, dict_data, as_list=True):
        values = dict_data.values()
        if as_list:
            values = list(values)
        return (values,)


class DictSubset:
    """Get a subset of a dict-like object by specifying keys."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dict_data": (
                    ANY_TYPE,
                    {"tooltip": "Any dict-like object"},
                ),
                "keys": (
                    ANY_TYPE,
                    {
                        "tooltip": "List of keys to extract (list-like object)",
                    },
                ),
            },
            "optional": {
                "ignore_missing": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Skip keys that don't exist instead of raising error",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("subset",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "DictSubset"
    DESCRIPTION = "Get a subset of a dict-like object by specifying keys"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, dict_data, keys, ignore_missing=False):
        subset = {}
        for key in keys:
            if ignore_missing and key not in dict_data:
                continue
            subset[key] = dict_data[key]
        return (subset,)
