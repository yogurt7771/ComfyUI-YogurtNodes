import json
import copy
from ..utils import ANY_TYPE, json_merge, json_get_path, json_set_path


class JsonParse:
    """JsonParse node.

    Parse JSON string to object
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "JSON string to parse",
                    },
                ),
            },
            "optional": {
                "strict": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use strict JSON parsing",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("json_object",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "JsonParse"
    DESCRIPTION = "Parse JSON string to object"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, json_string, strict=True):
        try:
            if strict:
                result = json.loads(json_string)
            else:
                # 宽松模式，可以处理一些非标准格式
                result = json.loads(json_string.replace("'", '"'))
            return (result,)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")


class JsonStringify:
    """JsonStringify node.

    Convert object to JSON string
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (
                    ANY_TYPE,
                    {"tooltip": "Data to convert to JSON string"},
                ),
            },
            "optional": {
                "indent": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8,
                        "tooltip": "Indentation level (0 for compact)",
                    },
                ),
                "ensure_ascii": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Ensure ASCII output",
                    },
                ),
                "sort_keys": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Sort dictionary keys",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_string",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "JsonStringify"
    DESCRIPTION = "Convert object to JSON string"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, data, indent=0, ensure_ascii=False, sort_keys=False):
        indent_val = None if indent == 0 else indent
        return (json.dumps(
            data,
            indent=indent_val,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            separators=(',', ':') if indent == 0 else (',', ': ')
        ),)


class JsonMerge:
    """JsonMerge node.

    Merge multiple JSON objects using deep merge
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json1": (
                    ANY_TYPE,
                    {"tooltip": "First JSON object"},
                ),
                "json2": (
                    ANY_TYPE,
                    {"tooltip": "Second JSON object"},
                ),
            },
            "optional": {
                "json3": (
                    ANY_TYPE,
                    {"tooltip": "Third JSON object (optional)"},
                ),
                "json4": (
                    ANY_TYPE,
                    {"tooltip": "Fourth JSON object (optional)"},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("merged_json",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "JsonMerge"
    DESCRIPTION = "Merge multiple JSON objects using deep merge"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, json1, json2, json3=None, json4=None):
        result = json_merge(json1, json2)
        if json3 is not None:
            result = json_merge(result, json3)
        if json4 is not None:
            result = json_merge(result, json4)
        return (result,)


class JsonGetPath:
    """JsonGetPath node.

    Get value from JSON object using JSONPath
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": (
                    ANY_TYPE,
                    {"tooltip": "JSON object or string"},
                ),
                "path": (
                    "STRING",
                    {
                        "default": "$",
                        "tooltip": "JSONPath expression (e.g., $.users[0].name)",
                    },
                ),
            },
            "optional": {
                "default_value": (
                    ANY_TYPE,
                    {
                        "tooltip": "Default value if path not found",
                    },
                ),
                "use_default": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Return default value instead of None",
                    },
                ),
                "raise_on_error": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Raise error if path not found",
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

    _NODE_NAME = "JsonGetPath"
    DESCRIPTION = "Get value from JSON object using JSONPath"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, json_data, path, default_value=None, use_default=False, raise_on_error=False):
        try:
            result = json_get_path(json_data, path, raise_on_nonexist=raise_on_error)
            if result is None and use_default:
                return (default_value,)
            return (result,)
        except (KeyError, ValueError) as e:
            if raise_on_error:
                raise e
            return (default_value if use_default else None,)


class JsonSetPath:
    """JsonSetPath node.

    Set value in JSON object using JSONPath
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": (
                    ANY_TYPE,
                    {"tooltip": "JSON object or string to modify"},
                ),
                "path": (
                    "STRING",
                    {
                        "default": "$.newfield",
                        "tooltip": "JSONPath expression (e.g., $.users[0].name)",
                    },
                ),
                "value": (
                    ANY_TYPE,
                    {"tooltip": "Value to set"},
                ),
            },
            "optional": {
                "create_missing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Create missing intermediate paths",
                    },
                ),
                "copy_data": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Create a copy instead of modifying original",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("modified_json",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "JsonSetPath"
    DESCRIPTION = "Set value in JSON object using JSONPath"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, json_data, path, value, create_missing=True, copy_data=True):
        if copy_data:
            data = copy.deepcopy(json_data)
        else:
            data = json_data
        
        result = json_set_path(data, path, value, raise_on_nonexist=not create_missing)
        return (result,)


class JsonValidate:
    """JsonValidate node.

    Validate JSON data structure
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (
                    ANY_TYPE,
                    {"tooltip": "Data to validate"},
                ),
            },
            "optional": {
                "check_type": (
                    ["any", "object", "array", "string", "number", "boolean", "null"],
                    {
                        "default": "any",
                        "tooltip": "Expected data type",
                    },
                ),
                "required_keys": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Required keys for objects (comma-separated)",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("is_valid", "error_message")
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "JsonValidate"
    DESCRIPTION = "Validate JSON data structure"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, data, check_type="any", required_keys=""):
        try:
            # Type validation
            if check_type != "any":
                if check_type == "object" and not isinstance(data, dict):
                    return (False, f"Expected object, got {type(data).__name__}")
                elif check_type == "array" and not isinstance(data, list):
                    return (False, f"Expected array, got {type(data).__name__}")
                elif check_type == "string" and not isinstance(data, str):
                    return (False, f"Expected string, got {type(data).__name__}")
                elif check_type == "number" and not isinstance(data, (int, float)):
                    return (False, f"Expected number, got {type(data).__name__}")
                elif check_type == "boolean" and not isinstance(data, bool):
                    return (False, f"Expected boolean, got {type(data).__name__}")
                elif check_type == "null" and data is not None:
                    return (False, f"Expected null, got {type(data).__name__}")

            # Required keys validation
            if required_keys and isinstance(data, dict):
                keys = [k.strip() for k in required_keys.split(",") if k.strip()]
                missing_keys = [k for k in keys if k not in data]
                if missing_keys:
                    return (False, f"Missing required keys: {', '.join(missing_keys)}")

            return (True, "Valid")
        except Exception as e:
            return (False, str(e))


class JsonPathExists:
    """JsonPathExists node.

    Check if a path exists in JSON object
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": (
                    ANY_TYPE,
                    {"tooltip": "JSON object or string"},
                ),
                "path": (
                    "STRING",
                    {
                        "default": "$",
                        "tooltip": "JSONPath expression to check",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("exists",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "JsonPathExists"
    DESCRIPTION = "Check if a path exists in JSON object"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, json_data, path):
        try:
            result = json_get_path(json_data, path, raise_on_nonexist=False)
            return (result is not None,)
        except Exception:
            return (False,)


class JsonDeepCopy:
    """JsonDeepCopy node.

    Create a deep copy of JSON object
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": (
                    ANY_TYPE,
                    {"tooltip": "JSON object to copy"},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("copied_json",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "JsonDeepCopy"
    DESCRIPTION = "Create a deep copy of JSON object"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, json_data):
        return (copy.deepcopy(json_data),)


class JsonFlatten:
    """JsonFlatten node.

    Flatten nested JSON object to flat key-value pairs
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": (
                    ANY_TYPE,
                    {"tooltip": "JSON object to flatten"},
                ),
            },
            "optional": {
                "separator": (
                    "STRING",
                    {
                        "default": ".",
                        "tooltip": "Separator for nested keys",
                    },
                ),
                "include_arrays": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Include array elements with index",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("flattened_json",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "JsonFlatten"
    DESCRIPTION = "Flatten nested JSON object to flat key-value pairs"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, json_data, separator=".", include_arrays=True):
        def _flatten(obj, parent_key="", sep="."):
            items = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, (dict, list)):
                        items.extend(_flatten(v, new_key, sep).items())
                    else:
                        items.append((new_key, v))
            elif isinstance(obj, list) and include_arrays:
                for i, v in enumerate(obj):
                    new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                    if isinstance(v, (dict, list)):
                        items.extend(_flatten(v, new_key, sep).items())
                    else:
                        items.append((new_key, v))
            else:
                return {parent_key: obj} if parent_key else obj
            return dict(items)

        return (_flatten(json_data, "", separator),)


class JsonUnflatten:
    """JsonUnflatten node.

    Unflatten flat JSON object back to nested structure
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flat_json": (
                    ANY_TYPE,
                    {"tooltip": "Flattened JSON object"},
                ),
            },
            "optional": {
                "separator": (
                    "STRING",
                    {
                        "default": ".",
                        "tooltip": "Separator used in flattened keys",
                    },
                ),
                "auto_detect_arrays": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically detect and create arrays",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("nested_json",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "JsonUnflatten"
    DESCRIPTION = "Unflatten flat JSON object back to nested structure"
    CATEGORY = "YogurtNodes/Logic"

    def execute(self, flat_json, separator=".", auto_detect_arrays=True):
        if not isinstance(flat_json, dict):
            return (flat_json,)

        result = {}
        
        for key, value in flat_json.items():
            parts = key.split(separator)
            current = result
            
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    # Decide whether to create array or object
                    next_part = parts[i + 1]
                    if auto_detect_arrays and next_part.isdigit():
                        current[part] = []
                    else:
                        current[part] = {}
                
                current = current[part]
                
                # Extend array if needed
                if isinstance(current, list):
                    next_idx = int(parts[i + 1])
                    while len(current) <= next_idx:
                        current.append({} if i + 2 < len(parts) else None)
                    if i + 2 < len(parts):
                        current = current[next_idx]
            
            # Set final value
            final_key = parts[-1]
            if isinstance(current, list):
                idx = int(final_key)
                while len(current) <= idx:
                    current.append(None)
                current[idx] = value
            else:
                current[final_key] = value
        
        return (result,)
