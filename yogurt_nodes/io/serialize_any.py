import pickle
from ..utils import ANY_TYPE


class SerializeAny:
    """Serialize Any node.

    Serialize any Python object to bytes using pickle
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (
                    ANY_TYPE,
                    {"tooltip": "The data to be serialized using pickle."},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("BYTES",)
    RETURN_NAMES = ("bytes_data",)
    FUNCTION = "serialize"
    OUTPUT_NODE = False

    _NODE_NAME = "Serialize Any"
    DESCRIPTION = "Serialize any Python object to bytes using pickle"
    CATEGORY = "YogurtNodes/IO"

    def serialize(self, data):
        try:
            serialized_data = pickle.dumps(data)
            return (serialized_data,)
        except Exception as e:
            raise ValueError(f"Failed to serialize data: {str(e)}")


class DeserializeAny:
    """Deserialize Any node.

    Deserialize bytes data to Python object using pickle
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bytes_data": (
                    "BYTES",
                    {"tooltip": "The bytes data to be deserialized using pickle."},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("data",)
    FUNCTION = "deserialize"
    OUTPUT_NODE = False

    _NODE_NAME = "Deserialize Any"
    DESCRIPTION = "Deserialize bytes data to Python object using pickle"
    CATEGORY = "YogurtNodes/IO"

    def deserialize(self, bytes_data):
        try:
            deserialized_data = pickle.loads(bytes_data)
            return (deserialized_data,)
        except Exception as e:
            raise ValueError(f"Failed to deserialize data: {str(e)}")
