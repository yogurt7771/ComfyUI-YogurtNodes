class StringToValue:
    """
    Get value from string
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "INT",
        "FLOAT",
    )
    RETURN_NAMES = ("string", "int", "float")
    OUTPUT_NODE = False

    FUNCTION = "string_to_value"

    _NODE_NAME = "String To Value"
    DESCRIPTION = "Get value from string"
    CATEGORY = "YogurtNodes/String"

    def string_to_value(self, text: str):
        try:
            int_result = int(text.strip())
        except ValueError:
            int_result = 0
        try:
            float_result = float(text.strip())
        except ValueError:
            float_result = 0.0
        return (text, int_result, float_result)
