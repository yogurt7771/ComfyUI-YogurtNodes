from regex import R


class AnyBridge:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("*", {}),
                "mode": (["raw value", "tensor shape"],),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    RETURN_TYPES = ("*", "STRING",)
    RETURN_NAMES = ("data", "text",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    _NODE_NAME = "Any Bridge"
    DESCRIPTION = "Any Bridge"
    CATEGORY = "YogurtNodes/IO"

    def execute(self, data, mode):
        if mode == "tensor shape":
            text = []

            def tensorShape(tensor):
                if isinstance(tensor, dict):
                    for k in tensor:
                        tensorShape(tensor[k])
                elif isinstance(tensor, list):
                    for i in range(len(tensor)):
                        tensorShape(tensor[i])
                elif hasattr(tensor, "shape"):
                    text.append(list(tensor.shape))

            tensorShape(data)
            text = str(text)
        else:
            text = str(data)

        return (data, text,)
