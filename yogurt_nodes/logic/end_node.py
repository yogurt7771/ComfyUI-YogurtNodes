from ..utils import ANY_TYPE


DATA_NUM = 8


class EndNode:
    """EndNode node.

    End
    """
    @classmethod
    def INPUT_TYPES(cls):
        items = {
            f"data{i}": (
                ANY_TYPE,
                {"tooltip": f"The {i}th data."},
            )
            for i in range(1, DATA_NUM + 1)
        }
        return {
            "optional": {**items},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    _NODE_NAME = "EndNode"
    DESCRIPTION = "End"

    def execute(
        self,
        **data,
    ):
        return ()
