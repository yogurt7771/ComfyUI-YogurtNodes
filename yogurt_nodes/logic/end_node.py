from ..utils import ANY_TYPE, DYNAMIC_INPUT_COUNT, make_dynamic_inputs


DATA_NUM = DYNAMIC_INPUT_COUNT


class EndNode:
    """EndNode node.

    End
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": make_dynamic_inputs(
                "data",
                ANY_TYPE,
                count=DATA_NUM,
                tooltip="The {index}th data.",
            ),
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
