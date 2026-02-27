print("Loading Yogurt Nodes...")

import inspect

from .yogurt_nodes import *
from .yogurt_nodes.v3_llm_adapter import (
    get_v3_adapter_status,
    wrap_llm_node_to_v3,
)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"
_v3_wrapped_count = 0

for name, obj in list(globals().items()):
    if inspect.isclass(obj) and hasattr(obj, "_NODE_NAME"):
        node_name = f"Yogurt{name}"
        wrapped_cls = wrap_llm_node_to_v3(node_name, obj)
        if wrapped_cls is not obj:
            _v3_wrapped_count += 1
        NODE_CLASS_MAPPINGS[node_name] = wrapped_cls
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = f"{obj._NODE_NAME} (Yogurt Nodes)"

print(f"Yogurt Nodes loaded: {NODE_DISPLAY_NAME_MAPPINGS.values()}")
print(f"Yogurt LLM V3 wrapped nodes: {_v3_wrapped_count}")
_v3_ready, _v3_detail = get_v3_adapter_status()
if not _v3_ready:
    print(f"Yogurt LLM V3 adapter unavailable: {_v3_detail}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
