print("Loading Yogurt Nodes...")

from .yogurt_nodes import *
from .yogurt_nodes.node_registry import build_node_mappings
from .yogurt_nodes.v3_llm_adapter import (
    get_v3_adapter_status,
    wrap_llm_node_to_v3,
)
from .yogurt_nodes.v3_node_adapter import (
    get_v3_node_adapter_status,
    wrap_node_to_v3,
)


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, _wrap_stats = build_node_mappings(
    globals(),
    wrap_llm_node_to_v3=wrap_llm_node_to_v3,
    wrap_node_to_v3=wrap_node_to_v3,
)
WEB_DIRECTORY = "./web"

print(f"Yogurt Nodes loaded: {NODE_DISPLAY_NAME_MAPPINGS.values()}")
print(f"Yogurt LLM V3 wrapped nodes: {_wrap_stats['llm_wrapped']}")
print(f"Yogurt Generic V3 wrapped nodes: {_wrap_stats['v3_wrapped']}")
_v3_ready, _v3_detail = get_v3_adapter_status()
if not _v3_ready:
    print(f"Yogurt LLM V3 adapter unavailable: {_v3_detail}")
_generic_v3_ready, _generic_v3_detail = get_v3_node_adapter_status()
if not _generic_v3_ready:
    print(f"Yogurt generic V3 adapter unavailable: {_generic_v3_detail}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
