from __future__ import annotations

import inspect
from typing import Callable


V3_CLASS_SUFFIX = "__V3"


def strip_v3_suffix(class_name: str) -> str:
    if class_name.endswith(V3_CLASS_SUFFIX):
        return class_name[: -len(V3_CLASS_SUFFIX)]
    return class_name


def build_node_mappings(
    namespace: dict[str, object],
    wrap_llm_node_to_v3: Callable[[str, type], type],
    wrap_node_to_v3: Callable[[str, type], type],
) -> tuple[dict[str, type], dict[str, str], dict[str, int]]:
    node_class_mappings: dict[str, type] = {}
    node_display_name_mappings: dict[str, str] = {}
    stats = {"llm_wrapped": 0, "v3_wrapped": 0}
    seen_classes: set[int] = set()

    for obj in namespace.values():
        if not inspect.isclass(obj) or not hasattr(obj, "_NODE_NAME"):
            continue
        obj_id = id(obj)
        if obj_id in seen_classes:
            continue
        seen_classes.add(obj_id)

        class_name = getattr(obj, "__name__", obj.__class__.__name__)
        public_class_name = strip_v3_suffix(class_name)
        node_name = f"Yogurt{public_class_name}"

        if class_name.endswith(V3_CLASS_SUFFIX):
            wrapped_cls = wrap_node_to_v3(node_name, obj)
            if wrapped_cls is not obj:
                stats["v3_wrapped"] += 1
        else:
            wrapped_cls = wrap_llm_node_to_v3(node_name, obj)
            if wrapped_cls is not obj:
                stats["llm_wrapped"] += 1

        node_class_mappings[node_name] = wrapped_cls
        node_display_name_mappings[node_name] = f"{obj._NODE_NAME} (Yogurt Nodes)"

    return node_class_mappings, node_display_name_mappings, stats
