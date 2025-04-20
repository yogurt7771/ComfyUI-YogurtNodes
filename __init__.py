print("Loading Yogurt Nodes...")

import inspect

from .yogurt_nodes import *


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for name, obj in list(globals().items()):
    if inspect.isclass(obj) and hasattr(obj, "_NODE_NAME"):
        NODE_CLASS_MAPPINGS[name] = obj
        NODE_DISPLAY_NAME_MAPPINGS[name] = f"{obj._NODE_NAME} (Yogurt Nodes)"

print(f"Yogurt Nodes loaded: {NODE_DISPLAY_NAME_MAPPINGS.values()}")
