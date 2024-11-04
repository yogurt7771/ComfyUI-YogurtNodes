import inspect

from .yogurt_nodes import *
import inspect

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for name, obj in list(globals().items()):
    if inspect.isclass(obj) and hasattr(obj, 'NODE_NAME'):
        NODE_CLASS_MAPPINGS[name] = obj
        NODE_DISPLAY_NAME_MAPPINGS[name] = obj.NODE_NAME
