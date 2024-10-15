from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from . import new_loader

NODE_CLASS_MAPPINGS.update(new_loader.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(new_loader.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]