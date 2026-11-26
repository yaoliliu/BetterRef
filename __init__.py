"""BetterRef custom nodes for native MiniMax H3 Image."""
import sys

from .nodes.h3_ref import NODES as NODE_CLASS_MAPPINGS, TITLES as NODE_DISPLAY_NAME_MAPPINGS

if getattr(getattr(sys.modules.get("server"), "PromptServer", None), "instance", None) is not None:
    from .nodes import h3_ref_previews

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
