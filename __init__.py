"""BetterRef custom nodes for MiniMax H3 Image and Qwen-Image 2.1."""
import sys

from .nodes.h3_ref import NODES, TITLES
from .nodes.qwen21 import QwenGenerate, QwenRole

NODE_CLASS_MAPPINGS = {**NODES, "BetterRefQwen21Role": QwenRole, "BetterRefQwen21Generate": QwenGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {**TITLES, "BetterRefQwen21Role": "Qwen 2.1 Reference Role",
                              "BetterRefQwen21Generate": "Qwen 2.1 BetterRef Generation"}

if getattr(getattr(sys.modules.get("server"), "PromptServer", None), "instance", None) is not None:
    from .nodes import h3_ref_previews

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
