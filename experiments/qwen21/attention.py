"""Use the processor shipped with the Qwen workflow in the standalone pilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen21.attention import RoleAttention

__all__ = ["RoleAttention"]
