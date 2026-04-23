"""GrowthPilot core package."""

__version__ = "5.0.0"

__all__ = [
    "__version__",
    "BaseAgent",
    "AgentState",
    "create_llm",
    "Settings",
    "get_settings",
]

from src.core.base import BaseAgent
from src.core.state import AgentState
from src.core.llm_factory import create_llm
from src.core.config import Settings, get_settings
