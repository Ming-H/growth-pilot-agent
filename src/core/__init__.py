"""GrowthPilot core package."""

__version__ = "8.0.0"

__all__ = [
    "__version__",
    "BaseAgent",
    "ExpertAgentBase",
    "create_llm",
    "Settings",
    "get_settings",
]

from src.core.base import BaseAgent
from src.core.config import Settings, get_settings
from src.core.expert import ExpertAgentBase
from src.core.llm_factory import create_llm
