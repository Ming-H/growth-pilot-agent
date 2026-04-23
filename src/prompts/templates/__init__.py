"""Prompt template sub-package.

Note: OrchestratorPrompt was removed when the old OrchestratorAgent
was replaced by GrowthPilotAgent (Chief Agent) in the new architecture.
"""
from src.prompts.templates.base import PromptTemplate
from src.prompts.templates.agent_prompts import (
    AdPrompt,
    ConversionPrompt,
    ProspectPrompt,
    RetentionPrompt,
    SubsidyPrompt,
)

__all__ = [
    "PromptTemplate",
    "ProspectPrompt",
    "ConversionPrompt",
    "SubsidyPrompt",
    "RetentionPrompt",
    "AdPrompt",
]
