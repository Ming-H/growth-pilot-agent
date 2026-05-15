"""GrowthPilot Prompt Template System.

Exports all prompt template classes for use by agents.

Note: OrchestratorPrompt was removed when the old OrchestratorAgent
was replaced by GrowthPilotAgent (Chief Agent) in the new architecture.
"""
from src.prompts.templates.agent_prompts import (
    AdPrompt,
    ConversionPrompt,
    ProspectPrompt,
    RetentionPrompt,
    SubsidyPrompt,
)
from src.prompts.templates.base import PromptTemplate

__all__ = [
    "PromptTemplate",
    "ProspectPrompt",
    "ConversionPrompt",
    "SubsidyPrompt",
    "RetentionPrompt",
    "AdPrompt",
]
