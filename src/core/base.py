"""Agent base class."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.core.state import AgentState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all GrowthPilot agents."""

    name: str = "base_agent"
    description: str = "Base agent"

    def __init__(self, llm: BaseChatModel, system_prompt: str = "") -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    @abstractmethod
    async def run(self, state: AgentState) -> dict[str, Any]:
        """Execute the agent logic. Returns a partial state update."""
        ...

    async def _invoke_llm(
        self,
        user_message: str,
        system_override: str | None = None,
    ) -> str:
        """Invoke LLM with system + user message, return content."""
        system = system_override or self.system_prompt
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=user_message))

        response = await self.llm.ainvoke(messages)
        return response.content if isinstance(response.content, str) else str(response.content)

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Try to extract JSON from LLM response."""
        try:
            # Try direct parse
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block
        for marker in ["```json", "```"]:
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start)
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    continue

        return {"raw_response": text}

    def _build_prompt_context(self, state: AgentState) -> str:
        """Build context string from current state for prompt injection."""
        parts = [f"用户查询: {state.get('query', '')}"]
        if state.get("prospect_results"):
            parts.append(f"潜客识别结果: 已完成")
        if state.get("budget"):
            parts.append(f"可用预算: {state['budget']}")
        return "\n".join(parts)
