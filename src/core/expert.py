"""Expert Agent base class for Tier 2 domain specialists.

Design references:
- OpenAI Agent-as-Tool: experts wrapped as callable tools for the Chief Agent
- Anthropic Prompt Chaining: internal pipeline is sequential deterministic steps
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class ExpertAgentBase(ABC):
    """Base class for Tier 2 expert agents.

    Each expert:
    1. Receives a dict of parameters from the Chief Agent
    2. Runs deterministic tools internally via _execute_pipeline()
    3. Optionally synthesizes with LLM via _build_synthesis_prompt()
    4. Returns a dict result

    This class is the bridge between the Chief Agent (Tier 1) and
    the deterministic ML tools (Tier 3).
    """

    name: str = "expert_base"
    description: str = "Base expert agent"

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self.llm = llm
        self._tools: dict[str, Any] = self._init_tools()

    # ------------------------------------------------------------------
    # Abstract methods — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _init_tools(self) -> dict[str, Any]:
        """Initialize and return deterministic tool instances as a dict."""
        ...

    @abstractmethod
    def _execute_pipeline(self, params: dict) -> dict:
        """Run the deterministic tool pipeline. Returns raw results dict.

        This contains the same logic as the old agent's _execute(),
        but uses params dict instead of AgentState.
        """
        ...

    @abstractmethod
    def _build_synthesis_prompt(self, params: dict, results: dict) -> str:
        """Build the LLM synthesis prompt from pipeline results."""
        ...

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        """Return the expert's system prompt for LLM synthesis."""
        return ""

    def can_handle(self, query: str) -> float:
        """Return confidence score (0-1) for handling this query.

        Default returns 0.5. Subclasses should override with
        domain-specific keyword matching or LLM-based routing.
        """
        return 0.5

    # ------------------------------------------------------------------
    # Main entry point (called by @tool wrapper)
    # ------------------------------------------------------------------

    async def analyze(self, params: dict | str) -> dict:
        """Main entry point. Takes params dict or JSON string, returns dict.

        This is the method that gets wrapped as a @tool for the Chief Agent.
        """
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return {"success": False, "errors": ["Invalid JSON params"]}

        errors: list[str] = []

        # Run deterministic pipeline
        try:
            results = self._execute_pipeline(params)
        except Exception as exc:
            logger.warning("[%s] Pipeline failed: %s", self.name, exc)
            errors.append(f"Pipeline: {exc}")
            results = {}

        # Optional LLM synthesis
        if self.llm:
            try:
                prompt = self._build_synthesis_prompt(params, results)
                llm_response = await self._invoke_llm(prompt)
                analysis = self._parse_json_response(llm_response)
                results["analysis"] = analysis
            except Exception as exc:
                logger.warning("[%s] LLM synthesis failed: %s", self.name, exc)
                results["analysis"] = {"summary": f"LLM synthesis unavailable: {exc}"}
                errors.append(f"LLM synthesis: {exc}")

        if errors:
            results["errors"] = errors

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _invoke_llm(self, user_message: str) -> str:
        """Invoke LLM with system + user message."""
        messages: list[Any] = []
        system = self._get_system_prompt()
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=user_message))

        response = await self.llm.ainvoke(messages)
        return response.content if isinstance(response.content, str) else str(response.content)

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any]:
        """Try to extract JSON from LLM response."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        for marker in ["```json", "```"]:
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start)
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    continue
        return {"raw_response": text}
