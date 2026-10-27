"""Expert Agent base class for Tier 2 domain specialists.

Design references:
- OpenAI Agent-as-Tool: experts wrapped as callable tools for the Chief Agent
- Anthropic Prompt Chaining: internal pipeline is sequential deterministic steps

Unified with BaseAgent to inherit hooks, middleware stack, self-eval,
and the shared _invoke_llm / _parse_json_response implementations.
"""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.core.base import BaseAgent

logger = logging.getLogger(__name__)


class ExpertAgentBase(BaseAgent):
    """Base class for Tier 2 expert agents.

    Each expert:
    1. Receives a dict of parameters from the Orchestrator
    2. Runs deterministic tools internally via _execute_pipeline()
    3. Optionally synthesizes with LLM via _build_synthesis_prompt()
    4. Returns a dict result

    Inherits from BaseAgent for:
    - Middleware stack (retry, logging, error handling)
    - Pre/Post hook lifecycle
    - Quality self-evaluation
    - Shared _invoke_llm and _parse_json_response

    This class is the bridge between the Orchestrator (Tier 1) and
    the deterministic ML tools (Tier 3).
    """

    name: str = "expert_base"
    description: str = "Base expert agent"

    def __init__(self, llm: BaseChatModel | None = None, **kwargs: Any) -> None:
        # BaseAgent requires llm, system_prompt, enable_self_eval
        super().__init__(
            llm=llm,  # type: ignore[arg-type]
            system_prompt="",
            enable_self_eval=False,
            **kwargs,
        )
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

    @staticmethod
    def can_handle(query: str) -> float:
        """Return confidence score (0-1) for handling this query.

        Default returns 0.5. Subclasses should override with
        domain-specific keyword matching or LLM-based routing.
        """
        return 0.5

    # ------------------------------------------------------------------
    # BaseAgent._execute implementation
    # ------------------------------------------------------------------

    async def _execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Implements BaseAgent._execute — delegates to analyze().

        When invoked through ``BaseAgent.run()`` the graph wraps params under
        the ``"params"`` key; direct calls pass the params dict itself.
        """
        params = state.get("params", state)
        return await self.analyze(params)

    # ------------------------------------------------------------------
    # Main entry point (called by graph nodes)
    # ------------------------------------------------------------------

    async def analyze(self, params: dict | str) -> dict:
        """Main entry point. Takes params dict or JSON string, returns dict.

        This is the method that gets called by the execute_node in the graph.
        """
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return {"success": False, "errors": ["Invalid JSON params"]}

        errors: list[str] = []
        degraded: list[str] = []

        # Run deterministic pipeline (offloaded to thread to avoid blocking the event loop)
        try:
            import asyncio

            results = await asyncio.to_thread(self._execute_pipeline, params)
        except Exception as exc:
            logger.warning("[%s] Pipeline failed: %s", self.name, exc)
            errors.append(f"Pipeline: {exc}")
            results = {}

        # LLM synthesis: present → interpret; absent → deterministic summary.
        # Absence of a brain is a mode (degraded), not a failure.
        if self.llm:
            try:
                prompt = self._build_synthesis_prompt(params, results)
                llm_response = await self._invoke_llm(prompt)
                analysis = self._parse_json_response(llm_response)
                results["analysis"] = analysis
            except Exception as exc:
                logger.warning("[%s] LLM synthesis failed: %s", self.name, exc)
                results["analysis"] = {
                    "summary": self._deterministic_summary(results),
                    "degraded": True,
                }
                degraded.append("llm_synthesis")
        else:
            results["analysis"] = {
                "summary": self._deterministic_summary(results),
                "degraded": True,
            }
            degraded.append("llm_synthesis")

        if errors:
            results["errors"] = errors
        if degraded:
            results["degraded"] = degraded

        # Explicit success contract: downstream nodes (evaluate/refine/report)
        # check this flag instead of relying on key absence.
        results["success"] = not errors

        return results

    # ------------------------------------------------------------------
    # Deterministic summary (zero-key degradation path)
    # ------------------------------------------------------------------

    @staticmethod
    def _deterministic_summary(results: dict) -> str:
        """Render a one-line deterministic summary from pipeline KPIs.

        Used when no LLM is configured (zero-key mode) or synthesis fails.
        Experts may override for nicer domain-specific phrasing; the default
        renders ``k=v`` pairs so every number stays tool-traceable.
        """
        kpis = results.get("kpis") or {}
        if kpis:
            rendered = "; ".join(
                f"{k}={ExpertAgentBase._fmt_num(v)}" for k, v in list(kpis.items())[:10]
            )
            return f"确定性管线结果（无 LLM 解读）: {rendered}"
        return "确定性管线完成（无 KPI 产出，详见结构化结果）"

    @staticmethod
    def _fmt_num(v: Any) -> str:
        """Format a number compactly for deterministic summaries."""
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 1 else f"{v:,.2f}"
        return str(v)

    # ------------------------------------------------------------------
    # Shared utility: safe tool execution
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_execute(
        tool_method: Any,
        error_prefix: str,
        errors: list[str],
        default: Any = None,
    ) -> Any:
        """Execute a tool method with unified error handling.

        Use this in _execute_pipeline() to avoid repeating try/except
        blocks for every tool call.

        Args:
            tool_method: Callable (usually ``lambda: tool.run(...)``)
            error_prefix: Prefix for the error message (e.g. "FeatureEngine")
            errors: List to append error messages to
            default: Fallback value on failure (defaults to empty dict)

        Returns:
            Tool result on success, *default* on failure.
        """
        try:
            return tool_method()
        except Exception as exc:
            logger.warning("%s failed: %s", error_prefix, exc)
            errors.append(f"{error_prefix}: {exc}")
            return default if default is not None else {}

    # ------------------------------------------------------------------
    # Shared utility: keyword-based confidence scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_confidence(query: str, keywords: list[str]) -> float:
        """Calculate confidence score based on keyword matches.

        Scoring:
        - 0 matches -> 0.0
        - 1 match   -> 0.6
        - 2+ matches -> min(0.9, 0.5 + matches * 0.1)

        Args:
            query: The user query string.
            keywords: Domain-specific keywords to match against.

        Returns:
            Confidence score in [0.0, 0.9].
        """
        query_lower = query.lower()
        matches = sum(1 for kw in keywords if kw.lower() in query_lower)
        if matches == 0:
            return 0.0
        if matches == 1:
            return 0.6
        return min(0.9, 0.5 + matches * 0.1)
