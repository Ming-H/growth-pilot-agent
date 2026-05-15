"""Context Engineering for GrowthPilot Agent.

Implements the Write / Select / Compress / Isolate pattern recommended by
LangChain and Anthropic for managing context in multi-agent systems.

- **Write**: Persist key context outside the context window.
- **Select**: Pull only relevant context into each agent invocation.
- **Compress**: Reduce verbose feedback into concise directives.
- **Isolate**: Scope each expert's context independently.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages context flow between graph nodes and expert agents."""

    @staticmethod
    def select_for_expert(
        state: dict[str, Any],
        expert_key: str,
    ) -> dict[str, Any]:
        """Select only the context relevant to a specific expert.

        Filters the full graph state down to what a single expert needs,
        preventing cross-contamination and reducing token waste.
        """
        query = state.get("query", "")
        scope = state.get("scope") or ""
        budget = state.get("budget") or 0
        plan = state.get("plan") or {}

        # Extract plan reasoning relevant to this expert
        expert_reasoning = ""
        if isinstance(plan, dict):
            reasoning = plan.get("reasoning", "")
            if expert_key in reasoning.lower():
                expert_reasoning = reasoning

        context_summary = plan.get("context_summary", "") if isinstance(plan, dict) else ""

        return {
            "query": query,
            "scope": scope,
            "budget": budget,
            "expert_key": expert_key,
            "plan_reasoning": expert_reasoning,
            "context_summary": context_summary,
        }

    @staticmethod
    def compress_feedback(
        quality_scores: dict[str, Any],
        max_chars: int = 500,
    ) -> str:
        """Compress quality scores into a concise refinement directive.

        Takes verbose evaluation results and produces a short, actionable
        feedback string that fits within token budgets.
        """
        parts: list[str] = []
        for expert_name, score_data in quality_scores.items():
            if not isinstance(score_data, dict):
                continue
            overall = score_data.get("overall", 0.0)
            if overall >= 0.7:
                continue
            reasoning = score_data.get("reasoning", "")
            if reasoning:
                parts.append(f"[{expert_name} score={overall:.2f}] {reasoning}")

        if not parts:
            return ""

        feedback = "Quality improvement needed: " + "; ".join(parts)
        if len(feedback) > max_chars:
            feedback = feedback[: max_chars - 3] + "..."
        return feedback

    @staticmethod
    def build_expert_params(
        state: dict[str, Any],
        expert_key: str,
        *,
        is_refinement: bool = False,
    ) -> dict[str, Any]:
        """Build isolated parameters for a single expert invocation.

        Combines selected context with optional refinement feedback,
        producing a clean parameter dict for the expert's analyze() call.
        """
        ctx = ContextManager.select_for_expert(state, expert_key)
        params: dict[str, Any] = {
            "query": ctx["query"],
            "scope": ctx["scope"],
            "budget": ctx["budget"],
        }

        if is_refinement:
            quality_scores = state.get("quality_scores", {})
            feedback = ContextManager.compress_feedback(quality_scores)
            if feedback:
                params["refinement_feedback"] = feedback
            params["is_refinement"] = True

        return params
