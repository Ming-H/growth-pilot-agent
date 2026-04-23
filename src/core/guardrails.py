"""Guardrails system for GrowthPilot Agent.

Three-layer guardrail system inspired by OpenAI Agents SDK Guardrails:
- Input guardrails validate queries before processing
- Planning guardrails check execution plans for feasibility
- Output guardrails validate final result quality
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GuardrailResult(BaseModel):
    """Result of a guardrail check."""

    blocked: bool = False
    reason: str = ""
    warnings: list[str] = Field(default_factory=list)


class GrowthPilotGuardrails:
    """Three-layer guardrail system for the Chief Agent.

    Layer 1 (Input): Validate query, budget, data_path
    Layer 2 (Planning): Check execution plan feasibility
    Layer 3 (Output): Validate final output quality
    """

    def __init__(
        self,
        *,
        max_plan_steps: int = 6,
        max_budget: float = 10_000_000,
        require_strategy: bool = True,
    ) -> None:
        self.max_plan_steps = max_plan_steps
        self.max_budget = max_budget
        self.require_strategy = require_strategy

    # ── Layer 1: Input Validation ──────────────────────────────────────

    def check_input(self, query: str, **kwargs: Any) -> GuardrailResult:
        """Validate input before processing."""
        warnings: list[str] = []

        # Query checks
        if not query or not query.strip():
            return GuardrailResult(blocked=True, reason="Query is empty")

        if len(query) > 5000:
            warnings.append("Query is very long, may be truncated")

        # Budget checks
        budget = kwargs.get("budget")
        if budget is not None:
            if budget < 0:
                return GuardrailResult(
                    blocked=True,
                    reason=f"Budget cannot be negative: {budget}",
                )
            if budget > self.max_budget:
                warnings.append(
                    f"Budget {budget} exceeds recommended maximum {self.max_budget}"
                )
            if budget == 0:
                warnings.append("Budget is 0, some analyses may be limited")

        # Data path checks
        data_path = kwargs.get("data_path", "")
        if data_path:
            import pathlib

            if not pathlib.Path(data_path).exists():
                warnings.append(f"Data path does not exist: {data_path}")

        return GuardrailResult(warnings=warnings)

    # ── Layer 2: Planning Validation ───────────────────────────────────

    def check_plan(self, plan: dict[str, Any]) -> GuardrailResult:
        """Validate execution plan before running."""
        warnings: list[str] = []
        steps = plan.get("steps", [])

        if not steps:
            return GuardrailResult(blocked=True, reason="Execution plan has no steps")

        if len(steps) > self.max_plan_steps:
            return GuardrailResult(
                blocked=True,
                reason=f"Plan has {len(steps)} steps, exceeds maximum {self.max_plan_steps}",
            )

        # Check expert names are valid
        valid_experts = {
            "prospect_analysis",
            "conversion_analysis",
            "subsidy_analysis",
            "retention_analysis",
            "ad_analysis",
        }
        for step in steps:
            expert = step.get("expert", "")
            if expert not in valid_experts:
                warnings.append(f"Unknown expert: {expert}")

        return GuardrailResult(warnings=warnings)

    # ── Layer 3: Output Validation ─────────────────────────────────────

    def check_output(self, result: dict[str, Any]) -> GuardrailResult:
        """Validate final output quality."""
        warnings: list[str] = []

        if not result.get("success", True):
            warnings.append("Analysis was not fully successful")

        if self.require_strategy:
            strategy = result.get("strategy_recommendation", "")
            if not strategy or len(strategy) < 20:
                warnings.append("Strategy recommendation is missing or too short")

        errors = result.get("errors", [])
        if len(errors) > 5:
            warnings.append(
                f"Many errors ({len(errors)}), results may be unreliable"
            )

        return GuardrailResult(warnings=warnings)
