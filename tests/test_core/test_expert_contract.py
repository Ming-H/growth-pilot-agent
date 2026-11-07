"""Honest-failure contract tests for ExpertAgentBase.analyze().

A12: pipeline-local ``errors`` (tool failures collected via
``_safe_execute``) must flip ``success`` to False — previously only a
whole-pipeline exception did, so an expert with every tool failing still
reported ``success=True``. Degradation (``llm=None``) is a mode, not a
failure, and must NOT flip it.
"""

from __future__ import annotations

import pytest

from src.core.expert import ExpertAgentBase


class _ProbeExpert(ExpertAgentBase):
    """Minimal concrete expert: pipeline output is injected per test."""

    name = "probe"

    def __init__(self, pipeline_result: dict):  # noqa: D107
        self._pipeline_result = pipeline_result
        super().__init__(llm=None)

    def _init_tools(self) -> dict:
        return {}

    def _execute_pipeline(self, params: dict) -> dict:
        return dict(self._pipeline_result)

    def _build_synthesis_prompt(self, params: dict, results: dict) -> str:  # pragma: no cover
        return ""

    def _get_system_prompt(self) -> str:  # pragma: no cover
        return ""


class TestHonestFailureContract:
    @pytest.mark.asyncio
    async def test_local_tool_errors_flip_success(self):
        result = await _ProbeExpert(
            {"kpis": {"a": 1.0}, "errors": ["ChurnPredictor: boom"]}
        ).analyze({"query": "q"})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_clean_pipeline_zero_key_still_success(self):
        """llm=None is a mode, not a failure (degraded != failed)."""
        result = await _ProbeExpert({"kpis": {"a": 1.0}}).analyze({"query": "q"})
        assert result["success"] is True
        assert result["analysis"].get("degraded")

    @pytest.mark.asyncio
    async def test_empty_errors_list_stays_success(self):
        result = await _ProbeExpert({"kpis": {"a": 1.0}, "errors": []}).analyze({"query": "q"})
        assert result["success"] is True
