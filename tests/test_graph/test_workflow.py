"""Tests for src.graph.workflow - LangGraph StateGraph construction."""

from __future__ import annotations

import pytest

from src.core.state import AgentState


class TestWorkflowBuild:
    """Tests for build_workflow (graph construction, no execution)."""

    def test_workflow_builds(self):
        """build_workflow returns a compiled graph without errors."""
        from src.graph.workflow import build_workflow

        app = build_workflow()
        assert app is not None

    def test_workflow_has_correct_nodes(self):
        """The compiled graph contains all expected nodes."""
        from src.graph.workflow import build_workflow

        app = build_workflow()

        # Access the compiled graph's node names
        # LangGraph compiled graph stores nodes internally
        node_names = set(app.get_graph().nodes.keys())

        expected_nodes = {
            "orchestrator",
            "prospect",
            "subsidy",
            "ad",
            "conversion",
            "retention",
            "synthesis",
            "report_gen",
            "skip_prospect",
            "skip_subsidy",
            "skip_ad",
            "skip_conversion",
            "skip_retention",
        }

        for node in expected_nodes:
            assert node in node_names, f"Missing node: {node}"


class TestRoutingHelpers:
    """Tests for the routing helper functions."""

    def test_should_run_prospect_full(self):
        from src.graph.workflow import _should_run_prospect
        state: AgentState = {"query": "", "data_path": "", "scope": "full", "errors": [], "metadata": []}  # type: ignore
        assert _should_run_prospect(state) == "prospect"

    def test_should_run_prospect_prospect_scope(self):
        from src.graph.workflow import _should_run_prospect
        state: AgentState = {"query": "", "data_path": "", "scope": "prospect", "errors": [], "metadata": []}  # type: ignore
        assert _should_run_prospect(state) == "prospect"

    def test_should_run_prospect_skip(self):
        from src.graph.workflow import _should_run_prospect
        state: AgentState = {"query": "", "data_path": "", "scope": "retention", "errors": [], "metadata": []}  # type: ignore
        assert _should_run_prospect(state) == "skip_prospect"

    def test_should_run_conversion_inapp(self):
        from src.graph.workflow import _should_run_conversion
        state: AgentState = {"query": "", "data_path": "", "scope": "inapp", "errors": [], "metadata": []}  # type: ignore
        assert _should_run_conversion(state) == "conversion"

    def test_should_run_retention_inapp(self):
        from src.graph.workflow import _should_run_retention
        state: AgentState = {"query": "", "data_path": "", "scope": "inapp", "errors": [], "metadata": []}  # type: ignore
        assert _should_run_retention(state) == "retention"
