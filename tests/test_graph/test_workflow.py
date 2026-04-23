"""Tests for src.graph.graph - LangGraph StateGraph construction."""

from __future__ import annotations

import pytest

from src.graph.state import AnalysisState, AnalysisStatus


class TestGraphBuild:
    """Tests for build_graph (graph construction, no execution)."""

    def test_graph_builds(self):
        """build_graph returns a StateGraph without errors."""
        from src.graph.graph import build_graph

        graph = build_graph()
        assert graph is not None

    def test_graph_has_correct_nodes(self):
        """The graph contains all expected nodes."""
        from src.graph.graph import build_graph

        graph = build_graph()

        # Access node names from the graph
        node_names = set(graph.nodes.keys())

        expected_nodes = {
            "plan",
            "execute",
            "evaluate",
            "approval",
            "report",
        }

        for node in expected_nodes:
            assert node in node_names, f"Missing node: {node}"

    def test_graph_compiles(self):
        """The graph compiles without errors."""
        from src.graph.graph import build_graph

        graph = build_graph()
        compiled = graph.compile()
        assert compiled is not None


class TestRoutingLogic:
    """Tests for the conditional edge routing functions."""

    def test_evaluate_routes_to_execute_on_refine(self):
        """After evaluate, if refinement needed and round < 2, route to execute."""
        from src.graph.graph import _route_after_evaluate

        state: AnalysisState = {
            "query": "test",
            "org_id": "org1",
            "user_id": "user1",
            "needs_refinement": True,
            "refinement_round": 0,
            "approval_required": False,
        }  # type: ignore
        assert _route_after_evaluate(state) == "execute"

    def test_evaluate_routes_to_approval_when_required(self):
        """After evaluate, if approval required, route to approval."""
        from src.graph.graph import _route_after_evaluate

        state: AnalysisState = {
            "query": "test",
            "org_id": "org1",
            "user_id": "user1",
            "needs_refinement": False,
            "refinement_round": 0,
            "approval_required": True,
        }  # type: ignore
        assert _route_after_evaluate(state) == "approval"

    def test_evaluate_routes_to_report_when_done(self):
        """After evaluate, if no refinement and no approval, route to report."""
        from src.graph.graph import _route_after_evaluate

        state: AnalysisState = {
            "query": "test",
            "org_id": "org1",
            "user_id": "user1",
            "needs_refinement": False,
            "refinement_round": 0,
            "approval_required": False,
        }  # type: ignore
        assert _route_after_evaluate(state) == "report"

    def test_evaluate_max_refinement_rounds(self):
        """After 2 refinement rounds, route to report even if needs_refinement."""
        from src.graph.graph import _route_after_evaluate

        state: AnalysisState = {
            "query": "test",
            "org_id": "org1",
            "user_id": "user1",
            "needs_refinement": True,
            "refinement_round": 2,
            "approval_required": False,
        }  # type: ignore
        assert _route_after_evaluate(state) == "report"

    def test_approval_routes_to_report_on_approve(self):
        """After approval, if approved, route to report."""
        from src.graph.graph import _route_after_approval

        state: AnalysisState = {
            "query": "test",
            "org_id": "org1",
            "user_id": "user1",
            "approved": True,
        }  # type: ignore
        assert _route_after_approval(state) == "report"

    def test_approval_routes_to_end_on_reject(self):
        """After approval, if rejected, route to END."""
        from src.graph.graph import _route_after_approval

        state: AnalysisState = {
            "query": "test",
            "org_id": "org1",
            "user_id": "user1",
            "approved": False,
        }  # type: ignore
        assert _route_after_approval(state) == "__end__"
