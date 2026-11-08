"""Tests for src.graph.graph - LangGraph StateGraph construction."""

from __future__ import annotations

from src.graph.state import AnalysisState


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

    def test_evaluate_routes_to_refine_when_needed(self):
        """After evaluate, if refinement needed and round < max, route to refine."""
        from src.graph.graph import _route_after_evaluate

        state: AnalysisState = {
            "query": "test",
            "org_id": "org1",
            "user_id": "user1",
            "needs_refinement": True,
            "refinement_round": 0,
            "approval_required": False,
        }  # type: ignore
        assert _route_after_evaluate(state) == "refine"

    def test_evaluate_routes_to_approval_when_required(self):
        """After evaluate, approval required AND budget ≥ threshold → approval."""
        from src.graph.graph import _route_after_evaluate

        state: AnalysisState = {
            "query": "test",
            "org_id": "org1",
            "user_id": "user1",
            "needs_refinement": False,
            "refinement_round": 0,
            "approval_required": True,
            "budget": 50_000.0,
        }  # type: ignore
        assert _route_after_evaluate(state) == "approval"

    def test_evaluate_low_budget_auto_approves_to_report(self):
        """A2: interrupt_before pauses ANY run routed into approval — a
        low-budget approval_required run would hang forever on the node
        that is supposed to auto-approve it. Route it to report instead
        (the agent-mode fix, applied to workflow mode)."""
        from src.graph.graph import _route_after_evaluate

        state: AnalysisState = {
            "query": "test",
            "org_id": "org1",
            "user_id": "user1",
            "needs_refinement": False,
            "refinement_round": 0,
            "approval_required": True,
            "budget": 500.0,  # < default threshold 10_000
        }  # type: ignore
        assert _route_after_evaluate(state) == "report"

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


class TestCompatSuccess:
    """A11: paused/rejected runs must NOT report success=True — the
    --json honest-exit-code contract depends on it."""

    def _compat(self, state: dict) -> dict:
        from src.graph.workflow import _build_compat_result

        return _build_compat_result(state, query="q", scope="full")

    def test_completed_is_success(self):
        from src.graph.state import AnalysisStatus

        r = self._compat({"status": AnalysisStatus.COMPLETED, "final_report": "R"})
        assert r["success"] is True
        assert r["status"] == "completed"

    def test_awaiting_approval_is_not_success(self):
        """HITL pause: ainvoke returns early with an empty report."""
        from src.graph.state import AnalysisStatus

        r = self._compat({"status": AnalysisStatus.AWAITING_APPROVAL, "final_report": ""})
        assert r["success"] is False
        assert r["status"] == "awaiting_approval"

    def test_rejected_is_not_success(self):
        """User rejected at the approval node: status stays awaiting_approval."""
        from src.graph.state import AnalysisStatus

        r = self._compat(
            {"status": AnalysisStatus.AWAITING_APPROVAL, "approved": False, "final_report": ""}
        )
        assert r["success"] is False

    def test_failed_is_not_success(self):
        from src.graph.state import AnalysisStatus

        r = self._compat({"status": AnalysisStatus.FAILED})
        assert r["success"] is False


class TestCompatSnapshotPassthrough:
    """A8: compat must pass through the state's kpi_snapshot verbatim."""

    def test_state_snapshot_passed_through(self):
        from src.graph.state import AnalysisStatus
        from src.graph.workflow import _build_compat_result

        state = {
            "status": AnalysisStatus.COMPLETED,
            "final_report": "R",
            "kpi_snapshot": {"budget": 9.0, "subsidy_expected_roi": 1.6},
            "expert_results": [{"expert": "subsidy", "kpis": {"budget": 123.0}}],
        }
        r = _build_compat_result(state, query="q", scope="full")
        assert r["kpi_snapshot"]["budget"] == 9.0  # state wins, not recompute

    def test_recompute_fallback_when_state_empty(self):
        from src.graph.state import AnalysisStatus
        from src.graph.workflow import _build_compat_result

        r = _build_compat_result(
            {
                "status": AnalysisStatus.COMPLETED,
                "final_report": "R",
                "expert_results": [{"expert": "subsidy", "kpis": {"x": 1.0}}],
            },
            query="q",
            scope="full",
        )
        assert r["kpi_snapshot"] == {"x": 1.0}
