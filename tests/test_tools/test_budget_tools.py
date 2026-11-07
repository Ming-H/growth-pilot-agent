"""Tests for the global budget orchestration tools."""

from __future__ import annotations

import pytest


class TestChannelResponseModel:
    """Concave response curves + bounded calibration."""

    def _make(self, budget: float = 200_000.0, context_kpis: dict | None = None):
        from src.tools.budget.response_curves import ChannelResponseModel

        return ChannelResponseModel(total_budget=budget, context_kpis=context_kpis or {})

    def test_channels_exposed(self):
        m = self._make()
        assert set(m.channels) == {"ad", "subsidy", "retention", "conversion"}

    def test_zero_spend_zero_gmv(self):
        m = self._make()
        grid = m.grid("ad")
        assert grid[0] == (0.0, 0.0)

    def test_curve_is_concave(self):
        """Marginal returns must be non-increasing (saturation)."""
        m = self._make()
        for channel in m.channels:
            grid = m.grid(channel)
            marginals = [grid[i + 1][1] - grid[i][1] for i in range(len(grid) - 1)]
            assert all(
                marginals[i + 1] <= marginals[i] + 1e-9 for i in range(len(marginals) - 1)
            ), f"{channel} marginal returns increased"

    def test_grid_respects_channel_cap(self):
        """Max grid spend <= min(max_share * budget, max_abs)."""
        m = self._make(budget=1_000_000.0)
        for channel in m.channels:
            cap = m.channel_cap(channel)
            assert max(s for s, _ in m.grid(channel)) <= cap + 1e-6

    def test_calibration_multiplier_applied_and_bounded(self):
        """observed ROI 4x the reference -> scale multiplier clamped at 2.0."""
        from src.tools.budget.response_curves import CALIBRATION_RULES

        ref = CALIBRATION_RULES["subsidy"]["reference_roi"]
        m = self._make(context_kpis={"subsidy_expected_roi": ref * 4.0})
        base = self._make()
        s_cal = m.grid("subsidy")[-1][1]
        s_base = base.grid("subsidy")[-1][1]
        assert s_cal == pytest.approx(s_base * 2.0, rel=1e-6)

    def test_no_calibration_keys_multiplier_one(self):
        m = self._make(context_kpis={"unrelated": 3.3})
        base = self._make()
        assert m.grid("ad") == base.grid("ad")

    def test_deterministic_same_input_same_output(self):
        a = self._make(context_kpis={"subsidy_expected_roi": 2.0}).grid("subsidy")
        b = self._make(context_kpis={"subsidy_expected_roi": 2.0}).grid("subsidy")
        assert a == b


class TestGlobalBudgetAllocator:
    """IP allocation with exact-enumeration fallback."""

    def _grids(self, budget: float = 200_000.0):
        from src.tools.budget.response_curves import ChannelResponseModel

        m = ChannelResponseModel(total_budget=budget)
        return {c: m.grid(c) for c in m.channels}

    def _alloc(self, budget: float = 200_000.0):
        from src.tools.budget.global_allocator import GlobalBudgetAllocator

        return GlobalBudgetAllocator().allocate(self._grids(budget), total_budget=budget)

    def test_allocation_respects_total_budget(self):
        r = self._alloc()
        assert r["status"] == "optimal"
        assert sum(r["allocation"].values()) <= 200_000.0 + 1e-6

    def test_best_channel_gets_largest_share(self):
        """Subsidy has the strongest low-spend response in the defaults."""
        r = self._alloc()
        assert max(r["allocation"], key=r["allocation"].get) in {"subsidy", "ad"}

    def test_per_channel_cap_respected(self):
        from src.tools.budget.response_curves import ChannelResponseModel

        m = ChannelResponseModel(total_budget=200_000.0)
        r = self._alloc()
        for channel, spend in r["allocation"].items():
            assert spend <= m.channel_cap(channel) + 1e-6

    def test_zero_budget_errors(self):
        """Allocator-level guard (the curve model raises its own ValueError)."""
        from src.tools.budget.global_allocator import GlobalBudgetAllocator

        r = GlobalBudgetAllocator().allocate(self._grids(), total_budget=0)
        assert "error" in r

    def test_expected_roi_consistent(self):
        r = self._alloc()
        spent = sum(r["allocation"].values())
        assert r["expected_incremental_gmv"] >= 0
        if spent > 0:
            assert r["expected_roi"] == round(r["expected_incremental_gmv"] / spent, 4)

    def test_enumeration_fallback_matches_ip(self):
        """With PuLP monkeypatched out, enumeration must find the same optimum."""
        import src.tools.budget.global_allocator as ga

        grids = self._grids()
        ip = ga.GlobalBudgetAllocator().allocate(grids, total_budget=200_000.0)
        real_has = ga.HAS_PULP
        ga.HAS_PULP = False
        try:
            enum = ga.GlobalBudgetAllocator().allocate(grids, total_budget=200_000.0)
        finally:
            ga.HAS_PULP = real_has
        assert enum["solver_method"] == "enumeration"
        assert enum["expected_incremental_gmv"] == pytest.approx(
            ip["expected_incremental_gmv"], rel=1e-9
        )

    def test_deterministic(self):
        a = self._alloc()
        b = self._alloc()
        assert a["allocation"] == b["allocation"]
        assert a["expected_incremental_gmv"] == b["expected_incremental_gmv"]


class TestBudgetResultModel:
    def test_validate_expert_result_marks_validated(self):
        from src.core.models import EXPERT_RESULT_MODELS
        from src.graph.nodes import _validate_expert_result

        assert "budget" in EXPERT_RESULT_MODELS
        result = {
            "success": True,
            "allocation": {"ad": 60_000.0, "subsidy": 90_000.0},
            "expected_incremental_gmv": 320_000.0,
            "expected_roi": 1.8,
            "kpis": {"budget_total": 200_000.0, "expected_roi": 1.8},
        }
        out = _validate_expert_result("budget", result)
        assert out["_validated"] is True
        assert out["agent_name"] == "budget"
