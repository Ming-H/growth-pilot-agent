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
