"""Channel response models for global budget orchestration.

Each growth channel is modelled with a concave (saturating) response
curve: incremental GMV as a function of spend, ``g(s) = scale * log1p(s / k)``.
Marginal returns decline — the property that makes cross-channel
allocation a real optimisation instead of "give everything to the best
channel".

Determinism: default parameters are constants; calibration from
``context_kpis`` (KPIs already produced by other experts in the same
run) applies a bounded multiplier to ``scale``. Same inputs → same
curve, always.
"""

from __future__ import annotations

import math
from typing import Any

# Channel defaults. scale/k shape the log1p saturation curve; max_share
# caps the share of the total budget a single channel may absorb;
# max_abs is an absolute ceiling (so a huge budget cannot push 60% of
# ¥10M into retention). Values are plausible-growth defaults — the
# honest label is data_source=default_curves until calibrated.
CHANNEL_DEFAULTS: dict[str, dict[str, float]] = {
    "ad": {"scale": 420_000.0, "k": 150_000.0, "max_share": 0.60, "max_abs": 600_000.0},
    "subsidy": {"scale": 380_000.0, "k": 90_000.0, "max_share": 0.60, "max_abs": 400_000.0},
    "retention": {"scale": 150_000.0, "k": 40_000.0, "max_share": 0.40, "max_abs": 150_000.0},
    "conversion": {"scale": 200_000.0, "k": 60_000.0, "max_share": 0.40, "max_abs": 200_000.0},
}

# First-order calibration: anchor a channel's scale to an ROI KPI a
# sibling expert may already have produced in this run. The multiplier
# is clamped — calibration is anchoring, not curve re-fitting.
#
# ANCHOR CONTRACT: every kpi_key here must be a key the anchoring expert
# actually emits in its ``kpis`` dict (see tests/test_tools/test_budget_tools.py
# ::TestCalibrationContract — it runs the expert and asserts the match).
# v9.1 shipped speculative keys for ad/retention/conversion that never
# matched (dead config, data_source stuck at default_curves); they were
# removed. Re-add a channel only together with its contract test.
#   ad        emits: ad_target_cpa / ad_win_rate / ad_impressions (no ROI yet)
#   retention emits: churn_high_risk_ratio / churn_model_auc (no ROI yet)
#   conversion emits: conversion_rate / funnel_bottleneck_rate (no ROI yet)
CALIBRATION_RULES: dict[str, dict[str, Any]] = {
    "subsidy": {"kpi_keys": ["subsidy_expected_roi", "expected_roi"], "reference_roi": 1.6},
}

_CALIBRATION_BOUNDS = (0.5, 2.0)
_GRID_STEPS = 10  # 0%, 10%, ..., 100% of the channel cap


def _as_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class ChannelResponseModel:
    """Saturating response curves per channel, optionally calibrated."""

    def __init__(self, total_budget: float, context_kpis: dict[str, Any] | None = None) -> None:
        if total_budget <= 0:
            raise ValueError("total_budget must be positive")
        self.total_budget = float(total_budget)
        self.context_kpis = context_kpis or {}
        self._multipliers = {
            channel: self._calibration_multiplier(channel) for channel in CHANNEL_DEFAULTS
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def channels(self) -> list[str]:
        return list(CHANNEL_DEFAULTS)

    def channel_cap(self, channel: str) -> float:
        d = CHANNEL_DEFAULTS[channel]
        return round(min(d["max_share"] * self.total_budget, d["max_abs"]), 2)

    def curve(self, channel: str) -> tuple[float, float]:
        """Return the (scale, k) parameters after calibration."""
        d = CHANNEL_DEFAULTS[channel]
        return d["scale"] * self._multipliers[channel], d["k"]

    def grid(self, channel: str) -> list[tuple[float, float]]:
        """(spend, incremental_gmv) grid from 0 to the channel cap."""
        cap = self.channel_cap(channel)
        scale, k = self.curve(channel)

        def g(s: float) -> float:
            return round(scale * math.log1p(s / k), 2) if s > 0 else 0.0

        return [
            (round(cap * i / _GRID_STEPS, 2), g(cap * i / _GRID_STEPS))
            for i in range(_GRID_STEPS + 1)
        ]

    def multiplier(self, channel: str) -> float:
        """The applied calibration multiplier (1.0 = default curve)."""
        return self._multipliers[channel]

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _calibration_multiplier(self, channel: str) -> float:
        rule = CALIBRATION_RULES.get(channel)
        if rule is None:  # no anchor KPI for this channel (yet) — default curve
            return 1.0
        observed = None
        for key in rule["kpi_keys"]:
            v = _as_float(self.context_kpis.get(key))
            if v is not None and v > 0:
                observed = v
                break
        if observed is None:
            return 1.0
        ratio = observed / float(rule["reference_roi"])
        return round(min(max(ratio, _CALIBRATION_BOUNDS[0]), _CALIBRATION_BOUNDS[1]), 4)
