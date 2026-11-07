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

# First-order calibration: anchor each channel's scale to an ROI KPI a
# sibling expert may already have produced in this run. The multiplier
# is clamped — calibration is anchoring, not curve re-fitting.
CALIBRATION_RULES: dict[str, dict[str, Any]] = {
    "ad": {"kpi_keys": ["ad_expected_roi", "ad_roi"], "reference_roi": 2.0},
    "subsidy": {"kpi_keys": ["subsidy_expected_roi", "expected_roi"], "reference_roi": 1.6},
    "retention": {"kpi_keys": ["retention_expected_roi", "winback_roi"], "reference_roi": 1.8},
    "conversion": {"kpi_keys": ["conversion_expected_roi", "coupon_roi"], "reference_roi": 2.2},
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
        rule = CALIBRATION_RULES[channel]
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
