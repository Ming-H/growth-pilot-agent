"""Cohort retention analysis module.

Computes retention matrices from order data, analyses retention curves,
and identifies inflection points where retention rate changes
significantly.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


class CohortAnalyzer:
    """Analyse user retention by cohort dimensions.

    Typical usage::

        ca = CohortAnalyzer()
        matrix = ca.analyze_retention_cohort(order_data, cohort_dim="signup_month")
        inflection = ca.find_retention_inflection(retention_curve)
    """

    def __init__(self, observation_window: int = 90) -> None:
        """Initialise analyser.

        Parameters
        ----------
        observation_window : int
            Maximum number of days to track retention (default 90).
        """
        self.observation_window = observation_window

    # ------------------------------------------------------------------
    # Cohort retention matrix
    # ------------------------------------------------------------------

    def analyze_retention_cohort(
        self,
        order_data: pd.DataFrame,
        cohort_dim: str = "signup_date",
        period: str = "W",
    ) -> pd.DataFrame:
        """Compute a retention matrix from order-level data.

        Parameters
        ----------
        order_data : pd.DataFrame
            Must contain:
                ``user_id``, ``order_date`` (datetime-like),
                and the column referenced by *cohort_dim*.
            If *cohort_dim* is ``"signup_date"``, the DataFrame must also
            have a ``signup_date`` column.  If ``"first_order_date"``, the
            first order per user is used as the cohort date.
        period : str
            Resampling frequency for cohort periods:
            ``"D"`` (daily), ``"W"`` (weekly), ``"M"`` (monthly).

        Returns
        -------
        pd.DataFrame
            A retention matrix where rows are cohorts, columns are period
            offsets (0, 1, 2, ...), and values are retention rates (0-1).
        """
        if order_data.empty:
            return pd.DataFrame()

        df = order_data.copy()
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df = df.dropna(subset=["order_date", "user_id"])

        if df.empty:
            return pd.DataFrame()

        # Determine cohort date per user
        if cohort_dim == "first_order_date" or cohort_dim not in df.columns:
            first_order = df.groupby("user_id")["order_date"].min().reset_index()
            first_order.columns = ["user_id", "cohort_date"]
        else:
            df[cohort_dim] = pd.to_datetime(df[cohort_dim], errors="coerce")
            first_order = (
                df[["user_id", cohort_dim]]
                .drop_duplicates("user_id")
                .rename(columns={cohort_dim: "cohort_date"})
            )

        df = df.merge(first_order, on="user_id", how="left")
        df = df.dropna(subset=["cohort_date"])

        # Cohort period (e.g. first week/month of signup)
        if period == "D":
            df["cohort_period"] = df["cohort_date"].dt.date
        elif period == "W":
            df["cohort_period"] = df["cohort_date"].dt.to_period("W").apply(
                lambda p: p.start_time.date()
            )
        elif period == "M":
            df["cohort_period"] = df["cohort_date"].dt.to_period("M").apply(
                lambda p: p.start_time.date()
            )
        else:
            df["cohort_period"] = df["cohort_date"].dt.to_period(period).apply(
                lambda p: p.start_time.date()
            )

        # Period offset (number of periods since cohort start)
        if period == "D":
            df["period_offset"] = (df["order_date"] - df["cohort_date"]).dt.days
        elif period == "W":
            df["period_offset"] = ((df["order_date"] - df["cohort_date"]).dt.days // 7)
        elif period == "M":
            delta = df["order_date"].dt.to_period("M") - df["cohort_date"].dt.to_period("M")
            df["period_offset"] = delta.apply(lambda p: p.n if hasattr(p, "n") else int(p))
        else:
            df["period_offset"] = ((df["order_date"] - df["cohort_date"]).dt.days // 7)

        # Clip to observation window
        max_offset_days = self.observation_window
        if period == "D":
            df = df[df["period_offset"] <= max_offset_days]
        elif period == "W":
            df = df[df["period_offset"] <= max_offset_days // 7]
        elif period == "M":
            df = df[df["period_offset"] <= max_offset_days // 30]

        df["period_offset"] = df["period_offset"].astype(int)

        # Cohort sizes
        cohort_sizes = (
            df.groupby("cohort_period")["user_id"].nunique().reset_index()
        )
        cohort_sizes.columns = ["cohort_period", "cohort_size"]

        # Retained users per (cohort, offset)
        retained = (
            df.groupby(["cohort_period", "period_offset"])["user_id"]
            .nunique()
            .reset_index()
        )
        retained.columns = ["cohort_period", "period_offset", "retained_users"]

        # Merge and compute rate
        retention = retained.merge(cohort_sizes, on="cohort_period", how="left")
        retention["retention_rate"] = retention["retained_users"] / retention["cohort_size"]

        # Pivot to matrix
        matrix = retention.pivot_table(
            index="cohort_period",
            columns="period_offset",
            values="retention_rate",
            aggfunc="first",
        )

        # Sort columns
        matrix = matrix.reindex(columns=sorted(matrix.columns))

        return matrix

    # ------------------------------------------------------------------
    # Inflection point detection
    # ------------------------------------------------------------------

    def find_retention_inflection(
        self,
        retention_curve: pd.Series | np.ndarray | list[float],
        smoothing_window: int = 3,
        prominence_threshold: float = 0.05,
    ) -> dict[str, Any]:
        """Find the inflection point in a retention curve.

        An inflection point is a period where the retention rate shows a
        significant change in slope (either a sharp drop or stabilisation).

        Parameters
        ----------
        retention_curve : array-like
            Retention rates ordered by period (0, 1, 2, ...).
            Period 0 should be 1.0 (or close to it).
        smoothing_window : int
            Window size for moving-average smoothing.
        prominence_threshold : float
            Minimum relative change to qualify as an inflection.

        Returns
        -------
        dict
            Keys:
                ``inflection_period``: int (period index),
                ``retention_at_inflection``: float,
                ``slope_before``: float,
                ``slope_after``: float,
                ``inflection_type``: "drop" | "stabilisation" | "recovery",
                ``smoothed_curve``: list of floats
        """
        curve = np.asarray(retention_curve, dtype=float).ravel()

        if len(curve) < 3:
            return {
                "inflection_period": None,
                "retention_at_inflection": None,
                "inflection_type": "insufficient_data",
                "smoothed_curve": curve.tolist(),
                "message": "Retention curve too short for inflection analysis",
            }

        # Smooth the curve
        if len(curve) >= smoothing_window:
            kernel = np.ones(smoothing_window) / smoothing_window
            smoothed = np.convolve(curve, kernel, mode="valid")
            # Pad to original length
            pad = (len(curve) - len(smoothed)) // 2
            smoothed = np.concatenate([
                np.full(pad, smoothed[0]),
                smoothed,
                np.full(len(curve) - len(smoothed) - pad, smoothed[-1]),
            ])
        else:
            smoothed = curve.copy()

        # Compute slopes (first derivative of smoothed curve)
        slopes = np.diff(smoothed)

        if len(slopes) < 2:
            return {
                "inflection_period": None,
                "retention_at_inflection": float(curve[0]),
                "inflection_type": "insufficient_data",
                "smoothed_curve": smoothed.tolist(),
                "message": "Not enough data points for slope analysis",
            }

        # Find local minima in slopes (biggest negative slope changes)
        # and local maxima (recovery / stabilisation points)
        try:
            local_min_indices = argrelextrema(slopes, np.less, order=2)[0]
            local_max_indices = argrelextrema(slopes, np.greater, order=2)[0]
        except Exception:
            local_min_indices = np.array([])
            local_max_indices = np.array([])

        # Also check the most negative slope directly
        max_drop_idx = int(np.argmin(slopes))

        # Select the most prominent inflection
        best_idx: int | None = None
        best_type: str = "drop"

        # Prefer identified local minima (significant drop-offs)
        if len(local_min_indices) > 0:
            # Pick the one with the steepest negative slope
            slopes_at_mins = slopes[local_min_indices]
            best_min = local_min_indices[np.argmin(slopes_at_mins)]
            if abs(slopes[best_min]) >= prominence_threshold:
                best_idx = int(best_min)
                best_type = "drop"

        # Check for stabilisation (slope approaching 0 after a drop)
        if len(local_max_indices) > 0:
            for idx in local_max_indices:
                if idx > 1 and slopes[idx] > 0 and abs(slopes[idx]) >= prominence_threshold * 0.5:
                    if best_idx is None or idx > best_idx:
                        best_idx = int(idx)
                        best_type = "stabilisation"

        # Fallback: use the period with the steepest single-period drop
        if best_idx is None:
            best_idx = max_drop_idx
            best_type = "drop"

        # Compute slopes before and after inflection
        if best_idx > 0:
            slope_before = float(slopes[best_idx - 1])
        else:
            slope_before = float(slopes[0])

        if best_idx < len(slopes) - 1:
            slope_after = float(slopes[best_idx + 1])
        else:
            slope_after = float(slopes[-1])

        # Classify inflection type more precisely
        if slope_before < -prominence_threshold and slope_after > -prominence_threshold * 0.5:
            inflection_type = "stabilisation"
        elif slope_after > prominence_threshold * 0.3:
            inflection_type = "recovery"
        else:
            inflection_type = "drop"

        return {
            "inflection_period": best_idx,
            "retention_at_inflection": float(curve[best_idx]) if best_idx < len(curve) else None,
            "slope_before": round(slope_before, 4),
            "slope_after": round(slope_after, 4),
            "inflection_type": inflection_type,
            "smoothed_curve": [round(v, 4) for v in smoothed.tolist()],
            "max_single_drop_period": int(max_drop_idx),
            "max_single_drop_rate": round(float(slopes[max_drop_idx]), 4),
        }

    # ------------------------------------------------------------------
    # Utility: build a synthetic retention curve
    # ------------------------------------------------------------------

    @staticmethod
    def generate_sample_retention_curve(
        n_periods: int = 30,
        initial_retention: float = 1.0,
        decay_rate: float = 0.05,
        noise_std: float = 0.02,
        inflection_period: int | None = 14,
        seed: int = 42,
    ) -> pd.Series:
        """Generate a synthetic retention curve for testing.

        Models a typical retention curve with an optional inflection
        point where decay rate changes.
        """
        rng = np.random.RandomState(seed)

        periods = np.arange(n_periods)
        curve = np.zeros(n_periods)

        for t in range(n_periods):
            if inflection_period and t < inflection_period:
                rate = initial_retention * np.exp(-decay_rate * t)
            elif inflection_period:
                val_at_inflection = initial_retention * np.exp(-decay_rate * inflection_period)
                rate = val_at_inflection * np.exp(-decay_rate * 0.3 * (t - inflection_period))
            else:
                rate = initial_retention * np.exp(-decay_rate * t)

            curve[t] = max(0, rate + rng.normal(0, noise_std))

        return pd.Series(curve, index=periods, name="retention_rate")
