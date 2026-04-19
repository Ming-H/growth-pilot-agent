"""User segmentation using RFM analysis and lifecycle stages.

Provides RFM (Recency-Frequency-Monetary) segmentation combined with
user lifecycle classification to support targeted marketing strategies.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class UserSegmentor:
    """RFM + lifecycle segmentation for ride-hailing users.

    Typical usage::

        seg = UserSegmentor()
        rfm = seg.rfm_segmentation(user_data)
        combined = seg.combined_segmentation(user_data)
    """

    # RFM score labels
    _RECENCY_LABELS = {5: "very_recent", 4: "recent", 3: "moderate", 2: "stale", 1: "inactive"}
    _FREQUENCY_LABELS = {5: "very_high", 4: "high", 3: "moderate", 2: "low", 1: "very_low"}
    _MONETARY_LABELS = {5: "very_high", 4: "high", 3: "moderate", 2: "low", 1: "very_low"}

    # Lifecycle stages
    _LIFECYCLE_STAGES = ["new", "active", "loyal", "at_risk", "churned"]

    def __init__(self, n_bins: int = 5) -> None:
        """Initialise segmentor.

        Parameters
        ----------
        n_bins : int
            Number of quantile bins for RFM scoring (typically 5).
        """
        self.n_bins = n_bins

    # ------------------------------------------------------------------
    # RFM segmentation
    # ------------------------------------------------------------------

    def rfm_segmentation(self, user_data: pd.DataFrame) -> pd.DataFrame:
        """Compute RFM (Recency-Frequency-Monetary) segmentation.

        Expected columns in *user_data*:
            user_id, last_ride_date (datetime or str), ride_count (int),
            total_spent (float)

        Returns a DataFrame with columns:
            user_id, recency_days, recency_score, recency_label,
            frequency_score, frequency_label, monetary_score,
            monetary_label, rfm_score, rfm_segment
        """
        if user_data.empty:
            return pd.DataFrame()

        df = user_data.copy()

        # Ensure required columns
        for col, default in [
            ("last_ride_date", pd.Timestamp.now()),
            ("ride_count", 0),
            ("total_spent", 0.0),
        ]:
            if col not in df.columns:
                df[col] = default

        df["last_ride_date"] = pd.to_datetime(df["last_ride_date"], errors="coerce")
        now = pd.Timestamp.now()
        df["recency_days"] = (now - df["last_ride_date"]).dt.days.fillna(999).astype(int)

        # Score each dimension 1-5 using quantile binning.
        # For recency, lower days = higher score (inverted).
        df["recency_score"] = self._quantile_score(df["recency_days"], invert=True)
        df["frequency_score"] = self._quantile_score(df["ride_count"], invert=False)
        df["monetary_score"] = self._quantile_score(df["total_spent"], invert=False)

        df["recency_label"] = df["recency_score"].map(self._RECENCY_LABELS).fillna("unknown")
        df["frequency_label"] = df["frequency_score"].map(self._FREQUENCY_LABELS).fillna("unknown")
        df["monetary_label"] = df["monetary_score"].map(self._MONETARY_LABELS).fillna("unknown")

        # Composite RFM score (sum of the three)
        df["rfm_score"] = df["recency_score"] + df["frequency_score"] + df["monetary_score"]

        # Segment labels based on RFM score
        df["rfm_segment"] = df["rfm_score"].apply(self._classify_rfm)

        result_cols = [
            "user_id", "recency_days",
            "recency_score", "recency_label",
            "frequency_score", "frequency_label",
            "monetary_score", "monetary_label",
            "rfm_score", "rfm_segment",
        ]
        available = [c for c in result_cols if c in df.columns]
        return df[available]

    # ------------------------------------------------------------------
    # Lifecycle segmentation
    # ------------------------------------------------------------------

    def lifecycle_segment(self, user_data: pd.DataFrame) -> pd.Series:
        """Assign a lifecycle stage to each user.

        Expected columns:
            user_id, days_since_signup (int), ride_count_30d (int),
            ride_count_90d (int), days_since_last_ride (int)

        Returns a Series indexed by ``user_id`` with lifecycle stage labels.
        """
        if user_data.empty:
            return pd.Series(dtype=str, name="lifecycle_stage")

        df = user_data.copy()

        # Defaults for missing columns
        for col, default in [
            ("days_since_signup", 365),
            ("ride_count_30d", 0),
            ("ride_count_90d", 0),
            ("days_since_last_ride", 999),
        ]:
            if col not in df.columns:
                df[col] = default

        stages = pd.Series("unknown", index=df.index, name="lifecycle_stage")

        # New users: signed up within 30 days
        is_new = df["days_since_signup"] <= 30
        stages[is_new] = "new"

        # Active: rides in last 30d, not new
        is_active = (~is_new) & (df["ride_count_30d"] >= 3)
        stages[is_active] = "active"

        # Loyal: heavy users (>8 rides in 30d or >20 in 90d)
        is_loyal = (~is_new) & (
            (df["ride_count_30d"] >= 8) | (df["ride_count_90d"] >= 20)
        )
        stages[is_loyal] = "loyal"

        # At-risk: not new, some activity but declining (last ride 14-60 days ago)
        is_at_risk = (
            (~is_new)
            & (~is_loyal)
            & (df["days_since_last_ride"] >= 14)
            & (df["days_since_last_ride"] < 60)
        )
        stages[is_at_risk] = "at_risk"

        # Churned: no ride in 60+ days
        is_churned = (~is_new) & (df["days_since_last_ride"] >= 60)
        stages[is_churned] = "churned"

        stages.index = df["user_id"].values
        return stages

    # ------------------------------------------------------------------
    # Combined segmentation
    # ------------------------------------------------------------------

    def combined_segmentation(self, user_data: pd.DataFrame) -> pd.DataFrame:
        """Produce combined RFM + lifecycle segmentation.

        Parameters
        ----------
        user_data : pd.DataFrame
            Must contain columns required by both :meth:`rfm_segmentation`
            and :meth:`lifecycle_segment`.

        Returns
        -------
        pd.DataFrame
            Indexed by ``user_id`` with RFM columns plus ``lifecycle_stage``.
        """
        rfm = self.rfm_segmentation(user_data)

        if rfm.empty:
            return pd.DataFrame()

        lifecycle = self.lifecycle_segment(user_data)

        combined = rfm.set_index("user_id") if "user_id" in rfm.columns else rfm
        combined["lifecycle_stage"] = lifecycle

        return combined

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _quantile_score(
        self,
        series: pd.Series,
        invert: bool = False,
    ) -> pd.Series:
        """Assign 1..n_bins quantile scores to a numeric series.

        For *invert=True*, the lowest values get the highest scores
        (used for recency where fewer days = better).
        """
        n = self.n_bins
        try:
            scores = pd.qcut(series, q=n, labels=False, duplicates="drop") + 1
        except ValueError:
            # Not enough unique values for n bins
            scores = pd.Series(np.ones(len(series), dtype=int), index=series.index)

        if invert:
            scores = (n + 1) - scores

        return scores.astype(int)

    @staticmethod
    def _classify_rfm(score: int) -> str:
        """Map total RFM score (3-15) to a segment label."""
        if score >= 13:
            return "champions"
        if score >= 10:
            return "loyal_customers"
        if score >= 7:
            return "potential_loyalists"
        if score >= 5:
            return "promising"
        if score >= 3:
            return "needs_attention"
        return "lost"
