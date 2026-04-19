"""User scoring and ranking module.

Combines predicted freight-intent probability with predicted customer
lifetime-value (LTV) into a composite score, then ranks and segments
users for downstream targeting.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class UserScorer:
    """Combine intent scores with LTV predictions to rank and segment users.

    The composite score is a weighted blend::

        composite = w_intent * intent_score + w_ltv * normalised_ltv

    where *normalised_ltv* is the min-max scaled LTV to [0, 1].
    """

    def __init__(
        self,
        intent_weight: float = 0.6,
        ltv_weight: float = 0.4,
        score_thresholds: tuple[float, float, float] = (0.7, 0.4, 0.2),
    ) -> None:
        self.intent_weight = intent_weight
        self.ltv_weight = ltv_weight
        self.score_thresholds = score_thresholds  # high / medium / low cut-offs

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score_users(
        self,
        intent_scores: pd.Series | np.ndarray,
        ltv_predictions: pd.Series | np.ndarray,
        user_ids: pd.Series | np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Compute a composite score for each user.

        Parameters
        ----------
        intent_scores : array-like of shape (n_users,)
            Predicted intent probabilities in [0, 1].
        ltv_predictions : array-like of shape (n_users,)
            Predicted lifetime values (monetary).
        user_ids : optional
            User identifiers. If ``None``, a RangeIndex is used.

        Returns
        -------
        pd.DataFrame
            Columns: ``intent_score``, ``ltv``, ``ltv_norm``, ``composite_score``.
        """
        intent_arr = np.asarray(intent_scores, dtype=float).ravel()
        ltv_arr = np.asarray(ltv_predictions, dtype=float).ravel()

        if len(intent_arr) != len(ltv_arr):
            raise ValueError(
                f"Length mismatch: intent_scores ({len(intent_arr)}) "
                f"!= ltv_predictions ({len(ltv_arr)})"
            )

        n = len(intent_arr)
        index = np.asarray(user_ids).ravel() if user_ids is not None else pd.RangeIndex(n)

        # Normalise LTV to [0, 1]
        ltv_min, ltv_max = ltv_arr.min(), ltv_arr.max()
        if ltv_max > ltv_min:
            ltv_norm = (ltv_arr - ltv_min) / (ltv_max - ltv_min)
        else:
            ltv_norm = np.zeros(n)

        composite = self.intent_weight * intent_arr + self.ltv_weight * ltv_norm

        result = pd.DataFrame(
            {
                "intent_score": intent_arr,
                "ltv": ltv_arr,
                "ltv_norm": ltv_norm,
                "composite_score": composite,
            },
            index=index,
        )
        result.index.name = "user_id"
        return result

    # ------------------------------------------------------------------
    # Rank
    # ------------------------------------------------------------------

    def rank_users(self, scored_users: pd.DataFrame) -> pd.DataFrame:
        """Sort users by composite score in descending order and add a rank column.

        Parameters
        ----------
        scored_users : pd.DataFrame
            Must contain ``composite_score`` column (output of :meth:`score_users`).

        Returns
        -------
        pd.DataFrame
            Sorted with additional ``rank`` column (1 = best).
        """
        if scored_users.empty or "composite_score" not in scored_users.columns:
            return scored_users

        ranked = scored_users.sort_values("composite_score", ascending=False).copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        return ranked

    # ------------------------------------------------------------------
    # Segment
    # ------------------------------------------------------------------

    def segment_by_score(
        self,
        scored_users: pd.DataFrame,
        thresholds: tuple[float, float, float] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Segment scored users into tiers based on composite score.

        Parameters
        ----------
        scored_users : pd.DataFrame
            Output of :meth:`score_users` (must contain ``composite_score``).
        thresholds : tuple of 3 floats, optional
            (high, medium, low) thresholds. Users with score >= *high* are
            "high_intent", >= *medium* are "medium_intent", >= *low* are
            "low_intent", rest are "cold".

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys: ``high_intent``, ``medium_intent``, ``low_intent``, ``cold``.
        """
        if thresholds is None:
            thresholds = self.score_thresholds

        high_t, med_t, low_t = thresholds

        segments: dict[str, pd.DataFrame] = {}

        if scored_users.empty:
            for key in ("high_intent", "medium_intent", "low_intent", "cold"):
                segments[key] = pd.DataFrame(columns=scored_users.columns)
            return segments

        scores = scored_users["composite_score"]

        segments["high_intent"] = scored_users[scores >= high_t].copy()
        segments["medium_intent"] = scored_users[
            (scores >= med_t) & (scores < high_t)
        ].copy()
        segments["low_intent"] = scored_users[
            (scores >= low_t) & (scores < med_t)
        ].copy()
        segments["cold"] = scored_users[scores < low_t].copy()

        for key, seg in segments.items():
            logger.info("Segment '%s': %d users", key, len(seg))

        return segments
