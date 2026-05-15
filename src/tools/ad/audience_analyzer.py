"""Audience analysis and lookalike expansion.

Segments audiences by behavioral and demographic features, and finds
lookalike users using cosine similarity on feature vectors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.linalg import norm

from src.tools.registry import ToolRegistry


@ToolRegistry.register("audience_analyzer")
class AudienceAnalyzer:
    """Analyze audience segments and perform lookalike expansion."""

    def analyze_audience(self, audience_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Build audience segment profiles from raw user data.

        Args:
            audience_data: List of dicts, each with keys like:
                - 'user_id': str
                - 'age': int
                - 'gender': str ('M' | 'F')
                - 'city_tier': int (1-4)
                - 'historical_orders': int
                - 'avg_order_value': float
                - 'days_since_last_order': int
                - 'ltv': float
                - 'segment': str (optional pre-labeled segment)

        Returns:
            Dict with:
                - 'total_users': int
                - 'segments': dict keyed by segment name
                - 'demographics': age/gender distribution
                - 'behavioral_summary': aggregate behavioral stats
        """
        if not audience_data:
            return {"total_users": 0, "segments": {}, "demographics": {}, "behavioral_summary": {}}

        total_users = len(audience_data)

        # --- Segment analysis ---
        segment_groups: dict[str, list[dict]] = {}
        for u in audience_data:
            seg = self._assign_segment(u)
            segment_groups.setdefault(seg, []).append(u)

        segments: dict[str, dict[str, Any]] = {}
        for seg_name, users in segment_groups.items():
            ltvs = [u.get("ltv", 0) for u in users]
            orders = [u.get("historical_orders", 0) for u in users]
            aovs = [u.get("avg_order_value", 0) for u in users if u.get("avg_order_value", 0) > 0]
            days_since = [u.get("days_since_last_order", 30) for u in users]

            segments[seg_name] = {
                "count": len(users),
                "pct_of_total": round(len(users) / total_users, 4),
                "avg_ltv": round(float(np.mean(ltvs)), 2),
                "avg_orders": round(float(np.mean(orders)), 1),
                "avg_aov": round(float(np.mean(aovs)), 2) if aovs else 0.0,
                "avg_recency_days": round(float(np.mean(days_since)), 1),
                "median_ltv": round(float(np.median(ltvs)), 2),
            }

        # --- Demographics ---
        ages = [u.get("age", 30) for u in audience_data]
        gender_counts: dict[str, int] = {}
        for u in audience_data:
            g = u.get("gender", "unknown")
            gender_counts[g] = gender_counts.get(g, 0) + 1

        age_bins = {"18-24": 0, "25-34": 0, "35-44": 0, "45-54": 0, "55+": 0}
        for a in ages:
            if a < 25:
                age_bins["18-24"] += 1
            elif a < 35:
                age_bins["25-34"] += 1
            elif a < 45:
                age_bins["35-44"] += 1
            elif a < 55:
                age_bins["45-54"] += 1
            else:
                age_bins["55+"] += 1

        demographics = {
            "age_distribution": {k: round(v / total_users, 4) for k, v in age_bins.items()},
            "age_mean": round(float(np.mean(ages)), 1),
            "age_median": round(float(np.median(ages)), 1),
            "gender_distribution": {k: round(v / total_users, 4) for k, v in gender_counts.items()},
        }

        # --- Behavioral summary ---
        all_orders = [u.get("historical_orders", 0) for u in audience_data]
        all_aov = [u.get("avg_order_value", 0) for u in audience_data if u.get("avg_order_value", 0) > 0]
        all_ltvs = [u.get("ltv", 0) for u in audience_data]
        all_recency = [u.get("days_since_last_order", 30) for u in audience_data]

        behavioral_summary = {
            "avg_orders_per_user": round(float(np.mean(all_orders)), 1),
            "avg_aov": round(float(np.mean(all_aov)), 2) if all_aov else 0.0,
            "avg_ltv": round(float(np.mean(all_ltvs)), 2),
            "avg_recency_days": round(float(np.mean(all_recency)), 1),
            "total_orders": sum(all_orders),
        }

        return {
            "total_users": total_users,
            "segments": segments,
            "demographics": demographics,
            "behavioral_summary": behavioral_summary,
        }

    def lookalike_expansion(
        self,
        seed_users: list[dict[str, Any]],
        all_users: list[dict[str, Any]],
        top_k: int = 10000,
    ) -> list[str]:
        """Find users similar to seed users using cosine similarity.

        Each user dict should contain numeric features used for vectorization:
        'age', 'historical_orders', 'avg_order_value', 'days_since_last_order',
        'ltv', 'city_tier', etc.

        Args:
            seed_users: List of seed user dicts (known good users).
            all_users: Pool of candidate users to search.
            top_k: Number of most similar users to return.

        Returns:
            List of user_ids (strings) sorted by similarity descending,
            excluding the seed users themselves.
        """
        if not seed_users or not all_users:
            return []

        feature_keys = ["age", "historical_orders", "avg_order_value", "days_since_last_order", "ltv", "city_tier"]

        # Build seed centroid
        seed_vectors = np.array([self._user_to_vector(u, feature_keys) for u in seed_users])
        centroid = np.mean(seed_vectors, axis=0)

        # Normalize centroid
        centroid_norm = norm(centroid)
        if centroid_norm == 0:
            return []
        centroid_normalized = centroid / centroid_norm

        # Collect seed user IDs for exclusion
        seed_ids = {str(u.get("user_id", "")) for u in seed_users}

        # Build candidate matrix (excluding seed users) and keep aligned uid list
        candidates: list[tuple[str, np.ndarray]] = []
        for u in all_users:
            uid = str(u.get("user_id", ""))
            if uid in seed_ids:
                continue
            vec = self._user_to_vector(u, feature_keys)
            if np.any(vec != 0):
                candidates.append((uid, vec))

        if not candidates:
            return []

        # Vectorized cosine similarity
        candidate_uids = [uid for uid, _ in candidates]
        candidate_matrix = np.array([vec for _, vec in candidates])  # (N, D)

        # Normalize rows
        row_norms = norm(candidate_matrix, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        candidate_normalized = candidate_matrix / row_norms

        # Cosine similarity = dot(candidate_normalized, centroid_normalized)
        similarities = candidate_normalized @ centroid_normalized  # (N,)

        # Sort by similarity descending
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [candidate_uids[i] for i in top_indices]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_segment(user: dict[str, Any]) -> str:
        """Assign a user to a behavioral segment."""
        ltv = user.get("ltv", 0)
        orders = user.get("historical_orders", 0)
        recency = user.get("days_since_last_order", 30)
        segment = user.get("segment")

        if segment:
            return segment

        if ltv >= 500 and orders >= 10:
            return "高价值忠实用户"
        elif ltv >= 200 and orders >= 5:
            return "中高价值用户"
        elif recency <= 7 and orders >= 2:
            return "活跃用户"
        elif recency <= 30:
            return "近期用户"
        elif orders >= 1:
            return "沉睡用户"
        else:
            return "新用户"

    @staticmethod
    def _user_to_vector(user: dict[str, Any], feature_keys: list[str]) -> np.ndarray:
        """Convert a user dict to a numeric feature vector."""
        defaults = {
            "age": 30,
            "historical_orders": 0,
            "avg_order_value": 0.0,
            "days_since_last_order": 30,
            "ltv": 0.0,
            "city_tier": 2,
        }
        values = []
        for key in feature_keys:
            v = user.get(key, defaults.get(key, 0))
            values.append(float(v))
        return np.array(values, dtype=np.float64)
