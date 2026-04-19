"""Feature engineering for ride-hailing user behavior data.

Extracts behavioral, temporal, and contextual features from raw ride-hailing
logs and user profiles to support freight intent prediction and user scoring.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default column sets – callers can pass DataFrames with these columns
# ---------------------------------------------------------------------------

_REQUIRED_LOG_COLUMNS = {"user_id", "ride_time", "distance", "destination_type"}
_OPTIONAL_LOG_COLUMNS = {"fare", "ride_duration"}

_REQUIRED_PROFILE_COLUMNS = {"user_id"}
_OPTIONAL_PROFILE_COLUMNS = {"city_tier", "has_freight_search", "has_large_item_search"}


class FeatureEngine:
    """Extract user-level features from ride-hailing event logs and profiles.

    Each public method returns a DataFrame indexed by ``user_id``.
    :meth:`build_feature_matrix` is the primary entry point that combines all
    feature groups into a single matrix ready for downstream ML pipelines.
    """

    # ------------------------------------------------------------------
    # Behaviour features
    # ------------------------------------------------------------------

    def extract_behavior_features(self, user_logs: pd.DataFrame) -> pd.DataFrame:
        """Compute per-user behavioural features from ride-level logs.

        Expected columns in *user_logs*:
            user_id, ride_time (datetime-like), distance (float),
            destination_type (str), fare (float, optional),
            ride_duration (float, optional)

        Returns a DataFrame indexed by ``user_id`` with columns:
            ride_count_30d, night_ride_ratio, weekend_ride_ratio,
            avg_ride_distance, median_ride_distance, avg_fare,
            destination_type_* (one-hot distribution of top-N types)
        """
        if user_logs.empty:
            return pd.DataFrame()

        df = user_logs.copy()
        self._coerce_log_columns(df)

        now = pd.Timestamp.now()
        df["ride_time"] = pd.to_datetime(df["ride_time"], errors="coerce")
        df = df.dropna(subset=["ride_time"])

        df["hour"] = df["ride_time"].dt.hour
        df["dow"] = df["ride_time"].dt.dayofweek  # 0=Mon
        df["is_night"] = df["hour"].between(22, 23) | df["hour"].between(0, 5)
        df["is_weekend"] = df["dow"] >= 5

        # Only consider last 30 days of rides
        cutoff = now - pd.Timedelta(days=30)
        df30 = df[df["ride_time"] >= cutoff]

        grouped = df30.groupby("user_id")

        records: list[dict[str, Any]] = []
        for uid, grp in grouped:
            rec: dict[str, Any] = {"user_id": uid}
            total = len(grp)
            rec["ride_count_30d"] = total
            rec["night_ride_ratio"] = grp["is_night"].sum() / max(total, 1)
            rec["weekend_ride_ratio"] = grp["is_weekend"].sum() / max(total, 1)
            rec["avg_ride_distance"] = grp["distance"].mean()
            rec["median_ride_distance"] = grp["distance"].median()

            if "fare" in grp.columns:
                rec["avg_fare"] = grp["fare"].mean()
            if "ride_duration" in grp.columns:
                rec["avg_ride_duration"] = grp["ride_duration"].mean()

            # Destination type distribution (top-6 types)
            dest_counts = grp["destination_type"].value_counts(normalize=True)
            for dtype, ratio in dest_counts.head(6).items():
                safe_name = f"dest_type_{self._safe_col(dtype)}"
                rec[safe_name] = ratio

            records.append(rec)

        result = pd.DataFrame.from_records(records).set_index("user_id").fillna(0)
        return result

    # ------------------------------------------------------------------
    # Temporal features
    # ------------------------------------------------------------------

    def extract_temporal_features(self, user_logs: pd.DataFrame) -> pd.DataFrame:
        """Compute per-user temporal features: activity trends and cyclical patterns.

        Returns a DataFrame indexed by ``user_id`` with columns:
            trend_7d, trend_14d, trend_30d  (ride counts),
            active_days_30d,
            hour_sin, hour_cos, dow_sin, dow_cos  (cyclical encodings of
                the user's mean activity hour / day-of-week)
        """
        if user_logs.empty:
            return pd.DataFrame()

        df = user_logs.copy()
        self._coerce_log_columns(df)

        now = pd.Timestamp.now()
        df["ride_time"] = pd.to_datetime(df["ride_time"], errors="coerce")
        df = df.dropna(subset=["ride_time"])

        records: list[dict[str, Any]] = []
        for uid, grp in df.groupby("user_id"):
            rec: dict[str, Any] = {"user_id": uid}
            times = grp["ride_time"]

            rec["trend_7d"] = (times >= now - pd.Timedelta(days=7)).sum()
            rec["trend_14d"] = (times >= now - pd.Timedelta(days=14)).sum()
            rec["trend_30d"] = (times >= now - pd.Timedelta(days=30)).sum()

            last30 = grp[times >= now - pd.Timedelta(days=30)]
            rec["active_days_30d"] = last30["ride_time"].dt.date.nunique()

            # Cyclical encodings based on the user's average activity patterns
            mean_hour = grp["ride_time"].dt.hour.mean()
            rec["hour_sin"] = np.sin(2 * np.pi * mean_hour / 24)
            rec["hour_cos"] = np.cos(2 * np.pi * mean_hour / 24)

            mean_dow = grp["ride_time"].dt.dayofweek.mean()
            rec["dow_sin"] = np.sin(2 * np.pi * mean_dow / 7)
            rec["dow_cos"] = np.cos(2 * np.pi * mean_dow / 7)

            records.append(rec)

        return pd.DataFrame.from_records(records).set_index("user_id").fillna(0)

    # ------------------------------------------------------------------
    # Context features
    # ------------------------------------------------------------------

    def extract_context_features(self, user_profile: pd.DataFrame) -> pd.DataFrame:
        """Extract contextual features from user profiles.

        Expected columns in *user_profile*:
            user_id, city_tier (int 1-5), has_freight_search (bool),
            has_large_item_search (bool)

        Returns a DataFrame indexed by ``user_id`` with columns:
            city_tier, has_freight_search, has_large_item_search
        """
        if user_profile.empty:
            return pd.DataFrame()

        df = user_profile.copy()

        # Fill missing columns with defaults
        for col, default in [
            ("city_tier", 3),
            ("has_freight_search", False),
            ("has_large_item_search", False),
        ]:
            if col not in df.columns:
                df[col] = default

        df["has_freight_search"] = df["has_freight_search"].astype(int)
        df["has_large_item_search"] = df["has_large_item_search"].astype(int)

        return (
            df[["user_id", "city_tier", "has_freight_search", "has_large_item_search"]]
            .set_index("user_id")
            .fillna({"city_tier": 3})
        )

    # ------------------------------------------------------------------
    # Composite matrix
    # ------------------------------------------------------------------

    def build_feature_matrix(self, raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine all feature groups into a single matrix.

        Parameters
        ----------
        raw_data : dict
            Must contain keys ``"user_logs"`` and ``"user_profile"``
            (both ``pd.DataFrame``).

        Returns
        -------
        pd.DataFrame
            Indexed by ``user_id``, one column per feature.
        """
        user_logs = raw_data.get("user_logs", pd.DataFrame())
        user_profile = raw_data.get("user_profile", pd.DataFrame())

        parts: list[pd.DataFrame] = []

        if not user_logs.empty:
            behavior = self.extract_behavior_features(user_logs)
            if not behavior.empty:
                parts.append(behavior)

            temporal = self.extract_temporal_features(user_logs)
            if not temporal.empty:
                parts.append(temporal)

        if not user_profile.empty:
            context = self.extract_context_features(user_profile)
            if not context.empty:
                parts.append(context)

        if not parts:
            return pd.DataFrame()

        # Join all on user_id index, using outer join to keep all users
        result = parts[0]
        for part in parts[1:]:
            result = result.join(part, how="outer")

        return result.fillna(0)

    # ------------------------------------------------------------------
    # Sample data generator (useful for testing)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_sample_data(
        n_users: int = 500,
        n_rides: int = 5000,
        seed: int = 42,
    ) -> dict[str, pd.DataFrame]:
        """Generate realistic sample ride-hailing data for testing.

        Returns a dict with keys ``"user_logs"`` and ``"user_profile"``.
        """
        rng = np.random.RandomState(seed)

        user_ids = np.arange(1, n_users + 1)

        # --- User profiles ---
        profiles = pd.DataFrame(
            {
                "user_id": user_ids,
                "city_tier": rng.choice([1, 2, 3, 4, 5], size=n_users, p=[0.05, 0.15, 0.35, 0.30, 0.15]),
                "has_freight_search": rng.random(n_users) < 0.08,
                "has_large_item_search": rng.random(n_users) < 0.10,
            }
        )

        # --- Ride logs ---
        ride_user_ids = rng.choice(user_ids, size=n_rides)
        base_time = pd.Timestamp.now()

        # Generate timestamps spread over last 60 days
        offsets_seconds = rng.randint(0, 60 * 24 * 3600, size=n_rides)
        ride_times = pd.to_datetime(
            [base_time - pd.Timedelta(seconds=int(s)) for s in offsets_seconds]
        )

        destination_types = rng.choice(
            ["residential", "office", "shopping", "airport", "industrial", "school", "hospital"],
            size=n_rides,
            p=[0.30, 0.25, 0.15, 0.08, 0.07, 0.10, 0.05],
        )

        distances = np.clip(rng.exponential(scale=8.0, size=n_rides), 0.5, 80)
        fares = distances * rng.uniform(2.0, 3.5, size=n_rides)
        durations = distances * rng.uniform(2.0, 4.0, size=n_rides)  # minutes

        logs = pd.DataFrame(
            {
                "user_id": ride_user_ids,
                "ride_time": ride_times,
                "distance": distances,
                "destination_type": destination_types,
                "fare": fares,
                "ride_duration": durations,
            }
        ).sort_values("ride_time", ascending=False).reset_index(drop=True)

        return {"user_logs": logs, "user_profile": profiles}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_log_columns(df: pd.DataFrame) -> None:
        """Ensure required log columns exist (fill with NaN if missing)."""
        for col in ["distance", "destination_type"]:
            if col not in df.columns:
                if col == "distance":
                    df[col] = np.nan
                else:
                    df[col] = "unknown"

    @staticmethod
    def _safe_col(name: str) -> str:
        """Sanitise a categorical value for use as a column name."""
        return str(name).lower().replace(" ", "_").replace("-", "_")
