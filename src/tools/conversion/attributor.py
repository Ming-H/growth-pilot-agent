"""Attributor - multi-touch attribution models.

Implements five attribution models: first-touch, last-touch, linear,
time-decay, and U-shaped. Each model distributes conversion credit across
touchpoints in a user journey.

Input DataFrame columns: user_id, channel, action, timestamp, converted, conversion_value
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Attributor:
    """Multi-touch attribution analysis across five models."""

    def first_touch_attribution(self, journeys: pd.DataFrame) -> dict[str, Any]:
        """Assign 100% credit to the first touchpoint in each converted journey.

        Args:
            journeys: DataFrame with columns [user_id, channel, action, timestamp,
                converted, conversion_value].

        Returns:
            Dict with per-channel attributed value and conversions.
        """
        self._validate_journeys(journeys)
        if journeys.empty:
            return self._empty_result("first_touch")

        converted = journeys[journeys["converted"] == 1].copy()
        if converted.empty:
            return self._empty_result("first_touch")

        converted["timestamp"] = pd.to_datetime(converted["timestamp"])
        # For each user, find the first touchpoint
        first_touches = converted.sort_values("timestamp").groupby("user_id").first()

        attribution = (
            first_touches.groupby("channel")
            .agg(
                attributed_conversions=("conversion_value", "count"),
                attributed_value=("conversion_value", "sum"),
            )
            .reset_index()
        )

        return self._format_result("first_touch", attribution, len(first_touches))

    def last_touch_attribution(self, journeys: pd.DataFrame) -> dict[str, Any]:
        """Assign 100% credit to the last touchpoint before conversion.

        Args:
            journeys: DataFrame with columns [user_id, channel, action, timestamp,
                converted, conversion_value].

        Returns:
            Dict with per-channel attributed value and conversions.
        """
        self._validate_journeys(journeys)
        if journeys.empty:
            return self._empty_result("last_touch")

        converted = journeys[journeys["converted"] == 1].copy()
        if converted.empty:
            return self._empty_result("last_touch")

        converted["timestamp"] = pd.to_datetime(converted["timestamp"])
        last_touches = converted.sort_values("timestamp").groupby("user_id").last()

        attribution = (
            last_touches.groupby("channel")
            .agg(
                attributed_conversions=("conversion_value", "count"),
                attributed_value=("conversion_value", "sum"),
            )
            .reset_index()
        )

        return self._format_result("last_touch", attribution, len(last_touches))

    def linear_attribution(self, journeys: pd.DataFrame) -> dict[str, Any]:
        """Distribute credit equally across all touchpoints in each journey.

        Args:
            journeys: DataFrame with columns [user_id, channel, action, timestamp,
                converted, conversion_value].

        Returns:
            Dict with per-channel attributed value and conversions.
        """
        self._validate_journeys(journeys)
        if journeys.empty:
            return self._empty_result("linear")

        converted = journeys[journeys["converted"] == 1].copy()
        if converted.empty:
            return self._empty_result("linear")

        converted["timestamp"] = pd.to_datetime(converted["timestamp"])

        # For each converted user, distribute value equally
        records: list[dict[str, Any]] = []
        for user_id, group in converted.groupby("user_id"):
            group_sorted = group.sort_values("timestamp")
            n_touches = len(group_sorted)
            if n_touches == 0:
                continue
            conv_value = group_sorted["conversion_value"].iloc[-1]
            credit_per_touch = conv_value / n_touches

            for _, row in group_sorted.iterrows():
                records.append(
                    {
                        "channel": row["channel"],
                        "credit": credit_per_touch,
                    }
                )

        if not records:
            return self._empty_result("linear")

        credit_df = pd.DataFrame(records)
        attribution = (
            credit_df.groupby("channel")
            .agg(
                attributed_conversions=("credit", "count"),
                attributed_value=("credit", "sum"),
            )
            .reset_index()
        )

        return self._format_result("linear", attribution, converted["user_id"].nunique())

    def time_decay_attribution(
        self, journeys: pd.DataFrame, half_life: int = 7
    ) -> dict[str, Any]:
        """Distribute credit with exponential time decay (more recent = more credit).

        Args:
            journeys: DataFrame with standard columns.
            half_life: decay half-life in days. Default 7.

        Returns:
            Dict with per-channel attributed value and conversions.
        """
        self._validate_journeys(journeys)
        if journeys.empty:
            return self._empty_result("time_decay")

        converted = journeys[journeys["converted"] == 1].copy()
        if converted.empty:
            return self._empty_result("time_decay")

        converted["timestamp"] = pd.to_datetime(converted["timestamp"])
        decay_lambda = np.log(2) / half_life

        records: list[dict[str, Any]] = []
        for user_id, group in converted.groupby("user_id"):
            group_sorted = group.sort_values("timestamp")
            n_touches = len(group_sorted)
            if n_touches == 0:
                continue

            conv_value = group_sorted["conversion_value"].iloc[-1]
            last_time = group_sorted["timestamp"].iloc[-1]

            # Compute decay weight for each touchpoint
            weights: list[float] = []
            for _, row in group_sorted.iterrows():
                days_before = (last_time - row["timestamp"]).total_seconds() / 86400.0
                weight = np.exp(-decay_lambda * days_before)
                weights.append(weight)

            total_weight = sum(weights)
            if total_weight <= 0:
                continue

            for i, (_, row) in enumerate(group_sorted.iterrows()):
                credit = conv_value * weights[i] / total_weight
                records.append({"channel": row["channel"], "credit": credit})

        if not records:
            return self._empty_result("time_decay")

        credit_df = pd.DataFrame(records)
        attribution = (
            credit_df.groupby("channel")
            .agg(
                attributed_conversions=("credit", "count"),
                attributed_value=("credit", "sum"),
            )
            .reset_index()
        )

        return self._format_result(
            "time_decay", attribution, converted["user_id"].nunique()
        )

    def u_shaped_attribution(self, journeys: pd.DataFrame) -> dict[str, Any]:
        """U-shaped (bathtub) attribution: 40% first, 40% last, 20% middle.

        Args:
            journeys: DataFrame with standard columns.

        Returns:
            Dict with per-channel attributed value and conversions.
        """
        self._validate_journeys(journeys)
        if journeys.empty:
            return self._empty_result("u_shaped")

        converted = journeys[journeys["converted"] == 1].copy()
        if converted.empty:
            return self._empty_result("u_shaped")

        converted["timestamp"] = pd.to_datetime(converted["timestamp"])

        records: list[dict[str, Any]] = []
        for user_id, group in converted.groupby("user_id"):
            group_sorted = group.sort_values("timestamp")
            n_touches = len(group_sorted)
            if n_touches == 0:
                continue

            conv_value = group_sorted["conversion_value"].iloc[-1]

            if n_touches == 1:
                # Single touch: 100% credit
                records.append(
                    {"channel": group_sorted.iloc[0]["channel"], "credit": conv_value}
                )
            elif n_touches == 2:
                # Two touches: 50/50
                for i in range(2):
                    records.append(
                        {
                            "channel": group_sorted.iloc[i]["channel"],
                            "credit": conv_value * 0.5,
                        }
                    )
            else:
                # 40% first, 40% last, 20% divided among middle
                middle_count = n_touches - 2
                records.append(
                    {
                        "channel": group_sorted.iloc[0]["channel"],
                        "credit": conv_value * 0.4,
                    }
                )
                records.append(
                    {
                        "channel": group_sorted.iloc[-1]["channel"],
                        "credit": conv_value * 0.4,
                    }
                )
                for i in range(1, n_touches - 1):
                    records.append(
                        {
                            "channel": group_sorted.iloc[i]["channel"],
                            "credit": conv_value * 0.2 / middle_count,
                        }
                    )

        if not records:
            return self._empty_result("u_shaped")

        credit_df = pd.DataFrame(records)
        attribution = (
            credit_df.groupby("channel")
            .agg(
                attributed_conversions=("credit", "count"),
                attributed_value=("credit", "sum"),
            )
            .reset_index()
        )

        return self._format_result(
            "u_shaped", attribution, converted["user_id"].nunique()
        )

    def compare_models(self, journeys: pd.DataFrame) -> dict[str, Any]:
        """Run all five models and produce a comparison table.

        Args:
            journeys: DataFrame with standard columns.

        Returns:
            Comparison dict with all models and a summary.
        """
        self._validate_journeys(journeys)

        models = {
            "first_touch": self.first_touch_attribution,
            "last_touch": self.last_touch_attribution,
            "linear": self.linear_attribution,
            "time_decay": lambda j: self.time_decay_attribution(j),
            "u_shaped": self.u_shaped_attribution,
        }

        results: dict[str, Any] = {}
        for model_name, model_fn in models.items():
            try:
                results[model_name] = model_fn(journeys)
            except Exception as e:
                results[model_name] = {"error": str(e)}
                logger.warning("Model %s failed: %s", model_name, e)

        # Build comparison table
        comparison_rows: list[dict[str, Any]] = []
        all_channels: set[str] = set()

        for model_name, result in results.items():
            if "error" in result:
                continue
            for ch_data in result.get("channel_attribution", []):
                all_channels.add(ch_data["channel"])
                comparison_rows.append(
                    {
                        "model": model_name,
                        "channel": ch_data["channel"],
                        "attributed_conversions": ch_data["attributed_conversions"],
                        "attributed_value": round(ch_data["attributed_value"], 2),
                    }
                )

        if not comparison_rows:
            return {"models": results, "comparison": [], "summary": "No data to compare"}

        comp_df = pd.DataFrame(comparison_rows)

        # Pivot for easy comparison
        summary_lines: list[str] = []
        for channel in sorted(all_channels):
            ch_data = comp_df[comp_df["channel"] == channel]
            if ch_data.empty:
                continue
            best_model = ch_data.loc[ch_data["attributed_value"].idxmax(), "model"]
            worst_model = ch_data.loc[ch_data["attributed_value"].idxmin(), "model"]
            max_val = ch_data["attributed_value"].max()
            min_val = ch_data["attributed_value"].min()
            summary_lines.append(
                f"{channel}: 最高归因={best_model}({max_val:.1f}), "
                f"最低归因={worst_model}({min_val:.1f})"
            )

        return {
            "models": results,
            "comparison": comp_df.to_dict(orient="records"),
            "total_converted_users": results.get("first_touch", {}).get(
                "total_converted_users", 0
            ),
            "summary": "\n".join(summary_lines) if summary_lines else "No comparison data",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_journeys(journeys: pd.DataFrame) -> None:
        """Validate that the journeys DataFrame has required columns."""
        if not isinstance(journeys, pd.DataFrame):
            raise TypeError("journeys must be a pandas DataFrame")

        required = {"user_id", "channel", "timestamp", "converted", "conversion_value"}
        missing = required - set(journeys.columns)
        if missing:
            raise ValueError(f"journeys DataFrame missing columns: {missing}")

    @staticmethod
    def _empty_result(model_name: str) -> dict[str, Any]:
        """Return an empty attribution result."""
        return {
            "model": model_name,
            "channel_attribution": [],
            "total_converted_users": 0,
            "total_attributed_value": 0.0,
            "message": "No converted journeys to attribute",
        }

    @staticmethod
    def _format_result(
        model_name: str, attribution: pd.DataFrame, total_users: int
    ) -> dict[str, Any]:
        """Format attribution DataFrame into a result dict."""
        channel_list = attribution.rename(
            columns={"attributed_conversions": "attributed_conversions"}
        ).to_dict(orient="records")

        total_value = float(attribution["attributed_value"].sum())

        return {
            "model": model_name,
            "channel_attribution": [
                {
                    "channel": row["channel"],
                    "attributed_conversions": int(row["attributed_conversions"]),
                    "attributed_value": round(float(row["attributed_value"]), 4),
                }
                for row in channel_list
            ],
            "total_converted_users": total_users,
            "total_attributed_value": round(total_value, 4),
        }
