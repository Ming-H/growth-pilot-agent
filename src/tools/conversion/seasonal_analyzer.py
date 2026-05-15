"""SeasonalAnalyzer - detect seasonal patterns and plan campaigns.

Detects periodicity in daily metrics, performs year-over-year comparison,
forecasts demand, and provides a campaign calendar for Didi freight.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal, stats

from src.tools.registry import register

logger = logging.getLogger(__name__)


@register("seasonal_analyzer", category="conversion")
class SeasonalAnalyzer:
    """Detect seasonal patterns and plan marketing campaigns."""

    def detect_seasonality(
        self,
        daily_metrics: pd.DataFrame,
        period: str | int = "auto",
    ) -> dict[str, Any]:
        """Detect seasonal periodicity in daily metrics.

        Args:
            daily_metrics: DataFrame with columns [date, value].
                `date` should be parseable as a date; `value` is the metric
                to analyse (e.g. order count, revenue).
            period: "auto" for auto-detection, or an integer for a specific
                period length in days (e.g. 7 for weekly, 365 for yearly).

        Returns:
            Dict with detected periods, strength, and decomposition info.
        """
        if daily_metrics.empty or "value" not in daily_metrics.columns:
            return {"error": "daily_metrics must have 'date' and 'value' columns"}

        values = daily_metrics["value"].values.astype(float)
        n = len(values)

        if n < 14:
            return {"error": "Need at least 14 days of data for seasonality detection"}

        # Remove trend using simple moving average
        window = min(7, n // 2)
        trend = pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().values
        detrended = values - trend

        # Auto-detect period using autocorrelation
        if period == "auto":
            detected_periods = self._auto_detect_periods(detrended, n)
        else:
            period_int = int(period)
            detected_periods = [
                {"period_days": period_int, "strength": self._period_strength(detrended, period_int)}
            ]

        # Compute overall seasonality strength (1 - variance_ratio)
        seasonal_strength = self._overall_seasonality_strength(detrended)

        # Find peak and trough days of week
        dow_pattern = self._day_of_week_pattern(daily_metrics)

        # Monthly pattern
        monthly_pattern = self._monthly_pattern(daily_metrics)

        return {
            "detected_periods": detected_periods,
            "seasonality_strength": round(seasonal_strength, 4),
            "is_seasonal": seasonal_strength > 0.3,
            "day_of_week_pattern": dow_pattern,
            "monthly_pattern": monthly_pattern,
            "data_points": n,
            "date_range": {
                "start": str(daily_metrics["date"].iloc[0]),
                "end": str(daily_metrics["date"].iloc[-1]),
            },
        }

    def year_over_year(
        self,
        current_data: pd.DataFrame,
        historical_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Compare current period metrics to the same period last year.

        Args:
            current_data: DataFrame with [date, value] for the current period.
            historical_data: DataFrame with [date, value] for the historical
                period (typically same calendar dates one year prior).

        Returns:
            YoY comparison with growth rates and significance.
        """
        for name, df in [("current_data", current_data), ("historical_data", historical_data)]:
            if df.empty or "value" not in df.columns:
                return {"error": f"{name} must have 'date' and 'value' columns"}

        current_vals = current_data["value"].values.astype(float)
        hist_vals = historical_data["value"].values.astype(float)

        current_mean = float(np.mean(current_vals))
        hist_mean = float(np.mean(hist_vals))
        current_sum = float(np.sum(current_vals))
        hist_sum = float(np.sum(hist_vals))

        # Growth rates
        mean_growth = (current_mean - hist_mean) / hist_mean if hist_mean != 0 else 0.0
        total_growth = (current_sum - hist_sum) / hist_sum if hist_sum != 0 else 0.0

        # Statistical significance (t-test)
        t_stat, p_value = stats.ttest_ind(current_vals, hist_vals, equal_var=False)

        # Daily comparison
        min_len = min(len(current_vals), len(hist_vals))
        daily_diff = current_vals[:min_len] - hist_vals[:min_len]

        # Find best and worst days
        if min_len > 0:
            best_day_idx = int(np.argmax(daily_diff))
            worst_day_idx = int(np.argmin(daily_diff))
        else:
            best_day_idx = worst_day_idx = 0

        return {
            "current_period": {
                "mean": round(current_mean, 2),
                "total": round(current_sum, 2),
                "std": round(float(np.std(current_vals)), 2),
                "days": len(current_vals),
            },
            "historical_period": {
                "mean": round(hist_mean, 2),
                "total": round(hist_sum, 2),
                "std": round(float(np.std(hist_vals)), 2),
                "days": len(hist_vals),
            },
            "mean_growth_rate": round(mean_growth, 4),
            "total_growth_rate": round(total_growth, 4),
            "statistical_significance": {
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_value), 6),
                "is_significant": p_value < 0.05,
            },
            "best_day": {
                "index": best_day_idx,
                "difference": round(float(daily_diff[best_day_idx]), 2) if min_len > 0 else 0.0,
            },
            "worst_day": {
                "index": worst_day_idx,
                "difference": round(float(daily_diff[worst_day_idx]), 2) if min_len > 0 else 0.0,
            },
        }

    def forecast_demand(
        self,
        historical_data: pd.DataFrame,
        horizon_days: int = 30,
    ) -> dict[str, Any]:
        """Simple demand forecast using seasonal decomposition + trend extrapolation.

        Args:
            historical_data: DataFrame with [date, value].
            horizon_days: number of days to forecast ahead.

        Returns:
            Forecast dict with predicted values and confidence intervals.
        """
        if historical_data.empty or "value" not in historical_data.columns:
            return {"error": "historical_data must have 'date' and 'value' columns"}

        values = historical_data["value"].values.astype(float)
        n = len(values)

        if n < 14:
            return {"error": "Need at least 14 days of data for forecasting"}

        # Step 1: Compute trend via linear regression
        x = np.arange(n, dtype=float)
        slope, intercept, r_value, _, _ = stats.linregress(x, values)
        trend = slope * x + intercept

        # Step 2: Compute seasonal component (7-day pattern)
        detrended = values - trend
        seasonal_pattern = self._compute_weekly_seasonal(detrended, n)

        # Step 3: Compute residual stats for confidence interval
        seasonal_full = np.array([seasonal_pattern[i % 7] for i in range(n)])
        residuals = detrended - seasonal_full
        residual_std = float(np.std(residuals))

        # Step 4: Forecast
        forecast_dates: list[str] = []
        forecast_values: list[float] = []
        forecast_lower: list[float] = []
        forecast_upper: list[float] = []

        last_date = pd.to_datetime(historical_data["date"].iloc[-1])
        for d in range(1, horizon_days + 1):
            future_x = n + d - 1
            trend_val = slope * future_x + intercept
            seasonal_val = seasonal_pattern[(n + d - 1) % 7]
            forecast = max(0.0, trend_val + seasonal_val)

            # Widen CI with horizon
            ci = 1.96 * residual_std * np.sqrt(1 + d / n)

            future_date = last_date + timedelta(days=d)
            forecast_dates.append(str(future_date.date()))
            forecast_values.append(round(forecast, 2))
            forecast_lower.append(round(max(0.0, forecast - ci), 2))
            forecast_upper.append(round(forecast + ci, 2))

        return {
            "horizon_days": horizon_days,
            "trend": {
                "slope_per_day": round(slope, 4),
                "intercept": round(intercept, 2),
                "r_squared": round(r_value**2, 4),
                "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat",
            },
            "weekly_seasonal_pattern": {
                "Mon": round(seasonal_pattern[0], 2),
                "Tue": round(seasonal_pattern[1], 2),
                "Wed": round(seasonal_pattern[2], 2),
                "Thu": round(seasonal_pattern[3], 2),
                "Fri": round(seasonal_pattern[4], 2),
                "Sat": round(seasonal_pattern[5], 2),
                "Sun": round(seasonal_pattern[6], 2),
            },
            "forecast": [
                {
                    "date": forecast_dates[i],
                    "predicted": forecast_values[i],
                    "lower_95": forecast_lower[i],
                    "upper_95": forecast_upper[i],
                }
                for i in range(horizon_days)
            ],
            "total_forecast_demand": round(sum(forecast_values), 2),
            "avg_daily_forecast": round(np.mean(forecast_values), 2),
        }

    def campaign_calendar(
        self, year: int | None = None
    ) -> list[dict[str, Any]]:
        """Return key campaign dates for Didi freight.

        Args:
            year: target year. Defaults to current year.

        Returns:
            List of campaign events with date ranges and context.
        """
        import datetime as _dt

        year = year or _dt.date.today().year

        campaigns: list[dict[str, Any]] = [
            {
                "name": "春节搬家季",
                "start": f"{year}-01-15",
                "end": f"{year}-02-10",
                "type": "seasonal",
                "description": "春节前搬家/货运需求高峰",
                "expected_demand_lift": 1.5,
                "recommended_channels": ["Banner", "Push", "SMS"],
            },
            {
                "name": "315品质节",
                "start": f"{year}-03-10",
                "end": f"{year}-03-15",
                "type": "brand",
                "description": "品质服务宣传，建立品牌信任",
                "expected_demand_lift": 1.1,
                "recommended_channels": ["Banner", "金刚位"],
            },
            {
                "name": "五一出行季",
                "start": f"{year}-04-25",
                "end": f"{year}-05-05",
                "type": "seasonal",
                "description": "劳动节出行/搬家需求增加",
                "expected_demand_lift": 1.3,
                "recommended_channels": ["Push", "Banner", "金刚位"],
            },
            {
                "name": "毕业季",
                "start": f"{year}-06-01",
                "end": f"{year}-07-15",
                "type": "seasonal",
                "description": "大学生毕业搬家需求高峰",
                "expected_demand_lift": 1.8,
                "recommended_channels": ["Push", "Banner", "SMS", "金刚位"],
                "target_segments": ["student", "new_user"],
            },
            {
                "name": "618大促",
                "start": f"{year}-06-10",
                "end": f"{year}-06-18",
                "type": "promotion",
                "description": "年中大促，配合电商物流需求",
                "expected_demand_lift": 1.4,
                "recommended_channels": ["Banner", "Push", "金刚位"],
            },
            {
                "name": "暑假搬家季",
                "start": f"{year}-07-01",
                "end": f"{year}-08-31",
                "type": "seasonal",
                "description": "暑期租房搬家高峰",
                "expected_demand_lift": 1.5,
                "recommended_channels": ["Banner", "Push", "SMS"],
            },
            {
                "name": "开学季",
                "start": f"{year}-08-20",
                "end": f"{year}-09-10",
                "type": "seasonal",
                "description": "新生入学搬家/行李寄送需求",
                "expected_demand_lift": 1.6,
                "recommended_channels": ["Push", "Banner", "SMS", "金刚位"],
                "target_segments": ["student", "new_user"],
            },
            {
                "name": "国庆黄金周",
                "start": f"{year}-09-28",
                "end": f"{year}-10-07",
                "type": "seasonal",
                "description": "国庆出行/搬家需求",
                "expected_demand_lift": 1.3,
                "recommended_channels": ["Banner", "Push"],
            },
            {
                "name": "双11",
                "start": f"{year}-11-01",
                "end": f"{year}-11-11",
                "type": "promotion",
                "description": "双11大促，电商物流+搬家双重需求",
                "expected_demand_lift": 2.0,
                "recommended_channels": ["金刚位", "Banner", "Push", "SMS"],
            },
            {
                "name": "双12",
                "start": f"{year}-12-05",
                "end": f"{year}-12-12",
                "type": "promotion",
                "description": "双12返场促销",
                "expected_demand_lift": 1.4,
                "recommended_channels": ["Banner", "Push"],
            },
            {
                "name": "年终搬家季",
                "start": f"{year}-12-20",
                "end": f"{year}-12-31",
                "type": "seasonal",
                "description": "年底搬家/办公室搬迁高峰",
                "expected_demand_lift": 1.5,
                "recommended_channels": ["Banner", "Push", "SMS"],
            },
        ]

        return campaigns

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_detect_periods(
        self, detrended: np.ndarray, n: int
    ) -> list[dict[str, Any]]:
        """Auto-detect dominant periods using autocorrelation."""
        # Compute autocorrelation for lags 1 to n//2
        max_lag = min(n // 2, 365)
        if max_lag < 2:
            return []

        acf_values = np.array([
            np.corrcoef(detrended[:-lag], detrended[lag:])[0, 1]
            if lag < n else 0.0
            for lag in range(1, max_lag + 1)
        ])

        # Find peaks in autocorrelation
        peaks, properties = signal.find_peaks(acf_values, height=0.1, distance=3)

        periods: list[dict[str, Any]] = []
        for peak in peaks:
            period_days = int(peak + 1)  # lag 0 -> period 1
            strength = float(acf_values[peak])
            label = self._period_label(period_days)
            periods.append({
                "period_days": period_days,
                "strength": round(strength, 4),
                "label": label,
            })

        # Sort by strength
        periods.sort(key=lambda x: x["strength"], reverse=True)
        return periods[:5]  # top 5 periods

    @staticmethod
    def _period_label(days: int) -> str:
        """Return a human-readable label for a period."""
        if days == 7:
            return "weekly"
        elif days == 14:
            return "bi-weekly"
        elif 28 <= days <= 31:
            return "monthly"
        elif 90 <= days <= 92:
            return "quarterly"
        elif 364 <= days <= 366:
            return "yearly"
        return f"{days}-day cycle"

    @staticmethod
    def _period_strength(detrended: np.ndarray, period: int) -> float:
        """Compute strength of a specific period via autocorrelation."""
        n = len(detrended)
        if period >= n:
            return 0.0
        corr = np.corrcoef(detrended[:-period], detrended[period:])[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0

    @staticmethod
    def _overall_seasonality_strength(detrended: np.ndarray) -> float:
        """Compute overall seasonality strength (fraction of variance explained by seasonal)."""
        n = len(detrended)
        if n < 14:
            return 0.0

        # Use 7-day pattern as proxy
        n_weeks = n // 7
        if n_weeks < 2:
            return 0.0

        reshaped = detrended[: n_weeks * 7].reshape(n_weeks, 7)
        seasonal = reshaped.mean(axis=0)
        seasonal_var = float(np.var(seasonal))
        total_var = float(np.var(detrended))

        if total_var <= 0:
            return 0.0
        return min(1.0, seasonal_var / total_var)

    def _day_of_week_pattern(
        self, daily_metrics: pd.DataFrame
    ) -> dict[str, float]:
        """Compute average value per day of week."""
        df = daily_metrics.copy()
        df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        pattern: dict[str, float] = {}
        means = df.groupby("dow")["value"].mean()
        for i, name in enumerate(dow_names):
            pattern[name] = round(float(means.get(i, 0.0)), 2)
        return pattern

    def _monthly_pattern(
        self, daily_metrics: pd.DataFrame
    ) -> dict[str, float]:
        """Compute average value per month."""
        df = daily_metrics.copy()
        df["month"] = pd.to_datetime(df["date"]).dt.month
        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        pattern: dict[str, float] = {}
        means = df.groupby("month")["value"].mean()
        for i, name in enumerate(month_names, 1):
            if i in means.index:
                pattern[name] = round(float(means[i]), 2)
        return pattern

    @staticmethod
    def _compute_weekly_seasonal(detrended: np.ndarray, n: int) -> list[float]:
        """Compute 7-day seasonal indices."""
        n_weeks = n // 7
        if n_weeks < 2:
            return [0.0] * 7

        reshaped = detrended[: n_weeks * 7].reshape(n_weeks, 7)
        seasonal = reshaped.mean(axis=0).tolist()
        return [float(s) for s in seasonal]
