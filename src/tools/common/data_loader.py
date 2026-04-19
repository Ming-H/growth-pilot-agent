"""Data loading utility with sample data generation.

Loads CSV files and generates realistic sample datasets for
demonstration and testing purposes. All random data uses seed 42
for reproducibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


class DataLoader:
    """Load data from files or generate realistic sample data."""

    # Reproducible seed for all sample data generation
    SEED = 42

    def load_csv(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load a CSV file into a DataFrame.

        Args:
            path: File path to the CSV.
            **kwargs: Additional keyword arguments passed to pd.read_csv.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            pd.errors.EmptyDataError: If the file is empty.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        return pd.read_csv(file_path, **kwargs)

    def load_sample_data(self, dataset_name: str) -> pd.DataFrame:
        """Generate realistic sample data for a given dataset type.

        Args:
            dataset_name: One of:
                - 'user_behavior'
                - 'funnel'
                - 'subsidy_experiment'
                - 'retention'
                - 'ad_campaign'
                - 'touchpoint_journey'
                - 'seasonal_history'

        Returns:
            DataFrame with columns appropriate to the dataset type.

        Raises:
            ValueError: If dataset_name is not recognized.
        """
        generators: dict[str, Callable[[], pd.DataFrame]] = {
            "user_behavior": self._gen_user_behavior,
            "funnel": self._gen_funnel,
            "subsidy_experiment": self._gen_subsidy_experiment,
            "retention": self._gen_retention,
            "ad_campaign": self._gen_ad_campaign,
            "touchpoint_journey": self._gen_touchpoint_journey,
            "seasonal_history": self._gen_seasonal_history,
        }

        if dataset_name not in generators:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {sorted(generators.keys())}"
            )

        return generators[dataset_name]()

    def validate_data(self, df: pd.DataFrame, required_columns: list[str]) -> bool:
        """Validate that a DataFrame contains all required columns.

        Args:
            df: DataFrame to validate.
            required_columns: Column names that must be present.

        Returns:
            True if all required columns exist, False otherwise.
        """
        if df is None or df.empty:
            return False
        missing = set(required_columns) - set(df.columns)
        return len(missing) == 0

    # ------------------------------------------------------------------
    # Sample data generators (all use SEED=42)
    # ------------------------------------------------------------------

    def _gen_user_behavior(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.SEED)
        n = 10000
        cities = ["上海", "北京", "深圳", "广州", "杭州", "成都", "武汉", "南京", "重庆", "苏州"]
        city_tiers = {"上海": 1, "北京": 1, "深圳": 1, "广州": 1, "杭州": 2, "成都": 2,
                      "武汉": 2, "南京": 2, "重庆": 3, "苏州": 2}
        city_choices = rng.choice(cities, size=n)
        genders = rng.choice(["M", "F"], size=n, p=[0.55, 0.45])

        ages = rng.integers(18, 60, size=n)
        orders = rng.geometric(0.15, size=n) - 1  # 0-based, most users have few orders
        orders = np.clip(orders, 0, 200)
        aov = np.clip(rng.lognormal(3.5, 0.8, size=n), 20, 2000).round(2)
        recency = rng.exponential(30, size=n).astype(int)
        recency = np.clip(recency, 0, 365)
        ltv = (orders * aov * rng.uniform(0.3, 0.9, size=n)).round(2)

        df = pd.DataFrame({
            "user_id": [f"U{i:06d}" for i in range(n)],
            "age": ages,
            "gender": genders,
            "city": city_choices,
            "city_tier": [city_tiers[c] for c in city_choices],
            "historical_orders": orders,
            "avg_order_value": aov,
            "days_since_last_order": recency,
            "ltv": ltv,
        })
        return df

    def _gen_funnel(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.SEED)
        stages = ["首页访问", "搜索", "查看报价", "下单", "支付成功"]
        base_counts = [100000, 65000, 38000, 18000, 14000]
        # Add daily variation
        dates = pd.date_range("2024-01-01", periods=90, freq="D")

        rows: list[dict[str, Any]] = []
        for date in dates:
            daily_factor = 1.0 + 0.15 * rng.standard_normal()
            for stage, base in zip(stages, base_counts):
                count = int(base * daily_factor * rng.uniform(0.9, 1.1))
                count = max(count, 0)
                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "stage": stage,
                    "count": count,
                })

        return pd.DataFrame(rows)

    def _gen_subsidy_experiment(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.SEED)
        n = 8000
        cities = ["上海", "北京", "深圳", "广州", "杭州", "成都"]

        groups = rng.choice(["control", "treatment_5", "treatment_10", "treatment_20"],
                            size=n, p=[0.4, 0.2, 0.2, 0.2])

        subsidy_map = {"control": 0, "treatment_5": 5, "treatment_10": 10, "treatment_20": 20}
        subsidies = np.array([subsidy_map[g] for g in groups])

        # Conversion probability increases with subsidy
        base_conv_prob = 0.12
        conv_lift = subsidies / 100 * 0.3  # Higher subsidy -> more conversions
        conv_probs = np.clip(base_conv_prob + conv_lift + rng.uniform(-0.02, 0.02, n), 0.01, 0.5)
        converted = rng.binomial(1, conv_probs).astype(bool)

        # Revenue (only for converted users)
        revenue = np.where(converted, np.clip(rng.lognormal(4.0, 0.6, n), 30, 500).round(2), 0)

        df = pd.DataFrame({
            "user_id": [f"U{i:06d}" for i in range(n)],
            "city": rng.choice(cities, size=n),
            "group": groups,
            "subsidy_amount": subsidies.astype(float),
            "converted": converted,
            "revenue": revenue,
            "age": rng.integers(18, 55, size=n),
        })
        return df

    def _gen_retention(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.SEED)
        # Cohort-based retention
        cohorts = pd.date_range("2024-01-01", periods=12, freq="MS")  # Monthly cohorts
        periods = list(range(0, 13))  # Month 0 to 12

        rows: list[dict[str, Any]] = []
        for cohort in cohorts:
            base_size = rng.integers(3000, 8000)
            for period in periods:
                # Retention follows a power-law decay
                rate = 0.4 * (period + 1) ** (-0.45)
                retained = int(base_size * rate * rng.uniform(0.8, 1.2))
                retained = min(retained, base_size)
                retained = max(retained, 0)
                rows.append({
                    "cohort": cohort.strftime("%Y-%m"),
                    "period": period,
                    "cohort_size": base_size,
                    "retained": retained,
                    "retention_rate": round(retained / base_size, 4) if base_size > 0 else 0,
                })

        return pd.DataFrame(rows)

    def _gen_ad_campaign(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.SEED)
        n = 2000  # 2000 daily records
        campaigns = ["品牌曝光-春季", "拉新-搜索", "拉新-信息流", "老客召回-推送", "节日大促-双十一"]
        channels = ["微信", "抖音", "百度", "快手", "美团"]

        dates = pd.date_range("2024-01-01", periods=90, freq="D")

        rows: list[dict[str, Any]] = []
        for date in dates:
            for campaign in rng.choice(campaigns, size=rng.integers(3, 6), replace=False):
                channel = rng.choice(channels)
                impressions = rng.integers(5000, 200000)
                ctr = rng.beta(2, 100)  # typical CTR distribution
                clicks = int(impressions * ctr)
                cvr = rng.beta(3, 60)  # typical CVR distribution
                conversions = int(clicks * cvr)
                spend = float(clicks * rng.uniform(0.5, 3.0))
                revenue = float(conversions * rng.lognormal(3.8, 0.5))

                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "campaign": campaign,
                    "channel": channel,
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "spend": round(spend, 2),
                    "revenue": round(revenue, 2),
                    "ctr": round(ctr, 4),
                    "cvr": round(cvr, 4),
                })

        return pd.DataFrame(rows)

    def _gen_touchpoint_journey(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.SEED)
        n_users = 5000
        channels = ["自然搜索", "信息流广告", "微信推送", "短信", "APP开屏", "好友分享", "品牌词搜索"]
        events = ["曝光", "点击", "访问", "注册", "下单"]

        rows: list[dict[str, Any]] = []
        for i in range(n_users):
            n_touchpoints = rng.integers(1, 8)
            for j in range(n_touchpoints):
                channel = rng.choice(channels)
                event = rng.choice(events, p=[0.3, 0.25, 0.2, 0.1, 0.15])
                days_offset = rng.integers(0, 30)
                rows.append({
                    "user_id": f"U{i:06d}",
                    "touchpoint_order": j + 1,
                    "channel": channel,
                    "event": event,
                    "timestamp_day_offset": days_offset,
                })

        return pd.DataFrame(rows)

    def _gen_seasonal_history(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.SEED)
        # 2 years of daily data
        dates = pd.date_range("2022-01-01", periods=730, freq="D")

        rows: list[dict[str, Any]] = []
        for date in dates:
            day_of_year = date.dayofyear
            # Seasonal pattern: peaks around Chinese New Year, May, and Nov
            seasonal = (
                30 * np.cos(2 * np.pi * day_of_year / 365)
                + 15 * np.cos(4 * np.pi * day_of_year / 365)
            )
            weekday = date.weekday()
            weekend_boost = 20 if weekday >= 5 else 0
            noise = rng.standard_normal() * 10
            orders = int(max(50, 200 + seasonal + weekend_boost + noise))

            revenue = orders * rng.uniform(60, 120)
            new_users = int(orders * rng.uniform(0.2, 0.5))
            avg_price = round(revenue / orders, 2) if orders > 0 else 0

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "day_of_week": weekday,
                "is_weekend": weekday >= 5,
                "orders": orders,
                "revenue": round(revenue, 2),
                "new_users": new_users,
                "avg_order_price": avg_price,
            })

        return pd.DataFrame(rows)
