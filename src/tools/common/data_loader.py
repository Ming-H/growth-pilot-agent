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

from src.tools.common.secure_loader import SecureDataLoader


class DataLoader:
    """Load data from files or generate realistic sample data."""

    # Reproducible seed for all sample data generation
    SEED = 42

    def __init__(self, base_dir: str | None = None) -> None:
        """Initialize data loader with optional base directory for security.

        Args:
            base_dir: Optional base directory to restrict file access.
        """
        self._secure_loader = SecureDataLoader(base_dir=base_dir)

    def load_csv(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load a CSV file into a DataFrame with security validation.

        Args:
            path: File path to the CSV.
            **kwargs: Additional keyword arguments passed to pd.read_csv.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the path escapes the base directory (if set).
            pd.errors.EmptyDataError: If the file is empty.
        """
        # Validate path with SecureDataLoader
        validated_path = self._secure_loader.validate_path(str(path))

        if not validated_path.exists():
            raise FileNotFoundError(f"CSV file not found: {validated_path}")

        return pd.read_csv(validated_path, **kwargs)

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

    def load_or_generate(
        self,
        data_path: str,
        domain: str,
        **kwargs: Any,
    ) -> list[dict] | pd.DataFrame:
        """Load data from file, or generate sample data if unavailable.

        Args:
            data_path: File path to load data from. If empty or file doesn't exist,
                sample data will be generated.
            domain: Domain type for sample data generation. One of:
                - 'prospect': User behavior data for prospecting
                - 'funnel': Conversion funnel data
                - 'subsidy_experiment': Subsidy A/B experiment data
                - 'retention': Retention/cohort analysis data
                - 'ad_campaign': Ad campaign performance data
                - 'rta_history': RTA bidding history for ad optimization
                - 'creative': Creative performance data
                - 'audience': Audience segment data
            **kwargs: Additional arguments passed to the generate_sample method.

        Returns:
            DataFrame if available, otherwise list of dicts for compatibility
            with existing tool interfaces.

        Raises:
            ValueError: If domain is not recognized.
        """
        if data_path and Path(data_path).exists():
            return self.load_csv(data_path)
        return self.generate_sample(domain, **kwargs)

    def generate_sample(
        self,
        domain: str,
        **kwargs: Any,
    ) -> list[dict] | pd.DataFrame:
        """Generate sample data for a specific domain.

        Args:
            domain: Domain type. See load_or_generate() for options.
            **kwargs: Optional parameters like n_users, n_records, etc.

        Returns:
            DataFrame or list of dicts with sample data.

        Raises:
            ValueError: If domain is not recognized.
        """
        generators: dict[str, Callable[[], list[dict] | pd.DataFrame]] = {
            "prospect": lambda: self.generate_prospect_data(
                n_users=kwargs.get("n_users", 500)
            ),
            "funnel": lambda: self.generate_funnel_data(
                n_users=kwargs.get("n_users", 1000)
            ),
            "subsidy_experiment": lambda: self.generate_subsidy_experiment_data(
                n_users=kwargs.get("n_users", 2000)
            ),
            "retention": lambda: self.generate_retention_data(
                n_users=kwargs.get("n_users", 800)
            ),
            "ad_campaign": lambda: self.generate_ad_campaign_data(
                n_campaigns=kwargs.get("n_campaigns", 10)
            ),
            "rta_history": lambda: self.generate_rta_history(
                n_records=kwargs.get("n_records", 200)
            ),
            "creative": lambda: self.generate_creative_data(
                n_creatives=kwargs.get("n_creatives", 8)
            ),
            "audience": lambda: self.generate_audience_data(
                n_users=kwargs.get("n_users", 100)
            ),
        }

        if domain not in generators:
            raise ValueError(
                f"Unknown domain '{domain}'. "
                f"Available: {sorted(generators.keys())}"
            )

        return generators[domain]()

    # ------------------------------------------------------------------
    # Domain-specific sample data generators
    # ------------------------------------------------------------------

    def generate_prospect_data(self, n_users: int = 500) -> pd.DataFrame:
        """Generate sample user behavior data for prospecting analysis.

        Args:
            n_users: Number of users to generate.

        Returns:
            DataFrame with columns: user_id, age, gender, city, city_tier,
            historical_orders, avg_order_value, days_since_last_order, ltv.
        """
        rng = np.random.default_rng(self.SEED)
        cities = ["上海", "北京", "深圳", "广州", "杭州", "成都", "武汉", "南京", "重庆", "苏州"]
        city_tiers = {"上海": 1, "北京": 1, "深圳": 1, "广州": 1, "杭州": 2, "成都": 2,
                      "武汉": 2, "南京": 2, "重庆": 3, "苏州": 2}
        city_choices = rng.choice(cities, size=n_users)
        genders = rng.choice(["M", "F"], size=n_users, p=[0.55, 0.45])

        ages = rng.integers(18, 60, size=n_users)
        orders = rng.geometric(0.15, size=n_users) - 1
        orders = np.clip(orders, 0, 200)
        aov = np.clip(rng.lognormal(3.5, 0.8, size=n_users), 20, 2000).round(2)
        recency = rng.exponential(30, size=n_users).astype(int)
        recency = np.clip(recency, 0, 365)
        ltv = (orders * aov * rng.uniform(0.3, 0.9, size=n_users)).round(2)

        return pd.DataFrame({
            "user_id": [f"U{i:06d}" for i in range(n_users)],
            "age": ages,
            "gender": genders,
            "city": city_choices,
            "city_tier": [city_tiers[c] for c in city_choices],
            "historical_orders": orders,
            "avg_order_value": aov,
            "days_since_last_order": recency,
            "ltv": ltv,
        })

    def generate_funnel_data(self, n_users: int = 1000) -> dict[str, int]:
        """Generate sample conversion funnel data.

        Args:
            n_users: Base number of users at funnel entry (scaled for realism).

        Returns:
            Dict with funnel stage counts: exposure, click, app_open, search,
            quote_view, order_confirm, first_order.
        """
        rng = np.random.default_rng(self.SEED)
        base = n_users
        # Apply conversion rates at each stage
        return {
            "exposure": int(base * 100),
            "click": int(base * 25),
            "app_open": int(base * 18),
            "search": int(base * 12),
            "quote_view": int(base * 9),
            "order_confirm": int(base * 5.5),
            "first_order": int(base * 3.2),
        }

    def generate_subsidy_experiment_data(self, n_users: int = 2000) -> pd.DataFrame:
        """Generate sample subsidy A/B experiment data.

        Args:
            n_users: Number of users in the experiment.

        Returns:
            DataFrame with columns: user_id, treatment, converted, revenue,
            subsidy_amount, price, demand.
        """
        rng = np.random.default_rng(self.SEED)
        groups = rng.choice(["control", "treatment_5", "treatment_10", "treatment_20"],
                            size=n_users, p=[0.4, 0.2, 0.2, 0.2])
        subsidy_map = {"control": 0, "treatment_5": 5, "treatment_10": 10, "treatment_20": 20}
        subsidies = np.array([subsidy_map[g] for g in groups])

        # Conversion probability increases with subsidy
        base_conv_prob = 0.12
        conv_lift = subsidies / 100 * 0.3
        conv_probs = np.clip(base_conv_prob + conv_lift + rng.uniform(-0.02, 0.02, n_users), 0.01, 0.5)
        converted = rng.binomial(1, conv_probs).astype(bool)
        revenue = np.where(converted, np.clip(rng.lognormal(4.0, 0.6, n_users), 30, 500).round(2), 0)

        return pd.DataFrame({
            "user_id": [f"U{i:06d}" for i in range(n_users)],
            "treatment": (subsidies > 0).astype(int),
            "converted": converted,
            "revenue": revenue,
            "subsidy_amount": subsidies.astype(float),
            "price": (rng.lognormal(3.0, 0.5, size=n_users) * 10).round(2),
            "demand": rng.poisson(3, size=n_users).astype(float),
        })

    def generate_retention_data(self, n_users: int = 800) -> pd.DataFrame:
        """Generate sample retention/cohort data for analysis.

        Args:
            n_users: Number of users for order-level data generation.

        Returns:
            DataFrame with columns: user_id, signup_date, order_date, is_active,
            days_since_signup. Suitable for CohortAnalyzer.
        """
        rng = np.random.default_rng(self.SEED)
        n_orders = n_users * 6  # Average 6 orders per user
        users = [f"U{i:05d}" for i in range(n_users)]
        signup_dates = pd.date_range("2024-01-01", periods=n_users, freq="h")
        rows: list[dict[str, Any]] = []

        for i in range(n_orders):
            uid = rng.choice(users)
            idx = users.index(uid)
            signup = signup_dates[idx]
            order_date = signup + pd.Timedelta(days=int(rng.exponential(15)))
            rows.append({
                "user_id": uid,
                "signup_date": signup,
                "order_date": order_date,
                "is_active": rng.random() > 0.3,
                "days_since_signup": int((order_date - signup).days),
            })

        return pd.DataFrame(rows)

    def generate_ad_campaign_data(self, n_campaigns: int = 10) -> pd.DataFrame:
        """Generate sample ad campaign performance data.

        Args:
            n_campaigns: Number of campaigns to generate (will generate daily records).

        Returns:
            DataFrame with columns: date, campaign, channel, impressions, clicks,
            conversions, spend, revenue, ctr, cvr.
        """
        rng = np.random.default_rng(self.SEED)
        campaigns = ["品牌曝光-春季", "拉新-搜索", "拉新-信息流", "老客召回-推送", "节日大促-双十一"]
        channels = ["微信", "抖音", "百度", "快手", "美团"]
        dates = pd.date_range("2024-01-01", periods=90, freq="D")

        rows: list[dict[str, Any]] = []
        for date in dates:
            for campaign in rng.choice(campaigns, size=rng.integers(3, 6), replace=False):
                channel = rng.choice(channels)
                impressions = rng.integers(5000, 200000)
                ctr = rng.beta(2, 100)
                clicks = int(impressions * ctr)
                cvr = rng.beta(3, 60)
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

    def generate_rta_history(self, n_records: int = 200) -> list[dict[str, Any]]:
        """Generate sample RTA (Real-Time Auction) bidding history.

        Ported from AdExpert._build_sample_rta_data().

        Args:
            n_records: Number of bidding records to generate.

        Returns:
            List of dicts with keys: user_features (dict with city_tier, intent_score),
            outcome (win/loss/no_bid), cpa, revenue.
        """
        rng = np.random.RandomState(self.SEED)
        data = []

        for i in range(n_records):
            tier = int(rng.choice([1, 2, 3, 4]))
            intent = round(float(rng.random()), 4)
            outcome = rng.choice(["win", "loss", "no_bid"], p=[0.15, 0.35, 0.5])
            cpa = round(float(rng.uniform(30, 150)), 2) if outcome == "win" else None
            revenue = round(float(rng.uniform(50, 300)), 2) if outcome == "win" else 0
            data.append({
                "user_features": {"city_tier": tier, "intent_score": intent},
                "outcome": outcome,
                "cpa": cpa,
                "revenue": revenue,
            })

        return data

    def generate_creative_data(self, n_creatives: int = 8) -> list[dict[str, Any]]:
        """Generate sample creative performance data.

        Ported from AdExpert._build_sample_creative_data().

        Args:
            n_creatives: Number of creatives to generate.

        Returns:
            List of dicts with keys: creative_id, impressions, clicks, conversions,
            spend, revenue.
        """
        rng = np.random.RandomState(self.SEED)
        data = []

        for i in range(n_creatives):
            impressions = int(rng.randint(10000, 200000))
            clicks = int(impressions * rng.uniform(0.02, 0.12))
            conversions = int(clicks * rng.uniform(0.02, 0.15))
            spend = round(float(rng.uniform(1000, 10000)), 2)
            revenue = round(float(spend * rng.uniform(0.5, 3.0)), 2)
            data.append({
                "creative_id": f"creative_{i + 1:03d}",
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "spend": spend,
                "revenue": revenue,
            })

        return data

    def generate_audience_data(self, n_users: int = 100) -> list[dict[str, Any]]:
        """Generate sample audience segment data.

        Ported from AdExpert._build_sample_audience_data().

        Args:
            n_users: Number of users in the audience.

        Returns:
            List of dicts with keys: user_id, age, gender, city_tier,
            historical_orders, avg_order_value, days_since_last_order, ltv, segment.
        """
        rng = np.random.RandomState(self.SEED)
        data = []
        segments = ["new_user", "active", "moderate", "dormant", "high_value"]

        for i in range(n_users):
            data.append({
                "user_id": f"user_{i + 1:04d}",
                "age": int(rng.randint(18, 55)),
                "gender": rng.choice(["M", "F"]),
                "city_tier": int(rng.choice([1, 2, 3, 4])),
                "historical_orders": int(rng.randint(0, 30)),
                "avg_order_value": round(float(rng.uniform(20, 200)), 2),
                "days_since_last_order": int(rng.randint(0, 120)),
                "ltv": round(float(rng.uniform(50, 500)), 2),
                "segment": rng.choice(segments),
            })

        return data

    # ------------------------------------------------------------------
    # Legacy sample data generators (all use SEED=42)
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
