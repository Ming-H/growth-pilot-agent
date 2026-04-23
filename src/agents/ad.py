"""AdExpert - RTA strategy, bid optimization, creative and audience analysis."""

from __future__ import annotations

import logging
from typing import Any

from src.core.expert import ExpertAgentBase
from src.prompts.templates.agent_prompts import AdPrompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool imports with fallback stubs
# ---------------------------------------------------------------------------
try:
    from src.tools.ad.rta_strategy import RTAStrategy
    from src.tools.ad.bid_optimizer import BidOptimizer
    from src.tools.ad.creative_analyzer import CreativeAnalyzer
    from src.tools.ad.audience_analyzer import AudienceAnalyzer
except ImportError:

    class _Stub:
        def __init__(self, *a: Any, **kw: Any) -> None: ...

    RTAStrategy = BidOptimizer = CreativeAnalyzer = AudienceAnalyzer = _Stub


class AdExpert(ExpertAgentBase):
    """Optimizes ad performance through RTA strategy and bid/creative analysis."""

    name = "ad"
    description = "广告投放专家 Agent"

    def _init_tools(self) -> dict[str, Any]:
        """Initialize and return deterministic tool instances as a dict."""
        return {
            "rta_strategy": RTAStrategy(),
            "bid_optimizer": BidOptimizer(),
            "creative_analyzer": CreativeAnalyzer(),
            "audience_analyzer": AudienceAnalyzer(),
        }

    def _execute_pipeline(self, params: dict) -> dict:
        """Run the deterministic tool pipeline. Returns raw results dict."""
        errors: list[str] = []

        rta_strategy = self._tools["rta_strategy"]
        bid_optimizer = self._tools["bid_optimizer"]
        creative_analyzer = self._tools["creative_analyzer"]
        audience_analyzer = self._tools["audience_analyzer"]

        # 1. Analyze RTA strategy
        rta_result: dict = {}
        rta_rules: list = []
        try:
            historical_data = self._build_sample_rta_data()
            rta_result = rta_strategy.build_rta_decision_rules(historical_data)
            rta_rules = rta_result.get("rules", [])
        except Exception as exc:
            logger.warning("RTAStrategy failed: %s", exc)
            rta_rules = []
            errors.append(f"RTAStrategy: {exc}")

        # 2. Optimize bids
        bid_result: dict = {}
        try:
            current_bid = 5.0
            target_cpa = 80.0
            cvr = 0.03
            optimized_bid = bid_optimizer.ecpc_bid(
                current_bid=current_bid,
                cvr=cvr,
                target_cpa=target_cpa,
            )
            bid_result = {
                "original_bid": current_bid,
                "optimized_bid": round(optimized_bid, 4),
                "target_cpa": target_cpa,
                "estimated_cvr": cvr,
            }
        except Exception as exc:
            logger.warning("BidOptimizer failed: %s", exc)
            bid_result = {}
            errors.append(f"BidOptimizer: {exc}")

        # 3. Detect creative fatigue
        creative_result: dict = {}
        fatigue_alerts: list = []
        try:
            sample_creative_data = self._build_sample_creative_data()
            creative_result = creative_analyzer.analyze_creative_performance(
                sample_creative_data,
            )
            for c in creative_result.get("underperformers", []):
                if isinstance(c, dict) and c.get("roi", 0) < 0.5:
                    fatigue_alerts.append(
                        f"Creative {c.get('creative_id', '?')} shows low ROI: {c.get('roi', 'N/A')}"
                    )
        except Exception as exc:
            logger.warning("CreativeAnalyzer failed: %s", exc)
            creative_result = {}
            errors.append(f"CreativeAnalyzer: {exc}")

        # 4. Analyze audience
        audience_result: dict = {}
        expansion_opportunities: list = []
        try:
            sample_audience = self._build_sample_audience_data()
            audience_result = audience_analyzer.analyze_audience(sample_audience)
            for seg_name, seg_data in audience_result.get("segments", {}).items():
                if isinstance(seg_data, dict) and seg_data.get("avg_ltv", 0) > 100:
                    expansion_opportunities.append(f"{seg_name}: high-value segment for expansion")
        except Exception as exc:
            logger.warning("AudienceAnalyzer failed: %s", exc)
            audience_result = {}
            errors.append(f"AudienceAnalyzer: {exc}")

        return {
            "rta_rules": rta_rules,
            "rta_metrics": rta_result.get("overall_metrics", {}),
            "bid_result": bid_result,
            "creative_result": creative_result,
            "fatigue_alerts": fatigue_alerts,
            "audience_result": audience_result,
            "expansion_opportunities": expansion_opportunities,
            "errors": errors,
        }

    def _build_synthesis_prompt(self, params: dict, results: dict) -> str:
        """Build the LLM synthesis prompt from pipeline results."""
        rta_metrics = results.get("rta_metrics", {})
        rta_win_rate = str(rta_metrics.get("win_rate", "")) if rta_metrics.get("win_rate") else ""
        rta_impressions = (
            str(rta_metrics.get("total_impressions", ""))
            if rta_metrics.get("total_impressions")
            else ""
        )

        bid_result = results.get("bid_result", {})
        bid_summary = ""
        if bid_result:
            bid_summary = f"{bid_result.get('original_bid', 'N/A')} -> {bid_result.get('optimized_bid', 'N/A')}"

        fatigue_alerts = results.get("fatigue_alerts", [])
        fatigue_str = str(fatigue_alerts) if fatigue_alerts else ""

        expansion = results.get("expansion_opportunities", [])
        expansion_str = str(expansion) if expansion else ""

        return AdPrompt().render(
            user_query=params.get("query", ""),
            season=params.get("season", "当前"),
            kpi_baseline=params.get("kpi_baseline", {}),
            memory_context=params.get("memory_context", {}),
            rta_win_rate=rta_win_rate,
            rta_impressions=rta_impressions,
            bid_summary=bid_summary,
            fatigue_alerts=fatigue_str,
            expansion_opportunities=expansion_str,
        )

    def _get_system_prompt(self) -> str:
        """Return the expert's system prompt for LLM synthesis."""
        template = AdPrompt()
        return f"{template.role_definition}\n\n{template.business_context}"

    def can_handle(self, query: str) -> float:
        """Return confidence score (0-1) for handling this query."""
        keywords = [
            "广告", "rta", "出价", "竞价", "创意",
            "投放", "audience", "bid", "creative", "ad",
        ]
        query_lower = query.lower()
        matches = sum(1 for kw in keywords if kw in query_lower)
        if matches >= 2:
            return 0.9
        if matches == 1:
            return 0.7
        return 0.0

    # ------------------------------------------------------------------
    # Sample data builders (for demo mode when no real data available)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_sample_rta_data() -> list[dict[str, Any]]:
        """Generate sample RTA historical data."""
        import numpy as np

        rng = np.random.RandomState(42)
        data = []
        for i in range(200):
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

    @staticmethod
    def _build_sample_creative_data() -> list[dict[str, Any]]:
        """Generate sample creative performance data."""
        import numpy as np

        rng = np.random.RandomState(42)
        data = []
        for i in range(8):
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

    @staticmethod
    def _build_sample_audience_data() -> list[dict[str, Any]]:
        """Generate sample audience data."""
        import numpy as np

        rng = np.random.RandomState(42)
        data = []
        segments = ["new_user", "active", "moderate", "dormant", "high_value"]
        for i in range(100):
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
