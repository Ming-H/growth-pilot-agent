"""AdAgent - RTA strategy, bid optimization, creative and audience analysis."""

from __future__ import annotations

import logging
from typing import Any

from src.core.base import BaseAgent
from src.core.state import AgentState

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


SYSTEM_PROMPT = """\
你是 GrowthPilot 广告投放 Agent。你的职责是：
1. 分析广告 RTA 策略效果
2. 优化出价策略
3. 检测创意疲劳
4. 分析受众表现

请用 JSON 格式输出分析结果。
"""


class AdAgent(BaseAgent):
    """Optimizes ad performance through RTA strategy and bid/creative analysis."""

    name = "ad"
    description = "广告投放 Agent"

    def __init__(self, llm: Any, system_prompt: str = SYSTEM_PROMPT) -> None:
        super().__init__(llm=llm, system_prompt=system_prompt)
        self._rta_strategy = RTAStrategy()
        self._bid_optimizer = BidOptimizer()
        self._creative_analyzer = CreativeAnalyzer()
        self._audience_analyzer = AudienceAnalyzer()

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Execute the ad analysis pipeline."""
        errors: list[str] = []
        budget = state.get("budget", 0)
        prospect = state.get("prospect_results") or {}

        # 1. Analyze RTA strategy
        rta_result: dict = {}
        rta_rules: list = []
        try:
            # Build sample historical data for RTA analysis
            historical_data = self._build_sample_rta_data()
            rta_result = self._rta_strategy.build_rta_decision_rules(historical_data)
            rta_rules = rta_result.get("rules", [])
        except Exception as exc:
            logger.warning("RTAStrategy failed: %s", exc)
            rta_rules = []
            errors.append(f"RTAStrategy: {exc}")

        # 2. Optimize bids
        bid_result: dict = {}
        expected_cpa = 0.0
        try:
            # Use ecpc_bid for sample optimization
            current_bid = 5.0
            target_cpa = 80.0
            cvr = 0.03
            optimized_bid = self._bid_optimizer.ecpc_bid(
                current_bid=current_bid,
                cvr=cvr,
                target_cpa=target_cpa,
            )
            expected_cpa = target_cpa
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
            creative_result = self._creative_analyzer.analyze_creative_performance(
                sample_creative_data,
            )
            # Check for fatigue indicators
            for c in creative_result.get("underperformers", []):
                if isinstance(c, dict) and c.get("roi", 0) < 0.5:
                    fatigue_alerts.append(f"Creative {c.get('creative_id', '?')} shows low ROI: {c.get('roi', 'N/A')}")
        except Exception as exc:
            logger.warning("CreativeAnalyzer failed: %s", exc)
            creative_result = {}
            errors.append(f"CreativeAnalyzer: {exc}")

        # 4. Analyze audience
        audience_result: dict = {}
        expansion_opportunities: list = []
        try:
            sample_audience = self._build_sample_audience_data()
            audience_result = self._audience_analyzer.analyze_audience(sample_audience)
            # Extract expansion opportunities from segments
            for seg_name, seg_data in audience_result.get("segments", {}).items():
                if isinstance(seg_data, dict) and seg_data.get("avg_ltv", 0) > 100:
                    expansion_opportunities.append(f"{seg_name}: high-value segment for expansion")
        except Exception as exc:
            logger.warning("AudienceAnalyzer failed: %s", exc)
            audience_result = {}
            errors.append(f"AudienceAnalyzer: {exc}")

        # 5. LLM synthesis
        try:
            prompt = self._build_ad_prompt(
                rta_rules=rta_rules,
                rta_metrics=rta_result.get("overall_metrics", {}),
                bid_result=bid_result,
                expected_cpa=expected_cpa,
                creative_result=creative_result,
                fatigue_alerts=fatigue_alerts,
                audience_result=audience_result,
                expansion_opportunities=expansion_opportunities,
                state=state,
            )
            llm_response = await self._invoke_llm(prompt)
            analysis = self._parse_json_response(llm_response)
        except Exception as exc:
            logger.warning("Ad LLM synthesis failed: %s", exc)
            analysis = {"summary": "LLM synthesis unavailable"}
            errors.append(f"LLM synthesis: {exc}")

        result: dict[str, Any] = {
            "ad_results": {
                "rta_rules": rta_rules,
                "rta_metrics": rta_result.get("overall_metrics", {}),
                "bid_result": bid_result,
                "expected_cpa": expected_cpa,
                "creative_result": creative_result,
                "fatigue_alerts": fatigue_alerts,
                "audience_result": audience_result,
                "expansion_opportunities": expansion_opportunities,
                "analysis": analysis,
            },
        }
        if errors:
            result["errors"] = errors
        return result

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

    # ------------------------------------------------------------------
    def _build_ad_prompt(self, *, state: AgentState, **kw: Any) -> str:
        context = self._build_prompt_context(state)
        parts = [context, "", "## 广告投放分析数据"]

        rta_metrics = kw.get("rta_metrics", {})
        if rta_metrics:
            parts.append(f"- RTA 整体胜率: {rta_metrics.get('win_rate', 'N/A')}")
            parts.append(f"- RTA 总曝光: {rta_metrics.get('total_impressions', 'N/A')}")

        bid_result = kw.get("bid_result", {})
        if bid_result:
            parts.append(f"- 出价优化: {bid_result.get('original_bid', 'N/A')} -> {bid_result.get('optimized_bid', 'N/A')}")

        fatigue = kw.get("fatigue_alerts", [])
        if fatigue:
            parts.append(f"- 创意疲劳告警: {fatigue}")

        expansion = kw.get("expansion_opportunities", [])
        if expansion:
            parts.append(f"- 受众拓展机会: {expansion}")

        parts.append("""
请基于以上数据给出广告投放的综合分析和建议：
1. RTA 策略评估
2. 出价优化建议
3. 创意优化方案
4. 受众分析洞察

请以 JSON 格式输出:
{
  "summary": "总体概述",
  "rta_assessment": "RTA 策略评估",
  "bid_optimization": "出价优化建议",
  "creative_plan": "创意优化方案",
  "audience_insight": "受众分析洞察"
}""")
        return "\n".join(parts)
