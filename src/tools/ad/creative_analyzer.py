"""Creative performance analysis, fatigue detection, and A/B testing.

Analyzes ad creative metrics to identify top performers, detect creative
fatigue, and run statistical significance tests between variants.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sp_stats

from src.tools.registry import ToolRegistry


@ToolRegistry.register("creative_analyzer")
class CreativeAnalyzer:
    """Analyze ad creative performance, detect fatigue, and run A/B tests."""

    def analyze_creative_performance(self, creative_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregate performance metrics for each creative.

        Args:
            creative_data: List of dicts, each with keys:
                - 'creative_id': str
                - 'impressions': int
                - 'clicks': int
                - 'conversions': int
                - 'spend': float
                - 'revenue': float

        Returns:
            Dict with:
                - 'creatives': list of per-creative metrics (CTR, CVR, CPC, CPA, ROI)
                - 'summary': aggregate stats
                - 'top_performers': top 3 by ROI
                - 'underperformers': bottom 3 by ROI
        """
        if not creative_data:
            return {"creatives": [], "summary": {}, "top_performers": [], "underperformers": []}

        creative_metrics: list[dict[str, Any]] = []
        for c in creative_data:
            impressions = c.get("impressions", 0)
            clicks = c.get("clicks", 0)
            conversions = c.get("conversions", 0)
            spend = c.get("spend", 0.0)
            revenue = c.get("revenue", 0.0)

            ctr = clicks / impressions if impressions > 0 else 0.0
            cvr = conversions / clicks if clicks > 0 else 0.0
            cpc = spend / clicks if clicks > 0 else 0.0
            cpa = spend / conversions if conversions > 0 else 0.0
            roi = revenue / spend if spend > 0 else 0.0

            creative_metrics.append({
                "creative_id": c.get("creative_id", "unknown"),
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "spend": round(spend, 2),
                "revenue": round(revenue, 2),
                "ctr": round(ctr, 4),
                "cvr": round(cvr, 4),
                "cpc": round(cpc, 2),
                "cpa": round(cpa, 2),
                "roi": round(roi, 4),
            })

        # Sort by ROI descending
        creative_metrics.sort(key=lambda x: x["roi"], reverse=True)

        total_impressions = sum(c["impressions"] for c in creative_metrics)
        total_clicks = sum(c["clicks"] for c in creative_metrics)
        total_conversions = sum(c["conversions"] for c in creative_metrics)
        total_spend = sum(c["spend"] for c in creative_metrics)
        total_revenue = sum(c["revenue"] for c in creative_metrics)

        summary = {
            "num_creatives": len(creative_metrics),
            "total_impressions": total_impressions,
            "total_clicks": total_clicks,
            "total_conversions": total_conversions,
            "total_spend": round(total_spend, 2),
            "total_revenue": round(total_revenue, 2),
            "avg_ctr": round(total_clicks / total_impressions, 4) if total_impressions > 0 else 0.0,
            "avg_cvr": round(total_conversions / total_clicks, 4) if total_clicks > 0 else 0.0,
            "overall_roi": round(total_revenue / total_spend, 4) if total_spend > 0 else 0.0,
        }

        top_performers = creative_metrics[:3]
        underperformers = creative_metrics[-3:] if len(creative_metrics) >= 3 else creative_metrics

        return {
            "creatives": creative_metrics,
            "summary": summary,
            "top_performers": top_performers,
            "underperformers": underperformers,
        }

    def detect_fatigue(
        self,
        creative_data: list[dict[str, Any]],
        threshold: float = 0.7,
    ) -> list[str]:
        """Detect fatigued creatives whose performance has degraded.

        A creative is considered fatigued when its recent CTR or CVR falls
        below ``threshold`` of its own peak (historical best).

        Args:
            creative_data: List of dicts, each with keys:
                - 'creative_id': str
                - 'period': str (e.g. "2024-01-15")
                - 'impressions': int
                - 'clicks': int
                - 'conversions': int
            threshold: Fraction of peak performance below which fatigue is
                       flagged (default 0.7 = 70%).

        Returns:
            List of creative IDs that show signs of fatigue.
        """
        if not creative_data:
            return []

        # Group by creative_id
        groups: dict[str, list[dict[str, Any]]] = {}
        for d in creative_data:
            cid = d.get("creative_id", "unknown")
            groups.setdefault(cid, []).append(d)

        fatigued: list[str] = []
        for cid, entries in groups.items():
            if len(entries) < 3:
                # Not enough data to detect fatigue reliably
                continue

            # Sort by period to get chronological order
            entries.sort(key=lambda x: x.get("period", ""))

            ctrs = []
            cvrs = []
            for e in entries:
                imp = e.get("impressions", 0)
                clk = e.get("clicks", 0)
                conv = e.get("conversions", 0)
                ctrs.append(clk / imp if imp > 0 else 0.0)
                cvrs.append(conv / clk if clk > 0 else 0.0)

            peak_ctr = max(ctrs)
            peak_cvr = max(cvrs)

            # Look at recent performance (last 3 periods)
            recent_ctrs = ctrs[-3:]
            recent_cvrs = cvrs[-3:]

            avg_recent_ctr = float(np.mean(recent_ctrs))
            avg_recent_cvr = float(np.mean(recent_cvrs))

            is_ctr_fatigued = peak_ctr > 0 and avg_recent_ctr < threshold * peak_ctr
            is_cvr_fatigued = peak_cvr > 0 and avg_recent_cvr < threshold * peak_cvr

            if is_ctr_fatigued or is_cvr_fatigued:
                fatigued.append(cid)

        return fatigued

    def ab_test_creatives(
        self,
        creative_a_data: dict[str, Any],
        creative_b_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Run statistical A/B test between two creatives.

        Tests both CTR (proportions z-test) and CVR differences.

        Args:
            creative_a_data: Dict with 'impressions', 'clicks', 'conversions'.
            creative_b_data: Same structure as creative_a_data.

        Returns:
            Dict with:
                - 'ctr_test': test results for click-through rate
                - 'cvr_test': test results for conversion rate
                - 'recommendation': 'A' | 'B' | 'inconclusive'
        """
        imp_a = creative_a_data.get("impressions", 0)
        clk_a = creative_a_data.get("clicks", 0)
        conv_a = creative_a_data.get("conversions", 0)

        imp_b = creative_b_data.get("impressions", 0)
        clk_b = creative_b_data.get("clicks", 0)
        conv_b = creative_b_data.get("conversions", 0)

        ctr_a = clk_a / imp_a if imp_a > 0 else 0.0
        ctr_b = clk_b / imp_b if imp_b > 0 else 0.0
        cvr_a = conv_a / clk_a if clk_a > 0 else 0.0
        cvr_b = conv_b / clk_b if clk_b > 0 else 0.0

        # --- CTR test (two-proportion z-test) ---
        ctr_test = self._proportion_test(clk_a, imp_a, clk_b, imp_b, label_a="A", label_b="B")

        # --- CVR test (two-proportion z-test) ---
        cvr_test = self._proportion_test(conv_a, clk_a, conv_b, clk_b, label_a="A", label_b="B")

        # Determine recommendation
        if ctr_test["significant"] or cvr_test["significant"]:
            # Use combined lift to decide
            a_score = ctr_a * cvr_a
            b_score = ctr_b * cvr_b
            if b_score > a_score * 1.05:
                recommendation = "B"
            elif a_score > b_score * 1.05:
                recommendation = "A"
            else:
                recommendation = "inconclusive"
        else:
            recommendation = "inconclusive"

        return {
            "ctr_test": {
                **ctr_test,
                "ctr_a": round(ctr_a, 4),
                "ctr_b": round(ctr_b, 4),
                "lift": round((ctr_b - ctr_a) / ctr_a, 4) if ctr_a > 0 else None,
            },
            "cvr_test": {
                **cvr_test,
                "cvr_a": round(cvr_a, 4),
                "cvr_b": round(cvr_b, 4),
                "lift": round((cvr_b - cvr_a) / cvr_a, 4) if cvr_a > 0 else None,
            },
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _proportion_test(
        successes_a: int,
        trials_a: int,
        successes_b: int,
        trials_b: int,
        label_a: str = "A",
        label_b: str = "B",
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Two-proportion z-test for comparing conversion/click rates."""
        if trials_a == 0 or trials_b == 0:
            return {
                "significant": False,
                "p_value": None,
                "z_statistic": None,
                "confidence_interval": None,
                "note": "Insufficient data (zero impressions/clicks)",
            }

        p_a = successes_a / trials_a
        p_b = successes_b / trials_b
        p_pool = (successes_a + successes_b) / (trials_a + trials_b)

        if p_pool == 0 or p_pool == 1:
            return {
                "significant": False,
                "p_value": None,
                "z_statistic": None,
                "confidence_interval": None,
                "note": "No variance in pooled proportion",
            }

        se = np.sqrt(p_pool * (1 - p_pool) * (1 / trials_a + 1 / trials_b))
        if se == 0:
            return {
                "significant": False,
                "p_value": None,
                "z_statistic": None,
                "confidence_interval": None,
                "note": "Zero standard error",
            }

        z = (p_b - p_a) / se
        # Two-tailed p-value
        p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))

        # Confidence interval for the difference
        se_diff = np.sqrt(p_a * (1 - p_a) / trials_a + p_b * (1 - p_b) / trials_b)
        ci_lower = (p_b - p_a) - 1.96 * se_diff
        ci_upper = (p_b - p_a) + 1.96 * se_diff

        return {
            "significant": p_value < alpha,
            "p_value": round(float(p_value), 6),
            "z_statistic": round(float(z), 4),
            "confidence_interval": (round(float(ci_lower), 4), round(float(ci_upper), 4)),
        }
