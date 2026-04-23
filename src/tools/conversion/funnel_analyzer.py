"""FunnelAnalyzer - analyze conversion funnel from exposure to first order.

Computes per-step conversion rates, identifies bottlenecks, and provides
diagnosis for drop-off at each funnel stage.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Default funnel stages for Didi freight
DEFAULT_FUNNEL_STAGES = [
    "exposure",       # 曝光
    "click",          # 点击
    "app_open",       # 打开APP
    "search",         # 搜索/发单
    "quote_view",     # 查看报价
    "order_confirm",  # 确认下单
    "first_order",    # 首单完成
]


@ToolRegistry.register("funnel_analyzer")
class FunnelAnalyzer:
    """Analyze conversion funnel from exposure to first order."""

    def analyze_funnel(
        self,
        funnel_data: dict[str, int] | pd.DataFrame,
        stages: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute per-step conversion rates and identify the biggest bottleneck.

        Args:
            funnel_data: either a dict mapping stage_name -> user_count, or a
                DataFrame with columns [stage, count]. If a DataFrame with more
                columns (e.g. segment), the first two are used.
            stages: ordered list of funnel stage names. Defaults to
                DEFAULT_FUNNEL_STAGES.

        Returns:
            Dict with per-stage metrics and bottleneck identification.
        """
        stages = stages or DEFAULT_FUNNEL_STAGES

        # Normalise input to dict[str, int]
        counts = self._normalise_funnel_input(funnel_data, stages)
        if counts is None:
            return {"error": "invalid funnel_data format", "stages": []}

        # Ensure stages are ordered and present
        ordered: list[str] = []
        for s in stages:
            if s in counts:
                ordered.append(s)
        if len(ordered) < 2:
            return {"error": "need at least 2 stages with data", "stages": []}

        stage_metrics: list[dict[str, Any]] = []
        overall_drop = 0.0
        bottleneck_stage = ordered[1]
        bottleneck_rate = 1.0

        for i, stage in enumerate(ordered):
            current = counts[stage]
            if i == 0:
                stage_metrics.append(
                    {
                        "stage": stage,
                        "count": current,
                        "stage_conversion_rate": 1.0,
                        "cumulative_conversion_rate": 1.0,
                        "drop_off_count": 0,
                        "drop_off_pct": 0.0,
                    }
                )
                continue

            prev = counts[ordered[i - 1]]
            rate = current / prev if prev > 0 else 0.0
            cum_rate = current / counts[ordered[0]] if counts[ordered[0]] > 0 else 0.0
            drop = prev - current
            drop_pct = drop / prev if prev > 0 else 0.0

            stage_metrics.append(
                {
                    "stage": stage,
                    "count": current,
                    "stage_conversion_rate": round(rate, 6),
                    "cumulative_conversion_rate": round(cum_rate, 6),
                    "drop_off_count": drop,
                    "drop_off_pct": round(drop_pct, 6),
                }
            )

            if rate < bottleneck_rate:
                bottleneck_rate = rate
                bottleneck_stage = stage

        total_conversion = (
            counts[ordered[-1]] / counts[ordered[0]] if counts[ordered[0]] > 0 else 0.0
        )

        return {
            "stages": stage_metrics,
            "total_users_entered": counts[ordered[0]],
            "total_users_converted": counts[ordered[-1]],
            "overall_conversion_rate": round(total_conversion, 6),
            "bottleneck": {
                "stage": bottleneck_stage,
                "stage_conversion_rate": round(bottleneck_rate, 6),
                "diagnosis_hint": self._quick_diagnosis(bottleneck_stage, bottleneck_rate),
            },
        }

    def bottleneck_diagnosis(
        self,
        step: str,
        segment_data: pd.DataFrame | dict[str, Any],
    ) -> dict[str, Any]:
        """Diagnose reasons for drop-off at a specific funnel step.

        Args:
            step: the funnel stage name to diagnose.
            segment_data: either a DataFrame with columns [segment, count_before,
                count_after] or a dict with the same structure. Can also include
                extra columns used for diagnosis (e.g. device, channel).

        Returns:
            Dict with probable reasons and recommended actions.
        """
        if isinstance(segment_data, dict):
            try:
                segment_data = pd.DataFrame(segment_data)
            except Exception:
                return {"error": "cannot parse segment_data into DataFrame"}

        if segment_data.empty:
            return {"error": "segment_data is empty"}

        # Normalise column names
        col_map = {
            "count_before": "before",
            "count_after": "after",
            "users_before": "before",
            "users_after": "after",
        }
        segment_data = segment_data.rename(columns=col_map)

        if "before" not in segment_data.columns or "after" not in segment_data.columns:
            return {"error": "segment_data must have 'before' and 'after' columns"}

        segment_data["conversion_rate"] = np.where(
            segment_data["before"] > 0,
            segment_data["after"] / segment_data["before"],
            0.0,
        )
        segment_data["drop_pct"] = 1.0 - segment_data["conversion_rate"]

        # Identify worst-performing segment
        worst_idx = segment_data["conversion_rate"].idxmin()
        worst = segment_data.loc[worst_idx].to_dict()

        # Identify best-performing segment
        best_idx = segment_data["conversion_rate"].idxmax()
        best = segment_data.loc[best_idx].to_dict()

        # Compute gap
        gap = best["conversion_rate"] - worst["conversion_rate"]

        reasons = self._generate_reasons(step, worst, gap)
        actions = self._generate_actions(step, worst, best)

        return {
            "step": step,
            "overall_conversion": round(
                segment_data["after"].sum() / segment_data["before"].sum(), 6
            )
            if segment_data["before"].sum() > 0
            else 0.0,
            "segment_breakdown": segment_data.to_dict(orient="records"),
            "worst_performing_segment": worst,
            "best_performing_segment": best,
            "performance_gap": round(gap, 6),
            "probable_reasons": reasons,
            "recommended_actions": actions,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise_funnel_input(
        self, funnel_data: dict[str, int] | pd.DataFrame, stages: list[str]
    ) -> dict[str, int] | None:
        """Convert various input formats to a stage->count dict."""
        if isinstance(funnel_data, dict):
            return {k: int(v) for k, v in funnel_data.items() if isinstance(v, (int, float))}

        if isinstance(funnel_data, pd.DataFrame):
            if funnel_data.empty:
                return None
            cols = funnel_data.columns.tolist()
            if len(cols) >= 2:
                return dict(
                    zip(
                        funnel_data.iloc[:, 0].astype(str),
                        funnel_data.iloc[:, 1].astype(int),
                    )
                )
        return None

    def _quick_diagnosis(self, stage: str, rate: float) -> str:
        """Return a quick one-liner diagnosis hint."""
        hints: dict[str, str] = {
            "click": "低CTR可能因创意不吸引或位置不佳，建议A/B测试素材",
            "app_open": "DeepLink失效或APP未安装，检查链接配置",
            "search": "用户未找到所需车型/路线，优化搜索推荐算法",
            "quote_view": "报价偏高或不透明，优化定价策略和展示方式",
            "order_confirm": "下单流程太长或支付环节有问题，简化流程",
            "first_order": "履约环节（司机接单/准时率）有问题，提升服务质量",
            "exposure": "曝光量不足，检查渠道投放配置",
        }
        base = hints.get(stage, "需进一步分析用户行为路径")
        if rate < 0.1:
            return f"严重瓶颈: {base}"
        elif rate < 0.3:
            return f"中等瓶颈: {base}"
        return base

    def _generate_reasons(
        self, step: str, worst_segment: dict, gap: float
    ) -> list[str]:
        """Generate probable reason strings for the bottleneck."""
        reasons: list[str] = []
        step_reasons: dict[str, list[str]] = {
            "click": [
                "创意素材与目标用户不匹配",
                "广告位不够显眼或被折叠",
                "文案缺乏吸引力或紧迫感",
            ],
            "app_open": [
                "DeepLink配置错误导致无法唤起APP",
                "用户未安装APP且下载引导不充分",
                "网络延迟导致跳转超时",
            ],
            "search": [
                "搜索结果与用户意图不匹配",
                "可用地域/车型覆盖不足",
                "搜索交互体验差（如输入困难）",
            ],
            "quote_view": [
                "报价明显高于用户预期",
                "报价展示缺乏价格拆解，用户不信任",
                "等待报价时间过长",
            ],
            "order_confirm": [
                "下单流程步骤过多，用户中途放弃",
                "支付方式不完善（缺少常用支付渠道）",
                "价格在确认时发生变化",
            ],
            "first_order": [
                "司机接单慢或取消率高",
                "履约过程中服务体验不佳",
                "缺少新用户订单确认/提醒机制",
            ],
        }

        base_reasons = step_reasons.get(step, ["该环节转化率偏低，需深入分析用户行为数据"])
        reasons.extend(base_reasons)

        if gap > 0.3:
            reasons.append(
                f"不同用户群体差异大（差距{gap:.1%}），建议分群优化策略"
            )

        return reasons

    def _generate_actions(
        self, step: str, worst_segment: dict, best_segment: dict
    ) -> list[str]:
        """Generate recommended actions."""
        actions: list[str] = []
        step_actions: dict[str, list[str]] = {
            "click": [
                "对低转化用户群A/B测试新创意素材",
                "优化广告位和展示时机",
                "增加紧迫感文案（限时/限量）",
            ],
            "app_open": [
                "检查并修复DeepLink配置",
                "对未安装用户优化应用商店跳转页",
                "增加Universal Link支持",
            ],
            "search": [
                "优化搜索算法提升结果相关性",
                "增加热门路线/快捷入口推荐",
                "改善搜索框交互体验",
            ],
            "quote_view": [
                "优化定价策略，对敏感用户提供优惠券",
                "增加价格透明度展示（里程/时长/费用拆分）",
                "缩短报价生成时间",
            ],
            "order_confirm": [
                "简化下单流程至3步以内",
                "增加一键支付和免密支付",
                "确认页增加信任背书（保险/保障）",
            ],
            "first_order": [
                "提升司机接单效率（调度优化/激励机制）",
                "增加新用户专属服务保障",
                "完善订单状态实时通知",
            ],
        }
        actions.extend(step_actions.get(step, ["收集更多用户行为数据进行深入分析"]))

        worst_seg_name = worst_segment.get("segment", "未知群体")
        actions.insert(0, f"优先优化 {worst_seg_name} 群体的 {step} 环节体验")

        return actions
