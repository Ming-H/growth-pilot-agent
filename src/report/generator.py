"""ReportGenerator - Produces a comprehensive markdown report from AgentState."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.core.state import AgentState

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates a full markdown report from the workflow state."""

    def generate_report(self, state: AgentState) -> str:
        """Generate a full markdown report.

        Parameters
        ----------
        state:
            The final ``AgentState`` after the workflow completes.

        Returns
        -------
        str
            Markdown report.
        """
        sections = [
            self._header(state),
            self._kpi_section(state),
            self._prospect_section(state),
            self._conversion_section(state),
            self._subsidy_section(state),
            self._retention_section(state),
            self._ad_section(state),
            self._experiment_section(state),
            self._seasonal_section(state),
            self._strategy_section(state),
            self._errors_section(state),
            self._footer(),
        ]
        # Filter out empty sections
        return "\n\n".join(s for s in sections if s.strip())

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _header(self, state: AgentState) -> str:
        query = state.get("query", "N/A")
        scope = state.get("scope", "full")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"# GrowthPilot 增长分析报告\n\n"
            f"> 生成时间: {now}\n\n"
            f"- **查询**: {query}\n"
            f"- **分析范围**: {scope}\n"
            f"- **数据路径**: {state.get('data_path', 'N/A')}\n"
            f"- **预算**: {state.get('budget', 'N/A')}"
        )

    def _kpi_section(self, state: AgentState) -> str:
        kpi = state.get("kpi_snapshot") or {}
        if not kpi:
            return ""
        lines = [
            "## KPI 快照\n",
            "| 指标 | 值 |",
            "|------|----|",
            f"| 用户总数 | {kpi.get('total_users', 'N/A')} |",
            f"| 意向模型 AUC | {kpi.get('intent_auc', 'N/A')} |",
            f"| 转化率 | {self._pct(kpi.get('conversion_rate'))} |",
            f"| 预期 ROI | {kpi.get('expected_roi', 'N/A')} |",
            f"| 流失风险比 | {self._pct(kpi.get('churn_risk_ratio'))} |",
            f"| 广告 CPA | {kpi.get('ad_cpa', 'N/A')} |",
            f"| 预算 | {kpi.get('budget', 'N/A')} |",
        ]
        return "\n".join(lines)

    def _prospect_section(self, state: AgentState) -> str:
        data = state.get("prospect_results") or {}
        if not data:
            return ""
        lines = ["## 潜客识别分析\n"]
        analysis = data.get("analysis", {})
        if analysis.get("summary"):
            lines.append(f"**概述**: {analysis['summary']}\n")
        if analysis.get("high_value_profile"):
            lines.append(f"**高价值用户画像**: {analysis['high_value_profile']}\n")
        if analysis.get("intent_insight"):
            lines.append(f"**转化意向洞察**: {analysis['intent_insight']}\n")
        if analysis.get("ltv_insight"):
            lines.append(f"**LTV 分布洞察**: {analysis['ltv_insight']}\n")
        if analysis.get("segment_strategy"):
            lines.append(f"**分层运营建议**: {analysis['segment_strategy']}\n")

        # Segment summary table
        seg = data.get("segment_summary", {})
        if seg:
            lines.append("### 用户分层概览\n")
            lines.append("| 分层 | 用户数 | 占比 |")
            lines.append("|------|--------|------|")
            for name, info in seg.items():
                if isinstance(info, dict):
                    lines.append(
                        f"| {name} | {info.get('count', 'N/A')} "
                        f"| {self._pct(info.get('ratio'))} |"
                    )
                else:
                    lines.append(f"| {name} | {info} | - |")

        return "\n".join(lines)

    def _conversion_section(self, state: AgentState) -> str:
        data = state.get("conversion_results") or {}
        if not data:
            return ""
        lines = ["## 转化策略分析\n"]
        analysis = data.get("analysis", {})
        if analysis.get("summary"):
            lines.append(f"**概述**: {analysis['summary']}\n")
        if analysis.get("reach_assessment"):
            lines.append(f"**触达策略评估**: {analysis['reach_assessment']}\n")
        if analysis.get("funnel_optimization"):
            lines.append(f"**漏斗优化建议**: {analysis['funnel_optimization']}\n")
        if analysis.get("coupon_recommendation"):
            lines.append(f"**优惠券策略建议**: {analysis['coupon_recommendation']}\n")
        if analysis.get("attribution_insight"):
            lines.append(f"**归因洞察**: {analysis['attribution_insight']}\n")
        if analysis.get("seasonal_adjustment"):
            lines.append(f"**季节性调整**: {analysis['seasonal_adjustment']}\n")

        # Funnel table
        funnel = data.get("funnel_result", {})
        stages = funnel.get("stages", [])
        if stages:
            lines.append("### 漏斗阶段\n")
            lines.append("| 阶段 | 用户数 | 阶段转化率 | 累计转化率 |")
            lines.append("|------|--------|-----------|-----------|")
            for stage in stages:
                if isinstance(stage, dict):
                    lines.append(
                        f"| {stage.get('stage', 'N/A')} "
                        f"| {stage.get('count', 'N/A')} "
                        f"| {self._pct(stage.get('stage_conversion_rate'))} "
                        f"| {self._pct(stage.get('cumulative_conversion_rate'))} |"
                    )
        if funnel.get("bottleneck"):
            bn = funnel["bottleneck"]
            lines.append(f"\n**瓶颈环节**: {bn.get('stage', 'N/A')} (转化率: {self._pct(bn.get('stage_conversion_rate'))})")
            if bn.get("diagnosis_hint"):
                lines.append(f"- 诊断: {bn['diagnosis_hint']}")

        return "\n".join(lines)

    def _subsidy_section(self, state: AgentState) -> str:
        data = state.get("subsidy_results") or {}
        if not data:
            return ""
        lines = ["## 补贴策略分析\n"]
        analysis = data.get("analysis", {})
        if analysis.get("summary"):
            lines.append(f"**概述**: {analysis['summary']}\n")
        if analysis.get("causal_assessment"):
            lines.append(f"**补贴效果评估**: {analysis['causal_assessment']}\n")
        if analysis.get("elasticity_insight"):
            lines.append(f"**价格弹性洞察**: {analysis['elasticity_insight']}\n")
        if analysis.get("budget_recommendation"):
            lines.append(f"**预算分配建议**: {analysis['budget_recommendation']}\n")
        if analysis.get("roi_strategy"):
            lines.append(f"**ROI 优化策略**: {analysis['roi_strategy']}\n")

        # ATE info
        ate = data.get("ate", {})
        if ate:
            lines.append(f"### 因果推断结果\n")
            lines.append(f"- 平均处理效应 (ATE): {ate}")
            lines.append(f"- 置信度: {self._pct(data.get('confidence', 0))}")

        return "\n".join(lines)

    def _retention_section(self, state: AgentState) -> str:
        data = state.get("retention_results") or {}
        if not data:
            return ""
        lines = ["## 用户留存分析\n"]
        analysis = data.get("analysis", {})
        if analysis.get("summary"):
            lines.append(f"**概述**: {analysis['summary']}\n")
        if analysis.get("churn_analysis"):
            lines.append(f"**流失风险分析**: {analysis['churn_analysis']}\n")
        if analysis.get("winback_strategy"):
            lines.append(f"**挽回策略**: {analysis['winback_strategy']}\n")
        if analysis.get("cohort_insight"):
            lines.append(f"**群组洞察**: {analysis['cohort_insight']}\n")
        if analysis.get("retention_recommendation"):
            lines.append(f"**留存优化建议**: {analysis['retention_recommendation']}\n")

        high_risk = data.get("high_risk_users", [])
        if high_risk:
            lines.append(f"### 高流失风险用户\n")
            lines.append(f"- 高风险用户数: {len(high_risk)}")
            churn_factors = data.get("churn_factors", [])
            if churn_factors:
                lines.append(f"- 主要流失因素: {', '.join(str(f) for f in churn_factors[:5])}")

        return "\n".join(lines)

    def _ad_section(self, state: AgentState) -> str:
        data = state.get("ad_results") or {}
        if not data:
            return ""
        lines = ["## 广告投放分析\n"]
        analysis = data.get("analysis", {})
        if analysis.get("summary"):
            lines.append(f"**概述**: {analysis['summary']}\n")
        if analysis.get("rta_assessment"):
            lines.append(f"**RTA 策略评估**: {analysis['rta_assessment']}\n")
        if analysis.get("bid_optimization"):
            lines.append(f"**出价优化建议**: {analysis['bid_optimization']}\n")
        if analysis.get("creative_plan"):
            lines.append(f"**创意优化方案**: {analysis['creative_plan']}\n")
        if analysis.get("audience_insight"):
            lines.append(f"**受众分析洞察**: {analysis['audience_insight']}\n")

        fatigue = data.get("fatigue_alerts", [])
        if fatigue:
            lines.append("### 创意疲劳告警\n")
            for alert in fatigue:
                lines.append(f"- {alert}")

        expansion = data.get("expansion_opportunities", [])
        if expansion:
            lines.append("### 受众拓展机会\n")
            for opp in expansion:
                lines.append(f"- {opp}")

        return "\n".join(lines)

    def _experiment_section(self, state: AgentState) -> str:
        data = state.get("experiment_results") or {}
        if not data:
            return ""
        lines = ["## 实验分析\n"]
        if data.get("summary"):
            lines.append(f"**概述**: {data['summary']}\n")
        for key, value in data.items():
            if key != "summary":
                lines.append(f"- **{key}**: {value}")
        return "\n".join(lines)

    def _seasonal_section(self, state: AgentState) -> str:
        data = state.get("seasonal_context") or {}
        if not data:
            return ""
        lines = ["## 季节性分析\n"]
        if data.get("current_season"):
            lines.append(f"**当前季节因素**: {data['current_season']}\n")
        if data.get("upcoming_events"):
            lines.append(f"**即将到来的事件**: {data['upcoming_events']}\n")
        if data.get("recommendation"):
            lines.append(f"**建议**: {data['recommendation']}\n")
        return "\n".join(lines)

    def _strategy_section(self, state: AgentState) -> str:
        summary = state.get("analysis_summary") or ""
        strategy = state.get("strategy_recommendation") or ""
        if not summary and not strategy:
            return ""
        lines = ["## 综合策略建议\n"]
        if summary:
            lines.append(f"### 分析摘要\n\n{summary}\n")
        if strategy:
            lines.append("### 关键策略建议\n")
            for idx, item in enumerate(strategy.split(";"), 1):
                item = item.strip()
                if item:
                    lines.append(f"{idx}. {item}")
        return "\n".join(lines)

    def _errors_section(self, state: AgentState) -> str:
        errors = state.get("errors") or []
        if not errors:
            return ""
        lines = ["## 执行告警\n"]
        for err in errors:
            lines.append(f"- {err}")
        return "\n".join(lines)

    def _footer(self) -> str:
        return "---\n*GrowthPilot Agent v4.0.0 - Freight Growth Multi-Agent System*"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pct(value: Any) -> str:
        """Format a numeric value as a percentage string."""
        if value is None:
            return "N/A"
        if isinstance(value, (int, float)):
            return f"{value:.1%}"
        return str(value)
