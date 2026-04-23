"""Structured return types for all GrowthPilot agents.

These Pydantic models replace the loose dict[str, Any] returns from each agent,
providing validation, serialization, and self-documenting contracts between
agents, tools, and the orchestrator.

Reference patterns:
- claude-cookbooks: every tool returns JSON with success/error fields
- ai-app-lab: State event-sourcing with typed models
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Base result model
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    """Base class for all agent results.

    Every agent returns a subclass of AgentResult. The ``success`` and
    ``errors`` fields follow the claude-cookbooks convention of always
    including an explicit status indicator.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    success: bool = True
    errors: list[str] = Field(default_factory=list)
    agent_name: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# LLM analysis envelope (shared by all agents)
# ---------------------------------------------------------------------------

class LLMAnalysis(BaseModel):
    """Structured wrapper for LLM synthesis output."""

    summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_response: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prospect agent result
# ---------------------------------------------------------------------------

class SegmentInfo(BaseModel):
    """Single user segment statistics."""

    count: int = 0
    ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class ProspectResult(AgentResult):
    """Result from ProspectAgent."""

    agent_name: str = "prospect"
    user_count: int = 0
    intent_metrics: dict[str, Any] = Field(default_factory=dict)
    segment_summary: dict[str, SegmentInfo] = Field(default_factory=dict)
    rfm_result_count: int = 0
    top_users_sample: list[dict[str, Any]] = Field(default_factory=list)
    analysis: LLMAnalysis = Field(default_factory=LLMAnalysis)


# ---------------------------------------------------------------------------
# Conversion agent result
# ---------------------------------------------------------------------------

class ConversionResult(AgentResult):
    """Result from ConversionAgent."""

    agent_name: str = "conversion"
    reach_result: dict[str, Any] = Field(default_factory=dict)
    funnel_result: dict[str, Any] = Field(default_factory=dict)
    slot_result: dict[str, Any] = Field(default_factory=dict)
    coupon_results: list[dict[str, Any]] = Field(default_factory=list)
    analysis: LLMAnalysis = Field(default_factory=LLMAnalysis)


# ---------------------------------------------------------------------------
# Subsidy agent result
# ---------------------------------------------------------------------------

class SubsidyResult(AgentResult):
    """Result from SubsidyAgent."""

    agent_name: str = "subsidy"
    ate: dict[str, Any] = Field(default_factory=dict)
    causal_insight: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    elasticity: dict[str, Any] = Field(default_factory=dict)
    price_sensitivity: dict[str, Any] = Field(default_factory=dict)
    budget_plan: dict[str, Any] = Field(default_factory=dict)
    expected_roi: float = 0.0
    allocation_plan: dict[str, Any] = Field(default_factory=dict)
    analysis: LLMAnalysis = Field(default_factory=LLMAnalysis)


# ---------------------------------------------------------------------------
# Retention agent result
# ---------------------------------------------------------------------------

class ChurnRisk(BaseModel):
    """Churn risk distribution summary."""

    high_risk_ratio: float = 0.0
    medium_risk_ratio: float = 0.0
    low_risk_ratio: float = 0.0
    train_auc: float = 0.0


class RetentionResult(AgentResult):
    """Result from RetentionAgent."""

    agent_name: str = "retention"
    nurture_plans: dict[str, Any] = Field(default_factory=dict)
    nurture_progress: dict[str, Any] = Field(default_factory=dict)
    churn_risk: ChurnRisk = Field(default_factory=ChurnRisk)
    high_risk_users: list[dict[str, Any]] = Field(default_factory=list)
    churn_factors: list[str] = Field(default_factory=list)
    winback_plans: dict[str, Any] = Field(default_factory=dict)
    winback_priority: list[str] = Field(default_factory=list)
    cohort_data: dict[str, Any] = Field(default_factory=dict)
    retention_curve: dict[str, Any] = Field(default_factory=dict)
    cohort_insight: str = ""
    analysis: LLMAnalysis = Field(default_factory=LLMAnalysis)


# ---------------------------------------------------------------------------
# Ad agent result
# ---------------------------------------------------------------------------

class BidResult(BaseModel):
    """Bid optimization details."""

    original_bid: float = 0.0
    optimized_bid: float = 0.0
    target_cpa: float = 0.0
    estimated_cvr: float = 0.0


class AdResult(AgentResult):
    """Result from AdAgent."""

    agent_name: str = "ad"
    rta_rules: list[dict[str, Any]] = Field(default_factory=list)
    rta_metrics: dict[str, Any] = Field(default_factory=dict)
    bid_result: BidResult = Field(default_factory=BidResult)
    expected_cpa: float = 0.0
    creative_result: dict[str, Any] = Field(default_factory=dict)
    fatigue_alerts: list[str] = Field(default_factory=list)
    audience_result: dict[str, Any] = Field(default_factory=dict)
    expansion_opportunities: list[str] = Field(default_factory=list)
    analysis: LLMAnalysis = Field(default_factory=LLMAnalysis)


# ---------------------------------------------------------------------------
# KPI Snapshot (built by OrchestratorAgent)
# ---------------------------------------------------------------------------

class KpiSnapshot(BaseModel):
    """Aggregate key metrics from all sub-agent results."""

    total_users: int = 0
    intent_auc: float = 0.0
    conversion_rate: float = 0.0
    expected_roi: float = 0.0
    churn_risk_ratio: float = 0.0
    ad_cpa: float = 0.0
    budget: float = 0.0
    scope: str = "full"


# ---------------------------------------------------------------------------
# Analysis Output (final orchestrator output)
# ---------------------------------------------------------------------------

class AnalysisOutput(AgentResult):
    """Top-level output from the OrchestratorAgent.

    Wraps all sub-agent results and the final strategy recommendation.
    """

    agent_name: str = "orchestrator"
    analysis_summary: str = ""
    strategy_recommendation: str = ""
    kpi_snapshot: KpiSnapshot = Field(default_factory=KpiSnapshot)
    scope: str = "full"
    agents_run: list[str] = Field(default_factory=list)
    metadata: list[dict[str, Any]] = Field(default_factory=list)

    # Sub-agent results (optional, present only if that agent was executed)
    prospect_results: ProspectResult | None = None
    conversion_results: ConversionResult | None = None
    subsidy_results: SubsidyResult | None = None
    retention_results: RetentionResult | None = None
    ad_results: AdResult | None = None


# ---------------------------------------------------------------------------
# New Architecture: Tool Schemas & Plan Models
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    """Input to the GrowthPilot system (replaces AgentState for new architecture)."""

    query: str
    data_path: str = ""
    budget: float | None = None
    scope: str | None = None


class AnalysisResult(BaseModel):
    """Output from the GrowthPilot system."""

    success: bool = True
    query: str = ""
    scope: str = "full"
    analysis_summary: str = ""
    strategy_recommendation: str = ""
    kpi_snapshot: dict[str, Any] = Field(default_factory=dict)
    expert_results: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    report: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanStep(BaseModel):
    """Single step in the Chief Agent's execution plan."""

    expert: str
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    parallel_group: int = 0


class ExecutionPlan(BaseModel):
    """The Chief Agent's execution plan."""

    reasoning: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    context_summary: str = ""


class ProspectToolInput(BaseModel):
    """Input schema for prospect_analysis tool."""
    query: str = Field(description="用户的增长分析问题")
    data_path: str = Field(default="", description="数据文件路径 (可选)")
    budget: float = Field(default=0, description="预算金额 (可选)")


class ConversionToolInput(BaseModel):
    """Input schema for conversion_analysis tool."""
    query: str = Field(description="用户的转化分析问题")
    data_path: str = Field(default="", description="数据文件路径 (可选)")
    budget: float = Field(default=0, description="预算金额 (可选)")
    prospect_results_json: str = Field(default="", description="前序潜客分析结果JSON (可选)")


class SubsidyToolInput(BaseModel):
    """Input schema for subsidy_analysis tool."""
    query: str = Field(description="用户的补贴分析问题")
    data_path: str = Field(default="", description="数据文件路径 (可选)")
    budget: float = Field(default=10000, description="预算金额")
    prospect_results_json: str = Field(default="", description="前序潜客分析结果JSON (可选)")


class RetentionToolInput(BaseModel):
    """Input schema for retention_analysis tool."""
    query: str = Field(description="用户的留存分析问题")
    data_path: str = Field(default="", description="数据文件路径 (可选)")
    conversion_results_json: str = Field(default="", description="前序转化分析结果JSON (可选)")


class AdToolInput(BaseModel):
    """Input schema for ad_analysis tool."""
    query: str = Field(description="用户的广告分析问题")
    data_path: str = Field(default="", description="数据文件路径 (可选)")
    budget: float = Field(default=0, description="预算金额 (可选)")
    prospect_results_json: str = Field(default="", description="前序潜客分析结果JSON (可选)")


# ---------------------------------------------------------------------------
# Utility: convert AgentResult to state-update dict
# ---------------------------------------------------------------------------

def result_to_state_update(result: AgentResult) -> dict[str, Any]:
    """Convert an AgentResult into the dict format expected by AgentState.

    This bridges the new typed models with the existing LangGraph state
    convention where each agent returns a partial state update dict.
    """
    data = result.model_dump(exclude_none=True, mode="python")
    # Remove base fields that belong in metadata, not as state keys
    base_keys = {
        "success", "errors", "agent_name", "timestamp",
    }
    meta = {k: data.pop(k) for k in base_keys if k in data}
    errors = meta.get("errors", [])
    out: dict[str, Any] = {}
    out.update(data)
    if errors:
        out["errors"] = errors
    return out
