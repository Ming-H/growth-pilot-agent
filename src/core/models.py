"""Structured return types for all GrowthPilot agents.

These Pydantic models provide validation, serialization, and
self-documenting contracts between agents, tools, and the orchestrator.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Base result model
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    """Base class for all agent results."""

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
# Expert agent results
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


class ConversionResult(AgentResult):
    """Result from ConversionAgent."""

    agent_name: str = "conversion"
    reach_result: dict[str, Any] = Field(default_factory=dict)
    funnel_result: dict[str, Any] = Field(default_factory=dict)
    slot_result: dict[str, Any] = Field(default_factory=dict)
    coupon_results: list[dict[str, Any]] = Field(default_factory=list)
    analysis: LLMAnalysis = Field(default_factory=LLMAnalysis)


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
# KPI Snapshot (built by report_node)
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
# Expert result model mapping (used by graph nodes for validation)
# ---------------------------------------------------------------------------

EXPERT_RESULT_MODELS: dict[str, type[AgentResult]] = {
    "prospect": ProspectResult,
    "conversion": ConversionResult,
    "subsidy": SubsidyResult,
    "retention": RetentionResult,
    "ad": AdResult,
}
