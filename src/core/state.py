"""LangGraph state definition."""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired, TypedDict


class AgentState(TypedDict):
    """Global state for the LangGraph workflow."""

    # Input
    query: str
    data_path: str
    budget: NotRequired[float]
    scope: NotRequired[str]

    # Sub-agent results
    prospect_results: NotRequired[dict[str, Any]]
    conversion_results: NotRequired[dict[str, Any]]
    subsidy_results: NotRequired[dict[str, Any]]
    retention_results: NotRequired[dict[str, Any]]
    ad_results: NotRequired[dict[str, Any]]

    # Experiment & attribution
    experiment_results: NotRequired[dict[str, Any]]

    # KPI & seasonal context
    kpi_snapshot: NotRequired[dict[str, Any]]
    seasonal_context: NotRequired[dict[str, Any]]

    # Aggregation
    analysis_summary: NotRequired[str]
    strategy_recommendation: NotRequired[str]
    report: NotRequired[str]

    # Metadata (accumulated via reducer)
    errors: Annotated[list[str], operator.add]
    metadata: Annotated[list[dict[str, Any]], operator.add]
