"""API Pydantic models for GrowthPilot Agent REST API.

Request / response schemas used by the FastAPI endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.core import __version__


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Request body for /api/v1/analyze and /api/v1/analyze/stream."""

    query: str = Field(default="", description="User query for growth analysis")
    scope: str = Field(default="full", description="Analysis scope: full|prospect|conversion|subsidy|retention|ad|inapp")
    data_path: str = Field(default="data/", description="Path to data directory")
    budget: float = Field(default=0.0, ge=0.0, description="Available budget")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AnalysisResponse(BaseModel):
    """Synchronous analysis result."""

    success: bool = True
    scope: str = "full"
    analysis_summary: str = ""
    strategy_recommendation: str = ""
    results: dict[str, Any] = Field(default_factory=dict)
    kpi_snapshot: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)
    agents_run: list[str] = Field(default_factory=list)


class AnalysisPersistedResponse(BaseModel):
    """Analysis result with DB persistence metadata."""

    id: str
    status: str
    scope: str
    kpi_snapshot: dict[str, Any] | None = None
    strategy_recommendation: str | None = None
    analysis_summary: str | None = None
    result: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)
    agents_run: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    created_at: str = ""
    completed_at: str | None = None


class AnalysisListResponse(BaseModel):
    """Paginated list of analyses."""

    items: list[AnalysisPersistedResponse] = Field(default_factory=list)
    total: int = 0
    limit: int = 20
    offset: int = 0


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = __version__
    agents: list[str] = Field(
        default_factory=lambda: [
            "orchestrator",
            "prospect",
            "conversion",
            "subsidy",
            "retention",
            "ad",
        ]
    )


class MemoryEntryResponse(BaseModel):
    """Single memory entry."""

    id: str = ""
    run_id: str = ""
    query: str = ""
    scope: str = ""
    results_summary: Any = None
    timestamp: float = 0.0


class MemoryResponse(BaseModel):
    """Memory list response."""

    memories: list[MemoryEntryResponse] = Field(default_factory=list)
    total: int = 0


class MemoryClearResponse(BaseModel):
    """Response after clearing memory."""

    removed_count: int = 0
    message: str = ""


# ---------------------------------------------------------------------------
# SSE event data model
# ---------------------------------------------------------------------------

class EventData(BaseModel):
    """SSE event payload for streaming analysis."""

    event_type: str = ""  # started | running | completed | failed | result
    agent: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
