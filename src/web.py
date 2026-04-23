"""FastAPI web application for GrowthPilot Agent API.

Provides RESTful endpoints for auth, synchronous analysis, SSE streaming,
health checks, and analysis persistence with multi-tenant org isolation.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from src.api_models import (
    AgentEvent,
    AnalysisListResponse,
    AnalysisPersistedResponse,
    AnalysisResponse,
    AnalysisStateResponse,
    AnalyzeRequest,
    ApprovalRequest,
    ApprovalResponse,
    EventData,
    HealthResponse,
    MemoryClearResponse,
    MemoryEntryResponse,
    MemoryResponse,
    ResumeResponse,
)
from src.auth.dependencies import get_current_user
from src.auth.jwt import create_access_token
from src.auth.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from src.auth.security import hash_password, verify_password
from src.core import __version__
from src.core.config import get_settings
from src.core.observability import setup_telemetry
from src.db.database import get_db, init_db
from src.db.models import Analysis, Organization, User
from src.memory.manager import MemoryManager

# Rate limiting (optional -- graceful degradation if slowapi is not installed)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address)
    _RATE_LIMITING_AVAILABLE = True
except ImportError:  # pragma: no cover
    limiter = None  # type: ignore[assignment]
    _rate_limit_exceeded_handler = None  # type: ignore[assignment]
    RateLimitExceeded = None  # type: ignore[assignment, misc]
    _RATE_LIMITING_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals (initialized in lifespan)
# ---------------------------------------------------------------------------

_memory_manager: MemoryManager | None = None

# Cached compiled graph for human-in-the-loop endpoints.
# Built lazily on first access so that the import-time cost is deferred.
_compiled_graph: Any | None = None


def _get_memory() -> MemoryManager:
    """Return the singleton MemoryManager instance."""
    global _memory_manager
    if _memory_manager is None:
        settings = get_settings()
        _memory_manager = MemoryManager(base_path=settings.memory_base_path)
    return _memory_manager


async def _get_graph() -> Any:
    """Return the singleton compiled LangGraph instance (lazy, cached).

    Calls :func:`build_compiled_graph` once and caches the result for the
    lifetime of the process.  This avoids rebuilding the graph (and
    re-initialising the checkpointer) on every request.
    """
    global _compiled_graph
    if _compiled_graph is None:
        from src.graph.graph import build_compiled_graph

        _compiled_graph = await build_compiled_graph()
        logger.info("Compiled LangGraph initialised and cached")
    return _compiled_graph


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Application lifespan: startup and shutdown hooks."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    )
    logger.info("GrowthPilot API starting (version=%s)", app.version)

    # Initialize OpenTelemetry tracing
    setup_telemetry(settings.otel_service_name, enabled=settings.otel_enabled)

    # Initialize database tables
    try:
        await init_db()
        logger.info("Database tables initialized")
    except Exception as exc:
        logger.warning("Database init failed (tables may already exist): %s", exc)

    yield
    logger.info("GrowthPilot API shutting down")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1")


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class AnalysisError(Exception):
    """Raised when the analysis workflow fails."""

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------


@router.post("/auth/register", response_model=TokenResponse, status_code=201)
async def register(
    req: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Create a new user and organization, return JWT token."""
    # Check email uniqueness
    existing = await db.execute(select(User).where(User.email == req.email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail="Email already registered")

    # Create organization
    org_name = req.org_name or f"{req.name or req.email.split('@')[0]}'s Org"
    org = Organization(name=org_name)
    db.add(org)
    await db.flush()

    # Create user with hashed password
    user = User(
        email=req.email,
        hashed_password=hash_password(req.password),
        name=req.name,
        org_id=org.id,
        role="owner",
    )
    db.add(user)
    await db.flush()

    # Generate JWT
    token = create_access_token(
        data={"sub": user.id, "org_id": org.id, "role": user.role}
    )

    return TokenResponse(
        access_token=token,
        user_id=user.id,
        org_id=org.id,
        role=user.role,
    )


@router.post("/auth/login", response_model=TokenResponse)
async def login(
    req: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Authenticate user and return JWT token."""
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    token = create_access_token(
        data={"sub": user.id, "org_id": user.org_id, "role": user.role}
    )

    return TokenResponse(
        access_token=token,
        user_id=user.id,
        org_id=user.org_id,
        role=user.role,
    )


@router.get("/auth/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user)) -> UserResponse:
    """Return current authenticated user info."""
    return UserResponse.model_validate(user)


# ---------------------------------------------------------------------------
# Helper: run workflow and build response
# ---------------------------------------------------------------------------


async def _run_analysis(request: AnalyzeRequest) -> dict[str, Any]:
    """Execute the Chief Agent workflow and return the final state."""
    from src.graph.workflow import run_workflow

    result = await run_workflow(
        query=request.query,
        data_path=request.data_path,
        budget=request.budget,
        scope=request.scope or "",
    )
    return result


def _build_analysis_response(
    state: dict[str, Any],
    run_id: str,
    request: AnalyzeRequest,
) -> AnalysisResponse:
    """Convert workflow state to API response model."""
    scope = state.get("scope", request.scope or "full")
    errors = state.get("errors", [])

    # Collect sub-agent results
    results: dict[str, Any] = {}
    for key in (
        "prospect_results",
        "conversion_results",
        "subsidy_results",
        "retention_results",
        "ad_results",
    ):
        val = state.get(key)
        if val is not None:
            results[key] = val

    # KPI snapshot
    kpi_snapshot = state.get("kpi_snapshot")

    # Track which agents ran
    agents_run: list[str] = []
    for key in (
        "prospect_results",
        "conversion_results",
        "subsidy_results",
        "retention_results",
        "ad_results",
    ):
        if state.get(key) is not None:
            agent_name = key.replace("_results", "")
            agents_run.append(agent_name)

    return AnalysisResponse(
        success=len(errors) == 0,
        scope=scope,
        analysis_summary=state.get("analysis_summary", ""),
        strategy_recommendation=state.get("strategy_recommendation", ""),
        results=results,
        kpi_snapshot=kpi_snapshot,
        errors=errors,
        agents_run=agents_run,
    )


def _analysis_to_response(a: Analysis) -> AnalysisPersistedResponse:
    """Convert an Analysis ORM object to an API response."""
    return AnalysisPersistedResponse(
        id=a.id,
        status=a.status,
        scope=a.scope,
        kpi_snapshot=a.kpi_snapshot,
        strategy_recommendation=a.strategy_recommendation,
        analysis_summary=(a.result or {}).get("analysis_summary") if a.result else None,
        result=a.result,
        errors=(a.result or {}).get("errors", []) if a.result else [],
        agents_run=(a.result or {}).get("agents_run", []) if a.result else [],
        duration_seconds=a.duration_seconds,
        cost_usd=a.cost_usd,
        created_at=a.created_at.isoformat() if a.created_at else "",
        completed_at=a.completed_at.isoformat() if a.completed_at else None,
    )


# ---------------------------------------------------------------------------
# Analysis endpoints (all require auth)
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=AnalysisPersistedResponse)
@limiter.limit("10/minute")
async def analyze(
    http_request: Request,
    request: AnalyzeRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AnalysisPersistedResponse:
    """Run a growth analysis, persist results to DB, and return.

    Requires authentication. Analysis is scoped to the user's org.
    """
    # Create Analysis record in running state
    analysis = Analysis(
        user_id=user.id,
        org_id=user.org_id,
        query=request.query,
        scope=request.scope or "full",
        budget=request.budget,
        status="running",
    )
    db.add(analysis)
    await db.flush()

    logger.info(
        "Analyze request: id=%s user=%s org=%s query=%s scope=%s",
        analysis.id, user.id, user.org_id, request.query, request.scope,
    )

    start_time = time.monotonic()
    try:
        state = await _run_analysis(request)
    except Exception as exc:
        # Mark analysis as failed
        elapsed = time.monotonic() - start_time
        analysis.status = "failed"
        analysis.error_message = str(exc)
        analysis.duration_seconds = elapsed
        analysis.completed_at = datetime.now(timezone.utc)
        await db.flush()
        logger.error("Workflow execution failed: %s", exc)
        raise AnalysisError(
            message="Analysis workflow failed",
            detail=str(exc),
        ) from exc

    elapsed = time.monotonic() - start_time

    # Update analysis record with results
    analysis.status = "completed"
    analysis.result = state
    analysis.kpi_snapshot = state.get("kpi_snapshot")
    analysis.strategy_recommendation = state.get("strategy_recommendation")
    analysis.duration_seconds = elapsed
    analysis.cost_usd = state.get("cost_usd", 0.0)
    analysis.completed_at = datetime.now(timezone.utc)
    await db.flush()

    # Also store in memory for backward compatibility
    try:
        memory = _get_memory()
        scope = state.get("scope", "full")
        summary = state.get("analysis_summary", "")
        memory.store(
            run_id=analysis.id,
            query=request.query,
            scope=scope,
            results_summary=summary,
        )
    except Exception as exc:
        logger.warning("Failed to store analysis in memory: %s", exc)

    return _analysis_to_response(analysis)


@router.post("/analyze/stream")
async def analyze_stream(
    request: AnalyzeRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> EventSourceResponse:
    """Run a growth analysis with SSE streaming.

    Requires authentication. Yields structured ``agent_event`` SSE events with
    stage tracking (plan/execute/evaluate/approval/report) and progress.  Also
    emits legacy ``message`` / ``result`` / ``error`` events for backward
    compatibility with older consumers.
    """
    # Create Analysis record in running state
    analysis = Analysis(
        user_id=user.id,
        org_id=user.org_id,
        query=request.query,
        scope=request.scope or "full",
        budget=request.budget,
        status="running",
    )
    db.add(analysis)
    await db.flush()
    analysis_id = analysis.id

    logger.info(
        "Stream request: id=%s user=%s org=%s query=%s",
        analysis_id, user.id, user.org_id, request.query,
    )

    def _sse_event(agent_event: AgentEvent) -> dict:
        """Build an SSE dict for a structured AgentEvent."""
        return {
            "event": "agent_event",
            "data": agent_event.model_dump_json(),
            "id": f"{analysis_id}-{agent_event.type}",
            "retry": 15000,
        }

    async def event_generator():
        """Generate SSE events from the workflow execution."""
        # ── Plan stage ──────────────────────────────────────────────────
        plan_event = AgentEvent(
            type="plan",
            message="Analyzing your request...",
            progress=0.1,
            data={"run_id": analysis_id, "query": request.query},
        )
        yield _sse_event(plan_event)
        # Legacy compat — old consumers expect a ``message`` event
        yield {
            "event": "message",
            "data": EventData(
                event_type="started",
                agent="orchestrator",
                data={"run_id": analysis_id, "query": request.query},
            ).model_dump_json(),
        }

        start_time = time.monotonic()
        try:
            state = await _run_analysis(request)
            elapsed = time.monotonic() - start_time

            # ── Execute stage — emit per-expert events ─────────────────
            selected_experts = list(state.get("agents_run", []))
            if not selected_experts:
                # Derive from result keys
                for key in (
                    "prospect_results",
                    "conversion_results",
                    "subsidy_results",
                    "retention_results",
                    "ad_results",
                ):
                    if state.get(key) is not None:
                        selected_experts.append(key.replace("_results", ""))

            expert_count = max(len(selected_experts), 1)
            for idx, expert_name in enumerate(selected_experts):
                progress = 0.2 + (0.4 * (idx / expert_count))
                exec_event = AgentEvent(
                    type="execute",
                    expert=expert_name,
                    message=f"Running {expert_name} analysis...",
                    progress=round(progress, 2),
                    data={"expert": expert_name, "index": idx, "total": expert_count},
                )
                yield _sse_event(exec_event)

            # Stream intermediate events from state (if provided by workflow)
            for evt in state.get("events", []):
                progress_event = AgentEvent(
                    type="progress",
                    expert=evt.get("agent"),
                    message=evt.get("message", ""),
                    progress=0.6,
                    data=evt,
                )
                yield _sse_event(progress_event)

            # ── Evaluate stage ──────────────────────────────────────────
            eval_event = AgentEvent(
                type="evaluate",
                message="Evaluating quality...",
                progress=0.7,
                data={"quality_scores": state.get("quality_scores", {})},
            )
            yield _sse_event(eval_event)

            # ── Approval stage ──────────────────────────────────────────
            approval_event = AgentEvent(
                type="approval",
                message="Waiting for approval...",
                progress=0.8,
                data={"approved": state.get("approved", True)},
            )
            yield _sse_event(approval_event)

            # ── Report stage ────────────────────────────────────────────
            report_event = AgentEvent(
                type="report",
                message="Generating report...",
                progress=0.9,
                data={},
            )
            yield _sse_event(report_event)

            # ── Complete stage — final result ───────────────────────────
            response = _build_analysis_response(state, analysis_id, request)
            complete_event = AgentEvent(
                type="complete",
                message="Analysis complete",
                progress=1.0,
                data=response.model_dump(),
            )
            yield _sse_event(complete_event)

            # Legacy compat — emit old-style result event
            final_event = EventData(
                event_type="result",
                agent="orchestrator",
                data=response.model_dump(),
            )
            yield {"event": "result", "data": final_event.model_dump_json()}

            # Update analysis record with results
            from src.db.database import get_session_factory
            session_factory = get_session_factory()
            async with session_factory() as update_db:
                result_db = await update_db.execute(
                    select(Analysis).where(Analysis.id == analysis_id)
                )
                analysis_obj = result_db.scalar_one_or_none()
                if analysis_obj:
                    analysis_obj.status = "completed"
                    analysis_obj.result = state
                    analysis_obj.kpi_snapshot = state.get("kpi_snapshot")
                    analysis_obj.strategy_recommendation = state.get("strategy_recommendation")
                    analysis_obj.duration_seconds = elapsed
                    analysis_obj.cost_usd = state.get("cost_usd", 0.0)
                    analysis_obj.completed_at = datetime.now(timezone.utc)
                    await update_db.commit()

            # Store in memory for backward compatibility
            try:
                memory = _get_memory()
                scope = state.get("scope", "full")
                summary = state.get("analysis_summary", "")
                memory.store(
                    run_id=analysis_id,
                    query=request.query,
                    scope=scope,
                    results_summary=summary,
                )
            except Exception as exc:
                logger.warning("Memory store failed in stream: %s", exc)

        except Exception as exc:
            elapsed = time.monotonic() - start_time
            logger.error("Stream workflow failed: %s", exc)

            # Emit structured error event
            error_event = AgentEvent(
                type="error",
                message=f"Analysis failed: {exc}",
                progress=0.0,
                data={"error": str(exc)},
            )
            yield _sse_event(error_event)

            # Mark analysis as failed
            try:
                from src.db.database import get_session_factory
                session_factory = get_session_factory()
                async with session_factory() as update_db:
                    result_db = await update_db.execute(
                        select(Analysis).where(Analysis.id == analysis_id)
                    )
                    analysis_obj = result_db.scalar_one_or_none()
                    if analysis_obj:
                        analysis_obj.status = "failed"
                        analysis_obj.error_message = str(exc)
                        analysis_obj.duration_seconds = elapsed
                        analysis_obj.completed_at = datetime.now(timezone.utc)
                        await update_db.commit()
            except Exception as db_exc:
                logger.warning("Failed to mark stream analysis as failed: %s", db_exc)

            # Legacy compat error event
            legacy_error = EventData(
                event_type="failed",
                agent="orchestrator",
                data={"error": str(exc)},
            )
            yield {"event": "error", "data": legacy_error.model_dump_json()}

    return EventSourceResponse(event_generator())


@router.get("/analyses", response_model=AnalysisListResponse)
async def list_analyses(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> AnalysisListResponse:
    """List analyses for the user's organization (paginated)."""
    # Count total
    count_q = select(func.count()).select_from(Analysis).where(
        Analysis.org_id == user.org_id
    )
    total = (await db.execute(count_q)).scalar() or 0

    # Fetch page
    q = (
        select(Analysis)
        .where(Analysis.org_id == user.org_id)
        .order_by(Analysis.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(q)
    analyses = result.scalars().all()

    return AnalysisListResponse(
        items=[_analysis_to_response(a) for a in analyses],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/analyses/{analysis_id}", response_model=AnalysisPersistedResponse)
async def get_analysis(
    analysis_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AnalysisPersistedResponse:
    """Get a specific analysis by ID, scoped to the user's org."""
    result = await db.execute(
        select(Analysis).where(
            Analysis.id == analysis_id,
            Analysis.org_id == user.org_id,
        )
    )
    analysis = result.scalar_one_or_none()

    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return _analysis_to_response(analysis)


# ---------------------------------------------------------------------------
# Human-in-the-loop endpoints (require auth)
# ---------------------------------------------------------------------------


@router.get("/analyses/{analysis_id}/state", response_model=AnalysisStateResponse)
async def get_analysis_state(
    analysis_id: str,
    user: User = Depends(get_current_user),
) -> AnalysisStateResponse:
    """Get the current analysis state from the LangGraph checkpoint.

    Reads the checkpoint state for the given thread_id (analysis_id) and
    returns key fields such as status, plan, selected_experts, and the
    list of next nodes the graph would execute.
    """
    config = {"configurable": {"thread_id": analysis_id}}
    graph = await _get_graph()

    state_snapshot = await graph.aget_state(config)

    if state_snapshot is None or state_snapshot.values is None or not state_snapshot.values:
        raise HTTPException(
            status_code=404,
            detail=f"No checkpoint state found for analysis {analysis_id}",
        )

    values = state_snapshot.values
    # status may be an enum; coerce to its value string
    raw_status = values.get("status")
    status_str = raw_status.value if hasattr(raw_status, "value") else str(raw_status)

    return AnalysisStateResponse(
        analysis_id=analysis_id,
        status=status_str,
        plan=values.get("plan"),
        selected_experts=values.get("selected_experts"),
        expert_results=values.get("expert_results"),
        quality_scores=values.get("quality_scores"),
        approved=values.get("approved"),
        approval_required=values.get("approval_required"),
        final_report=values.get("final_report"),
        next_nodes=list(state_snapshot.next) if state_snapshot.next else [],
        raw_state=values,
    )


@router.post("/analyses/{analysis_id}/approve", response_model=ApprovalResponse)
async def approve_analysis(
    analysis_id: str,
    body: ApprovalRequest,
    user: User = Depends(get_current_user),
) -> ApprovalResponse:
    """Submit a human approval decision for a paused analysis.

    If approved is False the analysis is marked as rejected (FAILED) and the
    graph is *not* resumed.  If approved is True the state is updated and the
    graph execution resumes from the approval checkpoint.
    """
    config = {"configurable": {"thread_id": analysis_id}}
    graph = await _get_graph()

    # Verify a checkpoint exists
    state_snapshot = await graph.aget_state(config)
    if state_snapshot is None or state_snapshot.values is None or not state_snapshot.values:
        raise HTTPException(
            status_code=404,
            detail=f"No checkpoint state found for analysis {analysis_id}",
        )

    if not body.approved:
        # Rejection path: update state and return without resuming
        from src.graph.state import AnalysisStatus

        await graph.aupdate_state(
            config,
            {
                "approved": False,
                "status": AnalysisStatus.FAILED,
            },
        )
        logger.info(
            "Analysis %s rejected by user %s (feedback: %s)",
            analysis_id, user.id, body.feedback,
        )
        return ApprovalResponse(
            analysis_id=analysis_id,
            approved=False,
            status="failed",
            feedback=body.feedback,
        )

    # Approval path: update state and resume graph execution
    from src.graph.state import AnalysisStatus

    await graph.aupdate_state(
        config,
        {
            "approved": True,
            "status": AnalysisStatus.REPORTING,
        },
    )

    logger.info(
        "Analysis %s approved by user %s, resuming execution",
        analysis_id, user.id,
    )

    # Resume the graph from the approval checkpoint (None = continue from interrupt)
    result = await graph.ainvoke(None, config)

    # result is the final state dict after the graph completes
    final_status = result.get("status") if result else None
    status_str = final_status.value if hasattr(final_status, "value") else str(final_status)

    return ApprovalResponse(
        analysis_id=analysis_id,
        approved=True,
        status=status_str or "completed",
        feedback=body.feedback,
        result=result,
    )


@router.post("/analyses/{analysis_id}/resume", response_model=ResumeResponse)
async def resume_analysis(
    analysis_id: str,
    user: User = Depends(get_current_user),
) -> ResumeResponse:
    """Resume a paused/interrupted analysis from its last checkpoint.

    Invokes the graph with ``None`` input so that LangGraph picks up from
    the point where it was interrupted (e.g. the approval node).
    """
    config = {"configurable": {"thread_id": analysis_id}}
    graph = await _get_graph()

    # Verify a checkpoint exists
    state_snapshot = await graph.aget_state(config)
    if state_snapshot is None or state_snapshot.values is None or not state_snapshot.values:
        raise HTTPException(
            status_code=404,
            detail=f"No checkpoint state found for analysis {analysis_id}",
        )

    logger.info("Resuming analysis %s from checkpoint", analysis_id)

    result = await graph.ainvoke(None, config)

    final_status = result.get("status") if result else None
    status_str = final_status.value if hasattr(final_status, "value") else str(final_status)

    return ResumeResponse(
        analysis_id=analysis_id,
        status=status_str or "completed",
        final_report=result.get("final_report") if result else None,
        result=result,
    )


# ---------------------------------------------------------------------------
# Health endpoint (no auth)
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


# ---------------------------------------------------------------------------
# Memory endpoints (no auth, backward compatible)
# ---------------------------------------------------------------------------


@router.get("/memory", response_model=MemoryResponse)
async def get_memory(limit: int = 20) -> MemoryResponse:
    """Retrieve recent memory entries."""
    memory = _get_memory()
    entries = memory.get_recent(limit=limit)
    memories = [
        MemoryEntryResponse(
            id=entry.get("id", ""),
            run_id=entry.get("run_id", ""),
            query=entry.get("query", ""),
            scope=entry.get("scope", ""),
            results_summary=entry.get("results_summary"),
            timestamp=entry.get("timestamp", 0.0),
        )
        for entry in entries
    ]
    return MemoryResponse(memories=memories, total=memory.count())


@router.delete("/memory", response_model=MemoryClearResponse)
async def clear_memory() -> MemoryClearResponse:
    """Clear all memory entries."""
    memory = _get_memory()
    count = memory.clear()
    return MemoryClearResponse(
        removed_count=count,
        message=f"Cleared {count} memory entries",
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="GrowthPilot Agent API",
        version=__version__,
        description="Multi-Agent user growth analysis platform",
        lifespan=lifespan,
    )

    # CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting (only if slowapi is available)
    if _RATE_LIMITING_AVAILABLE and limiter is not None:
        application.state.limiter = limiter
        application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # Register router
    application.include_router(router)

    # Exception handlers
    @application.exception_handler(AnalysisError)
    async def analysis_error_handler(
        request: Request, exc: AnalysisError
    ) -> JSONResponse:
        """Handle AnalysisError -> 500 with structured JSON."""
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": exc.message,
                "detail": exc.detail,
            },
        )

    @application.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Handle HTTPException with consistent JSON envelope."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.detail,
                "status_code": exc.status_code,
            },
        )

    @application.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler for unhandled exceptions."""
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": str(exc),
            },
        )

    return application


app = create_app()


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.web:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
