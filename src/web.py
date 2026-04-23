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
    AnalysisListResponse,
    AnalysisPersistedResponse,
    AnalysisResponse,
    AnalyzeRequest,
    EventData,
    HealthResponse,
    MemoryClearResponse,
    MemoryEntryResponse,
    MemoryResponse,
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals (initialized in lifespan)
# ---------------------------------------------------------------------------

_memory_manager: MemoryManager | None = None


def _get_memory() -> MemoryManager:
    """Return the singleton MemoryManager instance."""
    global _memory_manager
    if _memory_manager is None:
        settings = get_settings()
        _memory_manager = MemoryManager(base_path=settings.memory_base_path)
    return _memory_manager


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
async def analyze(
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

    Requires authentication. Yields real-time progress events as each agent
    starts/completes, followed by the final analysis result.
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

    async def event_generator():
        """Generate SSE events from the workflow execution."""
        # Yield initial event
        init_event = EventData(
            event_type="started",
            agent="orchestrator",
            data={"run_id": analysis_id, "query": request.query},
        )
        yield {"event": "message", "data": init_event.model_dump_json()}

        start_time = time.monotonic()
        try:
            state = await _run_analysis(request)
            elapsed = time.monotonic() - start_time

            # Stream intermediate events from state
            events = state.get("events", [])
            for evt in events:
                event_data = EventData(
                    event_type=evt.get("status", "running"),
                    agent=evt.get("agent", ""),
                    data=evt,
                )
                yield {"event": "message", "data": event_data.model_dump_json()}

            # Build and yield final result
            response = _build_analysis_response(state, analysis_id, request)
            final_event = EventData(
                event_type="result",
                agent="orchestrator",
                data=response.model_dump(),
            )
            yield {"event": "result", "data": final_event.model_dump_json()}

            # Update analysis record with results
            # We need a fresh session since the outer one may be closed
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

            error_event = EventData(
                event_type="failed",
                agent="orchestrator",
                data={"error": str(exc)},
            )
            yield {"event": "error", "data": error_event.model_dump_json()}

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
