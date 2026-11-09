"""Analysis endpoints: sync analyze, SSE stream, list/get, HITL state/approve/resume.

v9 fixes baked in:
- A1: org_id/user_id thread through run_workflow (multi-tenant memory).
- A2: SSE dispatches on the SAME mode resolution as the sync path.
- A3: HITL thread discipline — runs, /state, /approve, /resume all use
  ``thread_id_for(org_id, analysis_id)``.
- A4/A5: responses and DB persistence read the compat contract
  (metadata.cost_usd, kpi_snapshot), never raw graph state.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
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
    ResumeResponse,
)
from src.auth.dependencies import get_current_user
from src.core.config import get_settings
from src.db.database import get_db, get_session_factory
from src.db.models import Analysis, Organization, User
from src.graph.workflow import run_workflow
from src.web.deps import get_graph_for, thread_id_for
from src.web.rate_limit import limit_or_pass

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analyses"])

# Rate limiting — shared limiter (C1: a local Limiter instance plus a
# decorator that was never applied = dead config). slowapi requires the
# Request parameter to be literally named ``request``, hence the rename.
_limit_analyze = limit_or_pass("10/minute")


class AnalysisError(Exception):
    """Raised when the analysis workflow fails."""

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compat_to_response(compat: dict[str, Any], request: AnalyzeRequest) -> AnalysisResponse:
    """Map the compat result contract → AnalysisResponse."""
    agents_run = [
        item.get("expert", "")
        for item in compat.get("expert_results", {}).values()
        if item.get("expert")
    ]
    results = {f"{k}_results": v for k, v in compat.get("expert_results", {}).items()}
    return AnalysisResponse(
        success=compat.get("success", False),
        scope=compat.get("scope", "full"),
        analysis_summary=compat.get("analysis_summary", ""),
        strategy_recommendation=compat.get("strategy_recommendation", ""),
        results=results,
        kpi_snapshot=compat.get("kpi_snapshot"),
        errors=compat.get("errors", []),
        agents_run=agents_run,
    )


def _analysis_to_response(a: Analysis) -> AnalysisPersistedResponse:
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


async def _enforce_quota(db: AsyncSession, user: User) -> Organization | None:
    org = await db.get(Organization, user.org_id)
    if org and org.monthly_quota > 0 and org.usage_count >= org.monthly_quota:
        raise HTTPException(status_code=429, detail="Monthly analysis quota exceeded")
    return org


# ---------------------------------------------------------------------------
# Sync analyze
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=AnalysisPersistedResponse)
@_limit_analyze
async def analyze(
    request: Request,
    body: AnalyzeRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AnalysisPersistedResponse:
    """Run a growth analysis (mode from settings/request), persist, return."""
    org = await _enforce_quota(db, user)

    analysis = Analysis(
        user_id=user.id,
        org_id=user.org_id,
        query=body.query,
        scope=body.scope or "full",
        budget=body.budget,
        mode=body.mode or get_settings().orchestration_mode,
        status="running",
    )
    db.add(analysis)
    await db.flush()

    logger.info(
        "Analyze request: id=%s user=%s org=%s mode=%s query=%s",
        analysis.id,
        user.id,
        user.org_id,
        analysis.mode,
        body.query[:80],
    )

    start_time = time.monotonic()
    try:
        state = await run_workflow(
            query=body.query,
            data_path=body.data_path,
            budget=body.budget,
            scope=body.scope or "",
            mode=body.mode,
            org_id=user.org_id,
            user_id=user.id,
            approval_required=body.approval_required,
            thread_id=analysis.id,  # stable HITL thread
            session_id=body.session_id,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start_time
        analysis.status = "failed"
        analysis.error_message = str(exc)
        analysis.duration_seconds = elapsed
        analysis.completed_at = datetime.now(timezone.utc)
        await db.flush()
        logger.error("Workflow execution failed: %s", exc)
        raise AnalysisError(message="Analysis workflow failed", detail=str(exc)) from exc

    elapsed = time.monotonic() - start_time

    # Persist from the compat contract (A5: cost lives in metadata).
    # A11: paused (awaiting_approval) / rejected runs are NOT "completed" —
    # previously every run was marked completed, even with an empty report.
    compat_status = state.get("status", "")
    analysis.status = (
        "completed"
        if state.get("success")
        else ("awaiting_approval" if compat_status == "awaiting_approval" else "failed")
    )
    analysis.result = state
    analysis.kpi_snapshot = state.get("kpi_snapshot")
    analysis.strategy_recommendation = state.get("strategy_recommendation", "")
    analysis.duration_seconds = elapsed
    analysis.cost_usd = (state.get("metadata") or {}).get("cost_usd", 0.0)
    analysis.completed_at = datetime.now(timezone.utc)
    await db.flush()

    if org:
        org.usage_count += 1
    return _analysis_to_response(analysis)


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------


async def _persist_stream_status(
    analysis_id: str, *, status: str, error_message: str, elapsed: float
) -> None:
    """Best-effort outcome persistence for streaming runs (C4).

    Used from the generator's exception AND cancellation paths — a
    cancellation must also leave an honest row status behind.
    """
    try:
        session_factory = get_session_factory()
        async with session_factory() as update_db:
            result_db = await update_db.execute(select(Analysis).where(Analysis.id == analysis_id))
            analysis_obj = result_db.scalar_one_or_none()
            if analysis_obj:
                analysis_obj.status = status
                analysis_obj.error_message = error_message
                analysis_obj.duration_seconds = elapsed
                analysis_obj.completed_at = datetime.now(timezone.utc)
                await update_db.commit()
    except Exception as db_exc:
        logger.warning("Failed to persist stream status %s: %s", status, db_exc)


async def _get_org_analysis(db: AsyncSession, analysis_id: str, user: User) -> Analysis:
    """Fetch an analysis row scoped to the caller's org (C5).

    HITL endpoints previously read the row WITHOUT the org filter — the
    org-scoped thread_id kept the checkpoint private, but the row itself
    (mode, existence) leaked cross-org, and same-org members could act on
    analyses they did not create.
    """
    result = await db.execute(
        select(Analysis).where(Analysis.id == analysis_id, Analysis.org_id == user.org_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    return row


@router.post("/analyze/stream")
@_limit_analyze
async def analyze_stream(
    request: Request,
    body: AnalyzeRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> EventSourceResponse:
    """Run a growth analysis with SSE streaming.

    A2 fix: streams the graph for the RESOLVED mode (agent default) — the
    node→event map covers both topologies.
    """
    org = await _enforce_quota(db, user)

    analysis = Analysis(
        user_id=user.id,
        org_id=user.org_id,
        query=body.query,
        scope=body.scope or "full",
        budget=body.budget,
        mode=body.mode or get_settings().orchestration_mode,
        status="running",
    )
    db.add(analysis)
    await db.flush()
    analysis_id = analysis.id

    def _sse_event(agent_event: AgentEvent) -> dict:
        return {
            "event": "agent_event",
            "data": agent_event.model_dump_json(),
            "id": f"{analysis_id}-{agent_event.type}",
            "retry": 15000,
        }

    async def event_generator():
        plan_event = AgentEvent(
            type="plan",
            message="Analyzing your body...",
            progress=0.05,
            data={"run_id": analysis_id, "query": body.query},
        )
        yield _sse_event(plan_event)
        yield {
            "event": "message",
            "data": EventData(
                event_type="started",
                agent="orchestrator",
                data={"run_id": analysis_id, "query": body.query},
            ).model_dump_json(),
        }

        # Node→event map covering BOTH topologies (A2).
        _node_progress = {
            "plan": 0.1,
            "guard": 0.05,
            "agent": 0.2,
            "tools": 0.35,
            "execute": 0.3,
            "evaluate": 0.6,
            "refine": 0.7,
            "approval": 0.8,
            "report": 0.9,
            "kpi_gate": 0.95,
        }
        _node_type = {
            "plan": "plan",
            "guard": "plan",
            "agent": "execute",
            "tools": "execute",
            "execute": "execute",
            "evaluate": "evaluate",
            "refine": "evaluate",
            "approval": "approval",
            "report": "report",
            "kpi_gate": "report",
        }

        start_time = time.monotonic()
        settings = get_settings()
        mode = body.mode or settings.orchestration_mode
        compiled_graph = await get_graph_for(mode)
        initial_state = {
            "query": body.query,
            "scope": body.scope,
            "budget": body.budget,
            "org_id": user.org_id,
            "user_id": user.id,
            "session_id": body.session_id,
            "approval_required": body.approval_required,
            "expert_results": [],
            "execution_errors": [],
            "messages": [],
            "status": "pending",
        }
        config = {"configurable": {"thread_id": thread_id_for(user.org_id, analysis_id)}}

        try:
            async with asyncio.timeout(settings.sse_timeout_seconds):
                async for event in compiled_graph.astream_events(
                    initial_state, config=config, version="v2"
                ):
                    kind = event["event"]
                    name = event.get("name", "")
                    if kind in ("on_chain_start", "on_chain_end") and name in _node_progress:
                        phase = "start" if kind == "on_chain_start" else "end"
                        progress = _node_progress[name] + (0.05 if phase == "end" else 0)
                        yield _sse_event(
                            AgentEvent(
                                type=_node_type[name],
                                message=f"{phase.capitalize()} {name}",
                                progress=round(min(progress, 0.95), 2),
                                data={"node": name, "phase": phase},
                            )
                        )

            elapsed = time.monotonic() - start_time
            state_snapshot = await compiled_graph.aget_state(config)
            state = dict(state_snapshot.values) if state_snapshot and state_snapshot.values else {}

            # A4 fix: build the response from the compat contract, not raw
            # graph state keys.
            from src.graph.workflow import _build_compat_result

            compat = _build_compat_result(state, query=body.query, scope=body.scope or "")
            response = _compat_to_response(compat, body)
            yield _sse_event(
                AgentEvent(
                    type="complete",
                    message="Analysis complete",
                    progress=1.0,
                    data=response.model_dump(),
                )
            )
            yield {
                "event": "result",
                "data": EventData(
                    event_type="result", agent="orchestrator", data=response.model_dump()
                ).model_dump_json(),
            }

            session_factory = get_session_factory()
            async with session_factory() as update_db:
                result_db = await update_db.execute(
                    select(Analysis).where(Analysis.id == analysis_id)
                )
                analysis_obj = result_db.scalar_one_or_none()
                if analysis_obj:
                    analysis_obj.status = "completed"
                    analysis_obj.result = compat
                    analysis_obj.kpi_snapshot = compat.get("kpi_snapshot")
                    analysis_obj.duration_seconds = elapsed
                    analysis_obj.cost_usd = (compat.get("metadata") or {}).get("cost_usd", 0.0)
                    analysis_obj.completed_at = datetime.now(timezone.utc)
                    await update_db.commit()
            if org:
                org.usage_count += 1

        except asyncio.CancelledError:
            # C4: client disconnect cancels this generator — CancelledError
            # inherits BaseException and escapes `except Exception`, so the
            # row used to stay "running" forever. Persist the interruption
            # (the checkpoint survives — /resume can continue the run),
            # then let the cancellation propagate as required.
            _persist_stream_status(
                analysis_id,
                status="interrupted",
                error_message="SSE client disconnected; run interrupted (checkpoint retained)",
                elapsed=time.monotonic() - start_time,
            )
            raise
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            logger.error("Stream workflow failed: %s", exc)
            yield _sse_event(
                AgentEvent(
                    type="error",
                    message=f"Analysis failed: {exc}",
                    progress=0.0,
                    data={"error": str(exc)},
                )
            )
            _persist_stream_status(
                analysis_id, status="failed", error_message=str(exc), elapsed=elapsed
            )
            yield {
                "event": "error",
                "data": EventData(
                    event_type="failed", agent="orchestrator", data={"error": str(exc)}
                ).model_dump_json(),
            }

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# List / get
# ---------------------------------------------------------------------------


@router.get("/analyses", response_model=AnalysisListResponse)
async def list_analyses(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> AnalysisListResponse:
    count_q = select(func.count()).select_from(Analysis).where(Analysis.org_id == user.org_id)
    total = (await db.execute(count_q)).scalar() or 0
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
    result = await db.execute(
        select(Analysis).where(Analysis.id == analysis_id, Analysis.org_id == user.org_id)
    )
    analysis = result.scalar_one_or_none()
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return _analysis_to_response(analysis)


# ---------------------------------------------------------------------------
# Human-in-the-loop (A3: one thread-id discipline everywhere)
# ---------------------------------------------------------------------------


@router.get("/analyses/{analysis_id}/state", response_model=AnalysisStateResponse)
async def get_analysis_state(
    analysis_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AnalysisStateResponse:
    row = await _get_org_analysis(db, analysis_id, user)
    mode = row.mode or get_settings().orchestration_mode
    config = {"configurable": {"thread_id": thread_id_for(user.org_id, analysis_id)}}
    graph = await get_graph_for(mode)

    state_snapshot = await graph.aget_state(config)
    if state_snapshot is None or not state_snapshot.values:
        raise HTTPException(
            status_code=404, detail=f"No checkpoint state found for analysis {analysis_id}"
        )

    values = state_snapshot.values
    raw_status = values.get("status")
    status_str = raw_status.value if hasattr(raw_status, "value") else str(raw_status)
    return AnalysisStateResponse(
        analysis_id=analysis_id,
        status=status_str,
        plan=values.get("plan"),
        selected_experts=values.get("selected_experts"),
        subgoals=values.get("subgoals"),
        expert_results=values.get("expert_results"),
        quality_scores=values.get("quality_scores"),
        approved=values.get("approved"),
        approval_required=values.get("approval_required"),
        final_report=values.get("final_report"),
        degradations=values.get("degradations"),
        next_nodes=list(state_snapshot.next) if state_snapshot.next else [],
        raw_state=values,
    )


@router.post("/analyses/{analysis_id}/approve", response_model=ApprovalResponse)
async def approve_analysis(
    analysis_id: str,
    body: ApprovalRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ApprovalResponse:
    row = await _get_org_analysis(db, analysis_id, user)
    # C5: approve is only valid while the run is actually paused/pending —
    # without this guard two concurrent approves double-invoke the graph.
    if row.status not in {"running", "awaiting_approval"}:
        raise HTTPException(
            status_code=409,
            detail=f"Analysis is '{row.status}'; approve requires a running/awaiting_approval run",
        )
    mode = row.mode or get_settings().orchestration_mode
    config = {"configurable": {"thread_id": thread_id_for(user.org_id, analysis_id)}}
    graph = await get_graph_for(mode)

    state_snapshot = await graph.aget_state(config)
    if state_snapshot is None or not state_snapshot.values:
        raise HTTPException(
            status_code=404, detail=f"No checkpoint state found for analysis {analysis_id}"
        )

    from src.graph.state import AnalysisStatus

    if not body.approved:
        await graph.aupdate_state(config, {"approved": False, "status": AnalysisStatus.FAILED})
        if row:
            row.status = "rejected"
            await db.flush()
        logger.info("Analysis %s rejected by user %s", analysis_id, user.id)
        return ApprovalResponse(
            analysis_id=analysis_id, approved=False, status="failed", feedback=body.feedback
        )

    await graph.aupdate_state(config, {"approved": True, "status": AnalysisStatus.REPORTING})
    logger.info("Analysis %s approved by user %s, resuming", analysis_id, user.id)
    result = await graph.ainvoke(None, config)

    final_status = result.get("status") if result else None
    status_str = final_status.value if hasattr(final_status, "value") else str(final_status)
    if row:
        row.status = "completed" if status_str == "completed" else status_str or "completed"
        row.result = result
        row.completed_at = datetime.now(timezone.utc)
        await db.flush()
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
    db: AsyncSession = Depends(get_db),
) -> ResumeResponse:
    row = await _get_org_analysis(db, analysis_id, user)
    if row.status in {"completed", "rejected"}:
        raise HTTPException(
            status_code=409, detail=f"Analysis already '{row.status}' — nothing to resume"
        )
    mode = row.mode or get_settings().orchestration_mode
    config = {"configurable": {"thread_id": thread_id_for(user.org_id, analysis_id)}}
    graph = await get_graph_for(mode)

    state_snapshot = await graph.aget_state(config)
    if state_snapshot is None or not state_snapshot.values:
        raise HTTPException(
            status_code=404, detail=f"No checkpoint state found for analysis {analysis_id}"
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
