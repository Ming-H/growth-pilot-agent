"""KPI reconciliation gate node — the final node of both orchestration modes.

Runs after the report is produced: LLM-organised language is checked
against tool-computed facts. Violations degrade to the deterministic
template report (mode=on) or annotate (mode=annotate). Never calls an
LLM; ground truth = state ``kpi_snapshot`` ∪ L1 verified-fact ledger.

Memory write-back lives HERE (after the gate) so the episodic store only
ever receives a reconciled report.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from src.graph.state import AnalysisState, AnalysisStatus

logger = logging.getLogger(__name__)


async def kpi_gate_node(state: AnalysisState) -> dict[str, Any]:
    """KPI reconciliation gate + final memory write-back."""
    import time

    from src.core.config import get_settings
    from src.core.kpi_gate import run_gate

    started = time.monotonic()
    settings = get_settings()
    final_report = state.get("final_report") or ""

    # Extra ground truth: L1 verified-fact ledger values for this org.
    verified_values: list[float] = []
    try:
        from src.memory import MemoryStore

        store = MemoryStore(db_path=settings.memory_db_path)
        facts = await store.search_facts(state.get("org_id", "default"), limit=200)
        await store.close()
        verified_values = [f["value"] for f in facts if f.get("value") is not None]
    except Exception as exc:
        logger.debug("[kpi_gate] L1 ledger unavailable: %s", exc)

    # The deterministic template is the degradation target — render it
    # lazily only when needed (mode=on and violations exist).
    deterministic_report = ""
    if settings.kpi_gate == "on":
        try:
            from src.report.generator import ReportGenerator

            deterministic_report = ReportGenerator().generate_report(dict(state))
        except Exception as exc:
            logger.warning("[kpi_gate] template render failed: %s", exc)

    update = run_gate(
        final_report,
        dict(state),
        mode=settings.kpi_gate,
        threshold=settings.kpi_gate_threshold,
        deterministic_report=deterministic_report,
        verified_values=verified_values,
    )

    gated_report = update.get("final_report", final_report)
    kpi_snapshot = state.get("kpi_snapshot") or {}

    # ── Final memory write-back (post-gate) ──────────────────────────────
    analysis_id = f"run:{uuid.uuid4().hex[:12]}"
    try:
        from src.memory import MemoryStore

        store = MemoryStore(db_path=settings.memory_db_path)
        org_id = state.get("org_id", "default")
        await store.save_analysis(
            analysis_id=analysis_id,
            org_id=org_id,
            query=state.get("query", ""),
            scope=state.get("scope") or "full",
            report=gated_report,
            kpis=kpi_snapshot,
            session_id=state.get("session_id"),
        )
        for r in state.get("expert_results", []):
            await store.record_facts(org_id, r, run_fingerprint=analysis_id)
        if state.get("session_id"):
            await store.save_turn(
                state["session_id"],
                org_id=org_id,
                user_id=state.get("user_id", "anonymous"),
                role="assistant",
                content=gated_report,
                structured={"kpis": kpi_snapshot},
            )
        await store.close()
    except Exception as exc:
        logger.debug("[kpi_gate] memory write-back unavailable: %s", exc)

    result = update.get("kpi_gate_result") or {}
    dur_ms = round((time.monotonic() - started) * 1000)
    update["trace"] = [
        {
            "node": "kpi_gate",
            "dur_ms": dur_ms,
            "detail": (
                f"claims={result.get('claims', 0)}"
                f" violations={len(result.get('violations', []))}"
                f" mode={settings.kpi_gate}"
            ),
        }
    ]
    if result.get("ok") is False:
        logger.warning(
            "[kpi_gate] %d/%d claims untraceable — mode=%s applied",
            len(result.get("violations", [])),
            result.get("claims", 0),
            settings.kpi_gate,
        )
    else:
        logger.info("[kpi_gate] pass (%d claims reconciled, %dms)", result.get("claims", 0), dur_ms)
    update.setdefault("status", AnalysisStatus.COMPLETED)
    return update
