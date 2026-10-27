"""LangGraph node functions for the GrowthPilot StateGraph DAG.

Each node is a pure function that receives the current AnalysisState and
returns a partial state update dict.  The graph builder (src/graph/graph.py)
wires these nodes together with conditional edges.

Node contract:
    Input:  AnalysisState (full state, read-only)
    Output: dict with a subset of AnalysisState keys to update
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from src.graph.state import AnalysisState, AnalysisStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic result validation
# ---------------------------------------------------------------------------


def _validate_expert_result(expert_key: str, result: dict[str, Any]) -> dict[str, Any]:
    """Validate expert result against its Pydantic model.

    Validates at the graph boundary to ensure type safety between agents.
    Hook-injected metadata (keys with a ``_`` prefix, e.g. ``_trace``) is
    stripped before validation and re-attached afterwards — Pydantic treats
    underscore-prefixed names as private attributes.
    On validation failure, logs a warning but does not block the pipeline.
    """
    from src.core.models import EXPERT_RESULT_MODELS as _RESULT_MODELS

    model_cls = _RESULT_MODELS.get(expert_key)
    if not model_cls:
        return result

    private_meta = {k: v for k, v in result.items() if k.startswith("_")}
    public = {k: v for k, v in result.items() if not k.startswith("_")}

    try:
        validated = model_cls(**public)
        validated_dict = validated.model_dump(exclude_none=True)
        validated_dict["expert"] = expert_key
        validated_dict["_validated"] = True
        return {**validated_dict, **private_meta}
    except Exception as exc:
        logger.warning("[validate] Pydantic validation failed for %s: %s", expert_key, exc)
        return result


# Note: QUALITY_THRESHOLD and BUDGET_APPROVAL_THRESHOLD moved to config.py

PLAN_PROMPT = """\
You are the Chief Agent of a growth analytics platform.  Given the user query
below, decide which domain experts should be invoked to produce a comprehensive
analysis.

Available experts (pick one or more):
- prospect (用户获取): user scoring, segmentation, LTV prediction, intent modeling
- conversion (转化优化): funnel analysis, coupon design, reach planning, slot allocation
- subsidy (补贴策略): causal inference for subsidy ATE, elasticity estimation, budget optimization
- retention (用户留存): churn prediction, nurture planning, cohort analysis, win-back strategy
- ad (广告投放): RTA strategy, bid optimization, creative analysis, audience targeting

User query: {query}
Scope hint: {scope}

Reply with ONLY a JSON object:
{{"reasoning": "<brief reasoning>", "experts": ["<expert_key>", ...],
"context_summary": "<one-line summary>"}}
"""


# ---------------------------------------------------------------------------
# Helper: instantiate an expert by key
# ---------------------------------------------------------------------------


def _get_expert_cls(expert_key: str) -> type:
    """Lazy-import and return the ExpertAgentBase subclass for *expert_key*."""
    import_map: dict[str, str] = {
        "prospect": "src.agents.prospect.ProspectExpert",
        "conversion": "src.agents.conversion.ConversionExpert",
        "subsidy": "src.agents.subsidy.SubsidyExpert",
        "retention": "src.agents.retention.RetentionExpert",
        "ad": "src.agents.ad.AdExpert",
    }
    if expert_key not in import_map:
        raise ValueError(f"Unknown expert key: {expert_key}")
    module_path, class_name = import_map[expert_key].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


_expert_cache: dict[str, Any] = {}
_cache_ttl: float = 300.0  # seconds
_cache_timestamps: dict[str, float] = {}


def _clear_expert_cache() -> None:
    """Clear expert cache (for testing and config changes)."""
    _expert_cache.clear()
    _cache_timestamps.clear()


def _create_expert(expert_key: str) -> Any:
    """Create (or retrieve cached) expert instance with TTL-based expiration.

    When no LLM is configured the expert gets ``llm=None`` and runs the
    deterministic pipeline with a degraded summary — the zero-key brain.
    """
    import time

    now = time.monotonic()
    cached_time = _cache_timestamps.get(expert_key, 0.0)
    if expert_key in _expert_cache and (now - cached_time) < _cache_ttl:
        return _expert_cache[expert_key]

    from src.core.config import get_settings
    from src.core.llm_factory import create_resilient_llm

    llm = create_resilient_llm(role="default") if get_settings().llm_available else None
    cls = _get_expert_cls(expert_key)
    expert = cls(llm=llm)
    _expert_cache[expert_key] = expert
    _cache_timestamps[expert_key] = now
    return expert


# ═══════════════════════════════════════════════════════════════════════════
# Node implementations
# ═══════════════════════════════════════════════════════════════════════════


async def plan_node(state: AnalysisState) -> dict:  # noqa: D401
    """Chief Agent planning: parse the user query and select expert agents.

    Reads ``query``, ``scope``, and ``budget`` from the state.  Uses an LLM
    to decompose the query into an ExecutionPlan and picks the experts that
    should be invoked.

    Returns:
        dict with keys ``plan``, ``selected_experts``, ``status``.
    """
    from src.core.llm_factory import create_resilient_llm
    from src.guardrails.input_guard import validate_input

    query = state.get("query", "")
    scope = state.get("scope") or ""
    budget = state.get("budget")

    # ── Input guardrail ──────────────────────────────────────────────────
    guard_result = validate_input(query, budget)
    if not guard_result.passed:
        logger.warning("[plan_node] Input guardrail rejected: %s", guard_result.reason)
        return {
            "plan": {
                "reasoning": f"Input rejected: {guard_result.reason}",
                "experts": [],
                "context_summary": "",
            },
            "selected_experts": [],
            "status": AnalysisStatus.FAILED,
        }
    query = guard_result.sanitized_input

    # ── Memory retrieval (cross-session context) ────────────────────────
    # Best-effort: memory unavailability must never block an analysis.
    memory_context: list[dict[str, Any]] = []
    try:
        from src.core.config import get_settings
        from src.memory import MemoryStore

        store = MemoryStore(db_path=get_settings().memory_db_path)
        hits = await store.search_analyses(state.get("org_id", "default"), query, limit=3)
        # L2 aliases steer keyword routing with org vocabulary
        alias_hits = await store.resolve_aliases(state.get("org_id", "default"), query)
        if alias_hits.get("expert"):
            memory_context.append(
                {
                    "query": "",
                    "scope": "alias",
                    "summary": f"org 别名命中专家: {sorted(alias_hits['expert'])}",
                    "alias_experts": sorted(alias_hits["expert"]),
                }
            )
        for hit in hits:
            memory_context.append(
                {
                    "query": str(hit.get("query", ""))[:120],
                    "scope": hit.get("scope", ""),
                    "summary": str(hit.get("results_summary", ""))[:300],
                }
            )
        if memory_context:
            logger.info(
                "[plan_node] Injected %d memory entr(ies) into planning context",
                len(memory_context),
            )
        await store.close()
    except Exception as exc:
        logger.debug("[plan_node] Memory retrieval unavailable: %s", exc)

    memory_block = ""
    if memory_context:
        prior = "\n".join(
            f"  - [{m['scope']}] {m['query']} -> {m['summary']}" for m in memory_context
        )
        memory_block = f"\nRelevant prior analyses for this org (use as context):\n{prior}\n"

    llm = create_resilient_llm(role="default", agent_name="plan_node")

    # ── Zero-key path: keyword routing, no LLM call ──────────────────────
    if llm is None:
        selected_experts = _intent_classify(query)
        logger.info("[plan_node] plan:rule (no LLM) — selected %s", selected_experts)
        return {
            "plan": {
                "reasoning": "rule-based intent classification (no LLM configured)",
                "experts": selected_experts,
                "context_summary": query[:100],
                "source": "rule",
                "memory_context": memory_context,
            },
            "selected_experts": selected_experts,
            "status": AnalysisStatus.EXECUTING,
        }

    try:
        prompt = PLAN_PROMPT.format(query=query, scope=scope or "full") + memory_block
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        plan_data: dict[str, Any] = json.loads(content)
        selected_experts = [
            e
            for e in plan_data.get("experts", [])
            if e in ("prospect", "conversion", "subsidy", "retention", "ad")
        ]

        # Fallback: if LLM returned no valid experts, use intent classification
        if not selected_experts:
            selected_experts = _intent_classify(query)

        logger.info(
            "[plan_node] Selected experts: %s (reasoning: %s)",
            selected_experts,
            plan_data.get("reasoning", ""),
        )

        return {
            "plan": {**plan_data, "source": "llm", "memory_context": memory_context},
            "selected_experts": selected_experts,
            "status": AnalysisStatus.EXECUTING,
        }

    except Exception as exc:
        logger.warning("[plan_node] LLM planning failed, using intent classify: %s", exc)
        selected_experts = _intent_classify(query)
        return {
            "plan": {
                "reasoning": f"LLM planning failed ({exc}); intent classify fallback",
                "experts": selected_experts,
                "context_summary": query[:100],
                "source": "rule",
                "memory_context": memory_context,
            },
            "selected_experts": selected_experts,
            "status": AnalysisStatus.EXECUTING,
        }


def _intent_classify(query: str) -> list[str]:
    """Score each expert's confidence for the query using can_handle().

    Uses each ExpertAgentBase subclass's domain-specific keyword matching
    instead of a centralized keyword map. Falls back to all experts if
    none score above the threshold. Each expert class is imported in its
    own guard — a broken native lib (OSError, not just ImportError) must
    degrade that expert, never crash the router (A8).
    """
    import importlib

    expert_classes: dict[str, type] = {}
    for key, path in [
        ("prospect", "src.agents.prospect.ProspectExpert"),
        ("conversion", "src.agents.conversion.ConversionExpert"),
        ("subsidy", "src.agents.subsidy.SubsidyExpert"),
        ("retention", "src.agents.retention.RetentionExpert"),
        ("ad", "src.agents.ad.AdExpert"),
    ]:
        try:
            module_path, class_name = path.rsplit(".", 1)
            expert_classes[key] = getattr(importlib.import_module(module_path), class_name)
        except Exception as exc:  # ImportError AND OSError from native deps
            logger.warning("[_intent_classify] %s unavailable: %s", key, exc)

    experts = list(expert_classes.items())
    if not experts:  # nothing importable at all — pathological, but never crash
        logger.warning("[_intent_classify] no experts importable")
        return []

    scores = []
    for key, cls in experts:
        try:
            # Create a temporary instance to call can_handle
            score = cls.can_handle(query)
            scores.append((key, score))
        except Exception:
            # If can_handle fails, give neutral score
            scores.append((key, 0.3))

    # Select experts scoring >= 0.5
    selected = [name for name, score in scores if score >= 0.5]

    # Default to all if none qualify
    if not selected:
        selected = [name for name, _ in experts]

    logger.info(
        "[_intent_classify] Scores: %s -> Selected: %s",
        {k: round(s, 2) for k, s in scores},
        selected,
    )
    return selected


async def execute_node(state: AnalysisState) -> dict:  # noqa: D401
    """Parallel expert execution.

    Reads ``plan`` and ``selected_experts`` to dispatch work to the relevant
    ExpertAgent subclasses.  Results are merged via the ``expert_results``
    reducer (_replace_by_expert) so that refinement results replace earlier
    ones for the same expert key.

    Returns:
        dict with keys ``expert_results``, ``execution_errors``, ``status``.
    """
    selected_experts: list[str] = state.get("selected_experts", [])
    query = state.get("query", "")
    scope = state.get("scope") or ""
    budget = state.get("budget") or 0
    data_path = state.get("data_path") or ""
    memory_context = (state.get("plan") or {}).get("memory_context", [])

    if not selected_experts:
        logger.warning("[execute_node] No experts selected, nothing to execute")
        return {
            "expert_results": [],
            "execution_errors": [],
            "status": AnalysisStatus.EVALUATING,
        }

    from src.core.config import get_settings

    timeout = get_settings().expert_timeout_seconds

    async def _run_expert(expert_key: str) -> tuple[str, dict[str, Any]]:
        expert = _create_expert(expert_key)
        params = {
            "query": query,
            "scope": scope,
            "budget": budget,
            "data_path": data_path,
            "memory_context": memory_context,
        }
        async with asyncio.timeout(timeout):
            # Route through BaseAgent.run() so the full hook lifecycle
            # (tracing / logging / metrics) and middleware stack execute.
            result = await expert.run({"params": params})
        return expert_key, result

    # Run all experts in parallel
    coros = [_run_expert(key) for key in selected_experts]
    raw_results = await asyncio.gather(*coros, return_exceptions=True)

    expert_results: list[dict[str, Any]] = []
    execution_errors: list[str] = []

    for expert_key, outcome in zip(selected_experts, raw_results):
        if isinstance(outcome, Exception):
            # str(TimeoutError()) is empty — render timeouts explicitly
            if isinstance(outcome, asyncio.TimeoutError):
                outcome = TimeoutError(f"expert timed out after {timeout}s")
            err_msg = f"{expert_key}: {outcome}"
            execution_errors.append(err_msg)
            expert_results.append(
                {
                    "expert": expert_key,
                    "success": False,
                    "error": str(outcome),
                }
            )
            logger.warning("[execute_node] Expert %s failed: %s", expert_key, outcome)
        else:
            _, result = outcome  # type: ignore[misc]
            result["expert"] = expert_key

            # Pydantic validation at node boundary
            result = _validate_expert_result(expert_key, result)

            expert_results.append(result)

    # ── L1 fact ledger write-back (gated: success ∧ validated ∧ kpis) ────
    try:
        from src.core.config import get_settings as _gs
        from src.memory import MemoryStore

        _store = MemoryStore(db_path=_gs().memory_db_path)
        for res in expert_results:
            n = await _store.record_facts(
                state.get("org_id", "default"), res, run_fingerprint=str(budget)
            )
            if n:
                logger.debug("[execute_node] L1 ledger +%d facts (%s)", n, res.get("expert"))
        await _store.close()
    except Exception as exc:
        logger.debug("[execute_node] L1 ledger unavailable: %s", exc)

    return {
        "expert_results": expert_results,
        "execution_errors": execution_errors,
        "status": AnalysisStatus.EVALUATING,
    }


async def evaluate_node(state: AnalysisState) -> dict:  # noqa: D401
    """Quality evaluation of expert results.

    Reads ``expert_results`` and ``query`` and uses the Evaluator to score
    each expert's output on completeness, actionability, and data-grounding.

    Returns:
        dict with keys ``quality_scores``, ``needs_refinement``,
        ``refinement_round``, ``status``.
    """
    from src.core.evaluator import batch_evaluate

    expert_results_list: list[dict[str, Any]] = state.get("expert_results", [])
    query = state.get("query", "")

    if not expert_results_list:
        return {
            "quality_scores": {},
            "needs_refinement": False,
            "refinement_round": state.get("refinement_round", 0),
            "status": AnalysisStatus.REPORTING,
        }

    # Convert list to dict keyed by expert name for batch_evaluate.
    # Failed experts are scored 0 without an LLM call — re-running a failed
    # pipeline via refine would waste two rounds of LLM calls for nothing.
    expert_results_dict: dict[str, Any] = {}
    failed_scores: dict[str, Any] = {}
    for item in expert_results_list:
        key = item.get("expert", f"unknown_{id(item)}")
        if item.get("success") is False:
            failed_scores[key] = {
                "completeness": 0.0,
                "actionability": 0.0,
                "data_grounding": 0.0,
                "overall": 0.0,
                "reasoning": f"Expert execution failed: {item.get('error', 'unknown error')}",
                "evaluation_failed": True,
                "execution_failed": True,
            }
        else:
            expert_results_dict[key] = item

    # ── Zero-key path: deterministic scoring, no LLM judge ────────────────
    # Re-running a degraded pipeline via refine cannot improve anything
    # without a brain, so needs_refinement stays False.
    from src.core.config import get_settings

    settings = get_settings()
    if not settings.llm_available:
        rule_scores: dict[str, Any] = {}
        for key, item in expert_results_dict.items():
            kpis = item.get("kpis") or {}
            coverage = min(1.0, len(kpis) / 3) if kpis else 0.0
            rule_scores[key] = {
                "completeness": coverage,
                "actionability": coverage,
                "data_grounding": 1.0 if kpis else 0.3,
                "overall": round((coverage + coverage + (1.0 if kpis else 0.3)) / 3, 2),
                "reasoning": "deterministic scoring (no LLM configured)",
                "evaluation_mode": "rule",
            }
        quality_scores_dicts = {**rule_scores, **failed_scores}
        logger.info(
            "[evaluate_node] eval:rule scores=%s",
            {k: v["overall"] for k, v in rule_scores.items()},
        )
        return {
            "quality_scores": quality_scores_dicts,
            "needs_refinement": False,
            "refinement_round": state.get("refinement_round", 0),
            "status": AnalysisStatus.EVALUATING,
        }

    quality_scores = await batch_evaluate(expert_results_dict, query)
    quality_scores_dicts = {k: v.model_dump() for k, v in quality_scores.items()}
    quality_scores_dicts.update(failed_scores)

    needs_refinement = any(
        score.overall < settings.quality_threshold for score in quality_scores.values()
    )

    current_round = state.get("refinement_round", 0) or 0
    next_round = current_round + (1 if needs_refinement else 0)

    logger.info(
        "[evaluate_node] Scores: %s | needs_refinement=%s round=%d",
        {k: round(v.overall, 2) for k, v in quality_scores.items()},
        needs_refinement,
        next_round,
    )

    # Convert QualityScore objects to dicts for state storage
    # (failed experts were already added as zero scores above)
    return {
        "quality_scores": quality_scores_dicts,
        "needs_refinement": needs_refinement,
        "refinement_round": next_round,
        "status": AnalysisStatus.EVALUATING,
    }


async def refine_node(state: AnalysisState) -> dict:  # noqa: D401
    """Retry experts whose output fell below the quality threshold.

    Reads ``quality_scores`` and ``refinement_round``.  For each expert with
    an overall score below the threshold, re-invokes the expert with the
    evaluator's feedback appended as additional context.

    Returns:
        dict with keys ``expert_results``, ``execution_errors``,
        ``refinement_round``, ``needs_refinement``, ``status``.
    """
    quality_scores: dict[str, Any] = state.get("quality_scores", {})
    query = state.get("query", "")
    scope = state.get("scope") or ""
    budget = state.get("budget") or 0

    from src.core.config import get_settings

    settings = get_settings()
    # Find experts that need refinement.  Experts whose *execution* failed
    # (execution_failed marker) are excluded — re-running a crashed pipeline
    # is the job of retry middleware, not the quality loop.
    experts_to_refine = [
        name
        for name, score_data in quality_scores.items()
        if score_data.get("overall", 1.0) < settings.quality_threshold
        and not score_data.get("execution_failed")
    ]

    if not experts_to_refine:
        logger.info("[refine_node] No experts below threshold, skipping refinement")
        return {
            "expert_results": [],
            "execution_errors": [],
            "needs_refinement": False,
            "refinement_round": state.get("refinement_round", 0),
            "status": AnalysisStatus.REPORTING,
        }

    async def _refine_expert(expert_key: str) -> tuple[str, dict[str, Any]]:
        expert = _create_expert(expert_key)
        params = {
            "query": query,
            "scope": scope,
            "budget": budget,
            "is_refinement": True,
        }
        async with asyncio.timeout(settings.expert_timeout_seconds):
            result = await expert.run({"params": params})
        return expert_key, result

    coros = [_refine_expert(key) for key in experts_to_refine]
    raw_results = await asyncio.gather(*coros, return_exceptions=True)

    expert_results: list[dict[str, Any]] = []
    execution_errors: list[str] = []

    for expert_key, outcome in zip(experts_to_refine, raw_results):
        if isinstance(outcome, Exception):
            if isinstance(outcome, asyncio.TimeoutError):
                outcome = TimeoutError(f"expert timed out after {settings.expert_timeout_seconds}s")
            err_msg = f"refine:{expert_key}: {outcome}"
            execution_errors.append(err_msg)
            expert_results.append(
                {
                    "expert": expert_key,
                    "success": False,
                    "error": str(outcome),
                    "refinement_attempt": True,
                }
            )
        else:
            _, result = outcome  # type: ignore[misc]
            result["expert"] = expert_key
            result["refinement_attempt"] = True
            expert_results.append(result)

    # After refinement, mark needs_refinement as False to avoid infinite loop
    return {
        "expert_results": expert_results,
        "execution_errors": execution_errors,
        "needs_refinement": False,
        "refinement_round": state.get("refinement_round", 0),
        "status": AnalysisStatus.EVALUATING,
    }


async def approval_node(state: AnalysisState) -> dict:  # noqa: D401
    """Human approval checkpoint.

    When ``approval_required`` is True, the graph pauses here (LangGraph
    interrupt) and waits for a human decision.  The resumed call sets
    ``approved`` to True or False.

    For auto-approval: budget below threshold -> auto-approve.
    Otherwise set status to AWAITING_APPROVAL.

    Returns:
        dict with keys ``approved``, ``status``.
    """
    budget = state.get("budget") or 0
    approval_required = state.get("approval_required", False)

    from src.core.config import get_settings

    settings = get_settings()

    if not approval_required:
        # Auto-approve when approval is not required
        return {
            "approved": True,
            "status": AnalysisStatus.REPORTING,
        }

    # Budget-based auto-approval
    if budget < settings.budget_approval_threshold:
        logger.info(
            "[approval_node] Auto-approved (budget=%.2f < threshold=%.2f)",
            budget,
            settings.budget_approval_threshold,
        )
        return {
            "approved": True,
            "status": AnalysisStatus.REPORTING,
        }

    # Requires human approval — set status and wait
    logger.info(
        "[approval_node] Awaiting human approval (budget=%.2f >= threshold=%.2f)",
        budget,
        settings.budget_approval_threshold,
    )
    return {
        "approved": None,
        "status": AnalysisStatus.AWAITING_APPROVAL,
    }


async def report_node(state: AnalysisState) -> dict:  # noqa: D401
    """Generate the final analysis report.

    Reads ``expert_results``, ``quality_scores``, and the accumulated context
    to synthesize a structured report with summary, strategy recommendations,
    and a KPI snapshot.

    Returns:
        dict with keys ``final_report``, ``token_usage``, ``cost_usd``,
        ``status``.
    """
    from src.core.llm_factory import create_resilient_llm

    query = state.get("query", "")
    expert_results_list: list[dict[str, Any]] = state.get("expert_results", [])
    quality_scores: dict[str, Any] = state.get("quality_scores", {})

    # Build a summary of each expert's findings
    expert_summaries: list[str] = []
    for result in expert_results_list:
        expert_name = result.get("expert", "unknown")
        success = result.get("success", True)

        if not success:
            expert_summaries.append(
                f"### {expert_name}\nStatus: FAILED - {result.get('error', 'unknown error')}"
            )
            continue

        # Extract analysis summary if available
        analysis = result.get("analysis", {})
        if isinstance(analysis, dict):
            summary = analysis.get("summary", analysis.get("raw_response", ""))
        else:
            summary = str(analysis)

        score_info = quality_scores.get(expert_name, {})
        overall_score = score_info.get("overall", "N/A")

        expert_summaries.append(
            f"### {expert_name}\nQuality Score: {overall_score}\nSummary: {summary}\n"
        )

    report_prompt = f"""\
You are the Chief Agent of a growth analytics platform.  Synthesize the
following expert analyses into a comprehensive, actionable final report.

## User Query
{query}

## Expert Results
{chr(10).join(expert_summaries)}

## Instructions
Generate a structured report in Markdown with these sections:
1. **Executive Summary** (2-3 sentences)
2. **Key Findings** (bullet points from each expert)
3. **Strategy Recommendations** (prioritized, actionable)
4. **KPI Snapshot** (key metrics mentioned)
5. **Risk Assessment** (if any)

Be concise but specific. Use data and metrics from the expert results.
"""

    # Final synthesis uses the power role across all expert outputs. When
    # no LLM is configured (or synthesis fails) the deterministic template
    # ReportGenerator renders expert results — a report is always produced.
    llm = create_resilient_llm(role="power", agent_name="report_node")
    if llm is None:
        from src.report.generator import ReportGenerator

        final_report = ReportGenerator().generate_report(dict(state))
        token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        cost_usd = 0.0
        logger.info("[report_node] report:template (no LLM configured)")
    else:
        try:
            response = await llm.ainvoke(report_prompt)
            final_report = response.content if hasattr(response, "content") else str(response)

            from src.core.token_tracker import get_cost_tracker

            cost_report = get_cost_tracker().report()
            token_usage = {
                "input_tokens": cost_report["input_tokens"],
                "output_tokens": cost_report["output_tokens"],
                "total_tokens": cost_report["total_tokens"],
            }
            cost_usd = cost_report["total_cost_usd"]

        except Exception as exc:
            logger.warning("[report_node] LLM report generation failed: %s", exc)
            from src.report.generator import ReportGenerator

            final_report = ReportGenerator().generate_report(dict(state))
            token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            cost_usd = 0.0

    # ── Output guardrail ─────────────────────────────────────────────────
    from src.guardrails.output_guard import validate_output

    output_guard = validate_output(final_report)
    if not output_guard.passed:
        logger.warning("[report_node] Output guardrail rejected: %s", output_guard.reason)
        # Truncate or replace the report with a safe version
        final_report = (
            f"# Analysis Report\n\n"
            f"## Query\n{query}\n\n"
            f"*Report generation completed but output was filtered: {output_guard.reason}.*\n"
        )

    # ── Episodic memory write-back (best-effort) ─────────────────────────
    kpi_snapshot: dict[str, Any] = {}
    for r in state.get("expert_results", []):
        ek = r.get("kpis")
        if isinstance(ek, dict):
            kpi_snapshot.update(ek)
    try:
        import uuid as _uuid

        from src.core.config import get_settings as _gs
        from src.memory import MemoryStore

        _store = MemoryStore(db_path=_gs().memory_db_path)
        await _store.save_analysis(
            analysis_id=f"wf:{_uuid.uuid4().hex[:12]}",
            org_id=state.get("org_id", "default"),
            query=query,
            scope=state.get("scope") or "full",
            report=final_report,
            kpis=kpi_snapshot,
        )
        await _store.close()
    except Exception as exc:
        logger.debug("[report_node] Memory write-back unavailable: %s", exc)

    return {
        "final_report": final_report,
        "kpi_snapshot": kpi_snapshot,
        "token_usage": token_usage,
        "cost_usd": cost_usd,
        "status": AnalysisStatus.COMPLETED,
    }
