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
# Expert mapping: Chinese domain keyword -> expert key used in the graph
# ---------------------------------------------------------------------------

EXPERT_KEYWORD_MAP: dict[str, str] = {
    "用户获取": "prospect",
    "转化优化": "conversion",
    "补贴策略": "subsidy",
    "用户留存": "retention",
    "广告投放": "ad",
}

QUALITY_THRESHOLD: float = 0.7
BUDGET_APPROVAL_THRESHOLD: float = 10_000

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
{{"reasoning": "<brief reasoning>", "experts": ["<expert_key>", ...], "context_summary": "<one-line summary>"}}
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


def _create_expert(expert_key: str) -> Any:
    """Create an expert instance with an LLM from settings."""
    from src.core.llm_factory import create_llm
    from src.core.config import get_settings

    settings = get_settings()
    llm = create_llm(tier=settings.expert_model_tier)
    cls = _get_expert_cls(expert_key)
    return cls(llm=llm)


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
    from src.core.llm_factory import create_llm
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

    try:
        llm = create_llm(tier="default")
        prompt = PLAN_PROMPT.format(query=query, scope=scope or "full")
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        plan_data: dict[str, Any] = json.loads(content)
        selected_experts = [
            e for e in plan_data.get("experts", [])
            if e in ("prospect", "conversion", "subsidy", "retention", "ad")
        ]

        # Fallback: if LLM returned no valid experts, try keyword matching
        if not selected_experts:
            selected_experts = _keyword_fallback(query)

        logger.info(
            "[plan_node] Selected experts: %s (reasoning: %s)",
            selected_experts,
            plan_data.get("reasoning", ""),
        )

        return {
            "plan": plan_data,
            "selected_experts": selected_experts,
            "status": AnalysisStatus.EXECUTING,
        }

    except Exception as exc:
        logger.warning("[plan_node] LLM planning failed, using keyword fallback: %s", exc)
        selected_experts = _keyword_fallback(query)
        return {
            "plan": {
                "reasoning": f"LLM planning failed ({exc}); keyword fallback",
                "experts": selected_experts,
                "context_summary": query[:100],
            },
            "selected_experts": selected_experts,
            "status": AnalysisStatus.EXECUTING,
        }


def _keyword_fallback(query: str) -> list[str]:
    """Simple keyword-based expert selection when LLM planning is unavailable."""
    matched: list[str] = []
    for keyword, expert_key in EXPERT_KEYWORD_MAP.items():
        if keyword in query:
            matched.append(expert_key)
    # Default to all experts if nothing matched
    return matched or ["prospect", "conversion", "subsidy", "retention", "ad"]


async def execute_node(state: AnalysisState) -> dict:  # noqa: D401
    """Parallel expert execution.

    Reads ``plan`` and ``selected_experts`` to dispatch work to the relevant
    ExpertAgent subclasses.  Results are appended via the ``expert_results``
    reducer (operator.add) so that parallel branches merge automatically.

    Returns:
        dict with keys ``expert_results``, ``execution_errors``, ``status``.
    """
    selected_experts: list[str] = state.get("selected_experts", [])
    query = state.get("query", "")
    scope = state.get("scope") or ""
    budget = state.get("budget") or 0

    if not selected_experts:
        logger.warning("[execute_node] No experts selected, nothing to execute")
        return {
            "expert_results": [],
            "execution_errors": [],
            "status": AnalysisStatus.EVALUATING,
        }

    async def _run_expert(expert_key: str) -> tuple[str, dict[str, Any]]:
        expert = _create_expert(expert_key)
        params = {
            "query": query,
            "scope": scope,
            "budget": budget,
        }
        result = await expert.analyze(params)
        return expert_key, result

    # Run all experts in parallel
    coros = [_run_expert(key) for key in selected_experts]
    raw_results = await asyncio.gather(*coros, return_exceptions=True)

    expert_results: list[dict[str, Any]] = []
    execution_errors: list[str] = []

    for expert_key, outcome in zip(selected_experts, raw_results):
        if isinstance(outcome, Exception):
            err_msg = f"{expert_key}: {outcome}"
            execution_errors.append(err_msg)
            expert_results.append({
                "expert": expert_key,
                "success": False,
                "error": str(outcome),
            })
            logger.warning("[execute_node] Expert %s failed: %s", expert_key, outcome)
        else:
            _, result = outcome  # type: ignore[misc]
            result["expert"] = expert_key
            expert_results.append(result)

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

    # Convert list to dict keyed by expert name for batch_evaluate
    expert_results_dict: dict[str, Any] = {}
    for item in expert_results_list:
        key = item.get("expert", f"unknown_{id(item)}")
        expert_results_dict[key] = item

    quality_scores = await batch_evaluate(expert_results_dict, query)

    # Determine if any expert scored below threshold
    needs_refinement = any(
        score.overall < QUALITY_THRESHOLD
        for score in quality_scores.values()
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
    scores_dict = {k: v.model_dump() for k, v in quality_scores.items()}

    return {
        "quality_scores": scores_dict,
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

    # Find experts that need refinement
    experts_to_refine = [
        name for name, score_data in quality_scores.items()
        if score_data.get("overall", 1.0) < QUALITY_THRESHOLD
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
        feedback = quality_scores.get(expert_key, {}).get("reasoning", "")
        params = {
            "query": query,
            "scope": scope,
            "budget": budget,
            "refinement_feedback": feedback,
            "is_refinement": True,
        }
        result = await expert.analyze(params)
        return expert_key, result

    coros = [_refine_expert(key) for key in experts_to_refine]
    raw_results = await asyncio.gather(*coros, return_exceptions=True)

    expert_results: list[dict[str, Any]] = []
    execution_errors: list[str] = []

    for expert_key, outcome in zip(experts_to_refine, raw_results):
        if isinstance(outcome, Exception):
            err_msg = f"refine:{expert_key}: {outcome}"
            execution_errors.append(err_msg)
            expert_results.append({
                "expert": expert_key,
                "success": False,
                "error": str(outcome),
                "refinement_attempt": True,
            })
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

    if not approval_required:
        # Auto-approve when approval is not required
        return {
            "approved": True,
            "status": AnalysisStatus.REPORTING,
        }

    # Budget-based auto-approval
    if budget < BUDGET_APPROVAL_THRESHOLD:
        logger.info(
            "[approval_node] Auto-approved (budget=%.2f < threshold=%.2f)",
            budget,
            BUDGET_APPROVAL_THRESHOLD,
        )
        return {
            "approved": True,
            "status": AnalysisStatus.REPORTING,
        }

    # Requires human approval — set status and wait
    logger.info(
        "[approval_node] Awaiting human approval (budget=%.2f >= threshold=%.2f)",
        budget,
        BUDGET_APPROVAL_THRESHOLD,
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
    from src.core.llm_factory import create_llm

    query = state.get("query", "")
    expert_results_list: list[dict[str, Any]] = state.get("expert_results", [])
    quality_scores: dict[str, Any] = state.get("quality_scores", {})

    # Build a summary of each expert's findings
    expert_summaries: list[str] = []
    for result in expert_results_list:
        expert_name = result.get("expert", "unknown")
        success = result.get("success", True)

        if not success:
            expert_summaries.append(f"### {expert_name}\nStatus: FAILED - {result.get('error', 'unknown error')}")
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
            f"### {expert_name}\n"
            f"Quality Score: {overall_score}\n"
            f"Summary: {summary}\n"
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

    try:
        llm = create_llm(tier="default")
        response = await llm.ainvoke(report_prompt)
        final_report = response.content if hasattr(response, "content") else str(response)

        # Estimate token usage (rough approximation)
        input_tokens = len(report_prompt) // 4
        output_tokens = len(final_report) // 4
        token_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        # Rough cost estimate (glm-4.7 pricing approximation)
        cost_usd = (input_tokens * 0.000001 + output_tokens * 0.000002)

    except Exception as exc:
        logger.warning("[report_node] LLM report generation failed: %s", exc)
        # Fallback: generate a simple report from expert results
        findings = "\n".join(
            f"- {r.get('expert', 'unknown')}: {'completed' if r.get('success', True) else 'failed'}"
            for r in expert_results_list
        )
        final_report = f"# Analysis Report\n\n## Query\n{query}\n\n## Expert Findings\n{findings}\n"
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

    return {
        "final_report": final_report,
        "token_usage": token_usage,
        "cost_usd": cost_usd,
        "status": AnalysisStatus.COMPLETED,
    }
