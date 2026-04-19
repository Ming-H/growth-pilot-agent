"""LangGraph StateGraph workflow for GrowthPilot multi-agent system.

Node layout:
    orchestrator -> parallel(prospect, subsidy, ad) -> conversion -> retention -> report_gen -> END

The orchestrator decides which agents to run based on ``state["scope"]``.
Conditional routing skips nodes that the orchestrator deems unnecessary.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from src.core.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node functions (each wraps an agent and handles errors)
# ---------------------------------------------------------------------------


async def orchestrator_node(state: AgentState) -> dict[str, Any]:
    """Detect scope and decide which agents to run."""
    from src.agents.orchestrator import OrchestratorAgent
    from src.core.llm_factory import create_llm

    try:
        llm = create_llm()
        agent = OrchestratorAgent(llm)
        result = await agent.run(state)
        return result
    except Exception as exc:
        logger.error("Orchestrator node failed: %s", exc)
        return {
            "errors": [f"orchestrator: {exc}"],
            "analysis_summary": f"编排失败: {exc}",
        }


async def prospect_node(state: AgentState) -> dict[str, Any]:
    """Run prospect analysis."""
    from src.agents.prospect import ProspectAgent
    from src.core.llm_factory import create_llm

    try:
        llm = create_llm()
        agent = ProspectAgent(llm)
        return await agent.run(state)
    except Exception as exc:
        logger.error("Prospect node failed: %s", exc)
        return {"errors": [f"prospect: {exc}"]}


async def subsidy_node(state: AgentState) -> dict[str, Any]:
    """Run subsidy analysis."""
    from src.agents.subsidy import SubsidyAgent
    from src.core.llm_factory import create_llm

    try:
        llm = create_llm()
        agent = SubsidyAgent(llm)
        return await agent.run(state)
    except Exception as exc:
        logger.error("Subsidy node failed: %s", exc)
        return {"errors": [f"subsidy: {exc}"]}


async def ad_node(state: AgentState) -> dict[str, Any]:
    """Run ad analysis."""
    from src.agents.ad import AdAgent
    from src.core.llm_factory import create_llm

    try:
        llm = create_llm()
        agent = AdAgent(llm)
        return await agent.run(state)
    except Exception as exc:
        logger.error("Ad node failed: %s", exc)
        return {"errors": [f"ad: {exc}"]}


async def conversion_node(state: AgentState) -> dict[str, Any]:
    """Run conversion analysis."""
    from src.agents.conversion import ConversionAgent
    from src.core.llm_factory import create_llm

    try:
        llm = create_llm()
        agent = ConversionAgent(llm)
        return await agent.run(state)
    except Exception as exc:
        logger.error("Conversion node failed: %s", exc)
        return {"errors": [f"conversion: {exc}"]}


async def retention_node(state: AgentState) -> dict[str, Any]:
    """Run retention analysis."""
    from src.agents.retention import RetentionAgent
    from src.core.llm_factory import create_llm

    try:
        llm = create_llm()
        agent = RetentionAgent(llm)
        return await agent.run(state)
    except Exception as exc:
        logger.error("Retention node failed: %s", exc)
        return {"errors": [f"retention: {exc}"]}


async def report_gen_node(state: AgentState) -> dict[str, Any]:
    """Generate the final report."""
    from src.report.generator import ReportGenerator

    try:
        generator = ReportGenerator()
        report = generator.generate_report(state)
        return {"report": report}
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        return {
            "report": f"报告生成失败: {exc}",
            "errors": [f"report_gen: {exc}"],
        }


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

# The orchestrator sets metadata with the agents to run.
# If the orchestrator already ran all agents internally (its default), the
# graph-level parallel nodes are skipped.  We use the ``scope`` field plus
# metadata to decide.

_FULL_SCOPES = {"full"}
_PROSPECT_SCOPES = {"full", "prospect"}
_SUBSIDY_SCOPES = {"full", "subsidy"}
_AD_SCOPES = {"full", "ad"}
_CONVERSION_SCOPES = {"full", "inapp", "prospect", "conversion"}
_RETENTION_SCOPES = {"full", "inapp", "retention"}


def _should_run_prospect(state: AgentState) -> Literal["prospect", "skip_prospect"]:
    scope = state.get("scope", "full")
    return "prospect" if scope in _PROSPECT_SCOPES else "skip_prospect"


def _should_run_subsidy(state: AgentState) -> Literal["subsidy", "skip_subsidy"]:
    scope = state.get("scope", "full")
    return "subsidy" if scope in _SUBSIDY_SCOPES else "skip_subsidy"


def _should_run_ad(state: AgentState) -> Literal["ad", "skip_ad"]:
    scope = state.get("scope", "full")
    return "ad" if scope in _AD_SCOPES else "skip_ad"


def _should_run_conversion(state: AgentState) -> Literal["conversion", "skip_conversion"]:
    scope = state.get("scope", "full")
    return "conversion" if scope in _CONVERSION_SCOPES else "skip_conversion"


def _should_run_retention(state: AgentState) -> Literal["retention", "skip_retention"]:
    scope = state.get("scope", "full")
    return "retention" if scope in _RETENTION_SCOPES else "skip_retention"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_workflow() -> StateGraph:
    """Build and compile the LangGraph StateGraph.

    Returns a compiled graph ready for ``ainvoke()``.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("prospect", prospect_node)
    graph.add_node("subsidy", subsidy_node)
    graph.add_node("ad", ad_node)
    graph.add_node("conversion", conversion_node)
    graph.add_node("retention", retention_node)
    graph.add_node("report_gen", report_gen_node)

    # Skip nodes (no-op pass-throughs for conditional edges)
    graph.add_node("skip_prospect", lambda state: {})
    graph.add_node("skip_subsidy", lambda state: {})
    graph.add_node("skip_ad", lambda state: {})
    graph.add_node("skip_conversion", lambda state: {})
    graph.add_node("skip_retention", lambda state: {})

    # Entry
    graph.set_entry_point("orchestrator")

    # Orchestrator -> parallel branch (prospect / subsidy / ad)
    # After orchestrator, we fan out to three conditional branches that
    # each decide independently whether to run or skip.
    graph.add_conditional_edges("orchestrator", _should_run_prospect, {
        "prospect": "prospect",
        "skip_prospect": "skip_prospect",
    })
    graph.add_conditional_edges("orchestrator", _should_run_subsidy, {
        "subsidy": "subsidy",
        "skip_subsidy": "skip_subsidy",
    })
    graph.add_conditional_edges("orchestrator", _should_run_ad, {
        "ad": "ad",
        "skip_ad": "skip_ad",
    })

    # Parallel -> conversion fan-in
    # All three parallel branches (or their skips) converge on conversion.
    for src in ("prospect", "skip_prospect", "subsidy", "skip_subsidy", "ad", "skip_ad"):
        graph.add_conditional_edges(src, _should_run_conversion, {
            "conversion": "conversion",
            "skip_conversion": "skip_conversion",
        })

    # Conversion -> retention
    graph.add_conditional_edges("conversion", _should_run_retention, {
        "retention": "retention",
        "skip_retention": "skip_retention",
    })
    graph.add_conditional_edges("skip_conversion", _should_run_retention, {
        "retention": "retention",
        "skip_retention": "skip_retention",
    })

    # Retention -> report_gen -> END
    graph.add_edge("retention", "report_gen")
    graph.add_edge("skip_retention", "report_gen")
    graph.add_edge("report_gen", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


async def run_workflow(
    *,
    query: str = "",
    data_path: str = "",
    budget: float = 0,
    scope: str = "",
) -> AgentState:
    """Build and run the workflow with the given inputs.

    Returns the final ``AgentState`` after all nodes have executed.
    """
    app = build_workflow()

    initial_state: AgentState = {
        "query": query,
        "data_path": data_path,
        "budget": budget,
        "scope": scope or None,  # type: ignore[typeddict-item]
        "errors": [],
        "metadata": [],
    }

    result = await app.ainvoke(initial_state)
    return result
