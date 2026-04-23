"""GrowthPilot workflow — Chief Agent orchestration.

Replaces the old LangGraph StateGraph with the Chief Agent's ReAct loop.
The `run_workflow()` function signature is preserved for backward compatibility
with CLI and web API.

Design references:
- Anthropic Orchestrator-Workers: dynamic decomposition replaces hardcoded graph
- OpenAI Runner: execution engine for the agent loop
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Convenience runner (backward-compatible interface)
# ---------------------------------------------------------------------------


async def run_workflow(
    *,
    query: str = "",
    data_path: str = "",
    budget: float = 0,
    scope: str = "",
) -> dict[str, Any]:
    """Run the GrowthPilot analysis via the Chief Agent.

    This function preserves the old interface for CLI and web API compatibility.
    Internally it instantiates the Chief Agent and runs the ReAct loop.

    Returns a dict with keys: success, query, scope, analysis_summary,
    strategy_recommendation, expert_results, errors, report, metadata.
    """
    from src.core.chief import GrowthPilotAgent
    from src.core.config import get_settings
    from src.core.llm_factory import create_llm
    from src.tools.expert_tools import get_expert_tools

    settings = get_settings()

    # Create LLM and tools
    llm = create_llm(tier=settings.chief_model_tier)
    tools = get_expert_tools()

    # Create memory manager (optional — gracefully handle failure)
    memory_manager = None
    try:
        from src.memory.manager import MemoryManager

        memory_manager = MemoryManager()
    except Exception as exc:
        logger.warning("Memory manager unavailable: %s", exc)

    # Create and run the Chief Agent
    chief = GrowthPilotAgent(
        llm=llm,
        tools=tools,
        memory_manager=memory_manager,
        max_iterations=settings.chief_max_iterations,
        enable_evaluator=settings.chief_enable_evaluator,
    )

    result = await chief.run(
        query=query,
        data_path=data_path,
        budget=budget,
        scope_hint=scope,
    )

    # Generate report if report generator is available
    if result.get("analysis_summary"):
        try:
            from src.report.generator import ReportGenerator

            generator = ReportGenerator()
            report = generator.generate_report(result)
            result["report"] = report
        except Exception as exc:
            logger.warning("Report generation failed: %s", exc)
            result["report"] = result.get("analysis_summary", "")

    return result


# ---------------------------------------------------------------------------
# Legacy compatibility: keep build_workflow for any code that imports it
# ---------------------------------------------------------------------------


def build_workflow():
    """Deprecated: Use run_workflow() instead.

    Returns None. The old LangGraph StateGraph has been replaced
    by the Chief Agent's ReAct loop.
    """
    import warnings

    warnings.warn(
        "build_workflow() is deprecated. Use run_workflow() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return None
