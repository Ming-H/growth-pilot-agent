"""State schema for the agent-led orchestration mode.

Extends the workflow-mode ``AnalysisState`` with a LangGraph ``messages``
channel so the Chief Agent keeps a full conversation history (user query,
tool calls, tool results, final answer) across the agent⇄tools loop and
across human-approval interrupts.
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages

from src.graph.state import AnalysisState


class AgentState(AnalysisState):
    """Graph state for the agent-led mode (tool-calling loop).

    Inherited fields (query / budget / expert_results / quality_scores /
    approval / report ...) keep their original reducers; ``messages`` uses
    LangGraph's ``add_messages`` channel so appended messages merge
    correctly across parallel branches and checkpoint resumes.
    """

    # Full agent conversation: SystemMessage, HumanMessage, AIMessage
    # (with tool_calls) and ToolMessage entries.
    messages: Annotated[list[Any], add_messages]
