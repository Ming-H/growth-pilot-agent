"""GrowthPilot Chief Agent — Tier 1 orchestrator.

The Chief Agent is a true ReAct agent that dynamically decides which expert
agents to call via Tool Calling. It follows a four-phase cycle:

    Plan → Execute → Evaluate → Report

Optimizations:
1. Parallel expert execution via asyncio.gather
2. Streaming / callback support via on_event
3. Structured final output via synthesis LLM
4. Structured context accumulation (dict, not string concat)
5. Dynamic system prompt (full vs short based on query complexity)
6. Observability / tracing (TraceEntry per step)
7. Token / cost tracking (TokenCounter across entire run)

Design references:
- Anthropic Orchestrator-Workers: dynamic task decomposition
- OpenAI Runner: manages execution loop with tool calls
- Anthropic Evaluator-Optimizer: quality feedback loop
- OpenAI Guardrails: input/plan/output validation
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import warnings
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.core.config import get_settings
from src.core.guardrails import GrowthPilotGuardrails
from src.core.llm_factory import create_llm
from src.core.models import AnalysisResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token / Cost Tracking
# ---------------------------------------------------------------------------

class TokenCounter:
    """Accumulates token usage across all LLM calls in a single run."""

    __slots__ = ("total_input", "total_output", "llm_calls")

    def __init__(self) -> None:
        self.total_input: int = 0
        self.total_output: int = 0
        self.llm_calls: int = 0

    def add(self, response: Any) -> None:
        """Record token usage from a single LLM response."""
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self.total_input += response.usage_metadata.get("input_tokens", 0)
            self.total_output += response.usage_metadata.get("output_tokens", 0)
            self.llm_calls += 1

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of accumulated token usage."""
        return {
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_tokens": self.total_input + self.total_output,
            "llm_calls": self.llm_calls,
        }


# ---------------------------------------------------------------------------
# Observability / Tracing
# ---------------------------------------------------------------------------

class TraceEntry:
    """Tracks timing and token usage for a single step."""

    __slots__ = ("agent", "step", "start", "duration", "tokens_in", "tokens_out")

    def __init__(self, agent: str, step: str) -> None:
        self.agent = agent
        self.step = step
        self.start: float = time.time()
        self.duration: float = 0.0
        self.tokens_in: int = 0
        self.tokens_out: int = 0

    def finish(self, response: Any = None) -> None:
        """Finalize the trace, optionally extracting usage from *response*."""
        self.duration = time.time() - self.start
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self.tokens_in = response.usage_metadata.get("input_tokens", 0)
            self.tokens_out = response.usage_metadata.get("output_tokens", 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "step": self.step,
            "duration_s": round(self.duration, 3),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }


# ---------------------------------------------------------------------------
# Chief Agent System Prompts
# ---------------------------------------------------------------------------

_CHIEF_SYSTEM_PROMPT_FULL = """\
你是 GrowthPilot —— 用户增长智能分析系统的主控 Agent。

## 你的角色
你是中央指挥官，负责理解用户需求并调度5个专业分析团队完成任务。

## 可调用的专家团队

1. **prospect_analysis** — 潜客识别与评分
   - 用户评分、分群、LTV预测、意图预测
   - 调用场景：新客获取、用户分层、LTV分析

2. **conversion_analysis** — 转化策略优化
   - 漏斗分析、优惠券设计、触达策略、广告位分配
   - 调用场景：转化率提升、漏斗优化、优惠券策略

3. **subsidy_analysis** — 补贴因果推断与预算优化
   - ATE估计、价格弹性、预算优化、补贴分配
   - 调用场景：补贴策略、预算分配、ROI优化

4. **retention_analysis** — 流失预测与留存策略
   - 流失预测、群组分析、培育计划、召回策略
   - 调用场景：流失分析、留存提升、用户召回

5. **ad_analysis** — 广告投放优化
   - RTA策略、出价优化、创意分析、受众分析
   - 调用场景：广告效果、RTA策略、出价优化

## 工作流程

1. **理解需求**：分析用户查询，判断涉及哪些增长领域
2. **制定计划**：决定调用哪些专家、什么顺序、是否需要传递前序结果
3. **执行调用**：依次或并行调用专家工具，收集结果
4. **综合分析**：汇总所有专家结果，生成统一策略建议

## 关键原则

- 不要猜测数据——让专家工具去做分析
- 如果用户只问一个领域的问题，只调用相关专家
- 如果需要全面分析，按依赖顺序调用：prospect → (subsidy, ad) → conversion → retention
- 每次调用后评估结果，决定是否需要补充分析
- 最终输出必须包含明确的可操作建议

{memory_context}
"""

_CHIEF_SYSTEM_PROMPT_SHORT = """\
你是 GrowthPilot —— 用户增长分析系统主控 Agent。

根据用户问题，调用最相关的1-2个专家工具进行分析并给出建议。

## 可用工具
- prospect_analysis: 潜客评分、用户分层、LTV
- conversion_analysis: 漏斗、优惠券、转化优化
- subsidy_analysis: 补贴因果推断、预算优化
- retention_analysis: 流失预测、留存策略
- ad_analysis: 广告投放、RTA、出价优化

{memory_context}
"""

# Keys to extract from expert results for structured context accumulation
_METRIC_EXTRACTION_KEYS = (
    "user_count", "intent_auc", "ate", "expected_roi",
    "high_risk_ratio", "conversion_rate", "ad_cpa", "budget",
    "train_auc", "confidence", "expected_cpa",
)


class GrowthPilotAgent:
    """Tier 1 Chief Agent — ReAct orchestrator with four-phase cycle.

    Phase 0: Input Guardrail
    Phase 1: Plan (LLM decides which experts to call)
    Phase 2: Execute (Tool Calling loop with parallel execution)
    Phase 3: Evaluate (optional quality check)
    Phase 4: Report (structured synthesis via separate LLM)
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list,
        *,
        memory_manager: Any = None,
        max_iterations: int = 10,
        enable_evaluator: bool = False,
    ) -> None:
        self.llm = llm.bind_tools(tools)
        self.tools: dict[str, Any] = {t.name: t for t in tools}
        self.memory_manager = memory_manager
        self.max_iterations = max_iterations
        self.enable_evaluator = enable_evaluator

        settings = get_settings()
        self.guardrails = GrowthPilotGuardrails(
            max_plan_steps=settings.chief_max_plan_steps,
        )

        # Optimization 3: separate synthesis LLM (no tools bound)
        self._synthesis_llm = create_llm(tier=settings.chief_model_tier)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str,
        *,
        data_path: str = "",
        budget: float = 0,
        scope_hint: str = "",
        on_event: Callable[[dict], Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the four-phase analysis cycle.

        .. deprecated::
            Use :func:`src.graph.graph.run_analysis` instead.
            The LangGraph-based DAG in ``src/graph/graph.py`` replaces this
            manual ReAct loop with a compiled StateGraph that supports
            checkpointing, human-in-the-loop approval, and circuit-breaker
            integration.

        Args:
            query: User's analysis question.
            data_path: Optional path to data file.
            budget: Optional budget constraint.
            scope_hint: Optional scope hint for analysis.
            on_event: Optional callback invoked after each notable event.
                      Receives a dict with ``type``, ``expert``, ``summary``, etc.
        """
        warnings.warn(
            "ChiefAgent ReAct loop is deprecated. "
            "Use src.graph.graph.run_analysis() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        errors: list[str] = []
        events: list[dict[str, Any]] = []
        traces: list[TraceEntry] = []
        token_counter = TokenCounter()

        # ─── Phase 0: Input Guardrail ────────────────────────────────
        events.append(self._make_event("chief", "started", "input_validation"))
        if on_event:
            on_event({"type": "phase", "phase": "input_validation"})

        guard_result = self.guardrails.check_input(
            query, data_path=data_path, budget=budget
        )
        if guard_result.blocked:
            return AnalysisResult(
                success=False,
                query=query,
                errors=[guard_result.reason],
            ).model_dump()

        if guard_result.warnings:
            logger.info("[chief] Input warnings: %s", guard_result.warnings)

        # ─── Phase 1+2: Plan & Execute (ReAct loop) ──────────────────
        events.append(self._make_event("chief", "running", "react_loop"))
        if on_event:
            on_event({"type": "phase", "phase": "react_loop"})

        expert_results: dict[str, Any] = {}
        # Optimization 4: structured context accumulation
        accumulated_context: dict[str, dict[str, Any]] = {}

        # Build memory context
        memory_context_str = ""
        if self.memory_manager:
            try:
                memory_context_str = self.memory_manager.build_context(query)
            except Exception as exc:
                logger.warning("[chief] Memory context failed: %s", exc)

        # Optimization 5: dynamic system prompt selection
        system_prompt = self._build_system_prompt(query, memory_context_str)
        user_message = self._format_user_message(
            query, data_path=data_path, budget=budget, scope_hint=scope_hint
        )

        messages: list[Any] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        # ReAct loop
        final_response = None
        iteration = 0
        for iteration in range(self.max_iterations):
            events.append(
                self._make_event("chief", "running", f"iteration_{iteration + 1}")
            )

            trace = TraceEntry("chief", f"react_iteration_{iteration + 1}")
            try:
                response = await self.llm.ainvoke(messages)
                trace.finish(response)
                token_counter.add(response)
            except Exception as exc:
                trace.finish()
                logger.error("[chief] LLM invocation failed: %s", exc)
                errors.append(f"LLM call: {exc}")
                traces.append(trace)
                break
            traces.append(trace)

            # Check if LLM requests tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                messages.append(response)

                # Optimization 1: parallel expert execution
                tool_call_results = await self._execute_tool_calls_parallel(
                    response.tool_calls,
                    events=events,
                    errors=errors,
                    on_event=on_event,
                    traces=traces,
                )

                # Process results and accumulate context
                for tool_name, result in tool_call_results.items():
                    expert_results[tool_name] = result
                    self._accumulate_context(
                        accumulated_context, tool_name, result
                    )
            else:
                # LLM returned final answer (no more tool calls)
                final_response = response
                break

        # ─── Phase 3: Evaluate (optional) ────────────────────────────
        if self.enable_evaluator and expert_results:
            events.append(self._make_event("chief", "running", "evaluation"))
            if on_event:
                on_event({"type": "phase", "phase": "evaluation"})

            eval_trace = TraceEntry("chief", "evaluation")
            try:
                expert_results = await self._evaluate_and_refine(
                    expert_results,
                    query,
                )
                eval_trace.finish()
            except Exception as exc:
                eval_trace.finish()
                logger.warning("[chief] Evaluation phase failed: %s", exc)
            finally:
                traces.append(eval_trace)

        # ─── Phase 4: Report (structured synthesis) ──────────────────
        events.append(self._make_event("chief", "running", "report"))
        if on_event:
            on_event({"type": "phase", "phase": "report"})

        analysis_summary = ""
        strategy_recommendation = ""
        kpi_snapshot: dict[str, Any] = {}

        if final_response:
            content = (
                final_response.content
                if isinstance(final_response.content, str)
                else str(final_response.content)
            )
            analysis_summary = content
            strategy_recommendation = content
        elif expert_results:
            # Optimization 3: use synthesis LLM for structured output
            analysis_summary, strategy_recommendation, kpi_snapshot = (
                await self._structured_synthesis(
                    messages, expert_results, accumulated_context,
                    token_counter=token_counter, traces=traces,
                )
            )

        events.append(self._make_event("chief", "completed", "done"))
        if on_event:
            on_event({"type": "completed", "summary": analysis_summary[:200]})

        # Persist to memory
        if self.memory_manager:
            try:
                self.memory_manager.store(
                    run_id=str(uuid.uuid4()),
                    query=query,
                    scope=scope_hint or "dynamic",
                    results_summary=analysis_summary[:500],
                )
            except Exception as exc:
                logger.warning("[chief] Memory persist failed: %s", exc)

        # Optimization 6 & 7: include traces and token stats in metadata
        result = AnalysisResult(
            success=len(expert_results) > 0 or bool(final_response),
            query=query,
            scope=scope_hint or "dynamic",
            analysis_summary=analysis_summary,
            strategy_recommendation=strategy_recommendation,
            kpi_snapshot=kpi_snapshot,
            expert_results=expert_results,
            errors=errors,
            metadata={
                "events": events,
                "iterations": iteration + 1,
                "token_usage": token_counter.summary(),
                "traces": [t.to_dict() for t in traces],
            },
        )
        return result.model_dump()

    # ------------------------------------------------------------------
    # Optimization 1: Parallel Tool Execution
    # ------------------------------------------------------------------

    async def _execute_tool_calls_parallel(
        self,
        tool_calls: list[dict],
        *,
        events: list[dict[str, Any]],
        errors: list[str],
        on_event: Callable[[dict], Any] | None,
        traces: list[TraceEntry],
    ) -> dict[str, Any]:
        """Execute independent tool calls in parallel via asyncio.gather.

        Returns a dict mapping tool_name → raw result string.
        """
        tasks: list[tuple[dict, Any]] = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            events.append(
                self._make_event(tool_name, "running", "tool_call")
            )
            if tool_name in self.tools:
                tasks.append((tool_call, self.tools[tool_name].ainvoke(tool_args)))
            else:
                # Unknown tool — record error immediately
                errors.append(f"Unknown tool: {tool_name}")

        if not tasks:
            # Append ToolMessages for any unknown tools so the LLM loop stays valid
            for tool_call in tool_calls:
                if tool_call["name"] not in self.tools:
                    # We need access to messages, but we don't have it here.
                    # The caller will handle adding ToolMessages for unknown tools
                    # via the returned dict containing a sentinel.
                    pass
            return {}

        # Run all tool invocations concurrently
        raw_results = await asyncio.gather(
            *[t[1] for t in tasks], return_exceptions=True
        )

        results: dict[str, Any] = {}
        for (tool_call, _), raw in zip(tasks, raw_results):
            tool_name = tool_call["name"]

            trace = TraceEntry(tool_name, "tool_call")
            if isinstance(raw, Exception):
                trace.finish()
                logger.error("[chief] Tool %s failed: %s", tool_name, raw)
                result_str = json.dumps(
                    {"success": False, "errors": [str(raw)]},
                    ensure_ascii=False,
                )
                errors.append(f"{tool_name}: {raw}")
                events.append(
                    self._make_event(tool_name, "failed", str(raw))
                )
            else:
                trace.finish()
                result_str = raw
                events.append(
                    self._make_event(tool_name, "completed", "tool_done")
                )
                # Optimization 2: callback on expert completion
                if on_event:
                    summary = ""
                    try:
                        parsed = json.loads(result_str) if isinstance(result_str, str) else result_str
                        summary = parsed.get("analysis", {}).get("summary", "")
                    except (json.JSONDecodeError, AttributeError):
                        pass
                    on_event({
                        "type": "expert_completed",
                        "expert": tool_name,
                        "summary": (summary or str(result_str))[:200],
                    })
            traces.append(trace)
            results[tool_name] = result_str

        return results

    # ------------------------------------------------------------------
    # Optimization 3: Structured Final Synthesis
    # ------------------------------------------------------------------

    async def _structured_synthesis(
        self,
        messages: list[Any],
        expert_results: dict[str, Any],
        accumulated_context: dict[str, dict[str, Any]],
        *,
        token_counter: TokenCounter,
        traces: list[TraceEntry],
    ) -> tuple[str, str, dict[str, Any]]:
        """Use the synthesis LLM (no tools) to produce a structured report.

        Returns (analysis_summary, strategy_recommendation, kpi_snapshot).
        """
        # Build a concise expert results digest for the synthesis prompt
        expert_digest_parts: list[str] = []
        for name, raw in expert_results.items():
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                analysis = parsed.get("analysis", {})
                summary = analysis.get("summary", "")
                if summary:
                    expert_digest_parts.append(f"[{name}] {summary}")
                else:
                    expert_digest_parts.append(f"[{name}] 分析完成")
            except (json.JSONDecodeError, AttributeError):
                expert_digest_parts.append(f"[{name}] 分析完成")

        expert_digest = "\n".join(expert_digest_parts)
        context_json = json.dumps(accumulated_context, ensure_ascii=False, default=str)

        synthesis_prompt = f"""\
请基于以上所有专家分析结果，输出最终的结构化分析报告。

## 专家结果摘要
{expert_digest}

## 提取的关键指标
{context_json}

请输出：
1. **综合分析摘要**（200字以内，概括所有专家发现的核心洞察）
2. **可执行策略建议**（3-5条具体、可操作的建议，每条含预期效果）
3. **关键指标快照**（从专家结果中提取的核心KPI数值）
"""
        synthesis_messages = list(messages) + [HumanMessage(content=synthesis_prompt)]

        trace = TraceEntry("chief", "structured_synthesis")
        try:
            synthesis_response = await self._synthesis_llm.ainvoke(synthesis_messages)
            trace.finish(synthesis_response)
            token_counter.add(synthesis_response)

            content = (
                synthesis_response.content
                if isinstance(synthesis_response.content, str)
                else str(synthesis_response.content)
            )

            # Extract KPI snapshot from accumulated context
            kpi_snapshot = self._extract_kpi_snapshot(accumulated_context)

            # Split content into summary and strategy sections
            analysis_summary, strategy_recommendation = self._split_synthesis(content)
            return analysis_summary, strategy_recommendation, kpi_snapshot

        except Exception as exc:
            trace.finish()
            logger.warning("[chief] Structured synthesis failed, using fallback: %s", exc)
            # Fallback: build summary from raw expert results
            summary = self._build_summary_from_results(expert_results)
            kpi_snapshot = self._extract_kpi_snapshot(accumulated_context)
            return summary, summary, kpi_snapshot
        finally:
            traces.append(trace)

    # ------------------------------------------------------------------
    # Phase 3: Evaluator-Optimizer Loop
    # ------------------------------------------------------------------

    async def _evaluate_and_refine(
        self,
        expert_results: dict[str, Any],
        query: str,
        *,
        max_rounds: int = 2,
        threshold: float = 0.7,
    ) -> dict[str, Any]:
        """Evaluator-Optimizer loop: evaluate quality, refine if needed."""
        from src.core.evaluator import batch_evaluate

        for round_num in range(max_rounds):
            scores = await batch_evaluate(expert_results, query)

            # Log scores
            for name, score in scores.items():
                logger.info(f"[Round {round_num+1}] {name}: overall={score.overall:.2f}")

            # Find low-quality results
            low_quality = {
                name: score for name, score in scores.items()
                if score.overall < threshold
            }

            if not low_quality:
                logger.info(f"All experts pass quality threshold ({threshold})")
                break

            # Re-invoke low-quality experts with feedback
            for name, score in low_quality.items():
                logger.info(f"Re-invoking {name} with feedback: {score.reasoning[:100]}")
                try:
                    refined = await self._invoke_expert_with_feedback(
                        name, query, score.reasoning,
                    )
                    if refined:
                        expert_results[name] = refined
                except Exception as e:
                    logger.warning(f"Failed to refine {name}: {e}")

        return expert_results

    async def _invoke_expert_with_feedback(
        self,
        expert_name: str,
        query: str,
        feedback: str,
    ) -> Any:
        """Re-invoke an expert tool, appending quality feedback as context."""
        if expert_name not in self.tools:
            logger.warning(f"[chief] Unknown expert for refinement: {expert_name}")
            return None

        tool = self.tools[expert_name]
        # Build augmented args with the original query plus evaluator feedback
        refined_args = {
            "query": query,
            "additional_context": (
                f"[Quality Feedback - please improve] {feedback}"
            ),
        }
        try:
            result = await tool.ainvoke(refined_args)
            return result
        except Exception as exc:
            logger.warning("[chief] Expert refinement call failed for %s: %s", expert_name, exc)
            return None

    # ------------------------------------------------------------------
    # Optimization 4: Structured Context Accumulation
    # ------------------------------------------------------------------

    @staticmethod
    def _accumulate_context(
        accumulated: dict[str, dict[str, Any]],
        tool_name: str,
        result: Any,
    ) -> None:
        """Extract key metrics from a tool result into a structured dict."""
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
            metrics: dict[str, Any] = {}

            # Top-level metric keys
            for key in _METRIC_EXTRACTION_KEYS:
                if key in parsed:
                    metrics[key] = parsed[key]

            # Nested under "analysis"
            analysis = parsed.get("analysis", {})
            if isinstance(analysis, dict):
                for key in _METRIC_EXTRACTION_KEYS:
                    if key in analysis and key not in metrics:
                        metrics[key] = analysis[key]
                # Grab the text summary for reference
                if "summary" in analysis:
                    metrics["summary"] = str(analysis["summary"])[:300]

            # Churn risk nested data
            churn_risk = parsed.get("churn_risk", {})
            if isinstance(churn_risk, dict):
                for key in ("high_risk_ratio", "medium_risk_ratio", "low_risk_ratio", "train_auc"):
                    if key in churn_risk:
                        metrics[key] = churn_risk[key]

            if metrics:
                accumulated[tool_name] = metrics
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    # ------------------------------------------------------------------
    # Optimization 5: Dynamic System Prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _build_system_prompt(query: str, memory_context: str = "") -> str:
        """Choose a full or short system prompt based on query complexity."""
        is_simple = (
            len(query) < 20
            or any(kw in query for kw in ("?", "多少", "是什么", "查一下", "多少个"))
        )

        template = _CHIEF_SYSTEM_PROMPT_SHORT if is_simple else _CHIEF_SYSTEM_PROMPT_FULL

        memory_section = (
            f"\n## 历史分析上下文\n{memory_context}" if memory_context else ""
        )
        return template.format(memory_context=memory_section)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_user_message(
        query: str,
        *,
        data_path: str = "",
        budget: float = 0,
        scope_hint: str = "",
    ) -> str:
        """Format the user query with context."""
        parts = [f"## 用户查询\n{query}"]
        if data_path:
            parts.append(f"## 数据路径\n{data_path}")
        if budget > 0:
            parts.append(f"## 预算\n{budget:,.0f} 元")
        if scope_hint:
            parts.append(f"## 分析范围提示\n{scope_hint}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_summary_from_results(expert_results: dict[str, Any]) -> str:
        """Build a summary from expert tool results."""
        summaries = []
        for name, result_str in expert_results.items():
            try:
                result = (
                    json.loads(result_str)
                    if isinstance(result_str, str)
                    else result_str
                )
                analysis = result.get("analysis", {})
                summary = analysis.get("summary", "")
                if summary:
                    summaries.append(f"**{name}**: {summary}")
            except (json.JSONDecodeError, AttributeError):
                summaries.append(f"**{name}**: 分析完成")

        if summaries:
            return "\n\n".join(summaries)
        return "分析完成，但未生成详细摘要。"

    @staticmethod
    def _make_event(
        agent: str, status: str, step: str
    ) -> dict[str, Any]:
        """Create a simple event dict."""
        return {
            "agent": agent,
            "status": status,
            "step": step,
            "timestamp": time.time(),
        }

    @staticmethod
    def _extract_kpi_snapshot(
        accumulated_context: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Merge metrics from all experts into a single KPI snapshot."""
        snapshot: dict[str, Any] = {}
        for _expert, metrics in accumulated_context.items():
            for key in _METRIC_EXTRACTION_KEYS:
                if key in metrics and key not in snapshot:
                    snapshot[key] = metrics[key]
        return snapshot

    @staticmethod
    def _split_synthesis(content: str) -> tuple[str, str]:
        """Best-effort split of synthesis content into summary + strategy.

        Looks for common section markers; falls back to returning the whole
        content as both fields.
        """
        lower = content.lower()
        strategy_markers = [
            "## 策略建议", "## 可执行策略", "## 策略", "**策略建议**",
            "**可执行策略**", "策略建议", "## 建议",
        ]
        split_idx = -1
        for marker in strategy_markers:
            idx = lower.find(marker.lower())
            if idx != -1:
                split_idx = idx
                break

        if split_idx > 0:
            return content[:split_idx].strip(), content[split_idx:].strip()
        return content, content
