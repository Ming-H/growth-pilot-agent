"""Agent base class."""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.hooks import LoggingHook, MetricsHook, PostRunHook, PreRunHook, TracingHook
from src.middleware import AgentMiddleware, build_middleware_stack

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Self-evaluation prompt (internal)
# ---------------------------------------------------------------------------
_SELF_EVAL_SYSTEM_PROMPT = """\
你是一个严格的质量评估专家。你的任务是评估货运增长分析 Agent 的输出质量。

评估维度（各 0-1 分）：
1. **完整性** (completeness): 分析是否覆盖了所有关键维度，是否有遗漏
2. **可操作性** (actionability): 建议是否具体、可执行，是否有明确的时间线和负责人
3. **数据支撑度** (data_grounding): 结论是否有数据支撑，是否引用了具体指标

输出严格的 JSON 格式：
{
  "completeness": 0.0-1.0,
  "actionability": 0.0-1.0,
  "data_grounding": 0.0-1.0,
  "overall": 0.0-1.0,
  "reasoning": "简短评估理由 (1-2句话)"
}
"""

_SELF_EVAL_USER_TEMPLATE = """\
请评估以下 Agent 输出质量：

## Agent: {agent_name}

## 输出内容:
{output_text}

## 评估标准：
- 完整性：是否覆盖了业务分析所需的关键维度（用户分群、转化漏斗、ROI等）
- 可操作性：建议是否可直接执行（有具体金额、时间节点、渠道等）
- 数据支撑度：是否有具体数据指标支撑结论

请输出严格 JSON，不要包含其他内容。"""


class BaseAgent(ABC):
    """Abstract base class for all GrowthPilot agents."""

    name: str = "base_agent"
    description: str = "Base agent"
    model_tier: str = "default"

    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str = "",
        *,
        enable_self_eval: bool = False,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.enable_self_eval = enable_self_eval
        self._pre_hooks: list[PreRunHook] = []
        self._post_hooks: list[PostRunHook] = []
        self._memory: Any = None

        # --- Middleware stack (activated) ---
        self._middleware_stack: list[AgentMiddleware] = build_middleware_stack()

        # --- Auto-register default hooks ---
        # TracingHook implements both PreRunHook and PostRunHook
        tracing = TracingHook()
        self._pre_hooks.append(tracing)
        self._post_hooks.append(tracing)

        # LoggingHook implements both PreRunHook and PostRunHook
        log_hook = LoggingHook()
        self._pre_hooks.append(log_hook)
        self._post_hooks.append(log_hook)

        # MetricsHook (optional, lightweight)
        metrics_hook = MetricsHook()
        self._pre_hooks.append(metrics_hook)
        self._post_hooks.append(metrics_hook)

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent with hooks. Returns a partial state update."""
        state_dict = dict(state)

        # Execute pre-run hooks
        for hook in self._pre_hooks:
            if hasattr(hook, "on_pre_run"):
                state_dict = await hook.on_pre_run(self.name, state_dict)

        # Execute the core logic
        result = await self._execute(state)

        # --- Quality self-evaluation (optional) ---
        if self.enable_self_eval:
            result = await self._self_evaluate(result)

        # Execute post-run hooks
        for hook in self._post_hooks:
            if hasattr(hook, "on_post_run"):
                result = await hook.on_post_run(self.name, result, state_dict)

        return result

    @abstractmethod
    async def _execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Core agent logic. Subclasses must implement this."""
        ...

    # ------------------------------------------------------------------
    # Quality self-evaluation
    # ------------------------------------------------------------------

    async def _self_evaluate(self, result: dict[str, Any]) -> dict[str, Any]:
        """Evaluate output quality using the fast model tier.

        Scores the result on three dimensions:
        - completeness (0-1)
        - actionability (0-1)
        - data_grounding (0-1)

        Writes the score into ``result["_quality_score"]``.
        This is opt-in — only runs when ``enable_self_eval=True``.
        """
        try:
            from src.core.llm_factory import create_llm

            eval_llm = create_llm(tier="fast")

            # Serialize result for evaluation (skip internal keys)
            eval_output = {
                k: v for k, v in result.items()
                if not k.startswith("_") and k != "errors"
            }
            output_text = json.dumps(eval_output, ensure_ascii=False, default=str)
            # Truncate to avoid overly long prompts
            if len(output_text) > 4000:
                output_text = output_text[:4000] + "...(truncated)"

            user_msg = _SELF_EVAL_USER_TEMPLATE.format(
                agent_name=self.name,
                output_text=output_text,
            )

            messages = [
                SystemMessage(content=_SELF_EVAL_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
            response = await eval_llm.ainvoke(messages)
            content = response.content if isinstance(response.content, str) else str(response.content)

            # Parse evaluation result
            eval_result = self._parse_json_response(content)

            quality_score = {
                "completeness": float(eval_result.get("completeness", 0.5)),
                "actionability": float(eval_result.get("actionability", 0.5)),
                "data_grounding": float(eval_result.get("data_grounding", 0.5)),
                "overall": float(eval_result.get("overall", 0.5)),
                "reasoning": eval_result.get("reasoning", ""),
            }

            # Clamp values to [0, 1]
            for key in ("completeness", "actionability", "data_grounding", "overall"):
                quality_score[key] = max(0.0, min(1.0, quality_score[key]))

            result["_quality_score"] = quality_score
            logger.info(
                "[%s] Quality self-eval: overall=%.2f (completeness=%.2f, actionability=%.2f, data=%.2f)",
                self.name,
                quality_score["overall"],
                quality_score["completeness"],
                quality_score["actionability"],
                quality_score["data_grounding"],
            )
        except Exception as exc:
            logger.warning("[%s] Quality self-evaluation failed: %s", self.name, exc)
            result["_quality_score"] = {
                "completeness": 0.0,
                "actionability": 0.0,
                "data_grounding": 0.0,
                "overall": 0.0,
                "reasoning": f"Evaluation failed: {exc}",
            }

        return result

    # ------------------------------------------------------------------
    # LLM invocation
    # ------------------------------------------------------------------

    async def _invoke_llm(
        self,
        user_message: str,
        system_override: str | None = None,
    ) -> str:
        """Invoke LLM with system + user message, return content.

        The actual LLM call is wrapped through the middleware stack so that
        logging, retry, and error-handling middleware are applied in order.
        """
        system = system_override or self.system_prompt
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=user_message))

        # Build the innermost handler: the raw LLM call
        async def _raw_handler(request: dict[str, Any]) -> Any:
            return await self.llm.ainvoke(messages)

        # Wrap the handler through the middleware stack (outermost first)
        handler = _raw_handler
        for mw in reversed(self._middleware_stack):
            prev_handler = handler
            handler = self._make_mw_handler(mw, prev_handler)

        request_payload: dict[str, Any] = {
            "agent": self.name,
            "messages": [
                {"role": "system" if isinstance(m, SystemMessage) else "user", "content": m.content}
                for m in messages
            ],
        }
        response = await handler(request_payload)
        return response.content if isinstance(response.content, str) else str(response.content)

    @staticmethod
    def _make_mw_handler(mw: Any, next_handler: Any) -> Any:
        """Create a closure that delegates to *mw.wrap_model_call*."""
        async def _handler(request: dict[str, Any]) -> Any:
            return await mw.wrap_model_call(request, next_handler)
        return _handler

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Try to extract JSON from LLM response."""
        try:
            # Try direct parse
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block
        for marker in ["```json", "```"]:
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start)
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    continue

        return {"raw_response": text}

    def _build_prompt_context(self, state: dict[str, Any]) -> str:
        """Build context string from current state for prompt injection."""
        parts = [f"用户查询: {state.get('query', '')}"]
        if state.get("prospect_results"):
            parts.append("潜客识别结果: 已完成")
        if state.get("budget"):
            parts.append(f"可用预算: {state['budget']}")
        return "\n".join(parts)
