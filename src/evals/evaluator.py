"""AgentEvaluator - Multi-dimensional agent output evaluation.

Core evaluation logic with LLM-as-Judge for quality assessment.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from src.core.llm_factory import create_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class EvalDimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    score: float = Field(ge=0.0, le=1.0, description="Normalized score 0-1")
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Complete evaluation result for a single agent run."""

    agent_name: str = ""
    input_query: str = ""
    dimensions: dict[str, EvalDimensionScore] = Field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """Weighted average of all dimension scores."""
        weights = {
            "task_completion": 0.35,
            "tool_efficiency": 0.15,
            "output_quality": 0.35,
            "latency": 0.15,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for dim_name, dim_score in self.dimensions.items():
            w = weights.get(dim_name, 0.1)
            weighted_sum += dim_score.score * w
            total_weight += w
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    @property
    def task_completion(self) -> float:
        dim = self.dimensions.get("task_completion")
        return dim.score if dim else 0.0

    @property
    def tool_efficiency(self) -> float:
        dim = self.dimensions.get("tool_efficiency")
        return dim.score if dim else 0.0

    @property
    def output_quality(self) -> float:
        dim = self.dimensions.get("output_quality")
        return dim.score if dim else 0.0

    @property
    def latency(self) -> float:
        dim = self.dimensions.get("latency")
        return dim.score if dim else 0.0


# ---------------------------------------------------------------------------
# LLM-as-Judge prompt templates
# ---------------------------------------------------------------------------

_QUALITY_JUDGE_PROMPT = """\
你是一个专业的 AI Agent 输出质量评审专家。请对以下 Agent 输出进行质量评估。

## 评估背景
- Agent 名称: {agent_name}
- 用户查询: {input_query}
- 参考答案: {reference_answer}

## Agent 实际输出
{actual_output}

## 评估标准 (每项 0-10 分)

1. **准确性** (accuracy): 输出中的事实、数据、结论是否准确？是否与参考答案一致？
2. **完整性** (completeness): 输出是否完整覆盖了用户查询的所有方面？是否遗漏关键信息？
3. **相关性** (relevance): 输出是否紧密围绕用户查询？是否包含不相关的内容？
4. **可操作性** (actionability): 输出是否提供了具体、可执行的建议或方案？
5. **表达清晰度** (clarity): 输出的语言是否清晰、有条理？是否易于理解？

## 输出要求
请以 JSON 格式输出评估结果:
```json
{{
  "accuracy": <0-10>,
  "completeness": <0-10>,
  "relevance": <0-10>,
  "actionability": <0-10>,
  "clarity": <0-10>,
  "overall_score": <0-10>,
  "reason": "评估理由（50-200字）"
}}
```"""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class AgentEvaluator:
    """Multi-dimensional agent output evaluator.

    Evaluates agent runs across four dimensions:
    - task_completion: Checks if expected keys/info are present
    - tool_efficiency: Scores tool call patterns
    - output_quality: Uses LLM-as-Judge for quality scoring
    - latency: Normalizes response time
    """

    def __init__(self, llm: Any = None) -> None:
        """Initialize evaluator.

        Args:
            llm: Optional pre-configured LLM for judge. If None, creates
                 a fast-tier model via llm_factory.
        """
        self._llm = llm
        self._latency_benchmarks: dict[str, float] = {
            "excellent": 2.0,   # seconds
            "good": 5.0,
            "acceptable": 10.0,
            "slow": 30.0,
        }

    def _get_llm(self) -> Any:
        """Get or create the judge LLM instance."""
        if self._llm is None:
            self._llm = create_llm(tier="fast")
        return self._llm

    # ------------------------------------------------------------------
    # Main evaluation entry point
    # ------------------------------------------------------------------

    async def evaluate_agent_run(
        self,
        agent_name: str,
        input_query: str,
        actual_output: str | dict[str, Any],
        expected_output: str | dict[str, Any] | None = None,
        expected_keys: list[str] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        latency_seconds: float | None = None,
        reference_answer: str = "",
    ) -> EvalResult:
        """Evaluate a single agent run across all dimensions.

        Args:
            agent_name: Name of the agent being evaluated.
            input_query: The original user query / input.
            actual_output: The agent's actual output (str or dict).
            expected_output: Expected output for comparison (optional).
            expected_keys: Keys that should be present in the output.
            tool_calls: List of tool call records for efficiency scoring.
            latency_seconds: Wall-clock time of the agent run.
            reference_answer: Reference answer for LLM-as-Judge.

        Returns:
            EvalResult with dimension scores and overall score.
        """
        result = EvalResult(
            agent_name=agent_name,
            input_query=input_query,
        )

        # Normalize output to string
        output_str = (
            json.dumps(actual_output, ensure_ascii=False, indent=2)
            if isinstance(actual_output, dict)
            else str(actual_output)
        )

        # 1. Task completion
        result.dimensions["task_completion"] = self._eval_task_completion(
            output_str=output_str,
            expected_output=expected_output,
            expected_keys=expected_keys or [],
        )

        # 2. Tool efficiency
        result.dimensions["tool_efficiency"] = self._eval_tool_efficiency(
            tool_calls=tool_calls or [],
            agent_name=agent_name,
        )

        # 3. Output quality (LLM-as-Judge)
        result.dimensions["output_quality"] = await self._eval_output_quality(
            agent_name=agent_name,
            input_query=input_query,
            actual_output=output_str,
            reference_answer=reference_answer or (
                json.dumps(expected_output, ensure_ascii=False)
                if isinstance(expected_output, dict)
                else str(expected_output or "")
            ),
        )

        # 4. Latency
        result.dimensions["latency"] = self._eval_latency(latency_seconds)

        return result

    # ------------------------------------------------------------------
    # Dimension evaluators
    # ------------------------------------------------------------------

    def _eval_task_completion(
        self,
        output_str: str,
        expected_output: str | dict[str, Any] | None,
        expected_keys: list[str],
    ) -> EvalDimensionScore:
        """Evaluate task completion by checking key information presence.

        Scoring:
        - If expected_keys provided: ratio of found keys
        - If expected_output provided: string similarity heuristic
        - Otherwise: basic completeness check
        """
        output_lower = output_str.lower()
        details: dict[str, Any] = {}
        score = 0.0

        if expected_keys:
            found = [k for k in expected_keys if k.lower() in output_lower]
            missing = [k for k in expected_keys if k.lower() not in output_lower]
            score = len(found) / len(expected_keys) if expected_keys else 1.0
            details["found_keys"] = found
            details["missing_keys"] = missing
            details["key_coverage"] = f"{len(found)}/{len(expected_keys)}"

        elif expected_output is not None:
            expected_str = (
                json.dumps(expected_output, ensure_ascii=False)
                if isinstance(expected_output, dict)
                else str(expected_output)
            )
            # Simple token overlap ratio
            expected_tokens = set(expected_str.lower().split())
            output_tokens = set(output_lower.split())
            if expected_tokens:
                overlap = expected_tokens & output_tokens
                score = len(overlap) / len(expected_tokens)
            else:
                score = 1.0
            details["overlap_ratio"] = score
        else:
            # Basic check: non-empty output gets 0.5, substantial output gets 1.0
            score = 0.5 if len(output_str.strip()) > 0 else 0.0
            if len(output_str) > 100:
                score = 1.0
            details["check_type"] = "basic"

        reason = "Task completion assessment: "
        if score >= 0.8:
            reason += "Output covers most expected information."
        elif score >= 0.5:
            reason += "Output partially covers expected information."
        else:
            reason += "Output is missing significant expected information."

        return EvalDimensionScore(score=score, reason=reason, details=details)

    def _eval_tool_efficiency(
        self,
        tool_calls: list[dict[str, Any]],
        agent_name: str,
    ) -> EvalDimensionScore:
        """Evaluate tool call efficiency.

        Ideal: 3-6 tool calls for most agents. Too few = under-utilization,
        too many = wasted computation.
        """
        n_calls = len(tool_calls)
        details: dict[str, Any] = {
            "total_calls": n_calls,
        }

        # Optimal call counts per agent type
        optimal_ranges: dict[str, tuple[int, int]] = {
            "prospect": (4, 8),
            "conversion": (3, 7),
            "subsidy": (3, 6),
            "retention": (3, 7),
            "ad": (3, 6),
            "orchestrator": (0, 2),
        }
        low, high = optimal_ranges.get(agent_name, (2, 8))
        details["optimal_range"] = f"{low}-{high}"

        if n_calls == 0:
            # No tool calls - score depends on agent
            if agent_name == "orchestrator":
                score = 1.0
                reason = "Orchestrator does not make tool calls directly."
            else:
                score = 0.1
                reason = "No tool calls made - agent may not be functioning."
        elif low <= n_calls <= high:
            score = 1.0
            reason = f"Tool call count ({n_calls}) is within optimal range ({low}-{high})."
        elif n_calls < low:
            score = 0.5 + 0.5 * (n_calls / low)
            reason = f"Tool call count ({n_calls}) is below optimal ({low}-{high})."
        else:
            # Exponential decay beyond high
            excess = n_calls - high
            score = max(0.2, 1.0 / (1.0 + 0.2 * excess))
            reason = f"Tool call count ({n_calls}) exceeds optimal ({low}-{high})."

        # Check for duplicate calls
        if n_calls > 1:
            call_names = [c.get("tool", c.get("name", "")) for c in tool_calls]
            duplicates = len(call_names) - len(set(call_names))
            if duplicates > 0:
                score *= 0.9  # 10% penalty for duplicate calls
                details["duplicate_calls"] = duplicates
                reason += f" {duplicates} duplicate tool calls detected."

        details["tool_names"] = list(set(
            c.get("tool", c.get("name", "unknown")) for c in tool_calls
        ))

        return EvalDimensionScore(score=min(1.0, score), reason=reason, details=details)

    async def _eval_output_quality(
        self,
        agent_name: str,
        input_query: str,
        actual_output: str,
        reference_answer: str,
    ) -> EvalDimensionScore:
        """Use LLM-as-Judge to evaluate output quality.

        Sends the actual output along with the reference answer to a fast
        LLM model for structured quality scoring.
        """
        prompt = _QUALITY_JUDGE_PROMPT.format(
            agent_name=agent_name,
            input_query=input_query,
            actual_output=actual_output[:3000],  # Truncate to avoid token limits
            reference_answer=reference_answer[:2000] if reference_answer else "N/A",
        )

        try:
            llm = self._get_llm()
            from langchain_core.messages import HumanMessage
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content if isinstance(response.content, str) else str(response.content)

            # Parse judge response
            parsed = self._parse_judge_response(response_text)
            overall = parsed.get("overall_score", 5.0)
            score = min(1.0, max(0.0, overall / 10.0))
            reason = parsed.get("reason", "LLM judge assessment completed.")

            details = {
                "accuracy": parsed.get("accuracy", 0),
                "completeness": parsed.get("completeness", 0),
                "relevance": parsed.get("relevance", 0),
                "actionability": parsed.get("actionability", 0),
                "clarity": parsed.get("clarity", 0),
                "judge_raw_score": overall,
            }

            return EvalDimensionScore(score=score, reason=reason, details=details)

        except Exception as exc:
            logger.warning("LLM-as-Judge failed: %s", exc)
            return EvalDimensionScore(
                score=0.5,
                reason=f"LLM-as-Judge unavailable ({exc}), using default score.",
                details={"error": str(exc)},
            )

    def _eval_latency(self, latency_seconds: float | None) -> EvalDimensionScore:
        """Evaluate response latency.

        Scoring thresholds:
        - < 2s: 1.0 (excellent)
        - < 5s: 0.8 (good)
        - < 10s: 0.6 (acceptable)
        - < 30s: 0.3 (slow)
        - >= 30s: 0.1 (unacceptable)
        """
        if latency_seconds is None:
            return EvalDimensionScore(
                score=0.5,
                reason="Latency not measured.",
                details={},
            )

        bm = self._latency_benchmarks
        details: dict[str, Any] = {"latency_seconds": round(latency_seconds, 3)}

        if latency_seconds <= bm["excellent"]:
            score = 1.0
            reason = f"Excellent response time ({latency_seconds:.2f}s <= {bm['excellent']}s)."
        elif latency_seconds <= bm["good"]:
            ratio = (latency_seconds - bm["excellent"]) / (bm["good"] - bm["excellent"])
            score = 1.0 - 0.2 * ratio
            reason = f"Good response time ({latency_seconds:.2f}s)."
        elif latency_seconds <= bm["acceptable"]:
            ratio = (latency_seconds - bm["good"]) / (bm["acceptable"] - bm["good"])
            score = 0.8 - 0.2 * ratio
            reason = f"Acceptable response time ({latency_seconds:.2f}s)."
        elif latency_seconds <= bm["slow"]:
            ratio = (latency_seconds - bm["acceptable"]) / (bm["slow"] - bm["acceptable"])
            score = 0.6 - 0.3 * ratio
            reason = f"Slow response time ({latency_seconds:.2f}s)."
        else:
            score = 0.1
            reason = f"Unacceptably slow ({latency_seconds:.2f}s > {bm['slow']}s)."

        return EvalDimensionScore(score=score, reason=reason, details=details)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_judge_response(text: str) -> dict[str, Any]:
        """Parse JSON from LLM judge response."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON block
        for marker in ["```json", "```"]:
            if marker in text:
                start = text.index(marker) + len(marker)
                end_idx = text.find("```", start)
                if end_idx > start:
                    try:
                        return json.loads(text[start:end_idx].strip())
                    except (json.JSONDecodeError, ValueError):
                        continue

        # Fallback: try to find any JSON-like structure
        import re
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        return {"overall_score": 5.0, "reason": "Failed to parse judge response"}
