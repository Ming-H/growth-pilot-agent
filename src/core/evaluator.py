"""Quality evaluator for expert agent outputs.

Implements the Evaluator-Optimizer pattern:
evaluate expert output quality -> if below threshold, generate feedback -> re-invoke expert.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from src.core.llm_factory import create_llm

logger = logging.getLogger(__name__)


class QualityScore(BaseModel):
    """Quality assessment for a single expert's output."""

    completeness: float = Field(ge=0, le=1, description="Does the analysis cover all relevant aspects?")
    actionability: float = Field(ge=0, le=1, description="Are recommendations specific and implementable?")
    data_grounding: float = Field(ge=0, le=1, description="Are claims backed by data/metrics?")
    overall: float = Field(ge=0, le=1, description="Weighted composite score")
    reasoning: str = Field(default="", description="Feedback for improvement if score is low")
    evaluation_failed: bool = Field(default=False, description="Whether the evaluation itself failed")


EVAL_PROMPT = """You are a quality evaluator for growth analysis outputs.
Rate the following analysis on these dimensions (0.0 to 1.0):

- completeness: Does it cover all relevant aspects of the query?
- actionability: Are the recommendations specific enough to implement?
- data_grounding: Are claims and recommendations backed by data/metrics?

Expert: {expert_name}
Query: {query}
Analysis Result:
{analysis}

Respond with ONLY a JSON object:
{{"completeness": 0.0-1.0, "actionability": 0.0-1.0, "data_grounding": 0.0-1.0, "overall": 0.0-1.0, "reasoning": "brief feedback"}}

Be strict but fair. Most good analyses score 0.7-0.9. Only truly exceptional ones score above 0.9.
If the analysis is clearly incomplete or lacks substance, score below 0.6.
"""


async def evaluate_expert_output(
    expert_name: str,
    query: str,
    analysis: str,
    *,
    tier: str = "fast",
) -> QualityScore:
    """Evaluate a single expert's output quality using LLM-as-judge."""
    llm = create_llm(tier=tier)
    prompt = EVAL_PROMPT.format(
        expert_name=expert_name,
        query=query,
        analysis=analysis[:3000],  # Truncate to avoid token limits
    )
    try:
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        # Parse JSON from response
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        return QualityScore(**data)
    except Exception as e:
        logger.warning("Evaluation failed for %s: %s", expert_name, e)
        return QualityScore(
            completeness=0.5,
            actionability=0.5,
            data_grounding=0.5,
            overall=0.5,
            reasoning=f"Evaluation failed: {e}",
            evaluation_failed=True,
        )


async def batch_evaluate(
    expert_results: dict[str, Any],
    query: str,
    *,
    tier: str = "fast",
) -> dict[str, QualityScore]:
    """Evaluate multiple expert outputs in parallel."""
    import asyncio

    tasks = {
        name: evaluate_expert_output(name, query, str(result), tier=tier)
        for name, result in expert_results.items()
    }
    results: dict[str, QualityScore] = {}
    coros = list(tasks.values())
    names = list(tasks.keys())
    scores = await asyncio.gather(*coros, return_exceptions=True)
    for name, score in zip(names, scores):
        if isinstance(score, Exception):
            logger.warning("Evaluation error for %s: %s", name, score)
            results[name] = QualityScore(
                completeness=0.5,
                actionability=0.5,
                data_grounding=0.5,
                overall=0.5,
                reasoning=f"Error: {score}",
                evaluation_failed=True,
            )
        else:
            results[name] = score
    return results
