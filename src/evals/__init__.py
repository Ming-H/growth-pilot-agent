"""Agent evaluation framework for GrowthPilot.

Provides multi-dimensional evaluation of agent outputs:
- task_completion: Whether key expected information is present
- tool_efficiency: How efficiently tools were invoked
- output_quality: LLM-as-Judge quality scoring
- latency: Response time tracking
"""

from src.evals.dataset import EvalDataset, EvalSample
from src.evals.evaluator import AgentEvaluator, EvalResult
from src.evals.report import EvalReport

__all__ = [
    "AgentEvaluator",
    "EvalResult",
    "EvalDataset",
    "EvalSample",
    "EvalReport",
]
