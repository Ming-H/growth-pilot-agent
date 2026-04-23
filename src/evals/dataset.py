"""EvalDataset - Evaluation dataset management for GrowthPilot agents.

Loads evaluation samples from JSON and provides built-in sample data
covering all agent types in the freight growth domain.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class EvalSample(BaseModel):
    """A single evaluation sample.

    Attributes:
        id: Unique sample identifier.
        input_query: The user query to send to the agent.
        scope: Expected scope detection result (prospect/conversion/subsidy/retention/ad/full).
        expected_keys: Keys that should appear in the agent output.
        reference_answer: Reference answer for LLM-as-Judge comparison.
        agent_name: Which agent this sample primarily targets.
        metadata: Additional metadata (e.g., difficulty, tags).
    """

    id: str = ""
    input_query: str = ""
    scope: str = "full"
    expected_keys: list[str] = Field(default_factory=list)
    reference_answer: str = ""
    agent_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in sample data (freight growth scenarios)
# ---------------------------------------------------------------------------

_BUILTIN_SAMPLES: list[dict[str, Any]] = [
    # --- Prospect Agent ---
    {
        "id": "prospect_001",
        "input_query": "帮我分析最近一周的潜客识别情况，高意向用户有多少？",
        "scope": "prospect",
        "expected_keys": ["user_count", "intent_metrics", "segment_summary", "analysis"],
        "reference_answer": "应包含用户总数、意向模型AUC指标、高意向用户分群占比，以及基于特征工程的用户画像描述。",
        "agent_name": "prospect",
        "metadata": {"difficulty": "easy", "tags": ["意向预测", "用户分群"]},
    },
    {
        "id": "prospect_002",
        "input_query": "从滴滴出行用户中筛选出可能有搬家需求的用户群体，评估他们的LTV",
        "scope": "prospect",
        "expected_keys": ["user_count", "segment_summary", "rfm_result_count", "top_users_sample", "analysis"],
        "reference_answer": "应包含RFM分层结果、LTV预测值分布、高价值用户特征画像，以及触达优先级建议。",
        "agent_name": "prospect",
        "metadata": {"difficulty": "medium", "tags": ["LTV预测", "用户画像", "RFM分群"]},
    },
    {
        "id": "prospect_003",
        "input_query": "潜客评分模型的表现怎么样？能优化吗？",
        "scope": "prospect",
        "expected_keys": ["intent_metrics", "analysis"],
        "reference_answer": "应包含模型AUC、Accuracy等指标评估，模型瓶颈分析，以及特征工程优化建议。",
        "agent_name": "prospect",
        "metadata": {"difficulty": "medium", "tags": ["模型评估", "特征工程"]},
    },

    # --- Conversion Agent ---
    {
        "id": "conversion_001",
        "input_query": "分析本周货运转化漏斗数据，哪个环节流失最严重？",
        "scope": "conversion",
        "expected_keys": ["funnel_result", "analysis"],
        "reference_answer": "应包含各漏斗环节转化率（浏览→点击→下单→支付），流失最严重环节识别，以及优化建议。",
        "agent_name": "conversion",
        "metadata": {"difficulty": "easy", "tags": ["漏斗分析", "转化率"]},
    },
    {
        "id": "conversion_002",
        "input_query": "设计一批优惠券策略，提升新用户首单转化率，预算5万元",
        "scope": "conversion",
        "expected_keys": ["coupon_results", "analysis"],
        "reference_answer": "应包含优惠券面额设计、适用人群、预期核销率、ROI预估，以及预算分配方案。",
        "agent_name": "conversion",
        "metadata": {"difficulty": "medium", "tags": ["优惠券设计", "首单转化", "预算分配"]},
    },
    {
        "id": "conversion_003",
        "input_query": "金刚位和首页弹窗的流量位如何分配才能最大化转化？",
        "scope": "conversion",
        "expected_keys": ["slot_result", "analysis"],
        "reference_answer": "应包含各流量位的曝光量、CTR、CVR数据，slot分配建议，以及A/B测试方案。",
        "agent_name": "conversion",
        "metadata": {"difficulty": "hard", "tags": ["流量位分配", "金刚位", "A/B测试"]},
    },

    # --- Subsidy Agent ---
    {
        "id": "subsidy_001",
        "input_query": "评估当前补贴策略的ROI，哪些线路的补贴效果最好？",
        "scope": "subsidy",
        "expected_keys": ["ate", "elasticity", "expected_roi", "analysis"],
        "reference_answer": "应包含补贴因果效应(ATE)评估、各线路价格弹性分析、ROI排名，以及补贴效率改进建议。",
        "agent_name": "subsidy",
        "metadata": {"difficulty": "medium", "tags": ["ROI评估", "因果推断", "价格弹性"]},
    },
    {
        "id": "subsidy_002",
        "input_query": "下月预算200万，如何分配到各城市各线路实现最优ROI？",
        "scope": "subsidy",
        "expected_keys": ["budget_plan", "allocation_plan", "expected_roi", "analysis"],
        "reference_answer": "应包含城市/线路级预算分配方案、预期ROI预估、弹性系数约束，以及风险控制建议。",
        "agent_name": "subsidy",
        "metadata": {"difficulty": "hard", "tags": ["预算分配", "最优化", "ROI"]},
    },

    # --- Retention Agent ---
    {
        "id": "retention_001",
        "input_query": "分析本月用户流失情况，高流失风险用户有什么特征？",
        "scope": "retention",
        "expected_keys": ["churn_risk", "churn_factors", "analysis"],
        "reference_answer": "应包含流失风险分布（高/中/低比例）、模型AUC、流失关键因素排序，以及高流失用户画像。",
        "agent_name": "retention",
        "metadata": {"difficulty": "easy", "tags": ["流失预测", "用户画像"]},
    },
    {
        "id": "retention_002",
        "input_query": "设计一个流失用户挽回方案，针对30天未下单的用户",
        "scope": "retention",
        "expected_keys": ["winback_plans", "winback_priority", "analysis"],
        "reference_answer": "应包含挽回优先级排序、分群挽回策略（Push/优惠券/专属客服）、预期挽回率，以及执行时间表。",
        "agent_name": "retention",
        "metadata": {"difficulty": "medium", "tags": ["流失挽回", "Winback"]},
    },
    {
        "id": "retention_003",
        "input_query": "新用户培育计划执行效果怎么样？各群组的留存曲线如何？",
        "scope": "retention",
        "expected_keys": ["nurture_plans", "nurture_progress", "cohort_data", "analysis"],
        "reference_answer": "应包含培育计划完成度、各群组留存率曲线、D7/D30留存对比，以及培育策略优化建议。",
        "agent_name": "retention",
        "metadata": {"difficulty": "medium", "tags": ["用户培育", "群组分析", "留存曲线"]},
    },

    # --- Ad Agent ---
    {
        "id": "ad_001",
        "input_query": "优化抖音RTA投放策略，降低CPA到50元以下",
        "scope": "ad",
        "expected_keys": ["rta_rules", "rta_metrics", "expected_cpa", "analysis"],
        "reference_answer": "应包含RTA规则调优建议（出价/定向/频控）、预期CPA变化、竞胜率分析，以及投放策略建议。",
        "agent_name": "ad",
        "metadata": {"difficulty": "medium", "tags": ["RTA", "CPA优化", "抖音投放"]},
    },
    {
        "id": "ad_002",
        "input_query": "分析广告创意疲劳情况，哪些素材需要更换？",
        "scope": "ad",
        "expected_keys": ["creative_result", "fatigue_alerts", "analysis"],
        "reference_answer": "应包含各素材CTR衰减趋势、疲劳阈值告警、素材更换优先级，以及新素材方向建议。",
        "agent_name": "ad",
        "metadata": {"difficulty": "easy", "tags": ["创意分析", "素材疲劳"]},
    },

    # --- Full / Orchestrator ---
    {
        "id": "full_001",
        "input_query": "给我一份完整的货运增长分析报告",
        "scope": "full",
        "expected_keys": ["analysis_summary", "strategy_recommendation", "kpi_snapshot"],
        "reference_answer": "应包含全链路KPI快照（用户数、AUC、CVR、ROI、流失率、CPA）、综合分析摘要、策略建议。",
        "agent_name": "orchestrator",
        "metadata": {"difficulty": "hard", "tags": ["全链路分析", "策略综合"]},
    },
]


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class EvalDataset:
    """Evaluation dataset manager.

    Loads samples from JSON files or uses built-in freight growth samples.
    Provides filtering and statistics.
    """

    def __init__(self, samples: list[EvalSample] | None = None) -> None:
        self._samples: list[EvalSample] = samples or []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> EvalDataset:
        """Load dataset from a JSON file.

        The JSON file should contain a list of sample objects.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            samples = [EvalSample(**item) for item in raw]
        elif isinstance(raw, dict) and "samples" in raw:
            samples = [EvalSample(**item) for item in raw["samples"]]
        else:
            raise ValueError("Dataset JSON must be a list or {'samples': [...]}")

        logger.info("Loaded %d evaluation samples from %s", len(samples), p)
        return cls(samples=samples)

    @classmethod
    def from_builtin(cls) -> EvalDataset:
        """Load the built-in freight growth evaluation dataset."""
        samples = [EvalSample(**s) for s in _BUILTIN_SAMPLES]
        logger.info("Loaded %d built-in evaluation samples", len(samples))
        return cls(samples=samples)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def samples(self) -> list[EvalSample]:
        """All samples in the dataset."""
        return list(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)

    def get(self, sample_id: str) -> EvalSample | None:
        """Get a sample by its ID."""
        for s in self._samples:
            if s.id == sample_id:
                return s
        return None

    def filter_by_agent(self, agent_name: str) -> list[EvalSample]:
        """Filter samples targeting a specific agent."""
        return [s for s in self._samples if s.agent_name == agent_name]

    def filter_by_scope(self, scope: str) -> list[EvalSample]:
        """Filter samples by scope."""
        return [s for s in self._samples if s.scope == scope]

    def filter_by_difficulty(self, difficulty: str) -> list[EvalSample]:
        """Filter samples by difficulty level (easy/medium/hard)."""
        return [
            s for s in self._samples
            if s.metadata.get("difficulty") == difficulty
        ]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Get dataset summary statistics."""
        agent_counts: dict[str, int] = {}
        scope_counts: dict[str, int] = {}
        difficulty_counts: dict[str, int] = {}

        for s in self._samples:
            agent_counts[s.agent_name] = agent_counts.get(s.agent_name, 0) + 1
            scope_counts[s.scope] = scope_counts.get(s.scope, 0) + 1
            diff = s.metadata.get("difficulty", "unknown")
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        return {
            "total_samples": len(self._samples),
            "agents": agent_counts,
            "scopes": scope_counts,
            "difficulty": difficulty_counts,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        """Export dataset to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [s.model_dump() for s in self._samples]
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Exported %d samples to %s", len(data), p)
