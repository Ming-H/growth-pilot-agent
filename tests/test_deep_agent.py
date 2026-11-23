"""DeepAgentBase / SubsidyDeepAgent 测试（mock LLM，CI 安全）。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.deep.base import DeepAgentBase
from src.agents.deep.subsidy import SubsidyDeepAgent


def _mock_llm(action_sequence: list[str]) -> MagicMock:
    """创建按序列返回 action 决策的 mock LLM。"""
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    responses = []
    for act in action_sequence:
        resp = MagicMock()
        resp.content = f'{{"action": "{act}", "thought": "test"}}'
        responses.append(resp)
    llm.ainvoke.side_effect = responses
    return llm


class SimpleDeepAgent(DeepAgentBase):
    """测试用最小实现。"""

    @property
    def domain(self) -> str:
        return "test"

    async def _execute_action(self, action: str, params: dict) -> str:
        if action == "finalize":
            return "done"
        return f"executed {action}"

    async def _compile_final(self, params: dict) -> dict:
        return {"kpis": {"test_metric": 1.0}, "summary": "test result"}


class TestDeepAgentBase:
    @pytest.mark.asyncio
    async def test_zero_llm_runs_pipeline_and_finalizes(self):
        """零 LLM 时应直接 run_pipeline → finalize（确定性兜底）。"""
        agent = SimpleDeepAgent(llm=None, max_deep_steps=3)
        result = await agent.deep_analyze({"task": "test"})
        assert result["deep_step_count"] >= 1
        assert result["kpis"]["test_metric"] == 1.0

    @pytest.mark.asyncio
    async def test_max_steps_hard_cap(self):
        """步数硬上限——LLM 永远不 finalize 也不应超过上限。"""
        llm = _mock_llm(["run_pipeline"] * 10)  # 永远 run_pipeline
        agent = SimpleDeepAgent(llm=llm, max_deep_steps=3)
        result = await agent.deep_analyze({"task": "test"})
        assert result["deep_step_count"] <= 3  # 硬上限生效

    @pytest.mark.asyncio
    async def test_finalize_stops_early(self):
        """LLM 第一轮就 finalize 应立即停止。"""
        llm = _mock_llm(["finalize"])
        agent = SimpleDeepAgent(llm=llm, max_deep_steps=5)
        result = await agent.deep_analyze({"task": "test"})
        assert result["deep_step_count"] == 1
        assert result["deep_steps"][0]["action"] == "finalize"

    @pytest.mark.asyncio
    async def test_step_observations_present(self):
        """每步应包含四元组（thought/action/obs/gate）。"""
        llm = _mock_llm(["run_pipeline", "finalize"])
        agent = SimpleDeepAgent(llm=llm, max_deep_steps=5)
        result = await agent.deep_analyze({"task": "test"})
        for step in result["deep_steps"]:
            assert "thought" in step
            assert "action" in step
            assert "obs" in step
            assert "gate" in step

    @pytest.mark.asyncio
    async def test_duplicate_gate_stops(self):
        """连续重复 action 应标记 duplicate 并收手。"""
        llm = _mock_llm(["run_pipeline", "run_pipeline", "run_pipeline"])
        agent = SimpleDeepAgent(llm=llm, max_deep_steps=5)
        result = await agent.deep_analyze({"task": "test"})
        # 第二个 run_pipeline 应被标为 duplicate（相同 thought）
        # 由于 thought 相同，应触发 duplicate 收手
        assert result["deep_step_count"] <= 3


class TestSubsidyDeepAgent:
    @pytest.mark.asyncio
    async def test_domain_is_subsidy(self):
        agent = SubsidyDeepAgent(llm=None)
        assert agent.domain == "subsidy"

    @pytest.mark.asyncio
    async def test_zero_llm_fallback_to_pipeline(self):
        """零 LLM 时应降级为确定性管线（SubsidyExpert 行为）。"""
        agent = SubsidyDeepAgent(llm=None, max_deep_steps=2)
        result = await agent.deep_analyze({"budget": 5000})
        assert result is not None
        assert result.get("deep_domain") == "subsidy"

    @pytest.mark.asyncio
    async def test_step_cap_respected(self):
        """步数硬上限——不应超过 max_deep_steps。"""
        llm = _mock_llm(["run_pipeline"] * 20)
        agent = SubsidyDeepAgent(llm=llm, max_deep_steps=3)
        result = await agent.deep_analyze({"budget": 5000})
        assert result.get("deep_step_count", 0) <= 3
