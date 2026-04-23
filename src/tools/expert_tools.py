"""Expert agent tool wrappers for the Chief Agent.

Each expert is wrapped as a LangChain @tool with a structured input schema.
The Chief Agent uses Tool Calling to dynamically decide which experts to invoke.

Design reference:
- OpenAI Agent-as-Tool: agents wrapped as callable tools
- OpenAI @function_tool: clear schemas for LLM tool selection
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from src.core.models import (
    AdToolInput,
    ConversionToolInput,
    ProspectToolInput,
    RetentionToolInput,
    SubsidyToolInput,
)

logger = logging.getLogger(__name__)


def _create_expert(expert_cls: type, llm_tier: str = "default") -> Any:
    """Create an expert instance with an LLM."""
    from src.core.llm_factory import create_llm

    llm = create_llm(tier=llm_tier)
    return expert_cls(llm=llm)


# ---------------------------------------------------------------------------
# Prospect Analysis Tool
# ---------------------------------------------------------------------------


@tool(args_schema=ProspectToolInput)
async def prospect_analysis(
    query: str,
    data_path: str = "",
    budget: float = 0,
) -> str:
    """分析潜在高价值货运用户。

    用于：用户评分、分群、LTV预测、意图预测。
    何时调用：用户询问新客获取、用户评分、分群、LTV预测。
    输入：query(必填), data_path(可选), budget(可选)
    输出：JSON字符串，包含评分、分群、意图指标和LTV预测。
    """
    from src.agents.prospect import ProspectExpert

    expert = _create_expert(ProspectExpert)
    params = {"query": query, "data_path": data_path, "budget": budget}
    result = await expert.analyze(params)
    return json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Conversion Analysis Tool
# ---------------------------------------------------------------------------


@tool(args_schema=ConversionToolInput)
async def conversion_analysis(
    query: str,
    data_path: str = "",
    budget: float = 0,
    prospect_results_json: str = "",
) -> str:
    """分析转化漏斗并设计优化策略。

    用于：漏斗瓶颈分析、优惠券策略、触达策略、广告位分配。
    何时调用：用户询问转化率、漏斗优化、优惠券设计、触达规划。
    输入：query(必填), data_path(可选), budget(可选), prospect_results_json(前序结果)
    输出：JSON字符串，包含漏斗分析、触达策略、优惠券方案。
    """
    from src.agents.conversion import ConversionExpert

    expert = _create_expert(ConversionExpert)
    params: dict[str, Any] = {"query": query, "data_path": data_path, "budget": budget}
    if prospect_results_json:
        try:
            params["prospect_results"] = json.loads(prospect_results_json)
        except json.JSONDecodeError:
            params["prospect_results"] = {}
    result = await expert.analyze(params)
    return json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Subsidy Analysis Tool
# ---------------------------------------------------------------------------


@tool(args_schema=SubsidyToolInput)
async def subsidy_analysis(
    query: str,
    data_path: str = "",
    budget: float = 10000,
    prospect_results_json: str = "",
) -> str:
    """使用因果推断优化补贴分配。

    用于：估计补贴因果效应(ATE)、计算价格弹性、优化预算分配、规划补贴方案。
    何时调用：用户询问补贴、预算优化、ROI、价格敏感度、折扣策略。
    输入：query(必填), data_path(可选), budget(建议提供), prospect_results_json(前序结果)
    输出：JSON字符串，包含因果推断结果、弹性估计、预算方案、ROI预测。
    """
    from src.agents.subsidy import SubsidyExpert

    expert = _create_expert(SubsidyExpert)
    params: dict[str, Any] = {
        "query": query,
        "data_path": data_path,
        "budget": budget,
    }
    if prospect_results_json:
        try:
            params["prospect_results"] = json.loads(prospect_results_json)
        except json.JSONDecodeError:
            params["prospect_results"] = {}
    result = await expert.analyze(params)
    return json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Retention Analysis Tool
# ---------------------------------------------------------------------------


@tool(args_schema=RetentionToolInput)
async def retention_analysis(
    query: str,
    data_path: str = "",
    conversion_results_json: str = "",
) -> str:
    """分析用户留存并设计流失预防策略。

    用于：预测流失风险、设计培育计划、分析群组留存、规划召回策略。
    何时调用：用户询问流失、留存、用户生命周期、培育策略、召回。
    输入：query(必填), data_path(可选), conversion_results_json(前序结果)
    输出：JSON字符串，包含流失预测、培育计划、群组分析、召回策略。
    """
    from src.agents.retention import RetentionExpert

    expert = _create_expert(RetentionExpert)
    params: dict[str, Any] = {"query": query, "data_path": data_path}
    if conversion_results_json:
        try:
            params["conversion_results"] = json.loads(conversion_results_json)
        except json.JSONDecodeError:
            params["conversion_results"] = {}
    result = await expert.analyze(params)
    return json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Ad Analysis Tool
# ---------------------------------------------------------------------------


@tool(args_schema=AdToolInput)
async def ad_analysis(
    query: str,
    data_path: str = "",
    budget: float = 0,
    prospect_results_json: str = "",
) -> str:
    """通过RTA策略和出价管理优化广告投放。

    用于：构建RTA决策规则、优化出价、检测创意疲劳、分析受众。
    何时调用：用户询问广告、RTA、出价、创意表现、受众定向。
    输入：query(必填), data_path(可选), budget(可选), prospect_results_json(前序结果)
    输出：JSON字符串，包含RTA规则、出价优化、创意分析、受众洞察。
    """
    from src.agents.ad import AdExpert

    expert = _create_expert(AdExpert)
    params: dict[str, Any] = {
        "query": query,
        "data_path": data_path,
        "budget": budget,
    }
    if prospect_results_json:
        try:
            params["prospect_results"] = json.loads(prospect_results_json)
        except json.JSONDecodeError:
            params["prospect_results"] = {}
    result = await expert.analyze(params)
    return json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_expert_tools() -> list:
    """Return all expert tools for the Chief Agent."""
    return [
        prospect_analysis,
        conversion_analysis,
        subsidy_analysis,
        retention_analysis,
        ad_analysis,
    ]
