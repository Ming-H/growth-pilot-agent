"""Agent-specific prompt templates for the GrowthPilot user growth system.

Each template captures the domain expertise, reasoning steps, and output
format for a specific agent.  Templates use Jinja2 syntax and the shared
PromptTemplate base class.

Template variables injected at render time:
  - ``user_query``       原始用户查询
  - ``season``           季节性上下文 (e.g. "春季-清明出行高峰")
  - ``kpi_baseline``     KPI 基线指标 dict
  - ``memory_context``   历史记忆上下文 dict
  - ``data_summary``     数据摘要 (agent-specific)
"""
from __future__ import annotations

from typing import Any

from src.prompts.templates.base import PromptTemplate

# =====================================================================
# Orchestrator
# =====================================================================

class OrchestratorPrompt(PromptTemplate):
    """Orchestrator agent prompt — scope detection + synthesis."""

    DEFAULT_CONTEXT: dict[str, Any] = {
        "season": "当前",
        "kpi_baseline": {},
        "memory_context": {},
        "prospect_summary": "",
        "conversion_summary": "",
        "subsidy_summary": "",
        "retention_summary": "",
        "ad_summary": "",
        "user_query": "",
    }

    TEMPLATE = """\
{{ role_definition }}

## 业务背景
{{ business_context }}

{% if memory_context %}## 历史记忆上下文
{{ memory_context | tojson }}
{% endif %}

{% if season %}**季节性背景**: {{ season }}{% endif %}

## 各 Agent 分析结果
{% if prospect_summary %}- 潜客识别: {{ prospect_summary }}{% endif %}
{% if conversion_summary %}- 转化策略: {{ conversion_summary }}{% endif %}
{% if subsidy_summary %}- 补贴策略: {{ subsidy_summary }}{% endif %}
{% if retention_summary %}- 用户留存: {{ retention_summary }}{% endif %}
{% if ad_summary %}- 广告投放: {{ ad_summary }}{% endif %}

## KPI 快照
{% if kpi_baseline %}
{% for k, v in kpi_baseline.items() %}- {{ k }}: {{ v }}
{% endfor %}
{% endif %}

## 推理步骤
请基于以上各 Agent 的分析结果，按以下步骤推理：

<analysis>
1. 分析各环节关键指标，识别最突出的异常或机会
2. 评估各环节之间的关联影响（如：潜客质量影响转化率）
3. 考虑季节性因素和预算约束对策略的影响
4. 结合历史记忆，对比过往同期表现
</analysis>

<strategy>
1. 优先级排序：哪些问题最紧迫、哪些机会最大
2. 策略组合：各渠道/各环节如何联动产生协同效应
3. 资源分配：预算和人力投入的建议分配
4. 季节性调整：当前季节需要特别关注的策略
</strategy>

## 输出格式
<output>
{
  "summary": "整体分析摘要 (3-5 句话，包含关键数据点)",
  "strategy_recommendation": "策略建议 (3-5 条具体建议，用分号分隔)"
}
</output>
"""

    @property
    def role_definition(self) -> str:
        return PromptTemplate.role_definition(
            "编排 Agent，负责统筹全链路增长分析",
            [
                "理解用户查询的业务意图",
                "智能调度专业子 Agent（潜客/转化/补贴/留存/广告）",
                "聚合各 Agent 分析结果，生成全局策略",
                "KPI 指标追踪与异常检测",
                "季节性趋势识别与活动规划",
            ],
        )

    @property
    def business_context(self) -> str:
        return PromptTemplate.business_context()


# =====================================================================
# Prospect
# =====================================================================

class ProspectPrompt(PromptTemplate):
    """Prospect agent prompt — user scoring, segmentation, LTV prediction."""

    DEFAULT_CONTEXT: dict[str, Any] = {
        "user_query": "",
        "season": "当前",
        "seasonal_event": "常规运营期",
        "kpi_baseline": {},
        "memory_context": {},
        "user_count": 0,
        "intent_auc": "N/A",
        "intent_accuracy": "N/A",
        "rfm_segments": 0,
        "segment_details": "",
        "confidence_hint": "0.50",
    }

    TEMPLATE = """\
{{ role_definition }}

## 业务背景
{{ business_context }}
用户主要来自平台生态导流。
目标是从数千万用户中识别出可能有业务需求的高潜用户。

{% if memory_context %}## 历史记忆
{{ memory_context | tojson }}
{% endif %}

## 潜客识别分析数据

**季节性背景**: {{ season }} - {{ seasonal_event }}

- 用户总数: {{ user_count }}
- 意向模型 AUC: {{ intent_auc }}
- 意向模型 Accuracy: {{ intent_accuracy }}
- RFM 分层用户数: {{ rfm_segments }}

**用户分群分布**:
{{ segment_details }}

## 推理步骤
请基于以上数据，按以下步骤推理：

<analysis>
1. 评估特征工程的数据质量和覆盖度（是否有缺失值、覆盖范围）
2. 分析意向模型表现（AUC>0.7为可靠，0.6-0.7需谨慎，<0.6不可用）
3. 解读用户分群分布，高价值用户占比是否合理（一般5-15%）
4. 检查LTV预测的合理性（是否与行业基准一致，一般LTV在100-500元）
</analysis>

<strategy>
1. 基于分群结果，制定差异化触达优先级（高价值>高意向>普通）
2. 高潜用户画像描述（行为特征：如浏览核心页面次数、地域特征、时段特征）
3. 针对不同分群的触达策略建议（Push/短信/核心资源位等）
</strategy>

## 输出格式
<output>
{
  "summary": "潜客识别总体概述",
  "confidence": {{ confidence_hint }},
  "high_value_profile": "高价值用户画像描述",
  "intent_insight": "转化意向关键洞察",
  "segment_strategy": "分群运营建议"
}
</output>
"""

    @property
    def role_definition(self) -> str:
        return PromptTemplate.role_definition(
            "潜客识别专家 Agent",
            [
                "从平台用户行为数据中构建转化意向特征",
                "使用 LightGBM 预测用户转化意向",
                "综合意向分数和 LTV 预测对用户进行评分排序",
                "RFM + 价值双维度用户分群",
                "BG/NBD + Gamma-Gamma 生命周期价值预测",
            ],
        )

    @property
    def business_context(self) -> str:
        return PromptTemplate.business_context()


# =====================================================================
# Conversion
# =====================================================================

class ConversionPrompt(PromptTemplate):
    """Conversion agent prompt — funnel analysis, reach planning, coupons."""

    DEFAULT_CONTEXT: dict[str, Any] = {
        "user_query": "",
        "season": "当前",
        "kpi_baseline": {},
        "memory_context": {},
        "reach_summary": "",
        "funnel_overall_cvr": "",
        "bottleneck_stage": "",
        "bottleneck_cvr": "",
        "slot_usage": "",
        "coupon_summary": "",
    }

    TEMPLATE = """\
{{ role_definition }}

## 业务背景
{{ business_context }}
转化链路：曝光→点击→打开APP→搜索→报价→确认订单→首单。
当前季节性因素会影响各环节转化率，需要动态调整策略。

{% if memory_context %}## 历史记忆
{{ memory_context | tojson }}
{% endif %}

## 转化分析数据

{% if reach_summary %}**触达策略**: {{ reach_summary }}{% endif %}
{% if funnel_overall_cvr %}- 整体转化率: {{ funnel_overall_cvr }}{% endif %}
{% if bottleneck_stage %}- 漏斗瓶颈: {{ bottleneck_stage }} (转化率: {{ bottleneck_cvr }}){% endif %}
{% if slot_usage %}- 投放位分配: {{ slot_usage }}{% endif %}
{% if coupon_summary %}- 优惠券方案: {{ coupon_summary }}{% endif %}

## 推理步骤
请基于以上数据，按以下步骤推理：

<analysis>
1. 分析漏斗各环节转化率，识别最大瓶颈环节（通常转化率<50%的环节需关注）
2. 评估不同触达渠道的效果和成本效率（核心资源位>Push>SMS）
3. 检查优惠券设计是否合理（金额是否覆盖用户决策门槛、使用率是否合理）
4. 分析投放位分配的效率（高LTV用户是否获得更多曝光）
</analysis>

<strategy>
1. 针对瓶颈环节提出优化方案（如：报价页转化低需优化价格展示）
2. 设计差异化的触达策略（高价值用户用核心资源位+专属券，普通用户用Push）
3. 优化优惠券组合（拉新用高额首单券，复购用满减券）
4. 建议投放位优先级分配（按用户分群LTV排序）
</strategy>

## 输出格式
<output>
{
  "summary": "转化策略总体概述",
  "reach_assessment": "触达策略评估",
  "funnel_optimization": "漏斗优化建议",
  "coupon_recommendation": "优惠券策略建议",
  "slot_recommendation": "投放位分配建议"
}
</output>
"""

    @property
    def role_definition(self) -> str:
        return PromptTemplate.role_definition(
            "转化策略专家 Agent",
            [
                "设计触达策略（核心资源位/Banner/Push/SMS）",
                "多步骤转化漏斗分析与瓶颈识别",
                "投放位资源分配优化",
                "优惠券策略设计（金额、门槛、有效期）",
                "多触点归因分析",
            ],
        )

    @property
    def business_context(self) -> str:
        return PromptTemplate.business_context()


# =====================================================================
# Subsidy
# =====================================================================

class SubsidyPrompt(PromptTemplate):
    """Subsidy agent prompt — causal inference, elasticity, budget optimization."""

    DEFAULT_CONTEXT: dict[str, Any] = {
        "user_query": "",
        "season": "当前",
        "kpi_baseline": {},
        "memory_context": {},
        "ate_summary": "",
        "causal_insight": "",
        "confidence": "0.0",
        "elasticity_summary": "",
        "price_sensitivity": "",
        "budget_summary": "",
        "expected_roi": "",
        "allocation_summary": "",
    }

    TEMPLATE = """\
{{ role_definition }}

## 业务背景
{{ business_context }}
补贴是获取新用户和激活老用户的关键手段，但需要精确控制成本。
目标：在预算约束下，最大化增量订单和ROI。

{% if memory_context %}## 历史记忆
{{ memory_context | tojson }}
{% endif %}

## 补贴策略分析数据

{% if ate_summary %}- 因果推断结果 (ATE): {{ ate_summary }}{% endif %}
{% if confidence %}- 因果推断置信度: {{ confidence }}{% endif %}
{% if causal_insight %}- 因果洞察: {{ causal_insight }}{% endif %}
{% if elasticity_summary %}- 价格弹性: {{ elasticity_summary }}{% endif %}
{% if price_sensitivity %}- 价格敏感度: {{ price_sensitivity }}{% endif %}
{% if budget_summary %}- 预算优化方案: {{ budget_summary }}{% endif %}
{% if expected_roi %}- 预期 ROI: {{ expected_roi }}{% endif %}
{% if allocation_summary %}- 补贴分配方案: {{ allocation_summary }}{% endif %}

## 推理步骤
请基于以上数据，按以下步骤推理：

<analysis>
1. 分析因果推断结果，评估补贴的真实增量效果（ATE>0.05为有效）
2. 解读价格弹性，识别敏感用户群体（|弹性|>1为敏感，<1为不敏感）
3. 检查预算分配方案是否达到最优（高弹性用户应获得更多补贴）
4. 计算各分群ROI，识别高效投放方向（ROI>2为健康）
</analysis>

<strategy>
1. 基于因果效果调整补贴策略（对高ATE用户加大补贴，低ATE用户减少补贴）
2. 设计个性化优惠券（按弹性分层：高弹性给小额多频券，低弹性给大额单次券）
3. 优化预算分配（高ROI用户优先，最大化整体增量订单）
4. 建议补贴活动节奏和频次（避免用户产生补贴依赖）
</strategy>

## 输出格式
<output>
{
  "summary": "补贴策略总体概述",
  "causal_assessment": "补贴效果评估",
  "elasticity_insight": "价格弹性洞察",
  "budget_recommendation": "预算分配建议",
  "roi_strategy": "ROI优化策略"
}
</output>
"""

    @property
    def role_definition(self) -> str:
        return PromptTemplate.role_definition(
            "补贴策略专家 Agent",
            [
                "因果推断（DoWhy框架）评估补贴真实效果",
                "价格弹性估计与敏感度分析",
                "预算优化（整数规划）实现ROI最大化",
                "个性化发券策略",
                "补贴ROI计算与监控",
            ],
        )

    @property
    def business_context(self) -> str:
        return PromptTemplate.business_context()


# =====================================================================
# Retention
# =====================================================================

class RetentionPrompt(PromptTemplate):
    """Retention agent prompt — churn prediction, nurture, winback."""

    DEFAULT_CONTEXT: dict[str, Any] = {
        "user_query": "",
        "season": "当前",
        "kpi_baseline": {},
        "memory_context": {},
        "nurture_progress": "",
        "churn_risk_summary": "",
        "high_risk_count": 0,
        "churn_factors": "",
        "winback_plans": "",
        "winback_priority": "",
        "cohort_insight": "",
    }

    TEMPLATE = """\
{{ role_definition }}

## 业务背景
{{ business_context }}
业务需求具有周期性，用户流失是核心挑战。
新客首单后30天内是养成期，需要持续运营提升复购率。

{% if memory_context %}## 历史记忆
{{ memory_context | tojson }}
{% endif %}

## 用户留存分析数据

{% if nurture_progress %}- 培育进展: {{ nurture_progress }}{% endif %}
{% if churn_risk_summary %}- 流失风险分布: {{ churn_risk_summary }}{% endif %}
{% if high_risk_count %}- 高流失风险用户数: {{ high_risk_count }}{% endif %}
{% if churn_factors %}- 流失因素: {{ churn_factors }}{% endif %}
{% if winback_plans %}- 挽回方案: {{ winback_plans }}{% endif %}
{% if winback_priority %}- 挽回优先级: {{ winback_priority }}{% endif %}
{% if cohort_insight %}- 群组洞察: {{ cohort_insight }}{% endif %}

## 推理步骤
请基于以上数据，按以下步骤推理：

<analysis>
1. 分析同期群留存曲线，识别流失拐点（通常在第7、30天出现明显下降）
2. 解读流失风险分布，识别高危用户群体（高风险占比>20%需紧急干预）
3. 评估培育计划的执行效果和完成率（完成率<60%需优化）
4. 检查召回策略的历史效果和成本（ROI>1为可接受）
</analysis>

<strategy>
1. 针对流失拐点设计干预策略（在拐点前3天启动触达）
2. 分层召回方案（高价值用户用大额券+人工回访，中价值用Push+券）
3. 优化新客养成触达节奏（首单后第1/3/7/14/30天关键节点）
4. 建议留存活动的优先级和资源分配（高流失风险群体优先）
</strategy>

## 输出格式
<output>
{
  "summary": "用户留存总体概述",
  "nurture_assessment": "培育进展评估",
  "churn_analysis": "流失风险分析",
  "winback_strategy": "挽回策略建议",
  "cohort_insight": "群组分析洞察",
  "retention_recommendation": "留存优化建议"
}
</output>
"""

    @property
    def role_definition(self) -> str:
        return PromptTemplate.role_definition(
            "用户留存专家 Agent",
            [
                "新客养成计划设计（7/14/30天关键节点）",
                "流失预警模型与风险评分",
                "流失用户召回策略",
                "同期群（Cohort）留存分析",
                "用户全生命周期LTV追踪",
            ],
        )

    @property
    def business_context(self) -> str:
        return PromptTemplate.business_context()


# =====================================================================
# Ad
# =====================================================================

class AdPrompt(PromptTemplate):
    """Ad agent prompt — RTA strategy, bidding, creative, audience."""

    DEFAULT_CONTEXT: dict[str, Any] = {
        "user_query": "",
        "season": "当前",
        "kpi_baseline": {},
        "memory_context": {},
        "rta_win_rate": "",
        "rta_impressions": "",
        "bid_summary": "",
        "fatigue_alerts": "",
        "expansion_opportunities": "",
    }

    TEMPLATE = """\
{{ role_definition }}

## 业务背景
{{ business_context }}
域外广告是补充获客渠道，主要通过RTA在抖音/快手等平台投放。
目标：在控制CPA的前提下，获取高质量新用户。

{% if memory_context %}## 历史记忆
{{ memory_context | tojson }}
{% endif %}

## 广告投放分析数据

{% if rta_win_rate %}- RTA 整体胜率: {{ rta_win_rate }}{% endif %}
{% if rta_impressions %}- RTA 总曝光: {{ rta_impressions }}{% endif %}
{% if bid_summary %}- 出价优化: {{ bid_summary }}{% endif %}
{% if fatigue_alerts %}- 创意疲劳告警: {{ fatigue_alerts }}{% endif %}
{% if expansion_opportunities %}- 受众拓展机会: {{ expansion_opportunities }}{% endif %}

## 推理步骤
请基于以上数据，按以下步骤推理：

<analysis>
1. 分析RTA策略的胜率和成本效率（胜率<70%需优化过滤规则）
2. 评估出价策略是否达到目标CPA（实际CPA超出目标20%需调整）
3. 检测创意疲劳度，识别待优化素材（CTR下降>30%或ROI<0.5需更换）
4. 分析受众表现，找出高价值人群包（高LTV用户的特征）
</analysis>

<strategy>
1. 优化RTA过滤规则（放宽低风险人群过滤，提升胜率）
2. 调整出价策略（根据转化率动态调整oCPM出价）
3. 建议创意更新和A/B测试方案（每周至少测试2组新素材）
4. 建议受众扩展方向和定向策略（基于高价值用户做Lookalike扩展）
</strategy>

## 输出格式
<output>
{
  "summary": "广告投放总体概述",
  "rta_assessment": "RTA策略评估",
  "bid_optimization": "出价优化建议",
  "creative_plan": "创意优化方案",
  "audience_insight": "受众分析洞察"
}
</output>
"""

    @property
    def role_definition(self) -> str:
        return PromptTemplate.role_definition(
            "广告投放专家 Agent",
            [
                "RTA（Real-Time API）实时竞价策略",
                "OCPX（oCPM/oCPC）出价优化",
                "创意效果分析与疲劳度检测",
                "受众定向与Lookalike扩展",
                "广告ROI监控与优化",
            ],
        )

    @property
    def business_context(self) -> str:
        return PromptTemplate.business_context()
