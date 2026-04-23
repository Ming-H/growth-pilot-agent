# GrowthPilot Agent — 货运用户增长智能体系统设计文档

> 版本：v4.0 | 日期：2026-04-19
> 定位：基于滴滴货运真实业务经验的 Multi-Agent 用户增长平台
> 参考：LangAlpha 项目架构模式

---

## 一、项目定位

### 1.1 一句话描述

一个面向货运平台的 Multi-Agent 用户增长系统，覆盖**潜客识别 → 端内转化 → 补贴优化 → 留存召回 → 域外投放**全生命周期，以实验驱动增长决策，用 ML 工具精确计算、LLM 解读并生成策略。

### 1.2 为什么做这个

| 维度 | 说明 |
|------|------|
| **业务价值** | 货运用户增长全链路自动化分析——从潜客识别到留存召回 |
| **技术差异化** | 因果推断 + 整数规划 + A/B 实验平台 + 多触点归因，不是"LLM 看数据然后瞎说" |
| **系统定位** | 实验驱动的 Multi-Agent 增长平台，覆盖用户全生命周期 |

### 1.3 设计原则

- **工具先行、AI 解读** — 指标计算用 Python 精确完成，LLM 负责解读和策略生成
- **实验驱动** — 每个增长策略都通过 A/B 实验验证，不用猜测
- **中间件架构** — 通用逻辑（重试、日志、上下文管理）通过中间件层处理，不侵入业务代码
- **配置驱动** — Agent 行为、模型选择、工具加载通过 YAML 配置，不硬编码
- **模板化 Prompt** — Jinja2 模板管理，支持继承和组件化组合
- **端内为主、端外为辅** — 滴滴集团用户池是主引流渠道，域外广告是辅助渠道

---

## 二、业务背景

### 2.1 货运增长全景

```
                         货运用户增长体系
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
      端内引流（主渠道）                    域外投放（辅助渠道）
      滴滴集团 ~10 亿用户池                抖音 / 快手等平台
            │                                   │
            ▼                                   ▼
    ┌───────────────────┐               RTA + OCPX 广告
    │ 潜客识别           │                    │
    │ 从集团用户中识别    │                    │
    │ 货运意向用户       │                    │
    └────────┬──────────┘                    │
             │                               │
             ▼                               ▼
    ┌───────────────────┐            ┌──────────────┐
    │ 端内触达           │            │ 广告投放优化  │
    │ 金刚位/Banner/     │            │ 竞价策略/     │
    │ Push/SMS           │            │ 创意优化      │
    └────────┬──────────┘            └──────┬───────┘
             │                              │
             ▼                              │
    ┌───────────────────┐                   │
    │ 补贴转化           │                   │
    │ 折扣券/满减券      │                   │
    │ 因果推断驱动的     │                   │
    │ 个性化发券         │                   │
    └────────┬──────────┘                   │
             │                              │
             ▼                              ▼
    ┌───────────────────┐            ┌──────────────┐
    │ 首单转化           │            │ 新客引入      │
    └────────┬──────────┘            └──────┬───────┘
             │                              │
             ▼                              ▼
    ┌───────────────────┐            ┌──────────────┐
    │ 新客养成           │            │ 新客养成      │
    │ 首单后 7/14/30 天  │◄───────────┤              │
    │ 阶段性运营         │            └──────────────┘
    └────────┬──────────┘
             │
             ▼
    ┌───────────────────┐
    │ 留存运营           │
    │ 防流失 + 召回      │
    └───────────────────┘

    贯穿全链路：
    ┌─────────────────────────────────────────────┐
    │  实验平台 · 多触点归因 · 增长指标体系         │
    │  LTV 预测 · 季节性运营                        │
    └─────────────────────────────────────────────┘
```

### 2.2 两条获客渠道（并列）

| 维度 | 端内引流（主渠道） | 域外投放（辅助渠道） |
|------|-------------------|---------------------|
| **用户来源** | 滴滴集团 ~10 亿用户 | 抖音、快手等平台用户 |
| **核心动作** | 识别货运意向 → APP 内触达 | RTA 竞价 → 广告曝光 |
| **触达方式** | 金刚位、Banner、Push、SMS | 信息流广告、开屏广告 |
| **补贴方式** | 折扣券、满减券 | 优惠券 landing page |
| **获客成本** | 低（已有多亿活跃用户） | 较高（需要竞价购买流量） |
| **规模占比** | **80%+** | **20%** 以下 |

### 2.3 完整的用户生命周期

```
获取期                    激活期                    留存期
潜客识别 → 端内触达 → 首单转化 → 新客养成 → 常规运营
                     ↘                                ↓
                      域外广告 → 首单转化 → 新客养成   ↓
                                                    ↓
                                              衰退期 → 流失期
                                              流失预警 → 流失召回
```

### 2.4 转化漏斗（端内主通道）

```
集团用户池（~10 亿）
  ↓ 潜客识别模型筛选
高潜用户池（~数百万）
  ↓ 端内资源位曝光
看到触达（曝光率）
  ↓ 点击进入货运页面
点击（CTR）
  ↓ 浏览货运服务/查看价格
活跃（浏览深度）
  ↓ 领取优惠券
领券（领券率）
  ↓ 完成首单
首单转化（转化率）
  ↓ 7 天内复购
短期留存
  ↓ 30 天仍在使用
长期留存
```

### 2.5 季节性业务节奏

```
货运需求的季节性波动：

Q1（1-3 月）  春节后返工潮 · 春季装修启动      需求↑
Q2（4-6 月）  毕业季搬家高峰（6-7 月峰值）      需求↑↑
Q3（7-9 月）  暑期平稳 · 开学季（9 月小高峰）   需求→
Q4（10-12 月）年底搬家潮 · 双 11 电商货运       需求↑

关键节点：
  3 月：春季装修 → 大件运输需求
  6-7 月：毕业季 → 大学生搬家（价格敏感、体积小）
  9 月：开学季 → 学生行李托运
  11 月：双 11 → 电商退货/换货货运
  12-1 月：年底搬家 + 春运前搬家
```

### 2.6 增长指标体系

```
北极星指标：货运月活跃下单用户数（MAO）
  │
  ├── L1：新客获取量（月新增首单用户）
  │    ├── L2：端内渠道新客量 | 域外渠道新客量
  │    │    ├── L3：金刚位引流 | Push 引流 | 广告引流
  │    │
  ├── L1：首单转化率（高潜 → 首单）
  │    ├── L2：各分群转化率 | 各城市转化率
  │    │    ├── L3：折扣券核销率 | 满减券核销率
  │    │
  ├── L1：留存率（D7 / D30）
  │    ├── L2：各获客渠道留存 | 新客养成完成率
  │    │    ├── L3：各阶段复购率（Day1/7/14/30）
  │    │
  ├── L1：客单价（AOV）
  │    ├── L2：各城市客单价 | 各品类客单价
  │    │
  └── L1：ROI（补贴投入回报比）
       ├── L2：各渠道 ROI | 各分群 ROI
       │    ├── L3：补贴 ROI | 广告 ROI
```

---

## 三、系统架构

### 3.1 整体架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              用户层                                      │
│   CLI（Click + Rich）     │     Web UI（FastAPI + SSE 流式输出）          │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴────────────────────────────────────────────┐
│                       Orchestrator Agent                                 │
│    理解需求 → 任务分解 → 调度子 Agent → 聚合结果 → 生成报告               │
│    · KPI 指标追踪与异常检测                                               │
│    · 季节性趋势识别与活动规划                                             │
│    · 跨渠道预算全局优化（基于 LTV）                                       │
├──────────┬──────────┬──────────┬──────────┬──────────────────────────────┤
│Prospect  │Conversion│ Subsidy  │Retention │  AdAcquisition               │
│Agent     │Agent     │Agent     │Agent     │  Agent                       │
│(潜客识别) │(端内转化) │(补贴优化) │(留存召回) │  (域外投放)                  │
│          │          │          │          │                              │
│·特征工程 │·触达策略  │·因果推断  │·新客养成  │·RTA 竞价策略                 │
│·意向建模 │·漏斗分析  │·价格弹性  │·流失预警  │·OCPX 出价优化                │
│·用户评分 │·资源位分配 │·预算优化  │·流失召回  │·创意效果分析                 │
│·用户分群 │·券策略设计 │·发券方案  │·同期群分析 │·受众定向                    │
│·LTV 预测 │·多触点归因 │          │          │                              │
├──────────┴──────────┴──────────┴──────────┴──────────────────────────────┤
│                         工具层（纯 Python 计算）                           │
│                                                                           │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌─────────┐ │
│  │Prospect    │ │Conversion  │ │Subsidy     │ │Retention   │ │Ad       │ │
│  │Tools       │ │Tools       │ │Tools       │ │Tools       │ │Tools    │ │
│  │            │ │            │ │            │ │            │ │         │ │
│  │FeatureEng  │ │ReachPlanner│ │CausalEngine│ │NurturePlan │ │RTAStrat │ │
│  │IntentModel │ │FunnelAnaly │ │Elasticity  │ │ChurnPredict│ │BidOptim │ │
│  │UserScorer  │ │SlotAlloc   │ │BudgetSolver│ │WinbackPlan │ │CreativeA│ │
│  │Segmentor   │ │CouponDesign│ │SubsidyAlloc│ │CohortAnaly │ │AudienceA│ │
│  │LVTPredictor│ │Attributor  │ │            │ │            │ │         │ │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘ └─────────┘ │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────┐            │
│  │                    通用工具                                │            │
│  │  DataLoader │ Visualizer │ ExperimentPlatform │ ReportGen │            │
│  └───────────────────────────────────────────────────────────┘            │
├───────────────────────────────────────────────────────────────────────────┤
│                       中间件层（横切关注点）                                │
│                                                                           │
│  ErrorHandling │ Logging │ Retry │ ContextWindow │ Streaming              │
├───────────────────────────────────────────────────────────────────────────┤
│                       基础设施层                                           │
│                                                                           │
│  Config（YAML + .env）│ LLM Factory │ Prompt Templates │ State            │
└───────────────────────────────────────────────────────────────────────────┘
```

### 3.2 LangGraph 编排拓扑

```
                              START
                                │
                        orchestrator
                         ╱     │      ╲
                        ╱      │       ╲
          ┌─────────────┐ ┌─────────┐ ┌──────────┐
          │ prospect    │ │ subsidy │ │   ad     │   ← 可并行
          │ (潜客识别)   │ │(补贴优化)│ │(域外投放) │
          └──────┬──────┘ └────┬────┘ └─────┬────┘
                 │             │             │
          ┌──────┴──────┐      │             │
          │ conversion  │◄─────┘             │
          │ (端内转化)   │                    │
          └──────┬──────┘                     │
                 │                            │
          ┌──────┴──────┐                     │
          │ retention   │◄────────────────────┘
          │ (留存召回)   │
          └──────┬──────┘
                 │
          ┌──────┴──────┐
          │ report_gen  │
          └──────┬──────┘
                 │
               END

    ExperimentPlatform 和 Attribution 作为通用工具，
    所有 Agent 在分析过程中均可调用
```

### 3.3 执行场景

**场景一：全链路增长方案**

```
用户："帮我制定下周的货运增长方案"

Orchestrator:
  1. ProspectAgent: 从集团用户中识别本周高潜用户名单（3200 名）
     → 含 LTV 预测，按长期价值排序
  2. 并行:
     ├─ ConversionAgent: 设计端内触达方案 + 多触点归因分析
     ├─ SubsidyAgent: 因果推断分析最优发券额度和分群方案
     └─ AdAgent: 本周域外广告投放数据和优化建议
  3. RetentionAgent: 本周新客养成进度 + 流失预警报告
  4. ExperimentPlatform: 自动推荐可运行的 A/B 实验
  5. 季节性提醒: 当前处于毕业季前期，建议提前储备搬家场景物料
  6. 聚合输出完整增长方案 + KPI 看板
```

**场景二：补贴实验分析**

```
用户："上周的 10 元 vs 15 元满减券 A/B 实验结果怎么样？"

Orchestrator:
  1. 调度 SubsidyAgent + ExperimentPlatform:
     → 实验设计回顾：分组方式、样本量、运行时长
     → 统计检验：t 检验 + 序贯检验，显著性 p < 0.05
     → 护栏指标检查：客单价是否下降、7 天留存是否变差
     → CATE 分析：10 元和 15 元对不同城市/分群的异质性效应
     → 归因分析：这次转化中各触点的贡献度
     → LTV 视角：虽然 15 元短期 ROI 更低，但长期 LTV 更高
  2. 输出实验报告 + 决策建议
```

**场景三：毕业季活动规划**

```
用户："马上毕业季了，帮我规划搬家场景的增长活动"

Orchestrator:
  1. ProspectAgent: 筛选"大学生/年轻人 + 毕业季城市"的高潜用户
  2. ConversionAgent:
     → 季节性分析：去年同期毕业季的数据和转化规律
     → 触达策略：针对毕业生的 Push 文案和 Banner 创意
     → 券策略：毕业季专属折扣券设计
  3. SubsidyAgent: 基于历史同期因果效应，制定毕业季发券方案
  4. ExperimentPlatform: 设计活动前后的对比实验
  5. 输出毕业季增长活动方案
```

---

## 四、Agent 详细设计

### 4.1 Orchestrator Agent（编排 Agent）

**职责**：理解用户意图，分解任务，调度子 Agent，聚合结果。

**关键能力**：
- 意图识别：判断用户需要全链路分析还是某个环节的深度分析
- 任务分解：将复杂需求拆解为子 Agent 可执行的子任务
- 结果聚合：综合多个子 Agent 的输出，消除矛盾，生成一致的建议
- 全局预算优化：基于 LTV 预测，在端内补贴 + 域外广告间做全局预算分配
- KPI 追踪：输出当前周期各层级指标的变化趋势和异常告警
- 季节性感知：识别当前所处的业务周期，推荐对应的活动策略

### 4.2 ProspectAgent（潜客识别 Agent）

**职责**：从滴滴集团用户池中识别有货运意向的潜在用户，并预测其长期价值。

**工具集**：

| 工具 | 功能 | 技术实现 |
|------|------|---------|
| `FeatureEngine` | 用户行为特征工程 | 出行频次、时段分布、目的地类型、搜索行为等特征提取 |
| `IntentModel` | 货运意向预测模型 | LightGBM 二分类，输出概率分 + SHAP 特征重要性 |
| `UserScorer` | 用户价值评分 | 意向分 × LTV 预测分的综合评分 |
| `UserSegmentor` | 用户分群 | RFM 分群 + 生命周期分群 + 城市维度 |
| `LVTPredictor` | 用户长期价值预测 | **新增**：基于历史同期群数据预测每个用户的 LTV |

**输出格式**：
```json
{
  "total_analyzed": 5000000,
  "total_prospects": 15000,
  "high_intent_users": 3200,
  "segments": {
    "high_value_high_intent": {
      "count": 800, "avg_score": 0.89, "avg_predicted_ltv": 520,
      "profile": "高频出行、有搬家相关搜索、一线城市",
      "top_features": ["夜间出行频次", "搬家相关搜索", "大件物品相关行为"]
    },
    "medium_value_high_intent": {
      "count": 1200, "avg_score": 0.82, "avg_predicted_ltv": 280,
      "profile": "中等频次出行、目的地含物流园/批发市场"
    }
  },
  "ltv_summary": {
    "total_predicted_ltv": 2800000,
    "ltv_cac_ratio": 4.2,
    "recommendation": "LTV:CAC = 4.2:1，高于健康阈值 3:1，可加大投入"
  }
}
```

### 4.3 ConversionAgent（端内转化 Agent）

**职责**：设计端内触达策略，优化转化漏斗，分析多触点归因。

**工具集**：

| 工具 | 功能 | 技术实现 |
|------|------|---------|
| `ReachPlanner` | 触达策略规划 | 渠道选择（金刚位/Banner/Push/SMS）× 时机 × 频次 |
| `FunnelAnalyzer` | 转化漏斗分析 | 曝光→点击→浏览→领券→首单各环节转化率分析 |
| `SlotAllocator` | 资源位分配 | 金刚位/Banner 等有限资源位的最优用户群分配 |
| `CouponDesigner` | 券策略设计 | 折扣券 vs 满减券、面额、使用条件的设计与效果预估 |
| `Attributor` | 多触点归因 | **新增**：首次/末次/线性/时间衰减/U型归因模型 |
| `SeasonalAnalyzer` | 季节性分析 | **新增**：历史同期数据对比、季节性趋势预测、活动日历管理 |

**多触点归因说明**：

一个用户从识别到首单，通常经历多次触达。Attributor 负责分析各触点的贡献度：

```
用户路径示例：
  Day 1: 金刚位曝光（未点击）
  Day 3: Push 推送（点击但未领券）
  Day 5: Banner 展示（点击并领券）
  Day 6: SMS 提醒（核销券并首单）

归因问题：这次转化的功劳怎么分配？

首次触点归因：金刚位 100%
末次触点归因：SMS 100%
线性归因：各 25%
时间衰减归因：SMS 40% > Banner 30% > Push 20% > 金刚位 10%
U 型归因：金刚位 40% + SMS 40% + 中间 20%
```

**季节性运营说明**：

SeasonalAnalyzer 分析历史数据的周期性规律，为活动运营提供数据支撑：

```
功能：
1. 趋势识别：从历史数据中检测季节性模式（周/月/季度）
2. 同期对比：今年 vs 去年同期的增长指标对比
3. 活动日历：管理全年的关键活动节点
4. 需求预测：基于季节性模型预测未来 N 天的需求量
5. 活动复盘：对比活动期 vs 非活动期的增量效果
```

**输出格式**：
```json
{
  "reach_plan": {
    "primary_channel": "金刚位 + Push",
    "timing": "工作日晚高峰（18:00-20:00），周末全天",
    "frequency_cap": "每用户每周最多 3 次触达",
    "creative_message": "搬家运货？滴滴货运首单立享 8 折！"
  },
  "funnel_analysis": {
    "exposure_rate": 0.85,
    "click_rate": 0.12,
    "coupon_claim_rate": 0.45,
    "first_order_rate": 0.08,
    "bottleneck": "点击→领券环节流失最大（73%），建议优化领券页面"
  },
  "attribution": {
    "model": "time_decay",
    "channel_contribution": {
      "金刚位": 0.25,
      "Push": 0.20,
      "Banner": 0.35,
      "SMS": 0.20
    },
    "insight": "Banner 的归因贡献度最高（35%），建议增加 Banner 投放资源"
  },
  "seasonal_context": {
    "current_period": "毕业季前期（5 月）",
    "yoy_growth": "+22%",
    "recommendation": "6 月毕业季即将到来，建议提前 2 周储备搬家场景素材和专属券",
    "predicted_demand_lift": "+45%"
  }
}
```

### 4.4 SubsidyAgent（补贴策略 Agent）

**职责**：基于因果推断，制定个性化的最优补贴方案。

**工具集**：

| 工具 | 功能 | 技术实现 |
|------|------|---------|
| `CausalInferenceEngine` | 因果效应估计 | DoWhy：ATE/CATE 估计、反事实分析、反驳检验 |
| `ElasticityEstimator` | 价格弹性估计 | 工具变量法（IV），估计不同用户群对券额的敏感度 |
| `BudgetOptimizer` | 预算优化 | 整数规划（PuLP），基于 LTV 预测在预算约束下求解最优方案 |
| `SubsidyAllocator` | 补贴方案生成 | 基于因果结论，生成每个分群的发券类型和额度建议 |

**输出格式**：
```json
{
  "causal_findings": {
    "ate_subsidy_on_conversion": 0.12,
    "ate_interpretation": "发券使用户完单概率平均提升 12 个百分点",
    "cate_by_segment": {
      "北京_高潜用户": {"effect": 0.18, "elasticity": "high", "best_coupon": "满50减15"},
      "上海_中潜用户": {"effect": 0.07, "elasticity": "low", "best_coupon": "8折券"}
    },
    "refutation_passed": true
  },
  "subsidy_plan": {
    "total_budget": 100000,
    "total_users": 3200,
    "allocation": {
      "北京高潜用户": {
        "count": 500, "coupon_type": "满减券", "coupon_amount": 15,
        "threshold": 50, "expected_roi": 3.2,
        "predicted_ltv": 520
      },
      "上海高潜用户": {
        "count": 400, "coupon_type": "折扣券", "discount": 0.8,
        "max_discount": 20, "expected_roi": 2.8,
        "predicted_ltv": 280
      }
    },
    "expected_incremental_orders": 450,
    "overall_roi": 2.9,
    "ltv_cac_ratio": 3.8
  }
}
```

### 4.5 RetentionAgent（留存召回 Agent）

**职责**：管理新客养成、流失预警和流失召回。

**工具集**：

| 工具 | 功能 | 技术实现 |
|------|------|---------|
| `NurturePlanner` | 新客养成计划 | 首单后 7/14/30 天的阶段性运营策略 |
| `ChurnPredictor` | 流失预警模型 | LightGBM 预测用户流失概率，SHAP 归因 |
| `WinbackPlanner` | 召回策略 | 针对不同流失群体的个性化召回方案 |
| `CohortAnalyzer` | 同期群分析 | 按获客渠道/时间追踪留存曲线，定位衰减拐点 |

**输出格式**：
```json
{
  "nurture_status": {
    "new_users_7d": 450,
    "day1_retention": 0.35,
    "day7_retention": 0.18,
    "day30_retention": 0.08,
    "nurture_plan": {
      "day1": {"action": "满意度推送 + 8折复购券", "expected_conversion": 0.12},
      "day7": {"action": "二单激励券（满50减10）", "expected_conversion": 0.08}
    }
  },
  "churn_analysis": {
    "at_risk_users": 1200,
    "high_value_churn_risk": 350,
    "churn_reasons": {
      "price_sensitivity": 0.45,
      "service_quality": 0.30,
      "no_repeat_need": 0.25
    }
  },
  "winback_plan": {
    "high_value_churned": {
      "count": 350, "strategy": "外呼 + 大额券（满100减30）", "expected_recall_rate": 0.15
    },
    "price_sensitive_churned": {
      "count": 540, "strategy": "Push推送 + 折扣券", "expected_recall_rate": 0.08
    }
  },
  "cohort_analysis": {
    "channels": {
      "金刚位": {"d7_retention": 0.22, "d30_retention": 0.10, "avg_ltv": 450},
      "Push": {"d7_retention": 0.15, "d30_retention": 0.06, "avg_ltv": 280},
      "域外广告": {"d7_retention": 0.12, "d30_retention": 0.04, "avg_ltv": 180}
    },
    "insight": "金刚位获客的留存质量显著高于其他渠道，建议增加金刚位资源"
  }
}
```

### 4.6 AdAgent（域外投放 Agent）

**职责**：优化在抖音、快手等外部平台的广告投放策略。

**工具集**：

| 工具 | 功能 | 技术实现 |
|------|------|---------|
| `RTAStrategy` | RTA 竞价策略 | 基于用户价值的实时竞价决策 |
| `BidOptimizer` | OCPX 出价优化 | PID 控制器 + ECPC 策略调节出价 |
| `CreativeAnalyzer` | 创意效果分析 | A/B 测试分析、素材疲劳度检测 |
| `AudienceAnalyzer` | 受众分析 | 人群画像、相似人群扩展、重定向 |

---

## 五、通用工具层：实验平台

### 5.1 为什么实验平台是通用工具

实验不是某个 Agent 独有的能力，而是贯穿所有增长决策的基础设施：

- ProspectAgent 训练的新模型需要 A/B 测试验证效果
- ConversionAgent 的触达策略需要实验对比不同方案
- SubsidyAgent 的发券方案需要实验验证因果效应
- RetentionAgent 的召回策略需要实验评估效果

因此 ExperimentPlatform 作为通用工具，所有 Agent 都可调用。

### 5.2 ExperimentPlatform 设计

```python
class ExperimentPlatform:
    """增长实验平台 —— 支持全链路的 A/B/N 实验"""

    def design_experiment(
        self,
        hypothesis: str,               # 实验假设："15 元满减券比 10 元转化率高"
        metric: str,                    # 主指标："first_order_rate"
        guardrail_metrics: list[str],   # 护栏指标：["avg_order_value", "d7_retention"]
        expected_lift: float,           # 预期提升幅度：0.03
        baseline_rate: float,           # 基线转化率：0.08
        n_variants: int = 2,            # 实验组数（含对照组）
        alpha: float = 0.05,            # 显著性水平
        power: float = 0.8,             # 统计功效
    ) -> dict:
        """
        设计实验：
        1. 计算最小样本量（基于 MDE、alpha、power）
        2. 建议运行时长（基于日均流量）
        3. 生成分流方案
        """
        ...

    def analyze_experiment(
        self,
        experiment_data: pd.DataFrame,
        metric: str,
        method: str = "t_test",   # t_test / chi_square / sequential / bayesian
    ) -> dict:
        """
        分析实验结果：
        1. 各组的核心指标值和置信区间
        2. 统计显著性（p-value）
        3. 实际提升幅度和置信区间
        4. 护栏指标检查（是否触发告警）
        5. 样本比例偏差检测（SRM）
        """
        ...

    def sequential_test(
        self,
        daily_data: pd.DataFrame,
        metric: str,
        method: str = "sprt",     # SPRT / O'Brien-Fleming
        alpha_spend: float = 0.05,
    ) -> dict:
        """
        序贯检验：在实验运行期间持续监控，达到显著性即可提前停止。
        不需要等到预定的实验结束时间，节省时间。
        """
        ...

    def prioritize_experiments(
        self,
        experiment_backlog: list[dict],
    ) -> list[dict]:
        """
        ICE 框架排列实验优先级：
        Impact（预期影响） × Confidence（信心程度） × Ease（实施难度）
        """
        ...
```

### 5.3 实验分析输出格式

```json
{
  "experiment": {
    "name": "10元 vs 15元满减券效果对比",
    "hypothesis": "15 元满减券的首单转化率显著高于 10 元",
    "status": "completed",
    "duration": "14 天",
    "sample_size": 6400
  },
  "results": {
    "control": {"rate": 0.082, "ci": [0.074, 0.090]},
    "treatment": {"rate": 0.098, "ci": [0.089, 0.107]},
    "lift": 0.016,
    "lift_ci": [0.004, 0.028],
    "p_value": 0.008,
    "significant": true
  },
  "guardrails": {
    "avg_order_value": {"control": 68.5, "treatment": 65.2, "alert": false},
    "d7_retention": {"control": 0.18, "treatment": 0.19, "alert": false}
  },
  "recommendation": "15 元满减券的首单转化率显著高于 10 元（+16%，p=0.008），且护栏指标未触发告警。建议全量上线。"
}
```

---

## 六、新增工具详细设计

### 6.1 LVTPredictor（LTV 预测）

```python
class LVTPredictor:
    """用户长期价值预测 —— 基于历史同期群数据预测每个用户的 LTV"""

    def predict_ltv(
        self,
        user_features: pd.DataFrame,
        cohort_history: pd.DataFrame,  # 历史同期群的 LTV 实际数据
    ) -> pd.Series:
        """
        预测每个用户的 90 天/180 天 LTV。
        方法：基于 BG/NBD + Gamma-Gamma 模型，或 LightGBM 回归。
        """
        ...

    def ltv_by_channel(
        self,
        conversion_data: pd.DataFrame,
        ltv_predictions: pd.Series,
    ) -> dict:
        """
        按获客渠道统计 LTV：
        - 金刚位获客的平均 LTV
        - Push 获客的平均 LTV
        - 域外广告获客的平均 LTV
        → 用于跨渠道预算分配
        """
        ...

    def compute_ltv_cac_ratio(
        self,
        ltv_predictions: pd.Series,
        cac_by_channel: dict,   # 各渠道的获客成本
    ) -> dict:
        """
        计算 LTV:CAC 比值
        健康值：3:1 以上
        低于 3:1 的渠道需要优化或减少投入
        """
        ...
```

### 6.2 Attributor（多触点归因）

```python
class Attributor:
    """多触点归因分析 —— 评估各触达渠道对转化的真实贡献"""

    def first_touch Attribution(self, journeys: pd.DataFrame) -> dict: ...
    def last_touch_attribution(self, journeys: pd.DataFrame) -> dict: ...
    def linear_attribution(self, journeys: pd.DataFrame) -> dict: ...
    def time_decay_attribution(self, journeys: pd.DataFrame, half_life: int = 7) -> dict: ...
    def u_shaped_attribution(self, journeys: pd.DataFrame) -> dict: ...

    def compare_models(
        self,
        journeys: pd.DataFrame,
    ) -> dict:
        """
        对比不同归因模型下各渠道的贡献度
        输出：
        {
          "first_touch": {"金刚位": 0.40, "Push": 0.25, "Banner": 0.20, "SMS": 0.15},
          "last_touch":  {"金刚位": 0.15, "Push": 0.20, "Banner": 0.25, "SMS": 0.40},
          "time_decay":  {"金刚位": 0.25, "Push": 0.20, "Banner": 0.35, "SMS": 0.20},
          "u_shaped":    {"金刚位": 0.35, "Push": 0.10, "Banner": 0.20, "SMS": 0.35}
        }
        """
        ...

    def channel_roi_under_attribution(
        self,
        attribution_result: dict,
        channel_spend: dict,     # 各渠道投入成本
        order_values: dict,      # 各渠道带来的订单价值
    ) -> dict:
        """基于不同归因模型计算各渠道的真实 ROI"""
        ...
```

### 6.3 SeasonalAnalyzer（季节性分析）

```python
class SeasonalAnalyzer:
    """季节性分析 —— 识别业务周期性规律，支持活动规划"""

    def detect_seasonality(
        self,
        daily_metrics: pd.DataFrame,   # 日级别的历史指标数据
        period: str = "auto",          # auto / weekly / monthly / quarterly
    ) -> dict:
        """
        检测季节性模式：
        - 周内模式（工作日 vs 周末）
        - 月内模式（月初/月中/月末）
        - 年内模式（毕业季、搬家季等）
        """
        ...

    def year_over_year(
        self,
        current_data: pd.DataFrame,
        historical_data: pd.DataFrame,
    ) -> dict:
        """同期对比：今年 vs 去年同期的各指标变化"""
        ...

    def forecast_demand(
        self,
        historical_data: pd.DataFrame,
        horizon_days: int = 30,
    ) -> dict:
        """
        需求预测：基于季节性模型预测未来 N 天的需求量
        用于提前储备运营资源和优惠券
        """
        ...

    def campaign_calendar(
        self,
    ) -> list[dict]:
        """
        活动日历：全年的关键活动节点
        [
          {"date": "06-01", "name": "毕业季启动", "expected_lift": "+45%"},
          {"date": "11-01", "name": "双11大促", "expected_lift": "+30%"},
          {"date": "12-15", "name": "年底搬家潮", "expected_lift": "+35%"},
        ]
        """
        ...
```

---

## 七、中间件架构

### 7.1 中间件栈设计

```python
middleware_stack = [
    ToolErrorHandlingMiddleware(),           # 错误处理
    ToolArgumentParsingMiddleware(),         # 参数解析
    ModelRetryMiddleware(max_retries=3),     # 重试 + fallback
    ContextWindowMiddleware(max_tokens=8000),# 上下文管理
    StreamingMiddleware(),                   # SSE 流式输出
    LoggingMiddleware(level="INFO"),         # 日志记录
]
```

### 7.2 中间件接口

```python
class AgentMiddleware(ABC):
    @abstractmethod
    async def wrap_model_call(self, request: ModelRequest, handler: Callable) -> ModelResponse: ...

    @abstractmethod
    async def wrap_tool_call(self, request: ToolRequest, handler: CallableResponse: ...
```

---

## 八、配置管理

### 8.1 agent_config.yaml

```yaml
llm:
  primary:
    provider: openai
    model: gpt-4o
    temperature: 0.1
  fallback:
    - provider: deepseek
      model: deepseek-chat

agents:
  prospect:
    name: ProspectAgent
    description: 从集团用户中识别货运潜客 + LTV 预测
    tools: [FeatureEngine, IntentModel, UserScorer, UserSegmentor, LVTPredictor]
    max_iterations: 5

  conversion:
    name: ConversionAgent
    description: 端内触达策略 + 转化漏斗 + 归因分析 + 季节性运营
    tools: [ReachPlanner, FunnelAnalyzer, SlotAllocator, CouponDesigner, Attributor, SeasonalAnalyzer]
    max_iterations: 5

  subsidy:
    name: SubsidyAgent
    description: 因果推断驱动的补贴策略优化
    tools: [CausalInferenceEngine, ElasticityEstimator, BudgetOptimizer, SubsidyAllocator]
    max_iterations: 5

  retention:
    name: RetentionAgent
    description: 新客养成 + 流失预警与召回 + 同期群分析
    tools: [NurturePlanner, ChurnPredictor, WinbackPlanner, CohortAnalyzer]
    max_iterations: 5

  ad:
    name: AdAgent
    description: 域外广告投放优化（辅助渠道）
    tools: [RTAStrategy, BidOptimizer, CreativeAnalyzer, AudienceAnalyzer]
    max_iterations: 5

common_tools: [DataLoader, Visualizer, ExperimentPlatform]
```

---

## 九、Prompt 模板系统

### 9.1 模板结构

```
src/prompts/
├── config/
│   └── prompts.yaml
└── templates/
    ├── system.md.j2
    ├── orchestrator.md.j2
    ├── agents/
    │   ├── prospect.md.j2
    │   ├── conversion.md.j2
    │   ├── subsidy.md.j2
    │   ├── retention.md.j2
    │   └── ad.md.j2
    └── components/
        ├── output_format.md.j2
        ├── tool_guide.md.j2
        └── safety_rules.md.j2
```

---

## 十、数据模型

### 10.1 输入数据

```python
class UserBehaviorData(TypedDict):
    user_id: str
    ride_count_30d: int
    night_ride_ratio: float
    weekend_ride_ratio: float
    avg_ride_distance: float
    destination_types: list[str]
    search_keywords: list[str]
    app_usage_hours: list[int]
    has_large_item_search: bool
    days_since_last_ride: int
    lifetime_orders: int
    city: str
    city_tier: int
    has_freight_search: bool

class FunnelData(TypedDict):
    user_id: str
    exposed: bool
    clicked: bool
    browsed: bool
    claimed_coupon: bool
    first_order: bool
    channel: str
    coupon_type: str
    coupon_amount: float
    touchpoint_sequence: list[str]  # 多触点路径

class SubsidyExperimentData(TypedDict):
    user_id: str
    user_segment: str
    city: str
    experiment_group: str           # 实验分组
    coupon_type: str
    subsidy_amount: float
    converted: bool
    order_value: float              # 订单金额
    days_active_7d: int
    historical_orders: int

class RetentionData(TypedDict):
    user_id: str
    acquire_channel: str
    acquire_date: str
    first_order_date: str
    orders_7d: int
    orders_14d: int
    orders_30d: int
    last_order_date: str
    is_churned: bool
    predicted_ltv: float            # 预测 LTV

class TouchpointJourney(TypedDict):
    """多触点用户旅程"""
    user_id: str
    journey: list[dict]             # [{channel, action, timestamp}]
    converted: bool
    conversion_value: float

class AdCampaignData(TypedDict):
    campaign_id: str
    platform: str
    date: str
    impressions: int
    clicks: int
    conversions: int
    spend: float
    cpa: float
    creative_id: str
    audience_segment: str
    bid: float
```

### 10.2 状态模型

```python
class AgentState(TypedDict):
    # 输入
    query: str
    data_path: str

    # 子 Agent 结果
    prospect_results: NotRequired[dict]
    conversion_results: NotRequired[dict]
    subsidy_results: NotRequired[dict]
    retention_results: NotRequired[dict]
    ad_results: NotRequired[dict]

    # 实验结果
    experiment_results: NotRequired[dict]

    # KPI 与季节性
    kpi_snapshot: NotRequired[dict]
    seasonal_context: NotRequired[dict]

    # 汇聚
    analysis_summary: NotRequired[str]
    strategy_recommendation: NotRequired[str]
    report: NotRequired[str]

    # 元数据
    errors: Annotated[list[str], operator.add]
    metadata: Annotated[list[dict], operator.add]
```

---

## 十一、项目文件结构

```
growth-pilot-agent/
├── pyproject.toml
├── Makefile
├── agent_config.yaml
├── config.yaml
├── .env.example
│
├── src/
│   ├── __init__.py
│   ├── cli.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── prospect.py
│   │   ├── conversion.py
│   │   ├── subsidy.py
│   │   ├── retention.py
│   │   └── ad.py
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── prospect/
│   │   │   ├── feature_engine.py
│   │   │   ├── intent_model.py
│   │   │   ├── user_scorer.py
│   │   │   ├── segmentor.py
│   │   │   └── ltv_predictor.py          ← 新增
│   │   ├── conversion/
│   │   │   ├── reach_planner.py
│   │   │   ├── funnel_analyzer.py
│   │   │   ├── slot_allocator.py
│   │   │   ├── coupon_designer.py
│   │   │   ├── attributor.py             ← 新增
│   │   │   └── seasonal_analyzer.py      ← 新增
│   │   ├── subsidy/
│   │   │   ├── causal_engine.py
│   │   │   ├── elasticity.py
│   │   │   ├── budget_optimizer.py
│   │   │   └── subsidy_allocator.py
│   │   ├── retention/
│   │   │   ├── nurture_planner.py
│   │   │   ├── churn_predictor.py
│   │   │   ├── winback_planner.py
│   │   │   └── cohort_analyzer.py
│   │   ├── ad/
│   │   │   ├── rta_strategy.py
│   │   │   ├── bid_optimizer.py
│   │   │   ├── creative_analyzer.py
│   │   │   └── audience_analyzer.py
│   │   └── common/
│   │       ├── data_loader.py
│   │       ├── visualizer.py
│   │       ├── experiment_platform.py     ← 新增（通用）
│   │       └── experiment_analyzer.py
│   │
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── error_handling.py
│   │   ├── retry.py
│   │   ├── context_window.py
│   │   ├── logging.py
│   │   └── streaming.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── llm_factory.py
│   │   ├── state.py
│   │   └── base.py
│   │
│   ├── prompts/
│   │   ├── config/
│   │   │   └── prompts.yaml
│   │   └── templates/
│   │       ├── system.md.j2
│   │       ├── orchestrator.md.j2
│   │       ├── agents/
│   │       │   ├── prospect.md.j2
│   │       │   ├── conversion.md.j2
│   │       │   ├── subsidy.md.j2
│   │       │   ├── retention.md.j2
│   │       │   └── ad.md.j2
│   │       └── components/
│   │           ├── output_format.md.j2
│   │           ├── tool_guide.md.j2
│   │           └── safety_rules.md.j2
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   └── workflow.py
│   │
│   └── report/
│       ├── __init__.py
│       └── generator.py
│
├── data/
│   ├── user_behavior.csv
│   ├── funnel_data.csv
│   ├── subsidy_experiment.csv
│   ├── retention_data.csv
│   ├── ad_campaign.csv
│   └── seasonal_history.csv             ← 新增
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── tools/
│   │   │   ├── prospect/
│   │   │   ├── conversion/
│   │   │   ├── subsidy/
│   │   │   ├── retention/
│   │   │   ├── ad/
│   │   │   └── common/
│   │   │       └── test_experiment_platform.py  ← 新增
│   │   ├── agents/
│   │   └── middleware/
│   └── integration/
│       └── test_full_workflow.py
│
├── deploy/
│   └── Dockerfile
│
├── reports/
│
└── README.md
```

---

## 十二、CLI 设计

```bash
# 全链路增长分析
gpa analyze --data data/ --scope full --budget 100000

# 端内增长专项
gpa analyze --data data/ --scope inapp

# 单模块分析
gpa analyze --data data/ --scope prospect
gpa analyze --data data/ --scope conversion
gpa analyze --data data/ --scope subsidy
gpa analyze --data data/ --scope retention
gpa analyze --data data/ --scope ad

# 实验相关
gpa experiment design --hypothesis "15元比10元好" --metric first_order_rate
gpa experiment analyze --data data/subsidy_experiment.csv
gpa experiment list                      # 列出所有实验

# 归因分析
gpa attribution --model time_decay --data data/funnel_data.csv

# 季节性分析
gpa seasonal --forecast 30               # 预测未来 30 天需求
gpa seasonal --calendar                   # 查看全年活动日历

# 跨渠道预算优化（基于 LTV）
gpa optimize-budget --data data/ --total-budget 500000

# KPI 看板
gpa kpi                                  # 当前各层级指标快照

# 交互式模式
gpa chat
```

---

## 十三、技术栈

```toml
[project]
name = "growth-pilot-agent"
version = "4.0.0"
requires-python = ">=3.12"

dependencies = [
    # Agent 框架
    "langgraph>=0.2",
    "langchain-core>=0.3",

    # LLM
    "openai>=1.30",

    # ML / 统计
    "lightgbm>=4.0",
    "scikit-learn>=1.3",
    "dowhy>=0.11",
    "econml>=0.15",
    "pulp>=2.7",
    "statsmodels>=0.14",
    "shap>=0.44",
    "lifetimes>=0.11",            # BG/NBD + Gamma-Gamma LTV 模型
    "scipy>=1.11",                # 统计检验

    # 数据处理
    "pandas>=2.0",
    "numpy>=1.24",
    "pydantic>=2.0",

    # 可视化
    "matplotlib>=3.7",
    "seaborn>=0.12",

    # Prompt
    "jinja2>=3.1",
    "pyyaml>=6.0",

    # CLI
    "click>=8.1",
    "rich>=13.0",

    # 服务化
    "fastapi>=0.110",
    "sse-starlette>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
    "mypy>=1.10",
]
```

---

## 十四、实施阶段

### Phase 1：核心工具层（第 1-3 天）

- [ ] 搭建项目骨架（pyproject.toml、目录结构、配置管理）
- [ ] 实现 ProspectTools（FeatureEngine、IntentModel、UserScorer、Segmentor、**LVTPredictor**）
- [ ] 实现 ConversionTools（ReachPlanner、FunnelAnalyzer、CouponDesigner、**Attributor**、**SeasonalAnalyzer**）
- [ ] 实现 SubsidyTools（CausalInferenceEngine、ElasticityEstimator、BudgetOptimizer）
- [ ] 实现 RetentionTools（NurturePlanner、ChurnPredictor、CohortAnalyzer）
- [ ] 实现 AdTools（BidOptimizer、RTAStrategy、CreativeAnalyzer）
- [ ] **实现 ExperimentPlatform（通用实验平台）**
- [ ] 每个工具配套单元测试

### Phase 2：Agent + 中间件（第 4-5 天）

- [ ] 实现 Agent 基类 + 中间件栈
- [ ] 实现 5 个子 Agent
- [ ] 实现 Orchestrator Agent（含 KPI 追踪、季节性感知）
- [ ] Jinja2 Prompt 模板系统

### Phase 3：编排 + CLI（第 6-7 天）

- [ ] LangGraph StateGraph 编排
- [ ] CLI 命令实现（含 experiment / attribution / seasonal 子命令）
- [ ] 报告生成器
- [ ] 示例数据生成

### Phase 4：工程化（第 8-9 天）

- [ ] Docker 多阶段构建
- [ ] Makefile 命令集
- [ ] 集成测试
- [ ] README.md + 架构图

---

## 十五、系统亮点总结

**端内主渠道（占 80%+）：** 滴滴集团有近 10 亿用户，用 LightGBM 从行为数据中识别货运潜客，同时预测每个用户的长期价值（LTV）。通过 APP 内的金刚位、Banner、Push 等资源位触达，用多触点归因模型分析各渠道的真实贡献度。补贴策略用因果推断（DoWhy）评估真实因果效应，整数规划求解最优发券方案。

**实验驱动：** 每个增长策略都通过 A/B 实验验证。实验平台支持序贯检验，达到显著性即可提前决策。每个实验都有护栏指标，确保不会为了短期转化牺牲长期价值。

**季节性运营：** 货运有明显的季节性（毕业季搬家需求暴增 45%、年底搬家潮等）。系统基于历史数据做季节性趋势预测，提前规划活动和储备资源。

**留存体系：** 首单后有 7/14/30 天新客养成计划，流失预警模型识别高风险用户，同期群分析对比不同获客渠道的留存质量。

**技术架构：** Multi-Agent 架构，Chief Agent (ReAct) 编排 5 个 Expert Agent，工具层全部纯 Python 实现，LLM 负责解读和策略生成。
