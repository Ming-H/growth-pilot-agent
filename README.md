<div align="center">

# GrowthPilot Agent

**Production-Grade Multi-Agent User Growth Platform**

覆盖用户全生命周期 — 潜客识别 → 端内转化 → 补贴优化 → 留存召回 → 域外投放 — 以实验驱动增长决策

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-%E2%89%A50.2-green?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-%E2%89%A50.110-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Version](https://img.shields.io/badge/Version-6.0.0-orange)](./pyproject.toml)

</div>

---

## Architecture Overview

```
                           ┌──────────────────────────────────────────────────┐
                           │               LangGraph StateGraph               │
                           │                 (Compiled Graph)                 │
                           └─────────────────────┬────────────────────────────┘
                                                 │
                                                 ▼
                                      ┌─────────────────────┐
                                      │    ORCHESTRATOR      │
                                      │   Scope Detection    │
                                      │  (Intent Keywords)   │
                                      └─────────┬───────────┘
                                                │
                          ┌─────────────────────┼─────────────────────┐
                          │                     │                     │
                          ▼                     ▼                     ▼
                 ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
                 │   PROSPECT     │    │    SUBSIDY      │    │      AD        │
                 │  Agent Node    │    │  Agent Node     │    │  Agent Node    │
                 │                │    │                 │    │                │
                 │ · FeatureEngine│    │ · DoWhy Causal  │    │ · RTA Strategy │
                 │ · LightGBM     │    │ · Elasticity    │    │ · Bid Optim    │
                 │ · UserScorer   │    │ · PuLP Optim    │    │ · Creative     │
                 │ · LTV (BG/NBD) │    │ · Allocator     │    │ · Audience     │
                 └───────┬────────┘    └───────┬─────────┘    └───────┬────────┘
                         │                     │                      │
                         └──────────┬──────────┘──────────────────────┘
                                    │          Fan-in (parallel join)
                                    ▼
                          ┌──────────────────┐
                          │   CONVERSION      │
                          │  Agent Node       │
                          │                   │
                          │ · FunnelAnalyzer  │
                          │ · ReachPlanner    │
                          │ · SlotAllocator   │
                          │ · CouponDesigner  │
                          │ · Multi-Touch Attr│
                          └────────┬──────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │   RETENTION       │
                          │  Agent Node       │
                          │                   │
                          │ · ChurnPredictor  │
                          │ · CohortAnalyzer  │
                          │ · NurturePlanner  │
                          │ · WinbackPlanner  │
                          └────────┬──────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │   SYNTHESIS       │
                          │  (LLM Agg)        │
                          │                   │
                          │ · KPI Snapshot    │
                          │ · Strategy Summary│
                          └────────┬──────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │  REPORT GEN       │
                          │  (Markdown)       │
                          └────────┬──────────┘
                                   │
                                   ▼
                                  END
```

---

## Highlights

- **LangGraph StateGraph 编排** — 以声明式图拓扑驱动 Multi-Agent 协作，支持 Conditional Edge 动态路由（`scope` 关键字意图检测 → 按 scope 激活子集 Agent），编译后单次 `ainvoke()` 完成全链路
- **因果推断驱动的补贴策略** — 集成 DoWhy 进行因果效应估计（ATE/CATE），避免观察数据偏差；PuLP 整数规划求解最优预算分配，而非启发式规则
- **工具先行、AI 解读** — 每个 Agent 先调用确定性工具（LightGBM/Funnel/Churn 等）产出结构化结果，再由 LLM 做业务解读和策略综合，兼顾准确性 & 可解释性
- **三层模型分级** — `fast`（DeepSeek-Chat, 低延迟）、`default`（GPT-4o-mini, 均衡）、`power`（GPT-4o, 高推理），按 Agent 复杂度选层，成本可控
- **中间件栈** — 可插拔的 `AgentMiddleware` 拦截 LLM/Tool 调用链：`RetryMiddleware`（指数退避）→ `LoggingMiddleware`（耗时追踪）→ `ToolErrorHandlingMiddleware`（错误归一化），与 LangChain 的 `with_fallbacks` 互补
- **Hook 生命周期** — `PreRunHook / PostRunHook` 在 Agent 执行前后注入 Tracing、Logging、Metrics 等横切关注点，零侵入
- **持久化记忆系统** — 文件存储 + 自实现 TF-IDF 余弦相似度检索（中英文混合分词），跨 Session 注入历史分析上下文
- **Pydantic 结构化输出** — 每个 Agent 返回强类型 `AgentResult` 子类（`ProspectResult / SubsidyResult / ...`），自动校验、序列化，Agent 间契约清晰

---

## SaaS Features

- **Multi-tenant**: Organization-level data isolation with PostgreSQL
- **JWT Authentication**: Register/login with token-based access control
- **Role-based Authorization**: Admin/member roles with endpoint protection
- **Analysis Persistence**: All analyses saved to database with history and retrieval
- **Cost Tracking**: Per-agent token usage and cost attribution
- **OpenTelemetry**: Production-ready tracing and observability
- **Docker Compose**: One-command full-stack deployment (App + PostgreSQL)

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          User Interface Layer                               │
│                                                                             │
│   Click + Rich CLI           Interactive Chat           FastAPI + SSE      │
│   (gpa analyze/chat/kpi)     (scope-aware REPL)      Auth (JWT) + REST API │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────────┐
│                        Orchestration Engine                                 │
│                                                                             │
│   LangGraph StateGraph ── Conditional Edges ── Scope-based Routing          │
│   AgentState (TypedDict + Annotated Reducers)                               │
│   Event Stream (AgentEvent: started/running/completed/failed)               │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────────┐
│                           Agent Layer                                       │
│                                                                             │
│   BaseAgent (ABC)           Pre/Post Hooks          Middleware Stack        │
│   ├── OrchestratorAgent     ├── TracingHook          ├── ErrorHandling      │
│   ├── ProspectAgent         ├── LoggingHook          ├── RetryMiddleware    │
│   ├── ConversionAgent       └── MetricsHook          └── LoggingMiddleware │
│   ├── SubsidyAgent                                                          │
│   ├── RetentionAgent                                                        │
│   └── AdAgent                                                               │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────────┐
│                           Tool Layer                                        │
│                                                                             │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐            │
│   │  Prospect    │ │ Conversion  │ │   Subsidy    │ │  Retention│            │
│   │ FeatureEngine│ │ FunnelAnlyz │ │ DoWhy Causal │ │ ChurnPred │            │
│   │ IntentModel  │ │ ReachPlannr │ │ Elasticity   │ │ CohortAnl │            │
│   │ UserScorer   │ │ SlotAlloc   │ │ BudgetOptim  │ │ Nurture   │            │
│   │ LTV(BG/NBD)  │ │ CouponDsgn  │ │ Allocatr(PuLP│ │ Winback   │            │
│   └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘            │
│   ┌─────────────┐ ┌─────────────────────────────────────────────┐          │
│   │     Ad      │ │              Common                          │          │
│   │ RTAStrategy │ │ ExperimentPlatform · DataLoader · Visualizer│          │
│   │ BidOptimizer│ │ SecureLoader · Attributor · SeasonalAnalyzer│          │
│   │ CreativeAnl │ └─────────────────────────────────────────────┘          │
│   │ AudienceAnl │  ToolRegistry (decorator-based dynamic registration)     │
│   └─────────────┘                                                          │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────────┐
│                        Infrastructure Layer                                  │
│                                                                             │
│   LLM Factory (OpenAI / DeepSeek / Ollama + Fallback)                      │
│   Pydantic Settings (env vars, GPA_ prefix)                                │
│   Memory Manager (JSON persistence + TF-IDF retrieval)                     │
│   Report Generator (structured Markdown)                                   │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────────┐
│                        Database Layer (PostgreSQL)                           │
│                                                                             │
│   SQLAlchemy ORM (async)     Alembic Migrations      Multi-tenant Isolation│
│   Analysis Persistence       User/Org Models          Cost Tracking         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Matrix

| Agent | Responsibility | Tools | Model Tier |
|-------|---------------|-------|------------|
| **Orchestrator** | 意图识别、scope 路由、KPI 聚合、策略综合 | — (pure decision + LLM synthesis) | `power` |
| **Prospect** | 用户意向预测、用户评分排序、RFM 分群、LTV 预测 | `FeatureEngine` `IntentModel`(LightGBM) `UserScorer` `Segmentor` `LVTPredictor`(BG/NBD) | `default` |
| **Conversion** | 转化漏斗、触达规划、投放位分配、优惠券设计、多触点归因 | `FunnelAnalyzer` `ReachPlanner` `SlotAllocator` `CouponDesigner` `Attributor` | `default` |
| **Subsidy** | 因果推断(ATE/CATE)、价格弹性、预算优化、补贴分配 | `CausalInferenceEngine`(DoWhy) `ElasticityEstimator` `BudgetOptimizer`(PuLP) `SubsidyAllocator` | `power` |
| **Retention** | 流失预警、同期群分析、新客养成、流失召回 | `ChurnPredictor` `CohortAnalyzer` `NurturePlanner` `WinbackPlanner` | `default` |
| **Ad** | RTA 策略、出价优化、创意分析、受众定向 | `RTAStrategy` `BidOptimizer` `CreativeAnalyzer` `AudienceAnalyzer` | `default` |

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Docker Compose (Recommended)

```bash
cp .env.docker .env
# Edit .env — set GPA_LLM_API_KEY and JWT_SECRET
docker compose up -d
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Installation

```bash
# Clone
git clone https://github.com/yourname/growth-pilot-agent.git
cd growth-pilot-agent

# Install dependencies
uv sync

# Or with dev tools
uv sync --group dev
```

### Configuration

```bash
cp .env.example .env
# Edit .env — set GPA_LLM_API_KEY at minimum
```

Key environment variables (all prefixed with `GPA_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GPA_LLM_PROVIDER` | `openai` | LLM provider: `openai` / `deepseek` / `local` |
| `GPA_LLM_MODEL` | `gpt-4o` | Default model name |
| `GPA_LLM_API_KEY` | — | API key (omit for demo mode) |
| `GPA_LLM_BASE_URL` | — | Custom endpoint (e.g. Volcengine Ark) |
| `GPA_FALLBACK_PROVIDER` | `deepseek` | Fallback provider on failure |

### CLI Usage

```bash
# Full-pipeline analysis
uv run gpa analyze --scope full --query "分析本月用户增长情况" --budget 500000

# Scope-specific analysis
uv run gpa analyze --scope subsidy --query "优化补贴预算分配"

# Interactive chat mode
uv run gpa chat

# KPI snapshot
uv run gpa kpi

# Experiment design
uv run gpa experiment design --metric conversion_rate --budget 50000

# Attribution analysis
uv run gpa attribution --model shapley

# Seasonal forecast
uv run gpa seasonal --forecast --calendar
```

### Makefile Shortcuts

```bash
make dev          # Quick dev run (full scope, no data)
make run          # Full run with data/
make chat         # Interactive REPL
make test         # Unit tests
make test-all     # All tests including integration
make lint         # Ruff lint check
```

---

## API Usage

### Register

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "pass123", "name": "User", "org_name": "MyOrg"}'
```

### Login

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "pass123"}'
```

### Analyze

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "分析本周用户增长", "scope": "full", "budget": 50000}'
```

---

## Usage Examples

### 1. Full-Pipeline Growth Analysis

```bash
$ uv run gpa analyze --scope full --query "分析本月用户增长瓶颈并给出优化建议" --budget 500000
```

Output (abbreviated):

```
╭─────────────────── GrowthPilot Report ───────────────────╮
│ # GrowthPilot 增长分析报告                                 │
│                                                          │
│ ## KPI 快照                                              │
│ - 用户总数: 1,250,000                                     │
│ - 意向模型 AUC: 0.87                                      │
│ - 整体转化率: 8.5%                                        │
│ - 预期 ROI: 3.2x                                         │
│ - 流失风险率: 12.3%                                       │
│                                                          │
│ ## 潜客识别                                               │
│ - 意向评分完成: 32,000 高潜用户                            │
│ - LTV 预测: Top 20% 用户贡献 65% 收益                     │
│                                                          │
│ ## 策略建议                                               │
│ 1. 加大 RTA 投放预算 15%，CPA 当前低于目标 20%             │
│ 2. 针对新客 7 日关键窗口设计 push+nurture 组合策略          │
│ 3. 补贴弹性分析显示低净值用户对 5 元券敏感度最高            │
╰──────────────────────────────────────────────────────────╯

       Metric        │    Value
  ───────────────────┼───────────
     total_users     │ 1,250,000
     intent_auc      │    0.87
   conversion_rate   │    8.50%
    expected_roi     │    3.20
  churn_risk_ratio   │   12.30%
       ad_cpa        │  ¥186.00
```

### 2. Subsidy Experiment Analysis

```bash
$ uv run gpa analyze --scope subsidy --query "分析补贴的因果效应并优化预算分配" --budget 500000
```

The Subsidy Agent pipeline:
1. `CausalInferenceEngine` (DoWhy) estimates ATE of subsidy on conversion
2. `ElasticityEstimator` computes price sensitivity by segment
3. `BudgetOptimizer` (PuLP ILP) solves optimal allocation across segments
4. LLM synthesizes insights + ROI projection

### 3. Seasonal Operation Recommendations

```bash
$ uv run gpa seasonal --forecast --calendar
```

```
╭─────────────────── Seasonal Forecast ───────────────────╮
│  Period        │ Factor          │ Impact     │ Rec     │
│ Q4 Week 1-2   │ Double 11       │ +40% demand│ Max acq │
│ Q4 Week 3-4   │ Year-End Rush   │ +30% demand│ CVR opt │
│ Q3 Week 3-4   │ Summer Dip      │ -10% demand│ Retain  │
╰─────────────────────────────────────────────────────────╯
```

---

## Project Structure

```
growth-pilot-agent/
├── pyproject.toml              # Project metadata & dependencies (Hatchling)
├── Makefile                    # Dev shortcuts
├── .env.example                # Environment template
├── deploy/
│   └── Dockerfile              # Container deployment
├── docs/
│   ├── PRD.md                  # Product requirements
│   ├── DESIGN.md               # Design documents
│   └── TECHNICAL.md            # Technical deep-dive
└── src/
    ├── cli.py                  # Click + Rich CLI entry point
    ├── core/                   # Framework core
    │   ├── base.py             # BaseAgent ABC (hooks + middleware + retry)
    │   ├── config.py           # Pydantic Settings (GPA_ env prefix)
    │   ├── events.py           # AgentEvent dataclass (execution tracking)
    │   ├── hooks.py            # PreRunHook / PostRunHook lifecycle
    │   ├── llm_factory.py      # Multi-provider LLM factory + fallback
    │   ├── memory.py           # GrowthMemory (categorized insight persistence)
    │   ├── models.py           # Pydantic structured output models
    │   └── state.py            # AgentState TypedDict (LangGraph state)
    ├── agents/                 # Agent implementations
    │   ├── orchestrator.py     # Scope detection + LLM synthesis
    │   ├── prospect.py         # User scoring, segmentation, LTV
    │   ├── conversion.py       # Funnel, coupons, attribution
    │   ├── subsidy.py          # Causal inference, budget optimization
    │   ├── retention.py        # Churn, cohort, nurture, winback
    │   └── ad.py               # RTA, bid, creative, audience
    ├── graph/
    │   └── workflow.py         # LangGraph StateGraph builder + runners
    ├── tools/                  # Deterministic tool implementations
    │   ├── registry.py         # Decorator-based tool registry
    │   ├── common/             # Shared tools (data loader, experiment, viz)
    │   ├── prospect/           # FeatureEngine, IntentModel, UserScorer, LTV
    │   ├── conversion/         # Funnel, Reach, Slot, Coupon, Attribution
    │   ├── subsidy/            # DoWhy causal, Elasticity, PuLP optimizer
    │   ├── retention/          # Churn, Cohort, Nurture, Winback
    │   └── ad/                 # RTA, Bid, Creative, Audience
    ├── middleware/              # AgentMiddleware intercept layer
    │   └── __init__.py         # Retry / Logging / ErrorHandling middleware
    ├── memory/
    │   └── manager.py          # TF-IDF semantic memory (extract-store-retrieve-inject)
    └── report/
        └── generator.py        # Structured Markdown report builder
```

---

## Design Decisions

### Why LangGraph over CrewAI / AutoGen?

| Dimension | LangGraph | CrewAI | AutoGen |
|-----------|-----------|--------|---------|
| **Graph topology** | Explicit StateGraph with conditional edges | Implicit sequential/hierarchical | Conversation-based |
| **State management** | TypedDict with reducer annotations | Internal state | Dict passing |
| **Routing** | Declarative conditional edges per node | Role-based task assignment | Reply-based flow |
| **Observability** | Node-level event stream + LangSmith | Limited | Limited |
| **Flexibility** | Fan-out/fan-in, skip nodes, loops | Rigid patterns | Rigid patterns |

GrowthPilot requires **precise topology control** — 3 agents run in parallel, then sequential fan-in to conversion → retention → synthesis. LangGraph's `StateGraph` with `add_conditional_edges` gives us first-class support for this. CrewAI's role-based paradigm would force us into rigid sequential patterns; AutoGen's conversation-driven model doesn't map well to a data pipeline.

### Why Tools-First, AI-Interpret?

Each Agent follows a strict pattern:

```
1. Tool execution → deterministic, testable, structured output (dict)
2. LLM interpretation → business insight + strategy over tool results
```

This is deliberate. LightGBM predictions, DoWhy causal estimates, PuLP optimization results are **mathematically grounded** — they should not be hallucinated by an LLM. The LLM's role is limited to:
- Interpreting tool outputs in business context
- Synthesizing cross-agent insights
- Generating actionable recommendations

This architecture ensures the system remains **auditable** and **deterministic** at the data layer while leveraging LLM reasoning at the strategy layer.

### Why Multi-Tier Model Routing?

Not all agents need GPT-4o. A `fast` tier (DeepSeek-Chat) handles simple formatting and extraction tasks at ~1/50th the cost. The `power` tier (GPT-4o) is reserved for:
- **Orchestrator synthesis** — requires strong reasoning across multiple data sources
- **Subsidy causal interpretation** — nuanced causal effect explanation

The `default` tier (GPT-4o-mini) handles the rest — scoring interpretation, funnel analysis, churn recommendations. This 3-tier approach keeps per-run cost under $0.05 for most queries.

### Why TypedDict State over Pydantic for LangGraph?

LangGraph's state channels use `Annotated[..., operator.add]` reducers for list accumulation (errors, metadata, events). Pydantic models don't natively support this pattern. We use `TypedDict` for the LangGraph state and Pydantic (`AgentResult` hierarchy) for agent return values — bridging via `result_to_state_update()`.

---

## Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Agent Framework** | LangGraph >= 0.2 | StateGraph orchestration, conditional routing |
| | LangChain Core >= 0.3 | Base abstractions, message types |
| | LangChain OpenAI >= 1.1 | ChatOpenAI provider |
| **LLM Providers** | OpenAI (GPT-4o / GPT-4o-mini) | Primary reasoning |
| | DeepSeek (deepseek-chat) | Cost-efficient fast tier |
| | Ollama (local) | Offline / air-gapped deployment |
| **ML / Causal** | LightGBM >= 4.0 | Intent prediction, churn scoring |
| | DoWhy >= 0.11 | Causal inference (ATE/CATE estimation) |
| | PuLP >= 2.7 | Integer programming for budget optimization |
| | scikit-learn >= 1.3 | Feature engineering, model metrics |
| | SHAP >= 0.44 | Model interpretability |
| | lifetimes >= 0.11 | BG/NBD + Gamma-Gamma LTV prediction |
| | statsmodels >= 0.14 | Statistical testing |
| | scipy >= 1.11 | Optimization, distributions |
| **Data** | pandas >= 2.0 | Data manipulation |
| | numpy >= 1.24 | Numerical computation |
| | Pydantic >= 2.0 | Structured output, validation, settings |
| **Visualization** | matplotlib >= 3.7 | Charts and plots |
| | seaborn >= 0.12 | Statistical visualization |
| **CLI / UI** | Click >= 8.1 | Command-line interface |
| | Rich >= 13.0 | Terminal formatting (tables, panels, progress) |
| **Server** | FastAPI >= 0.110 | Web API framework |
| | uvicorn >= 0.29 | ASGI server |
| | SSE-Starlette >= 2.0 | Server-sent events streaming |
| **Build / Dev** | Hatchling | Build backend |
| | uv | Dependency management |
| | Ruff >= 0.4 | Linting + formatting |
| | pytest >= 8.0 | Testing |

---

## License

This project is licensed under the [MIT License](./LICENSE).
