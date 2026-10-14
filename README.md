<div align="center">

# GrowthPilot Agent

**Production-Grade Multi-Agent User Growth Platform**

覆盖用户全生命周期 — 潜客识别 → 端内转化 → 补贴优化 → 留存召回 → 域外投放 — 以实验驱动增长决策

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-%E2%89%A50.2-green?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-%E2%89%A50.110-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-452%20passed-brightgreen)](./tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

</div>

---

## Architecture Overview

```
                        ┌────────────────────────────────────────────┐
                        │            LangGraph StateGraph            │
                        │   (compiled w/ checkpoint + interrupt)     │
                        └───────────────────────┬────────────────────┘
                                                ▼
                 ┌────────────────────────────────────────────────┐
                 │  plan — Chief Agent: guardrail → memory 检索    │
                 │  → LLM 路由选择专家 (keyword fallback)           │
                 └───────────────────────┬────────────────────────┘
                          无专家选中 → END │ 选定专家集
                                                ▼
                 ┌────────────────────────────────────────────────┐
                 │  execute — 专家并行 (asyncio.gather + timeout) │
                 │  每个 Expert = BaseAgent.run() 完整生命周期:     │
                 │  hooks(tracing/logging/metrics) → middleware   │
                 │  stack(retry/logging/error-handling) → 确定性   │
                 │  工具管线 → LLM 综合 → Pydantic 边界校验         │
                 └───────────────────────┬────────────────────────┘
                                                ▼
                 ┌────────────────────────────────────────────────┐
                 │  evaluate — LLM-as-judge 三维打分               │
                 │  (completeness / actionability / data-grounding)│
                 └───────────────────────┬────────────────────────┘
                     低于阈值 & 轮次<max   │        通过
                           ▼              └───────────────┐
                 ┌──────────────────┐                      │
                 │ refine — 携带评估 │                      │
                 │ 反馈重跑低分专家   │                      │
                 └────────┬─────────┘                      │
                          └──→ evaluate (循环, ≤2 轮)      ▼
                                              ┌────────────────────┐
                                              │ approval — HITL    │
                                              │ budget≥阈值时中断  │
                                              │ 等待人工审批        │
                                              └─────────┬──────────┘
                                                拒绝 → END │ 批准
                                                           ▼
                                              ┌────────────────────┐
                                              │ report — power 档  │
                                              │ LLM 综合 + 输出护栏 │
                                              │ (fallback: 确定性   │
                                              │  模板报告)          │
                                              └────────────────────┘
```

---

## Highlights（每一条都对应真实实现）

- **LangGraph StateGraph 编排** — `plan → execute → evaluate → (refine | approval | report)` 声明式 DAG；`expert_results` 使用自定义 reducer（按 expert 键替换），refine 循环不会重复累积结果
- **工具先行、AI 解读** — 每个 Expert 先跑确定性工具（LightGBM / DoWhy / PuLP / BG-NBD）产出结构化结果，LLM 只做业务解读与综合，数据层可审计
- **Evaluator-Optimizer 质量循环** — LLM-as-judge 三维打分，低于阈值自动携带反馈重跑（≤2 轮）；执行失败的专家直接记 0 分跳过，不浪费重试
- **Human-in-the-Loop** — `interrupt_before=["approval"]` + PostgreSQL checkpointer，高预算分析暂停等待人工审批，`thread_id` 断点恢复
- **韧性三件套，全部生效** — 指数退避重试（读自配置）→ 熔断器按 **(provider, model, endpoint) 进程级共享** → 跨厂商 fallback；自我 fallback（同端点）会被自动检测并禁用
- **Hook + Middleware 生命周期** — `execute_node` 经由 `BaseAgent.run()` 调用每个专家：Tracing/Logging/Metrics hook 与 Retry/Logging/ErrorHandling 中间件栈在真实链路中执行
- **三层模型分级** — `fast`（评估）/ `default`（规划、专家）/ `power`（最终综合报告）路由到不同模型与温度；tier 配置完全由环境变量驱动
- **持久化记忆接入主管线** — plan 节点按 org 检索历史分析（ChromaDB 向量检索，无依赖时回退 TF-IDF），命中上下文注入规划与专家 prompt
- **成本按运行归因** — LangChain callback 逐 agent 记录 token/成本，每次分析开始时重置，报告输出真实 input/output 拆分
- **输入/输出护栏** — 精确到明确攻击短语的中英文注入检测（误报优先收敛，正常业务用语不拦截）+ 输出脱敏检查
- **离线评估** — `gpa eval --router` 对内置数据集确定性评估路由命中率（当前 14/14），`gpa eval --full` 跑全链路质量评估
- **多租户 SaaS** — JWT 认证、org 级数据隔离、分析持久化（PostgreSQL + async SQLAlchemy）、SSE 流式、限流

---

## Quick Start

### Docker Compose

```bash
cp .env.docker .env
# 编辑 .env —— 至少设置 GPA_LLM_API_KEY 与 GPA_JWT_SECRET
docker compose up -d
# API: http://localhost:8000  文档: http://localhost:8000/docs
```

### 本地开发

```bash
uv sync --group dev
cp .env.example .env    # 编辑 .env 设置 GPA_LLM_API_KEY

# 测试（CI 同款环境变量）
GPA_DEMO_MODE=true GPA_LLM_API_KEY=sk-test-placeholder \
DB_URL=sqlite+aiosqlite:///./test.db \
uv run python -m pytest tests/ -v

# Lint + 格式（CI 双重检查）
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

### CLI

```bash
# 全链路分析
uv run gpa analyze --scope full --query "分析本月用户增长瓶颈" --budget 500000

# 指定数据文件（prospect / retention 专家会真实加载）
uv run gpa analyze --scope prospect --data data/user_behavior.csv

# 离线路由评估（无 LLM、秒级、CI 可跑）
uv run gpa eval --router

# 全链路质量评估（需要 API key）
uv run gpa eval --full --limit 5

uv run gpa chat        # 交互模式
uv run gpa kpi         # KPI 快照
```

### Web API

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"pass123","name":"User","org_name":"MyOrg"}'

curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query":"分析本周用户增长","scope":"full","budget":50000}'
```

---

## Configuration

所有配置走 `GPA_` 前缀环境变量（`src/core/config.py`）：

| Variable | Default | Description |
|----------|---------|-------------|
| `GPA_LLM_PROVIDER` | `openai` | `openai` / `deepseek` / `local`（OpenAI 兼容端点均可） |
| `GPA_LLM_MODEL` | `glm-4.7` | 主模型 |
| `GPA_LLM_API_KEY` | — | API key（demo 模式可为占位符） |
| `GPA_LLM_BASE_URL` | bigmodel.cn | OpenAI 兼容端点 |
| `GPA_FALLBACK_PROVIDER` | `deepseek` | 跨厂商容灾端点 |
| `GPA_FALLBACK_API_KEY` | — | fallback 专用 key（不设则 fallback 不可用） |
| `GPA_FALLBACK_BASE_URL` | — | fallback 专用端点 |
| `GPA_MODEL_TIERS_*` | 见 config | fast/default/power 三档模型路由 |
| `GPA_QUALITY_THRESHOLD` | `0.7` | 评估低于此分数触发 refine |
| `GPA_MAX_REFINEMENT_ROUNDS` | `2` | 质量循环上限 |
| `GPA_BUDGET_APPROVAL_THRESHOLD` | `10000` | 超过此预算触发人工审批 |
| `GPA_DEMO_MODE` | `false` | 允许占位 key + MemorySaver checkpoint |

> **Fallback 说明**：fallback 与主模型相同（provider+model+端点一致）时会被自动禁用并告警——自我容灾没有意义。要启用真实容灾，配置一个不同厂商的 `GPA_FALLBACK_*`。

---

## Agent Matrix

| Expert | 职责 | 工具 | KPIs (标准化输出) |
|--------|------|------|-------------------|
| **prospect** | 意向预测、评分排序、RFM 分群、LTV | `FeatureEngine` `IntentModel`(LightGBM) `UserScorer` `UserSegmentor` `LTVPredictor`(BG/NBD) | `total_users` `intent_auc` |
| **conversion** | 漏斗、触达规划、投放位、优惠券、归因 | `FunnelAnalyzer` `ReachPlanner` `SlotAllocator` `CouponDesigner` `Attributor` | `conversion_rate` `funnel_bottleneck_rate` |
| **subsidy** | 因果推断(ATE)、弹性、预算优化 | `CausalInferenceEngine`(DoWhy) `ElasticityEstimator` `BudgetOptimizer`(PuLP) `SubsidyAllocator` | `expected_roi` `subsidy_ate` |
| **retention** | 流失预警、同期群、养成、召回 | `ChurnPredictor` `CohortAnalyzer` `NurturePlanner` `WinbackPlanner` | `churn_high_risk_ratio` `churn_model_auc` |
| **ad** | RTA、出价、创意、受众 | `RTAStrategy` `BidOptimizer` `CreativeAnalyzer` `AudienceAnalyzer` | `ad_target_cpa` `ad_win_rate` 等 |

每个专家结果携带标准化 `kpis` 字典，workflow 层单点聚合为 `kpi_snapshot`。

---

## Design Decisions

### Why LangGraph over CrewAI / AutoGen?

| Dimension | LangGraph | CrewAI | AutoGen |
|-----------|-----------|--------|---------|
| 拓扑控制 | 显式 StateGraph + 条件边 | 隐式串行/层级 | 会话驱动 |
| 状态管理 | TypedDict + reducer 注解 | 内部状态 | dict 传递 |
| 中断恢复 | `interrupt_before` + checkpointer | 有限 | 有限 |
| 观测性 | 节点级事件 + LangSmith | 有限 | 有限 |

本项目的 refine 循环、HITL 审批中断、按 expert 键替换的 reducer 都依赖 LangGraph 的一等支持。

### Why Tools-First, AI-Interpret?

LightGBM 预测、DoWhy 因果估计、PuLP 优化结果是**数学事实**，不应由 LLM 幻觉生成。LLM 的角色严格限定在：解读工具输出的业务含义、跨专家综合、生成可执行建议。这保证数据层**确定性、可审计**，策略层才有推理弹性。

### Why TypedDict State + Pydantic Boundary?

LangGraph 的状态通道需要 `Annotated[..., reducer]` 支持并行合并，Pydantic 模型不原生支持。方案：图状态用 TypedDict + 自定义 reducer，Agent 边界用 Pydantic（`EXPERT_RESULT_MODELS`）做结构化校验，两套体系各司其职。

### 演示数据说明（诚实声明）

无外部数据时，管线生成**与特征相关的合成数据**：意向标签由特征线性组合 + sigmoid + Bernoulli 噪声合成（约 8% 基率），LTV 由同一信号驱动对数正态分布。因此演示环境下的 AUC 等指标有信息量但非业务真实值——接入 `--data` 后 prospect/retention 使用真实数据。

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| 编排 | LangGraph StateGraph + Checkpointer（PostgreSQL / Memory） |
| LLM | OpenAI 兼容端点（GLM / GPT / DeepSeek / Ollama），三层 tier 路由 |
| ML | LightGBM · DoWhy · PuLP · lifetimes(BG/NBD) · SHAP · statsmodels |
| 服务 | FastAPI · SSE · slowapi · JWT · async SQLAlchemy · Alembic |
| 记忆 | ChromaDB 向量检索（TF-IDF 无依赖回退） |
| 质量 | pytest（452 用例）· Ruff（lint + format）· OpenTelemetry |

---

## License

[MIT](./LICENSE)
