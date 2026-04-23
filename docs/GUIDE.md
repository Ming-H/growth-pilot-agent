# GrowthPilot Agent 产品使用手册

> 版本：v5.0 | 日期：2026-04-21
> 适用对象：增长工程师、数据分析师、产品运营

---

## 目录

1. [概述](#1-概述)
2. [安装指南](#2-安装指南)
3. [配置说明](#3-配置说明)
4. [CLI 使用](#4-cli-使用)
5. [Web API 使用](#5-web-api-使用)
6. [Scope 说明](#6-scope-说明)
7. [Agent 输出格式](#7-agent-输出格式)
8. [记忆系统](#8-记忆系统)
9. [常见问题 FAQ](#9-常见问题-faq)
10. [架构说明](#10-架构说明)

---

## 1. 概述

### 1.1 GrowthPilot Agent 是什么

GrowthPilot Agent 是一个面向货运平台的 Multi-Agent 用户增长智能系统，覆盖用户全生命周期：**潜客识别 → 端内转化 → 补贴优化 → 留存召回 → 域外投放**。

系统基于 LangGraph 编排 5 个专业子 Agent，每个 Agent 配备独立的 ML 工具链（LightGBM、DoWhy、PuLP 等），由 LLM 负责解读数据并生成策略建议。这不是"LLM 看数据然后瞎说"，而是工具精确计算 + AI 解读的架构。

### 1.2 核心能力

| 能力 | 说明 |
|------|------|
| 潜客识别 | 从集团用户池中识别货运意向用户，LightGBM 意向建模 + LTV 预测 |
| 转化优化 | 多步骤漏斗分析、触达策略规划、优惠券设计、投放位分配 |
| 补贴策略 | DoWhy 因果推断评估真实效果，整数规划预算优化 |
| 留存召回 | 流失预警模型、同期群分析、新客养成、流失用户召回 |
| 广告投放 | RTA 实时竞价策略、OCPX 出价优化、创意疲劳检测、受众定向 |
| 实验平台 | A/B/N 实验设计与分析、序贯检验、护栏指标监控 |

### 1.3 适合谁用

- **增长工程师**：用 CLI 快速跑全链路分析，获取策略建议
- **数据分析师**：对接数据源进行深度分析，生成结构化报告
- **产品运营**：通过 Web API 接入内部系统，实现自动化增长决策
- **增长团队负责人**：通过 KPI Snapshot 和综合策略建议做决策

### 1.4 接入方式

| 方式 | 技术栈 | 适用场景 |
|------|--------|---------|
| **CLI** | Click + Rich | 本地分析、调试、快速查询 |
| **Web API** | FastAPI + SSE | 系统集成、生产环境、流式输出 |

---

## 2. 安装指南

### 2.1 环境要求

| 依赖 | 版本要求 |
|------|---------|
| Python | >= 3.12 |
| uv（推荐） | >= 0.1 |
| pip | >= 24.0（备选） |

### 2.2 安装步骤

**方式一：使用 uv（推荐）**

```bash
# 克隆项目
git clone <repo-url> growth-pilot-agent
cd growth-pilot-agent

# 安装依赖
uv sync

# 开发环境（含 pytest、ruff）
uv sync --group dev
```

**方式二：使用 pip**

```bash
cd growth-pilot-agent

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# 安装
pip install -e .
```

### 2.3 快速验证

```bash
# 查看系统信息（验证安装成功）
gpa info

# 或通过 uv
uv run gpa info
```

输出示例：

```
┌──────────────────────────────────────┐
│         GrowthPilot System Info       │
├────────────────┬─────────────────────┤
│ Property       │ Value               │
├────────────────┼─────────────────────┤
│ Version        │ 4.0.0               │
│ LLM Provider   │ openai              │
│ LLM Model      │ gpt-4o              │
│ API Key Set    │ No (demo mode)      │
│ Log Level      │ INFO                │
│ Python         │ 3.12.x             │
└────────────────┴─────────────────────┘
```

### 2.4 Makefile 快捷命令

```bash
make install       # 安装依赖
make install-dev   # 安装开发依赖
make dev           # 以 demo 模式运行全链路分析
make run           # 运行全链路分析（指定 data/ 目录）
make chat          # 启动交互式对话模式
make test          # 运行单元测试
make test-all      # 运行所有测试
make lint          # 代码检查
make lint-fix      # 自动修复代码问题
make clean         # 清理缓存和报告
make setup-data    # 初始化示例数据
```

### 2.5 核心依赖说明

| 包名 | 用途 |
|------|------|
| langgraph >= 0.2 | Agent 编排引擎 |
| langchain-core >= 0.3 | LLM 抽象层 |
| openai >= 1.30 | OpenAI API 客户端 |
| lightgbm >= 4.0 | 意向模型 / 流失预测 |
| scikit-learn >= 1.3 | ML 工具链基础 |
| dowhy >= 0.11 | 因果推断引擎 |
| pulp >= 2.7 | 整数规划（预算优化） |
| statsmodels >= 0.14 | 统计检验 |
| lifetimes >= 0.11 | BG/NBD LTV 预测 |
| click >= 8.1 | CLI 框架 |
| rich >= 13.0 | 终端美化输出 |
| fastapi >= 0.110 | Web API 框架 |
| sse-starlette >= 2.0 | SSE 流式输出 |
| pydantic >= 2.0 | 数据校验 |
| pydantic-settings >= 2.13 | 环境变量配置 |
| pandas >= 2.0 | 数据处理 |
| matplotlib >= 3.7 | 可视化 |
| jinja2 >= 3.1 | Prompt 模板引擎 |

---

## 3. 配置说明

### 3.1 环境变量配置（.env）

复制 `.env.example` 创建 `.env` 文件：

```bash
cp .env.example .env
```

`.env` 完整配置项说明：

```bash
# ========== LLM 主模型配置 ==========
GPA_LLM_PROVIDER=openai          # 提供商: openai / deepseek / local
GPA_LLM_MODEL=gpt-4o             # 模型名称
GPA_LLM_API_KEY=sk-***           # API Key（必填，否则进入 demo 模式）
GPA_LLM_BASE_URL=                # 自定义端点（可选，用于代理或私有部署）
GPA_LLM_TEMPERATURE=0.1          # 生成温度

# ========== Fallback 模型 ==========
GPA_FALLBACK_PROVIDER=deepseek   # 主模型失败时的备选提供商
GPA_FALLBACK_MODEL=deepseek-chat # 备选模型

# ========== 模型分层配置 ==========
# fast 层：简单格式化、JSON 解析等轻量任务
FAST_PROVIDER=deepseek
FAST_MODEL=deepseek-chat
FAST_TEMPERATURE=0.3

# default 层：标准 Agent 分析任务
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4o-mini
DEFAULT_TEMPERATURE=0.5

# power 层：复杂推理、综合分析（Orchestrator 使用）
POWER_PROVIDER=openai
POWER_MODEL=gpt-4o
POWER_TEMPERATURE=0.7

# ========== 记忆系统 ==========
MEMORY_BASE_PATH=./data/memory   # 记忆存储路径

# ========== 重试配置 ==========
GPA_MAX_RETRIES=3                # 最大重试次数
GPA_RETRY_MIN_WAIT=1             # 最小等待时间（秒）
GPA_RETRY_MAX_WAIT=10            # 最大等待时间（秒）

# ========== 通用配置 ==========
GPA_LOG_LEVEL=INFO               # 日志级别: DEBUG / INFO / WARNING / ERROR
GPA_DATA_DIR=data/               # 数据目录
GPA_OUTPUT_DIR=reports/          # 报告输出目录
```

> **注意**：所有环境变量以 `GPA_` 为前缀，由 `pydantic-settings` 自动加载。

### 3.2 模型分层说明

系统采用三级模型分层架构，根据任务复杂度自动选择模型：

| 层级 | 默认模型 | 适用场景 | Agent |
|------|---------|---------|-------|
| `fast` | deepseek-chat | 简单格式化、JSON 解析、自评打分 | 辅助任务 |
| `default` | gpt-4o-mini | 标准分析、策略解读 | Prospect / Conversion / Subsidy / Retention / Ad |
| `power` | gpt-4o | 综合推理、多维度分析、全局优化 | Orchestrator / Synthesis |

### 3.3 Demo 模式

如果不配置 `GPA_LLM_API_KEY`，系统自动进入 **Demo 模式**：

- 所有 ML 工具使用合成数据运行
- LLM 调用被跳过，分析结果为摘要文本
- KPI 展示示例数据
- 适合快速体验和开发调试

```bash
# 不设置 API Key，直接运行（Demo 模式）
gpa analyze --scope full
```

---

## 4. CLI 使用

GrowthPilot 提供基于 Click + Rich 的 CLI 工具，入口命令为 `gpa`。

### 4.1 全局选项

```
gpa [--verbose/-v]  <command> [options]
```

| 选项 | 说明 |
|------|------|
| `--verbose` / `-v` | 启用 DEBUG 级别日志，输出详细执行过程 |

### 4.2 analyze — 全链路分析

这是核心命令，运行完整的增长分析 pipeline。

**命令格式**：

```
gpa analyze [OPTIONS]
```

**选项**：

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--query` / `-q` | string | `""` | 自然语言查询 |
| `--scope` / `-s` | string | `full` | 分析范围 |
| `--data` / `-d` | string | `""` | 数据文件或目录路径 |
| `--budget` / `-b` | float | `0.0` | 可用预算 |
| `--output` / `-o` | string | `""` | 报告输出文件路径 |

**示例 1：全链路分析**

```bash
gpa analyze -q "帮我分析本周增长数据"
```

输出：

```
┌──────────────────────────────────────┐
│          GrowthPilot Analysis        │
│                                      │
│ Scope: full                          │
│ Data: (none)                         │
│ Budget: (none)                       │
│ Query: 帮我分析本周增长数据           │
└──────────────────────────────────────┘

⠋ Running workflow...

┌──────────────────────────────────────────────────────┐
│               GrowthPilot Report                      │
│                                                       │
│ # GrowthPilot 增长分析报告                             │
│ ## KPI 快照                                           │
│ ...                                                   │
└──────────────────────────────────────────────────────┘

        KPI Snapshot
┌─────────────────┬──────────┐
│ Metric          │ Value    │
├─────────────────┼──────────┤
│ total_users     │ 500      │
│ intent_auc      │ 0.85     │
│ conversion_rate │ 8.00%    │
│ expected_roi    │ 2.80     │
│ churn_risk_ratio│ 12.00%   │
│ ad_cpa          │ 80.00    │
└─────────────────┴──────────┘
```

**示例 2：指定 scope 分析补贴效果**

```bash
gpa analyze -q "分析补贴效果" --scope subsidy --budget 50000
```

输出：

```
┌──────────────────────────────────────┐
│          GrowthPilot Analysis        │
│                                      │
│ Scope: subsidy                       │
│ Data: (none)                         │
│ Budget: 50000.0                      │
│ Query: 分析补贴效果                   │
└──────────────────────────────────────┘

⠋ Running workflow...

┌──────────────────────────────────────────────────────┐
│ 补贴策略分析                                          │
│                                                       │
│ **概述**: ...                                         │
│ **补贴效果评估**: ATE=0.12, p=0.003, 显著             │
│ **预算分配建议**: ...                                  │
│ **ROI 优化策略**: ...                                 │
└──────────────────────────────────────────────────────┘
```

**示例 3：详细模式**

```bash
gpa -v analyze -q "上周实验结果" --scope subsidy --data data/experiment.csv
```

详细模式会输出 DEBUG 级别日志，包括每个工具的执行过程、中间结果等。

**示例 4：保存报告到文件**

```bash
gpa analyze -q "全链路增长方案" --output reports/weekly-report.md
```

### 4.3 chat — 交互式对话

启动交互式对话模式，支持连续查询。

**命令格式**：

```
gpa chat [--scope/-s SCOPE]
```

**示例**：

```bash
gpa chat
```

```
┌──────────────────────────────────────┐
│       GrowthPilot Interactive Mode   │
│                                      │
│ Default scope: full                  │
│ Type your query and press Enter.     │
│ Type 'quit' or 'exit' to stop.      │
│ Type 'scope <name>' to change scope. │
│ Type 'kpi' to see current KPI snap. │
└──────────────────────────────────────┘

> 帮我看看本周的潜客数据
Running analysis with scope=full...
┌──────────────────────────────────────┐
│ GrowthPilot Report                   │
│ ...                                  │
└──────────────────────────────────────┘

> scope subsidy
Scope changed to: subsidy

> 分析补贴ROI
Running analysis with scope=subsidy...
...

> kpi
       KPI Snapshot
┌─────────────────┬──────────┐
│ Metric          │ Value    │
├─────────────────┼──────────┤
│ total_users     │ 1,250,000│
│ conversion_rate │ 8.5%     │
│ churn_rate_30d  │ 12.3%    │
│ roi             │ 3.2x     │
└─────────────────┴──────────┘

> quit
Goodbye!
```

**交互命令**：

| 命令 | 说明 |
|------|------|
| 自然语言查询 | 直接输入问题，运行分析 |
| `scope <name>` | 切换分析范围 |
| `kpi` | 显示当前 KPI 快照 |
| `help` | 显示帮助信息 |
| `quit` / `exit` / `q` | 退出交互模式 |

### 4.4 experiment — 实验设计与分析

**设计实验**：

```bash
gpa experiment design --metric conversion_rate --budget 10000
```

输出：

```json
{
  "experiment_name": "conversion_rate_optimization_test",
  "hypothesis": "Optimizing conversion_rate will improve overall conversion",
  "primary_metric": "conversion_rate",
  "variants": [
    {"name": "control", "description": "Current strategy"},
    {"name": "treatment_a", "description": "Optimized targeting"},
    {"name": "treatment_b", "description": "Enhanced incentive"}
  ],
  "sample_size": 10000,
  "duration_days": 14,
  "significance_level": 0.05,
  "power": 0.8,
  "budget": 10000.0,
  "status": "designed"
}
```

**分析实验结果**：

```bash
gpa experiment analyze --data results.csv --metric conversion_rate
```

输出：

```json
{
  "experiment": "conversion_rate",
  "winner": "treatment_a",
  "lift": 0.12,
  "p_value": 0.003,
  "confidence": 0.95,
  "recommendation": "Roll out treatment_a as the new default"
}
```

### 4.5 attribution — 归因分析

```bash
gpa attribution --model time_decay --data data/touchpoints.csv
```

输出：

```
     Attribution Results (time_decay)
┌─────────────┬───────────────────┬─────────────────────┐
│ Channel     │ Attribution Weight│ Revenue Contribution │
├─────────────┼───────────────────┼─────────────────────┤
│ Search Ads  │ 35%               │ ¥125,000            │
│ Social Ads  │ 25%               │ ¥89,000             │
│ Direct      │ 20%               │ ¥71,000             │
│ Referral    │ 12%               │ ¥43,000             │
│ Email       │ 8%                │ ¥29,000             │
└─────────────┴───────────────────┴─────────────────────┘
```

支持的归因模型：`last_touch` | `first_touch` | `linear` | `shapley`

### 4.6 seasonal — 季节性分析

```bash
gpa seasonal --forecast --calendar
```

输出季节性预测和活动日历：

```
           Seasonal Forecast
┌──────────────┬──────────────┬───────────┬──────────────────────────────┐
│ Period       │ Factor       │ Impact    │ Recommendation               │
├──────────────┼──────────────┼───────────┼──────────────────────────────┤
│ Q2 Week 1-2  │ Freight Peak │ +25%      │ Increase ad budget by 20%    │
│ Q3 Week 1-2  │ Mid-Year     │ +15%      │ Prepare promotional          │
│ Q4 Week 1-2  │ Double 11    │ +40%      │ Maximize acquisition budget  │
└──────────────┴──────────────┴───────────┴──────────────────────────────┘
```

### 4.7 kpi — KPI 快照

```bash
gpa kpi
```

输出当前 KPI 指标概览：

```
         Current KPI Snapshot
┌──────────────────────────┬──────────┐
│ Metric                   │ Value    │
├──────────────────────────┼──────────┤
│ Total Users              │ 1,250,000│
│ Monthly Active Users     │ 450,000  │
│ New Users This Month     │ 32,000   │
│ Conversion Rate          │ 8.5%     │
│ Average Order Value      │ ¥2,800   │
│ Customer Lifetime Value  │ ¥15,600  │
│ Churn Rate 30d           │ 12.3%    │
│ Retention Rate 30d       │ 87.7%    │
│ Nps Score                │ 42       │
│ Ad Spend                 │ ¥580,000 │
│ Cac                      │ ¥186     │
│ Roi                      │ 3.2x     │
└──────────────────────────┴──────────┘
```

### 4.8 info — 系统信息

```bash
gpa info
```

显示系统配置和可用 Agent 列表。

---

## 5. Web API 使用

GrowthPilot 提供 FastAPI 驱动的 Web API，支持同步和 SSE 流式两种模式。

### 5.1 启动服务

```bash
uvicorn src.web:app --host 0.0.0.0 --port 8000
```

### 5.2 POST /api/v1/analyze — 同步分析

提交分析请求，等待完整结果返回。

**请求**：

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "帮我分析本周增长数据",
    "scope": "full",
    "budget": 50000,
    "data_path": "data/weekly.csv"
  }'
```

**请求体 Schema**：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | string | 否 | `""` | 自然语言查询 |
| `scope` | string | 否 | `"full"` | 分析范围 |
| `budget` | float | 否 | `0` | 可用预算 |
| `data_path` | string | 否 | `""` | 数据文件路径 |

**响应**：

```json
{
  "status": "success",
  "scope": "full",
  "kpi_snapshot": {
    "total_users": 500,
    "intent_auc": 0.85,
    "conversion_rate": 0.08,
    "expected_roi": 2.8,
    "churn_risk_ratio": 0.12,
    "ad_cpa": 80.0,
    "budget": 50000,
    "scope": "full"
  },
  "prospect_results": { ... },
  "conversion_results": { ... },
  "subsidy_results": { ... },
  "retention_results": { ... },
  "ad_results": { ... },
  "analysis_summary": "本周整体增长态势良好...",
  "strategy_recommendation": "1. 加大毕业季投放; 2. 优化领券环节; ...",
  "report": "# GrowthPilot 增长分析报告\n...",
  "errors": []
}
```

### 5.3 POST /api/v1/analyze/stream — SSE 流式输出

提交分析请求，通过 Server-Sent Events 实时推送执行进度和结果。

**请求**：

```bash
curl -X POST http://localhost:8000/api/v1/analyze/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "分析补贴效果",
    "scope": "subsidy",
    "budget": 10000
  }'
```

**SSE 事件流**：

```
event: agent_started
data: {"agent": "orchestrator", "step": "scope_detection", "progress": 0}

event: agent_progress
data: {"agent": "orchestrator", "step": "scope_detection", "progress": 50}

event: agent_completed
data: {"agent": "orchestrator", "step": "scope_detection", "progress": 100}

event: agent_started
data: {"agent": "subsidy", "step": "", "progress": 0}

event: agent_progress
data: {"agent": "subsidy", "step": "analysis", "progress": 50}

event: agent_completed
data: {"agent": "subsidy", "step": "", "progress": 100}

event: synthesis_completed
data: {"analysis_summary": "...", "strategy_recommendation": "..."}

event: report_ready
data: {"report": "# GrowthPilot 增长分析报告\n..."}

event: done
data: {"status": "success"}
```

**SSE 事件类型**：

| 事件 | 说明 |
|------|------|
| `agent_started` | Agent 开始执行 |
| `agent_progress` | 执行进度更新 |
| `agent_completed` | Agent 执行完成 |
| `agent_failed` | Agent 执行失败 |
| `tool_called` | 工具被调用 |
| `tool_completed` | 工具执行完成 |
| `synthesis_completed` | 综合分析完成 |
| `report_ready` | 报告生成完成 |
| `done` | 全部完成 |

### 5.4 GET /api/v1/health — 健康检查

```bash
curl http://localhost:8000/api/v1/health
```

**响应**：

```json
{
  "status": "healthy",
  "version": "4.0.0",
  "llm_provider": "openai",
  "llm_model": "gpt-4o",
  "api_key_configured": true,
  "memory_entries": 42,
  "python_version": "3.12.x"
}
```

---

## 6. Scope 说明

Scope 决定了哪些 Agent 会被调度执行，是控制分析范围和成本的核心机制。

### 6.1 Scope 总览

| Scope | 中文名 | 调度 Agent | 适用场景 |
|-------|--------|-----------|---------|
| `full` | 全链路 | prospect, subsidy, ad → conversion → retention | 周报/月报、全面体检 |
| `prospect` | 潜客识别 | prospect → conversion | 新用户获取分析 |
| `conversion` | 转化优化 | conversion | 漏斗/触达/优惠券优化 |
| `subsidy` | 补贴策略 | subsidy | 发券策略、预算优化 |
| `retention` | 留存召回 | retention | 流失预警、召回策略 |
| `ad` | 广告投放 | ad | RTA/出价/创意优化 |
| `inapp` | 端内运营 | conversion → retention | 站内 Push/Banner 运营 |

### 6.2 执行拓扑

```
orchestrator (scope 检测)
    │
    ├── 并行 ── prospect / subsidy / ad (根据 scope 决定是否运行)
    │
    ├── conversion (根据 scope 决定是否运行)
    │
    ├── retention (根据 scope 决定是否运行)
    │
    ├── synthesis (LLM 综合分析，始终运行)
    │
    └── report_gen (报告生成，始终运行)
```

### 6.3 智能检测

当不指定 `--scope` 时，Orchestrator 会根据查询内容自动检测 scope：

```bash
# 自动识别为 subsidy scope（包含"补贴"关键词）
gpa analyze -q "分析补贴效果"

# 自动识别为 retention scope（包含"留存"关键词）
gpa analyze -q "最近用户流失严重怎么办"

# 无匹配关键词时默认 full scope
gpa analyze -q "本周增长数据怎么样"
```

**关键词映射**：

| Scope | 触发关键词 |
|-------|-----------|
| `prospect` | 潜客、获客、拉新、新用户、prospect、acquisition、评分、画像 |
| `conversion` | 转化、漏斗、conversion、funnel、优惠券、coupon、归因、attribution |
| `subsidy` | 补贴、优惠、budget、subsidy、预算、弹性、elasticity、ROI |
| `retention` | 留存、流失、挽回、retention、churn、winback、nurture、培育、群组 |
| `ad` | 广告、投放、RTA、出价、bid、创意、creative、受众、audience、ad |
| `inapp` | 站内、in-app、inapp、push、推送、消息 |

---

## 7. Agent 输出格式

每个 Agent 产出结构化结果，存储在 workflow state 中并写入最终报告。

### 7.1 ProspectAgent 输出

**state key**: `prospect_results`

```json
{
  "user_count": 500,
  "intent_metrics": {
    "auc": 0.85,
    "accuracy": 0.92
  },
  "segment_summary": {
    "high_value_high_intent": {
      "count": 80,
      "ratio": 0.16
    },
    "medium_value_high_intent": {
      "count": 120,
      "ratio": 0.24
    },
    "low_intent": {
      "count": 300,
      "ratio": 0.60
    }
  },
  "rfm_result_count": 5,
  "top_users_sample": [
    {"user_id": "U001", "score": 0.95, "ltv": 520}
  ],
  "analysis": {
    "summary": "从集团用户中识别出高潜用户...",
    "confidence": 0.70,
    "high_value_profile": "高频出行、一线城市、有搬家搜索行为",
    "intent_insight": "意向模型 AUC=0.85，模型可靠",
    "segment_strategy": "高价值高意向用户优先用金刚位触达..."
  }
}
```

**关键字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_count` | int | 分析的用户总数 |
| `intent_metrics.auc` | float | 意向模型 AUC（>0.7 可靠） |
| `segment_summary` | dict | 用户分群统计（名称 → {count, ratio}） |
| `top_users_sample` | list | Top 用户样本（前 10 名） |
| `analysis.summary` | str | LLM 生成的总体概述 |
| `analysis.confidence` | float | 分析置信度（0-1） |
| `analysis.high_value_profile` | str | 高价值用户画像描述 |
| `analysis.segment_strategy` | str | 分群运营建议 |

### 7.2 ConversionAgent 输出

**state key**: `conversion_results`

```json
{
  "reach_result": {
    "strategies": {
      "high_value": {
        "primary_channel": "金刚位",
        "creative": "搬家运货？首单立享8折！",
        "timing": "工作日晚高峰 18:00-20:00"
      }
    }
  },
  "funnel_result": {
    "overall_conversion_rate": 0.032,
    "stages": [
      {"stage": "exposure", "count": 100000, "stage_conversion_rate": 0.25},
      {"stage": "click", "count": 25000, "stage_conversion_rate": 0.72},
      {"stage": "app_open", "count": 18000, "stage_conversion_rate": 0.67},
      {"stage": "search", "count": 12000, "stage_conversion_rate": 0.75},
      {"stage": "first_order", "count": 3200, "stage_conversion_rate": null}
    ],
    "bottleneck": {
      "stage": "click → app_open",
      "stage_conversion_rate": 0.72,
      "diagnosis_hint": "点击后 APP 打开率偏低"
    }
  },
  "slot_result": {
    "total_slots_used": 5,
    "total_slots_available": 8
  },
  "coupon_results": [
    {"segment": "new_user", "coupon_type": "折扣券", "amount": 15, "threshold": 50}
  ],
  "analysis": {
    "summary": "转化策略总体概述",
    "reach_assessment": "触达策略评估",
    "funnel_optimization": "漏斗优化建议",
    "coupon_recommendation": "优惠券策略建议",
    "slot_recommendation": "投放位分配建议"
  }
}
```

### 7.3 SubsidyAgent 输出

**state key**: `subsidy_results`

```json
{
  "ate": {
    "ate": 0.12,
    "ci_lower": 0.08,
    "ci_upper": 0.16,
    "p_value": 0.003,
    "significant_at_05": true,
    "method": "backdoor"
  },
  "causal_insight": "补贴 ATE=0.1200 (p=0.0030), 显著, 95% CI=[0.0800, 0.1600]",
  "confidence": 0.9,
  "elasticity": {
    "elasticity": -1.8,
    "significant_at_05": true,
    "interpretation": "需求富有弹性"
  },
  "price_sensitivity": {
    "elasticity": -1.8,
    "most_sensitive": "new_user",
    "least_sensitive": "high_value"
  },
  "budget_plan": {
    "allocation": {
      "new_user": {"coupon_amount": 15, "user_count": 500},
      "active": {"coupon_amount": 10, "user_count": 300}
    },
    "total_budget_used": 40000,
    "expected_incremental_orders": 450,
    "method": "integer_programming"
  },
  "expected_roi": 2.8,
  "allocation_plan": { ... },
  "analysis": {
    "summary": "补贴策略总体概述",
    "causal_assessment": "补贴效果评估",
    "elasticity_insight": "价格弹性洞察",
    "budget_recommendation": "预算分配建议",
    "roi_strategy": "ROI优化策略"
  }
}
```

**关键字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `ate.ate` | float | 平均处理效应（>0.05 为有效） |
| `ate.p_value` | float | 显著性 p 值 |
| `confidence` | float | 因果推断置信度 |
| `elasticity.elasticity` | float | 价格弹性（\|e\|>1 为敏感） |
| `budget_plan.total_budget_used` | float | 分配使用的预算 |
| `expected_roi` | float | 预期 ROI（>2 为健康） |

### 7.4 RetentionAgent 输出

**state key**: `retention_results`

```json
{
  "nurture_plans": {
    "active": "weekly_push",
    "at_risk": "personalized_offer"
  },
  "nurture_progress": {
    "completion_rate": 0.65,
    "active_plans": 3
  },
  "churn_risk": {
    "high_risk_ratio": 0.12,
    "medium_risk_ratio": 0.25,
    "low_risk_ratio": 0.63,
    "train_auc": 0.88
  },
  "high_risk_users": [
    {"user_id": "high_risk_0", "risk_score": 0.85}
  ],
  "churn_factors": ["低活跃度", "无近30天订单", "价格敏感型用户"],
  "winback_plans": {
    "high_value_churned": {
      "action": "大额优惠券+专属客服回访",
      "budget_share": 0.4
    },
    "medium_risk": {
      "action": "个性化Push+小券引导",
      "budget_share": 0.35
    }
  },
  "winback_priority": ["high_value_churned", "medium_risk", "low_engagement"],
  "cohort_data": {
    "cohort_2024_q1": {"day_7": 0.45, "day_30": 0.28, "day_90": 0.15}
  },
  "retention_curve": {
    "day_1": 0.75, "day_7": 0.45, "day_30": 0.28, "day_90": 0.15
  },
  "cohort_insight": "首个群组留存分析：拐点出现在第 3 周 (类型: drop)",
  "analysis": {
    "summary": "用户留存总体概述",
    "nurture_assessment": "培育进展评估",
    "churn_analysis": "流失风险分析",
    "winback_strategy": "挽回策略建议",
    "cohort_insight": "群组分析洞察",
    "retention_recommendation": "留存优化建议"
  }
}
```

### 7.5 AdAgent 输出

**state key**: `ad_results`

```json
{
  "rta_rules": [
    {"condition": "intent_score > 0.6 AND city_tier <= 2", "action": "bid"},
    {"condition": "intent_score < 0.3", "action": "no_bid"}
  ],
  "rta_metrics": {
    "win_rate": 0.72,
    "total_impressions": 50000
  },
  "bid_result": {
    "original_bid": 5.0,
    "optimized_bid": 2.4,
    "target_cpa": 80.0,
    "estimated_cvr": 0.03
  },
  "expected_cpa": 80.0,
  "creative_result": {
    "underperformers": [
      {"creative_id": "creative_005", "roi": 0.4}
    ]
  },
  "fatigue_alerts": [
    "Creative creative_005 shows low ROI: 0.4"
  ],
  "audience_result": {
    "segments": {
      "high_value": {"avg_ltv": 320}
    }
  },
  "expansion_opportunities": [
    "high_value: high-value segment for expansion"
  ],
  "analysis": {
    "summary": "广告投放总体概述",
    "rta_assessment": "RTA策略评估",
    "bid_optimization": "出价优化建议",
    "creative_plan": "创意优化方案",
    "audience_insight": "受众分析洞察"
  }
}
```

### 7.6 Orchestrator 综合输出

**state key**: `kpi_snapshot`, `analysis_summary`, `strategy_recommendation`

```json
{
  "kpi_snapshot": {
    "total_users": 500,
    "intent_auc": 0.85,
    "conversion_rate": 0.08,
    "expected_roi": 2.8,
    "churn_risk_ratio": 0.12,
    "ad_cpa": 80.0,
    "budget": 50000,
    "scope": "full"
  },
  "analysis_summary": "整体增长态势良好，潜客模型 AUC 达到 0.85，转化率 8% 有提升空间...",
  "strategy_recommendation": "1. 加大毕业季投放预算; 2. 优化领券环节体验; 3. 对高流失风险用户启动召回"
}
```

---

## 8. 记忆系统

GrowthPilot 内置跨会话记忆系统，基于 TF-IDF 语义搜索，实现分析经验的持久化和自动检索。

### 8.1 工作原理

记忆系统采用 4 层架构：

```
Extract（提取） → Store（存储） → Retrieve（检索） → Inject（注入）
```

1. **Extract**：从分析结果中提取关键信息（查询、scope、分析摘要）
2. **Store**：以 JSON 格式持久化到 `data/memory/memory_store.json`
3. **Retrieve**：使用 TF-IDF 余弦相似度进行语义检索，带时间衰减权重
4. **Inject**：将相关历史记忆构建为上下文，注入 Agent Prompt

### 8.2 记忆存储结构

每条记忆条目包含：

```json
{
  "id": "a1b2c3d4e5f6...",
  "run_id": "run_20260421_001",
  "query": "分析补贴效果",
  "scope": "subsidy",
  "results_summary": {
    "ate": 0.12,
    "expected_roi": 2.8
  },
  "timestamp": 1713686400.0
}
```

### 8.3 查看历史记忆

```bash
# 查看记忆文件
cat data/memory/memory_store.json | python -m json.tool

# 查看最近的记忆条目数
python -c "
from src.memory.manager import MemoryManager
mm = MemoryManager()
print(f'Total memories: {mm.count()}')
for entry in mm.get_recent(limit=5):
    print(f'  [{entry[\"scope\"]}] {entry[\"query\"][:50]}')
"
```

### 8.4 清除记忆

```python
from src.memory.manager import MemoryManager

mm = MemoryManager()
removed = mm.clear()
print(f"Cleared {removed} memory entries")
```

或直接删除存储文件：

```bash
rm data/memory/memory_store.json
```

### 8.5 记忆检索示例

当用户发起新查询时，系统自动检索相关历史：

```
用户查询: "上周补贴实验结果怎么样"

系统自动检索到相关记忆:
  - 记忆 1 [2026-04-20 14:30] (scope: subsidy, run: run_20260420)
    原始查询: 分析10元vs15元补贴效果
    分析摘要: {"ate": 0.12, "winner": "treatment_a", ...}

  - 记忆 2 [2026-04-19 10:00] (scope: subsidy, run: run_20260419)
    原始查询: 补贴预算优化
    分析摘要: {"expected_roi": 2.8, "allocation": {...}}
```

### 8.6 配置

在 `.env` 中配置记忆存储路径：

```bash
MEMORY_BASE_PATH=./data/memory
```

默认存储在项目根目录的 `data/memory/` 下。

---

## 9. 常见问题 FAQ

### Q1: 没有配置 API Key 能用吗？

**可以**。系统会自动进入 Demo 模式，使用合成数据运行所有 ML 工具。LLM 摘要功能会被跳过，但工具层的计算结果（意向模型、漏斗分析、因果推断等）仍然正常输出。适合快速体验和开发调试。

```bash
# 不需要任何配置，直接运行
gpa analyze --scope full
```

### Q2: 支持哪些 LLM 提供商？

当前支持三种：

| 提供商 | `GPA_LLM_PROVIDER` | 说明 |
|--------|-------------------|------|
| OpenAI | `openai` | GPT-4o / GPT-4o-mini（推荐） |
| DeepSeek | `deepseek` | deepseek-chat（性价比高） |
| Local | `local` | 本地模型（需配合 GPA_LLM_BASE_URL） |

可以通过 `GPA_LLM_BASE_URL` 配置自定义端点，支持兼容 OpenAI API 的代理服务。

### Q3: 如何选择合适的 scope？

- **日常巡检**：用 `full`，获取全链路 KPI
- **针对性问题**：根据问题选择对应 scope，例如补贴问题用 `subsidy`
- **不确定时**：不用指定 `--scope`，让系统自动检测
- **节省成本**：针对性 scope 只调度相关 Agent，减少 LLM 调用

### Q4: 数据文件格式要求？

支持 CSV 和 Parquet 格式：

```bash
# CSV 文件
gpa analyze -d data/users.csv --scope prospect

# Parquet 文件
gpa analyze -d data/orders.parquet --scope conversion

# 目录（会扫描目录下的文件）
gpa analyze -d data/ --scope full
```

如果没有本地数据文件，系统会自动生成合成数据用于 demo。

### Q5: 如何切换到更便宜的模型？

修改 `.env` 中的模型配置：

```bash
# 使用 DeepSeek 降低成本
GPA_LLM_PROVIDER=deepseek
GPA_LLM_MODEL=deepseek-chat
GPA_LLM_API_KEY=your-deepseek-key

# 或者只降级 default 层（保持 power 层用 GPT-4o）
DEFAULT_PROVIDER=deepseek
DEFAULT_MODEL=deepseek-chat
```

### Q6: 分析速度慢怎么办？

1. **缩小 scope**：用 `--scope subsidy` 替代 `full`，只运行相关 Agent
2. **降低模型**：将 `default` 层从 gpt-4o-mini 切换到 deepseek-chat
3. **使用 demo 数据**：避免加载大文件，系统会使用内置合成数据
4. **关闭 verbose**：不使用 `-v` 选项，减少日志 I/O

### Q7: 报告输出到哪里？

- **终端显示**：默认通过 Rich 格式化在终端输出
- **文件保存**：使用 `--output` 指定文件路径
  ```bash
  gpa analyze -q "周报" --output reports/weekly.md
  ```
- **默认目录**：`reports/` 目录（可通过 `GPA_OUTPUT_DIR` 配置）

### Q8: 如何在 Docker 中运行？

```bash
# 构建镜像
make docker-build

# 运行
make docker-run

# 或手动指定
docker build -f deploy/Dockerfile -t growth-pilot-agent .
docker run --rm -it \
  -v $(PWD)/data:/app/data \
  -e GPA_LLM_API_KEY=your-key \
  growth-pilot-agent
```

---

## 10. 架构说明

### 10.1 系统架构概览

```
┌─────────────────────────────────────────────────────┐
│                     用户接入层                        │
│   CLI (Click + Rich)     │     Web API (FastAPI+SSE) │
└─────────────┬───────────────────────┬───────────────┘
              │                       │
┌─────────────┴───────────────────────┴───────────────┐
│                  LangGraph StateGraph                 │
│                                                       │
│  orchestrator → parallel(prospect,subsidy,ad)        │
│               → conversion → retention               │
│               → synthesis → report_gen → END         │
├──────────┬──────────┬──────────┬──────────┬──────────┤
│Prospect  │Conversion│ Subsidy  │Retention │   Ad     │
│Agent     │Agent     │Agent     │Agent     │  Agent   │
├──────────┴──────────┴──────────┴──────────┴──────────┤
│                   工具层 (纯 Python 计算)              │
│  FeatureEngine, IntentModel, CausalEngine, ...       │
├──────────────────────────────────────────────────────┤
│                   中间件层                             │
│  Retry │ Memory │ Logging │ Prompt Templates         │
├──────────────────────────────────────────────────────┤
│                   基础设施层                           │
│  Config │ LLM Factory │ State │ Model Tiers          │
└──────────────────────────────────────────────────────┘
```

### 10.2 详细文档索引

| 文档 | 内容 | 读者 |
|------|------|------|
| **DESIGN.md** | 业务背景、Agent 详细设计、工具集说明、输出格式、配置管理 | 产品、业务 |
| **TECHNICAL.md** | 分层架构、组件设计、Hook 系统、模型分层、事件系统、项目结构 | 开发工程师 |
| **PRD.md** | 产品需求定义、用户故事、优先级 | 产品经理 |

### 10.3 核心设计原则

1. **工具先行、AI 解读** — 指标计算用 Python 精确完成，LLM 负责解读和策略生成
2. **实验驱动** — 每个增长策略都通过 A/B 实验验证
3. **配置驱动** — Agent 行为、模型选择通过 .env 配置，不硬编码
4. **端内为主、端外为辅** — 滴滴集团用户池是主引流渠道

### 10.4 技术栈一览

| 层级 | 技术 |
|------|------|
| 编排引擎 | LangGraph StateGraph |
| LLM | OpenAI / DeepSeek（通过 langchain-openai） |
| ML | LightGBM, scikit-learn, SHAP, lifetimes |
| 因果推断 | DoWhy |
| 优化 | PuLP (整数规划) |
| 统计 | statsmodels, scipy |
| 数据 | pandas, numpy |
| CLI | Click + Rich |
| Web | FastAPI + sse-starlette + uvicorn |
| 配置 | pydantic-settings + .env |
| 模板 | Jinja2 |
| 记忆 | TF-IDF 语义搜索 + JSON 持久化 |

---

*GrowthPilot Agent v5.0 - Freight Growth Multi-Agent System*
