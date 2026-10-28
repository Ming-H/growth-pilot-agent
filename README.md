# GrowthPilot Agent

基于 LangGraph 的多智能体用户增长分析平台。系统围绕一条核心原则构建：**确定性工具计算事实，LLM 负责解读与组织语言，出口处的 KPI 校验闸门保证报告中出现的每个关键数字都能回溯到工具产出**。无法追溯的数字断言会被闸门拦截，报告自动降级为确定性模板版本并附差异说明。

平台支持两种运行形态：零 LLM 依赖的确定性模式（关键词路由 + 规则打分 + 模板报告，全程可复现），以及接入 LLM 后的智能编排模式（规划、专家调度、质量评审由模型驱动）。两种模式运行在同一张状态图上，所有降级点均带有显式标记。

## 功能特性

| 能力 | 说明 |
|---|---|
| **五领域专家分析** | 潜客识别（意向预测/评分/RFM 分群/LTV）、转化优化（漏斗/触达/优惠券/多触点归因）、补贴策略（ATE 因果推断/价格弹性/整数规划预算分配）、留存召回（流失预警/同期群/召回策略）、广告投放（RTA/出价/创意疲劳） |
| **双编排模式** | `agent`（默认）：LLM 动态决定专家调度与收敛时机；`workflow`：固定 DAG，关键词路由，不依赖 LLM 决策 |
| **KPI 校验闸门** | 报告中 KPI 语境的数字断言逐条与 `kpi_snapshot` 及事实台账核对（含百分比/倍数/千分位格式与显示精度容差）；支持严格拦截、差异标注、关闭三档 |
| **结构化规划** | 分析前生成子目标清单（专家/问题/预算），确定性校验器执行白名单过滤、去重与封顶；子目标状态实时回写 |
| **三层记忆** | L0 会话记忆（多轮上下文继承）、L1 事实台账（写入门禁：仅成功且校验通过的计算结果入库）、L2 组织别名词典（驱动路由）；SQLite + FTS5 全文检索 |
| **LLM 双通道** | 本地推理（Ollama 原生协议 / vLLM 等 OpenAI 兼容端点，无需 API key）与云端 API（智谱 GLM / DeepSeek 等）仅通过 `GPA_LLM_BASE_URL` 一个变量切换 |
| **多租户 Web API** | JWT 认证、组织级数据隔离、分析持久化、SSE 流式进度、人工审批（HITL）断点恢复 |
| **可观测性** | 产品态（成功静默）、诊断态（`-v`：节点级耗时与降级标记）、机器态（`--json`：完整状态输出与语义化退出码） |
| **内置评测** | 路由命中率（确定性、CI 安全）、零 LLM 回归基线（含报告可复现性断言）、LLM 评审全链路评测 |
| **工程韧性** | LLM 熔断器（进程级共享）、跨厂商降级、指数退避重试、专家级超时、按请求的 token 成本归因、输入注入防护与输出检查 |

## 系统架构

```
                    ┌────────────────────────────────────────────────────┐
   agent 模式(默认)  │ guard → plan → agent ⇄ tools → evaluate ─┬→ agent │
                    │      (子目标规划) (LLM/启发式)             │(反馈≤N轮)
                    │                                          ↓
                    │       approval(高预算暂停) → report → kpi_gate → END
                    ├────────────────────────────────────────────────────┤
   workflow 模式    │ plan → execute(并行专家) → evaluate ⇄ refine         │
                    │       → approval → report → kpi_gate → END         │
                    ├────────────────────────────────────────────────────┤
   专家层           │ prospect / conversion / subsidy / retention / ad    │
                    │ (确定性 ML 管线 + Hook/中间件生命周期)               │
                    ├────────────────────────────────────────────────────┤
   工具层           │ LightGBM · HiGHS 整数规划 · statsmodels 因果推断     │
                    │ · BG/NBD · 漏斗/归因/RTA 策略                       │
                    ├────────────────────────────────────────────────────┤
   记忆层           │ L0 会话 · L1 事实台账 · L2 别名 (SQLite + FTS5)      │
                    └────────────────────────────────────────────────────┘
```

**执行要点**

- **规划（Planning）**：`plan` 节点产出结构化子目标，校验器只做过滤（白名单交集、按专家去重、上限 5 个、空问题剔除），不引入模型未指定的任务；无 LLM 时退化为关键词路由（标记 `plan:rule`）。
- **记忆（Memory）**：L1 台账只存数字事实，写入条件为 `success ∧ Pydantic 校验通过 ∧ KPI 非空`；读取不豁免出口校验——台账中的数字同样受闸门约束。
- **工具（Tools）**：面向 LLM 的工具规格包含用途、调用时机、参数与失败模式四部分，模型可依据契约自行解读失败原因；所有工具为内部确定性实现，广告专家对合成数据显式标注 `data_source: synthetic`。
- **执行循环（ReAct）**：每步记录 thought/action/observation/gate 四元组；步数硬上限（默认 12）防止弱模型死循环，本地小模型连续无有效工具调用时自动切换确定性调度。

## 快速开始

### 环境要求

- Python 3.12+、[uv](https://docs.astral.sh/uv/)
- 可选：PostgreSQL（生产部署；默认使用 SQLite/Demo 模式）、Ollama 或 vLLM（本地 LLM）

### 安装与首次运行

```bash
git clone https://github.com/Ming-H/growth-pilot-agent.git
cd growth-pilot-agent
uv sync --group dev

# 确定性模式运行（无需任何 LLM 配置）
GPA_DEMO_MODE=true uv run gpa analyze --scope full --query "分析本月用户增长" --budget 50000
```

### LLM 配置

编辑 `.env`（参照 `.env.example`）。通道由 `GPA_LLM_BASE_URL` 的主机地址自动判定：

| 通道 | 端点示例 | 凭证 | 说明 |
|---|---|---|---|
| 本地 · Ollama 原生 | `http://localhost:11434` | 无需 | openEuler 上通过官方脚本安装 Ollama 后即可使用 |
| 本地 · OpenAI 兼容 | `http://localhost:8000` | 无需 | vLLM / SGLang 等自建服务，自动补全 `/v1` 路径 |
| 云端 · 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `GPA_LLM_API_KEY` | Coding Plan 用户使用 `/api/coding/paas/v4` |
| 云端 · 其他 | DeepSeek / OpenAI 等兼容端点 | `GPA_LLM_API_KEY` | 任意 OpenAI 兼容服务 |

```ini
# .env 示例 —— 本地通道
GPA_LLM_BASE_URL=http://localhost:11434
GPA_LLM_MODEL=qwen3.5:14b

# .env 示例 —— 云端通道
GPA_LLM_API_KEY=your-key
GPA_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GPA_LLM_MODEL=glm-4.7
```

角色预设（fast/default/power）仅控制采样温度与可选的按角色模型覆盖（`GPA_ROLE_MODELS`），不涉及独立端点配置。跨厂商降级（`GPA_FALLBACK_BASE_URL/MODEL`）仅在云端通道之间生效；本地通道不可用时的降级路径为确定性模式。

## CLI 使用

```bash
gpa analyze --scope full --query "分析本月用户增长" --budget 50000   # 标准分析
gpa analyze --scope subsidy --budget 50000 --query "补贴 ROI" -v     # 诊断模式（stderr 输出节点耗时与降级标记）
gpa analyze --scope full --json | jq '.kpi_snapshot'                 # 机器模式（完整状态 JSON；失败或闸门拦截时退出码为 1）
gpa analyze --mode workflow ...                                      # 固定 DAG 模式
gpa chat                                                             # 多轮对话（会话记忆；/new 新会话，/facts 查询事实台账）
gpa eval --router                                                    # 路由评测（确定性，14/14）
gpa eval --heuristic                                                 # 零 LLM 回归基线（含报告可复现性断言）
gpa eval --full                                                      # 全链路 LLM 评审（需配置 key）
gpa kpi                                                              # KPI 快照
gpa info                                                             # 系统信息（通道/模型/闸门配置）
```

`-v` 诊断输出示例：

```
┌─ Node Trace ──────────────────────────────┐
│ plan      │ 1255 │ source=rule n=1        │
│ tools     │   43 │ subsidy:ok             │
│ report    │  412 │ chars=775              │
│ kpi_gate  │  428 │ claims=2 violations=0  │
└───────────────────────────────────────────┘
degradations: plan:rule, eval:rule, report:template
```

## Web API

```bash
make web    # http://localhost:8000/docs
```

| 端点 | 功能 |
|---|---|
| `POST /api/v1/auth/register` `POST /auth/login` `GET /auth/me` | JWT 认证 |
| `POST /api/v1/analyze` | 同步分析（支持 `mode` / `approval_required` / `session_id` 参数），结果持久化 |
| `POST /api/v1/analyze/stream` | SSE 流式分析，按图节点实时推送阶段事件 |
| `GET /api/v1/analyses` `GET /analyses/{id}` | 分析历史与详情（组织隔离） |
| `GET /api/v1/analyses/{id}/state` | 查询 LangGraph checkpoint 状态（子目标/评分/审批/降级标记） |
| `POST /api/v1/analyses/{id}/approve` `POST /analyses/{id}/resume` | 人工审批与断点恢复 |
| `GET/DELETE /api/v1/memory` | 组织记忆查询与清理 |
| `GET/PUT /api/v1/memory/aliases` | L2 别名词典维护 |
| `GET /api/v1/memory/facts` | L1 事实台账查询 |

人工审批流程：请求携带 `approval_required=true` 且预算超过阈值（`GPA_BUDGET_APPROVAL_THRESHOLD`，默认 ¥10,000）时，执行在审批节点暂停并持久化 checkpoint；通过 `/approve` 提交决策后从中断点恢复。

## 配置参考

常用环境变量（完整列表见 `src/core/config.py` 与 `.env.example`，均带 `GPA_` 前缀）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GPA_LLM_BASE_URL` | 智谱 paas/v4 | LLM 端点；主机地址决定本地/云端通道 |
| `GPA_LLM_MODEL` | `glm-4.7` | 模型名 |
| `GPA_LLM_API_KEY` | 空 | 云端通道必填；本地通道无需 |
| `GPA_ORCHESTRATION_MODE` | `agent` | 编排模式：`agent` / `workflow` |
| `GPA_KPI_GATE` | `on` | 闸门模式：`on`（拦截降级）/ `annotate`（标注差异）/ `off` |
| `GPA_KPI_GATE_THRESHOLD` | `0.0` | 允许的不可追溯断言比例（0 为零容忍） |
| `GPA_AGENT_MAX_STEPS` | `12` | agent 循环步数上限 |
| `GPA_MEMORY_DB_PATH` | `./data/memory/memory.db` | 记忆库路径（SQLite） |
| `GPA_BUDGET_APPROVAL_THRESHOLD` | `10000` | 触发人工审批的预算阈值（CNY） |
| `GPA_DB_URL` | PostgreSQL 本机 | 应用数据库；测试可用 `sqlite+aiosqlite:///...` |
| `GPA_DEMO_MODE` | `false` | 演示模式（放宽启动校验） |

## 测试与评测

```bash
make test          # 全量测试（509 用例）
make lint          # ruff 检查与格式校验
gpa eval --router  # 路由命中率评测（离线）
gpa eval --heuristic  # 零 LLM 回归基线：断言成功/路由/KPI/报告/两次运行报告一致
```

测试套件包含一条核心回归约束：**相同输入在零 LLM 模式下产出字节级一致的报告**。该约束用于拦截数值计算链路中的不确定性（模型训练线程调度、合成数据时间基准、状态跨请求泄漏等），任何破坏该约束的改动都会在 CI 中暴露。

## 目录结构

```
src/
├── agent/            # agent 模式：规划节点 / Chief 循环 / 工具规格 / 步级观测
├── graph/            # workflow DAG · 节点实现 · KPI 闸门节点 · 图运行时缓存
├── core/             # 配置（双通道）· LLM 工厂 · KPI 闸门 · 专家基类
│                     # · 质量评审 · 成本归因 · 输入输出护栏
├── agents/           # 五个领域专家（确定性管线 + 综合提示词）
├── tools/            # 确定性 ML 工具（LightGBM / HiGHS / statsmodels / BG-NBD）
├── memory/           # 三层记忆存储（L0/L1/L2 + FTS5 检索）
├── web/              # FastAPI 应用（app / deps / routers）
├── evals/            # 内置评测数据集与评测器
└── cli.py            # gpa 命令行入口
docs/DESIGN-v9.md     # v9 架构设计文档
tests/                # 测试套件
```

## 从 v8 升级

| v8 配置 | v9 替代 | 说明 |
|---|---|---|
| `GPA_LLM_PROVIDER` | 删除 | 通道由 `GPA_LLM_BASE_URL` 自动判定 |
| `GPA_MODEL_TIERS`（JSON） | `GPA_ROLE_TEMPERATURES` / `GPA_ROLE_MODELS` | 角色预设替代分层端点 |
| `GPA_FALLBACK_PROVIDER` | `GPA_FALLBACK_BASE_URL` + `GPA_FALLBACK_MODEL` | 降级仅限云端通道之间 |
| `GPA_MEMORY_BASE_PATH` | `GPA_MEMORY_DB_PATH` | 目录树合并为单一 SQLite 文件（旧数据自动迁移） |
| `GPA_EXPERT_MODEL_TIER` | 删除 | 专家统一使用 default 角色 |
| `DB_URL` | `GPA_DB_URL` | 修正前缀缺失 |

行为变更、问题修复与依赖调整的完整清单见 [CHANGELOG.md](CHANGELOG.md)。

## License

MIT
