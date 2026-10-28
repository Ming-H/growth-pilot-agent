<div align="center">

# GrowthPilot Agent

**数字对得上账，才允许出门 —— 对账闸门式的用户增长 Agent**

</div>

> **增长的每个数字都要能对账。**

**封闭世界约束下的增长分析 Agent**（v9）：确定性工具计算事实（KPI），LLM 只负责组织语言——**对账闸门拥有"能不能出门"的最终决定权**。报告中的 KPI 数字断言必须能回溯到专家工具的产出（`kpi_snapshot` ∪ L1 事实台账）；对不上的报告自动降级为确定性模板报告并附对账附录。**编造的数字出不了门。**

> **零 key 是默认，不是上限。** LLM 双轨通道：openEuler 本地小模型（Ollama 原生 / vLLM OpenAI 兼容端点，零 key 零费用零外呼）或远端 API key（智谱 GLM / DeepSeek / 任意 OpenAI 兼容端点，`.env` 改一行切换）。没有 LLM 时启发式大脑接管，走同一张图，逐点降级且全程可见。**可靠性由架构保证，不靠模型堆料。**

## 核心特性

| 特性 | 一句话 | 锚点 |
|---|---|---|
| **对账闸门** | 报告中 KPI 语境的数字断言逐条与工具产出对账（`8%`≡`0.08`±显示精度）；违规即降级确定性模板+对账附录 | `src/core/kpi_gate.py`，测试 `test_kpi_gate.py` |
| **封闭世界契约** | 事实只来自内部确定性工具（LightGBM/HiGHS/statsmodels）；LLM 参数知识只组织语言不产出事实 | 分工宪法，DESIGN-v9 §1 |
| **Agent 四支柱** | Planning（子目标清单+只删不造校验）/ Memory（L0 会话+L1 事实台账+L2 别名）/ Tool Use（ACI 自描述+失败模式）/ ReAct（步级四元组观测） | DESIGN-v9 §3 |
| **双大脑** | 启发式规则模式（零 key、可复现回归基线）与 LLM 模式**同一张图**；降级点全带标记（`plan:rule`/`eval:rule`/`report:template`…） | `gpa eval --heuristic` |
| **确定性优先** | LightGBM 单线程确定性训练、合成数据固定时间锚点、BLAS 线程收敛——同输入零 key 模式产出**字节一致**的报告 | `test_zero_key.py::test_report_byte_identical` |
| **LLM 双轨通道** | base_url 一行切脑：localhost 自动识别本地（Ollama 原生/vLLM），远端智谱/DeepSeek 走 API key | DESIGN-v9 §2 |
| **HITL 审批** | 高预算分析 interrupt 暂停，checkpoint 持久化，thread 纪律统一（org:analysis_id） | `/analyses/{id}/approve` |
| **评测内建** | 路由命中 / 启发式回归基线 / 全链路 LLM 评审三档，一条命令复现 | `gpa eval --router \| --heuristic \| --full` |
| **观测三档** | 产品态（成功静默）/ `-v` 诊断态（节点级 trace+耗时）/ `--json` 机器态（整态+诚实退出码） | `gpa analyze -v / --json` |
| **多租户 SaaS** | JWT 认证、org 级隔离（记忆/台账/线程全按 org 作用域）、分析持久化、SSE 流式 | `src/web/` |

## 30 秒跑起来（零 key）

```bash
uv sync --group dev
GPA_DEMO_MODE=true uv run gpa analyze --scope full --query "分析本月用户增长"
```

不需要任何 LLM 配置——启发式大脑立即接管：关键词路由 → 确定性专家管线 → 模板报告，全程带降级标记。要接 LLM 大脑见下节。

## LLM 配置：双轨通道

所有配置写 `.env`（已 gitignore）。**通道由 `GPA_LLM_BASE_URL` 自动识别，换脑不改代码：**

| 轨道 | 端点 | 需要 | 说明 |
|---|---|---|---|
| 本地·Ollama 原生 | `http://localhost:11434` | 装 [Ollama](https://ollama.com) + pull 模型 | **零 key 零费用零外呼**；openEuler 上 `curl -fsSL https://ollama.com/install.sh \| sh` 即装；原生协议对 think/工具调用行为最好 |
| 本地·OpenAI 兼容 | `http://localhost:8000` | vLLM / SGLang 起服务 | 自动追加 `/v1`；`GPA_LLM_COMPAT_MODE=true` 可强制 Ollama 也走此轨（逃生门） |
| 远端·智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `GPA_LLM_API_KEY` | 本仓默认端点；Coding Plan 用户改 `/api/coding/paas/v4` |
| 远端·任意兼容 | DeepSeek / OpenAI / … | `GPA_LLM_API_KEY` | 任意 OpenAI 兼容 chat/completions |

```bash
cp .env.example .env   # 改三行即可切轨道
# 本地轨道示例（什么都不用填，base_url 指向 localhost 即免 key）:
# GPA_LLM_BASE_URL=http://localhost:11434
# GPA_LLM_MODEL=qwen3.5:14b
# 远端轨道示例（智谱）:
# GPA_LLM_API_KEY=你的key
# GPA_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
# GPA_LLM_MODEL=glm-4.7
```

角色预设（fast/default/power）只影响温度与可选的按角色模型覆盖（`GPA_ROLE_MODELS`），不再有每角色独立端点。跨厂商兜底（`GPA_FALLBACK_BASE_URL/MODEL`）仅远端→远端有意义——本地轨道的"兜底"就是确定性降级。**小模型工具调用弱不会死循环**：步数上限（默认 12）+ 连续空调用自动翻转启发式大脑（`chief_tools:weak`）。

## CLI 一览

```bash
gpa analyze --scope full --query "分析本月用户增长"            # 产品态：成功即静默，只出报告+KPI 表
gpa analyze --scope subsidy --budget 50000 -v                 # 诊断态：节点级 trace+耗时+降级标记（stderr）
gpa analyze --scope full --json | jq '.degradations'          # 机器态：整态 JSON，失败/闸门降级退出码 1
gpa analyze --mode workflow ...                               # 固定 DAG 降级模式
gpa chat                                                      # 多轮对话（L0 会话记忆，/new /facts）
gpa eval --router                                             # 确定性路由评测（离线 CI 安全，14/14）
gpa eval --heuristic                                          # 零 key 回归基线（含字节一致性断言）
gpa eval --full                                               # 全链路 LLM 评审（需 key）
gpa kpi / gpa info                                            # KPI 快照 / 系统信息
```

`-v` 诊断态示例（stderr，产品态成功时静默）：

```
┌─ Node Trace ───────────────────────────────┐
│ plan      │ 1255 │ source=rule n=1         │
│ tools     │   43 │ subsidy:ok              │
│ report    │  412 │ chars=775               │
│ kpi_gate  │  428 │ claims=2 violations=0   │
└────────────────────────────────────────────┘
degradations: plan:rule, eval:rule, report:template
```

## 架构

```
                    ┌────────────────────────────────────────────────────┐
   agent 模式(默认)  │ guard → plan → agent ⇄ tools → evaluate ─┬→ agent │
                    │        (子目标)  (LLM/启发式)              │(反馈≤2轮)
                    │                                          ↓
                    │              approval(高预算 interrupt) → report → kpi_gate → END
                    ├────────────────────────────────────────────────────┤
   workflow 模式    │ plan → execute(并行专家) → evaluate ⇄ refine → approval → report → kpi_gate → END
                    ├────────────────────────────────────────────────────┤
   专家层           │ prospect / conversion / subsidy / retention / ad    │
                    │ (确定性 ML 管线 + BaseAgent 生命周期 hook/中间件)    │
                    ├────────────────────────────────────────────────────┤
   工具层           │ LightGBM 意向/流失 · HiGHS 整数规划 · statsmodels    │
                    │ 因果推断 · BG/NBD LTV · 漏斗/归因/RTA               │
                    ├────────────────────────────────────────────────────┤
   记忆层           │ L0 会话(近轮全量+更早结构化压缩) · L1 事实台账        │
                    │ (写闸=成功∧校验∧有KPI，读不豁免闸门) · L2 别名       │
                    └────────────────────────────────────────────────────┘
```

**Agent 四支柱与边界**：

| 支柱 | 落地 | 边界（不破坏对账底线） |
|------|------|------|
| **Planning** | `plan` 节点输出结构化子目标（≤5）；校验器**只删不造**（白名单交集/去重/封顶/空问题丢弃）；子目标清单带 ☐/☑/☒ 状态进 chief 提示词 | 校验器永不新增 LLM 没点名的专家；无 LLM 退关键词先验（`plan:rule`） |
| **Memory** | L0 会话（近 3 轮全量+更早压缩为结构化摘要，**永不继承自由文本**）；L1 事实台账（键=org+domain+metric，写闸=success∧validated∧kpis）；L2 别名（组织词汇表驱动路由） | 台账只存数字不存散文；**读不豁免**——命中数字照样过对账闸门 |
| **Tool Use** | ACI 式自描述规格（用途/何时调用/参数/**失败模式**）；工具失败返回可解释 digest 而非栈回溯 | 白名单只有内部确定性工具；ad 专家诚实标注 `data_source=synthetic` |
| **ReAct** | 步级四元组 thought/action/obs/gate 进 state；`-v` 逐行可重放 | 步数硬上限 12 + 幂等子目标状态 + 连续空调用翻转启发式大脑 |

**长程可靠性**——单步 95% 可靠的任务 20 步只剩 ~36% 成功，该改的是传播结构：

```
P(错误答案) ≤ Σₖ eₖ·gₖ    eₖ=第k步出错率（模型决定）  gₖ=错误穿过该步闸门的存活率（代码决定）
```

四个杠杆：能确定化的环节不给模型（模板报告/确定性打分）；verify-as-you-go（闸门拦截推进到步级）；结构化传递（字段复制不文本重释）；并行分支不乘法（专家各自过闸）。失败形态是**安全失败**（降级/对账附录/诚实标记），不是自信的错误答案。

## 30 秒理解对账闸门

```python
from src.core.kpi_gate import reconcile
reconcile("转化率 8%，ROI 2.1x", {"conversion_rate": 0.08, "expected_roi": 2.1})
# → {'claims': 2, 'violations': [], 'ok': True}        ✓ 8%≡0.08 对上账

reconcile("转化率 99% 创历史新高", {"conversion_rate": 0.08})
# → {'claims': 1, 'violations': [...], 'ok': False}    ✗ 编造被拦，报告降级模板+对账附录
```

## Web API

```bash
make web    # http://localhost:8000/docs
```

JWT 多租户：`POST /auth/register` → `POST /analyze`（sync）/ `POST /analyze/stream`（SSE，按模式分发节点事件）→ `GET /analyses/{id}/state`（checkpoint 状态）→ `POST /analyses/{id}/approve`（HITL 恢复，线程纪律 `org:analysis_id`）。记忆端点：`GET/DELETE /memory`、`GET/PUT /memory/aliases`（L2）、`GET /memory/facts`（L1 台账）。

## 测试与评测

```bash
make test          # 全量（500+ 用例）
gpa eval --router  # 确定性路由评测：14/14
gpa eval --heuristic  # 零 key 回归基线：success/路由/KPI/报告/字节一致 五断言
```

确定性不变式保护所有后续重构：**同输入 + 零 key 模式 ⇒ 字节一致的报告**——它真实抓到过三个潜伏 bug（LightGBM 多线程训练不确定、合成数据用墙钟时间锚点导致 recency 特征漂移、共享 checkpoint 线程跨运行泄漏状态）。

## 目录

```
src/
├── agent/            # agent 模式: plan/chief 循环/工具规格/四元组观测
├── graph/            # workflow DAG · nodes · gate_node(对账闸门) · runtime(图缓存)
├── core/             # config(双轨通道) · llm_factory · kpi_gate · expert 基类
│                     # · evaluator · token_tracker(按运行归因) · guardrails
├── agents/           # 5 个领域专家（确定性管线 + Jinja2 综合提示词）
├── tools/            # 确定性 ML 工具（LightGBM/HiGHS/statsmodels/BG-NBD）
├── memory/           # store.py: L0/L1/L2 + FTS5 情景检索（SQLite，异步）
├── web/              # FastAPI 包: app + routers/{auth,analyses,memory}
├── evals/            # 内置评测集 + 评测器
└── cli.py            # gpa 命令（观测三档 / eval 三模式 / chat L0）
docs/DESIGN-v9.md     # 四支柱设计文档（本 README 的完整推理链）
```

## 配置迁移（v8 → v9）

| v8 | v9 | 说明 |
|---|---|---|
| `GPA_LLM_PROVIDER` | 删除 | 通道由 `GPA_LLM_BASE_URL` 自动识别 |
| `GPA_MODEL_TIERS`（JSON） | `GPA_ROLE_TEMPERATURES` / `GPA_ROLE_MODELS` | 角色预设替代分层端点 |
| `GPA_FALLBACK_PROVIDER` | `GPA_FALLBACK_BASE_URL` + `GPA_FALLBACK_MODEL` | 兜底仅远端→远端 |
| `GPA_MEMORY_BASE_PATH` | `GPA_MEMORY_DB_PATH` | SQLite 单文件替代目录树 |
| `GPA_EXPERT_MODEL_TIER` | 删除 | 专家统一 default 角色 |
| `DB_URL` | `GPA_DB_URL` | 修复前缀缺失（旧值从未生效） |

新增：`GPA_KPI_GATE`（on/annotate/off）、`GPA_KPI_GATE_THRESHOLD`、`GPA_AGENT_MAX_STEPS`、`GPA_LLM_COMPAT_MODE`。

## FAQ

- **没有 key 能用吗？** 能，且是默认。启发式大脑跑同一张图，报告带 `plan:rule`/`report:template` 等可见降级标记；`gpa eval --heuristic` 是可复现的回归基线。
- **本地小模型工具调用很弱怎么办？** 步数上限兜底 + 连续空调用自动翻转到启发式大脑（`chief_tools:weak`）；或 `GPA_LLM_COMPAT_MODE=true` 换 OpenAI 兼容端点试 function calling。
- **闸门会不会误伤合法数字？** 词表邻近域限定（数字须靠近 KPI 词才算断言）、显示精度容差、`annotate` 模式与 `GPA_KPI_GATE_THRESHOLD` 可调；行为被 `test_kpi_gate.py` 钉死。
- **HITL 审批怎么用？** `POST /analyze` 带 `approval_required=true` 且预算 ≥ 阈值（默认 ¥10,000）→ 运行在 approval 前 interrupt → `POST /analyses/{id}/approve` 恢复。
- **macOS 上 LightGBM 报 libomp 缺失？** `brew install libomp`。求解器已切 HiGHS（pulp 自带 CBC 二进制在 ARM Mac 上不可用）。
- **v8 的记忆数据会丢吗？** 不会，首次启动自动把 `memory_store.json` 导入 SQLite（原文件保留为 `.migrated`）。

## Roadmap

- [x] v9.0.0 双轨通道 · 四支柱 · 对账闸门 · 确定性基线 · Web 层修复
- [ ] v9.1 LLM 通道真机验收矩阵（Ollama 小档/大档 × agent/workflow × 闸门）
- [ ] v9.2 真实数据接口（广告平台连接器替换 synthetic 标注；实验数据 schema 导入）
- [ ] v9.3 多轮会话的台账继承策略与 eₖ/gₖ 实测表

## License

MIT
