# v9 架构设计（DESIGN-v9）

| 项 | 内容 |
|---|---|
| 状态 | 已实施（2026-10-28）——双通道 LLM、Agent 四支柱、KPI 校验闸门、三层记忆、确定性回归基线、Web 层修复全部落地；测试 509 绿 |
| 前置 | v8.0.0 agent-led refactor（Chief Agent 工具调用循环为主范式） |
| 目标 | 在拉长任务链（规划/记忆/多轮会话）的同时保证报告可信：事实由确定性工具计算，LLM 只做语言组织，出口闸门逐条校验数字可追溯性，把不可追溯断言的存活率压到零 |

---

## 1. 设计原则（一切能力的边界）

1. **事实唯一来源**：报告中的每个事实（数字/方向）只能来自内部确定性工具的产出（专家 `kpis`、L1 事实台账）。LLM 参数知识可用于语言组织，不得产出事实。
2. **LLM 只拥有"怎么查、怎么讲"**：规划检索顺序、解读数字、组织报告结构——这是 Chief Agent 的全部自由度。确定性代码拥有"对不对、能不能出门"。
3. **记忆回写必须过闸**：进入 L1 台账的只有 `success ∧ _validated ∧ kpis 非空` 的结构化数字；情景记忆只保存过闸后的报告。**读不豁免**——台账命中的数字照样受对账闸门约束。
4. **降级必须可见**：任何确定性接管点都在 `state["degradations"]` 留标记（`plan:rule` / `eval:rule` / `report:template` / `kpi_gate:degraded` / `kpi_gate:annotated` / `agent_loop:capped` / `chief_tools:weak`），`--json` 退出码如实反映。

## 2. LLM 双通道

`GPA_LLM_BASE_URL` 是唯一通道开关，host 自动识别：

- **localhost/127.0.0.1/::1 → 本地轨道**（零 key）：
  - Ollama 形态（:11434 或路径含 `/api`）→ `langchain-ollama` 原生协议。修复了 v8 的一个窟窿：`langchain-community`（ChatOllama 的宿主）从来不在依赖里，"local" provider 一旦被用到就会 ImportError。
  - 其余本地端点（openEuler 上 vLLM/SGLang，通常 :8000）→ OpenAI 兼容，无路径自动补 `/v1`。`GPA_LLM_COMPAT_MODE=true` 强制 Ollama 也走此轨（逃生门）。
- **其他 host → 远端轨道**（需 key）：智谱 `paas/v4`、`coding/paas/v4`、DeepSeek、任意 OpenAI 兼容。

角色预设（fast/default/power）只是**同一通道上的温度与可选模型覆盖**，终结 v8 的 model_tiers 多端点配置。跨厂商兜底只在远端→远端有意义——本地轨道的诚实兜底就是确定性降级，而不是偷偷打外网。

**`create_llm` 返回 None 是降级哨兵**：远端无 key（demo_mode 不再伪造可用性——v8 的占位 key 只会在调用时失败，把"没脑子"伪装成"网络错误"）→ 下游每个 LLM 决策点拿 None 走确定性孪生。

## 3. Agent 四支柱

### 3.1 Planning

- `agent_plan_node`（role=fast）：LLM 输出 `{subgoals: [{expert, question, budget}]}`。
- **确定性校验器只删不造**：白名单交集、按专家去重、≤5 封顶、空问题丢弃。校验器**永不新增** LLM 没点名的专家——会造活的校验器等于代码替模型编造工作量。
- 子目标清单带 ☐/☑/☒ 实时状态进 chief 系统提示词；`agent_tools_node` 按检查点纪律更新状态（`kpis 非空 ∧ success` → done，否则 failed）。
- 无 LLM → 关键词先验（`_intent_classify`），标记 `plan:rule`。

### 3.2 Memory（三层，SQLite 单文件）

- **L0 会话**：近 3 轮全量 + 更早轮压缩为结构化摘要（query 头 + KPIs）——**永不继承自由文本作为事实**。chat 的 `/new` 开新会话；图内 assistant 轮由闸门节点写入（保证入库的是过闸报告）。
- **L1 事实台账**（`verified_facts`）：键 `(org, domain, metric_key, period, run_fingerprint)`，值只存数字。**写闸**如宪法第 3 条；period 用 `''` 而非 NULL（SQLite UNIQUE 视 NULL 互不相同，去重会失效——实测踩过）。
- **L2 别名**（`aliases`）：组织词汇表（如 货主→prospect），`_intent_classify` 命中即抬分——组织黑话驱动路由而不用改代码。`PUT /api/v1/memory/aliases` 维护。
- 情景检索：FTS5（jieba 预分词）+ 按词 LIKE 回退，RRF(k=60) 融合 + recency boost；FTS5 不可用时 LIKE 是一等公民。全部 aiosqlite 异步、单写者串行——v8 的"在事件循环里跑同步文件 IO + 起线程池跑 asyncio.run"整套搬掉。

### 3.3 Tool Use

- ACI 式自描述规格：每个工具带【用途】【何时调用】【参数】【失败模式】；`failure_modes` 结构化字段让 chief 从契约解读失败，而不是看栈回溯。
- 白名单只有内部确定性工具。ad 专家诚实标注 `data_source=synthetic`（无视 data_path 的现状写进契约，接入真实广告平台在路线图）。
- digest 按字段截断（summary 400 / kpis 20 / errors 3），不再切 JSON 字符串产生非法 JSON。

### 3.4 ReAct 与观测

- 步级四元组 `{thought, action, obs, gate}` 进 `agent_steps`；节点级 `{node, dur_ms, detail}` 进 `trace`。`-v` 逐行渲染，`--json` 可重放。
- 步数硬上限（`agent_max_steps=12`）：超限注入收手指令，再超强制路由 evaluate，标记 `agent_loop:capped`——弱本地模型不再烧到 LangGraph 递归上限。
- 弱工具调用翻转：本地通道首个有效步之后仍无 tool_calls 且有 pending 子目标 → 翻转启发式大脑，标记 `chief_tools:weak`。

## 4. 对账闸门（kpi_gate）

**规则**：提取报告中 KPI 词邻近域（±24 字符）内的数字断言（百分比/倍数/千分位/小数），逆格式化后必须匹配 `kpi_snapshot ∪ L1 台账` 中任一值（显示精度容差：`8%`≡`0.08`，`8.0%`≡0.0801）。

**判定**：
- `on`（默认）：违规 ⇒ 报告整体替换为确定性模板 + 对账附录（违规行号/数字/上下文表），标记 `kpi_gate:degraded`——**安全失败**优于自信的错误答案。
- `annotate`：保留报告附差异表（`kpi_gate:annotated`）。
- `off`：跳过。`GPA_KPI_GATE_THRESHOLD` 容忍违规比例（0=零容忍）。

**接线**：`kpi_gate_node` 是两种模式的**最终节点**，且记忆回写在闸门之后——情景库只存过闸报告。闸门纯正则+算术，零 LLM 零网络。

**词表纪律**（防误伤）：数字必须邻近 KPI 词（转化率/ROI/AUC/流失/预算/…）才算断言——日期、章节号、"3 月 15 日"不进闸门。合法但不在 kpis 的量（预算额、高风险用户数）补进快照真值集，而不是放松闸门。

## 5. 双大脑与长程可靠性

启发式大脑与 LLM 大脑**同一张图**：

| LLM 决策点 | 确定性孪生 | 标记 |
|---|---|---|
| plan 子目标规划 | 关键词路由先验 | `plan:rule` |
| chief 工具调用决策 | 按 pending 子目标合成 tool_calls | （启发式步的 gate 字段） |
| 专家 LLM 综合 | `k=v` 确定性摘要（success 不受影响——没脑子是模式不是失败） | 结果 `degraded` 字段 |
| evaluate LLM 裁判 | 完成率×KPI 覆盖率确定性打分，不进 refine | `eval:rule` |
| report LLM 综合 | ReportGenerator 模板 | `report:template` |

```
P(错误答案) ≤ Σₖ eₖ·gₖ    eₖ=第k步出错率（模型决定）  gₖ=错误穿过第k步闸门的存活率（代码决定）
```

四杠杆：能确定化的不给模型（模板/确定性打分/启发式大脑）；verify-as-you-go（子目标检查点+闸门）；结构化传递（subgoals/messages/台账是字段复制不是文本重释）；并行分支不乘法（专家各自过闸）。失败形态=降级+标记+对账附录，非自信错误。

## 6. 确定性不变式（保护一切后续重构）

**同输入 + 零 key ⇒ 字节一致的报告**（`test_zero_key.py` + `gpa eval --heuristic`）。它在本轮开发中真实抓到四个潜伏 bug：

1. LightGBM 多线程训练不确定（分群数 36↔39 漂移）→ `deterministic=True/force_row_wise/num_threads=1` + conftest 进程级单线程环境 + 管线内 `threadpool_limits(1)`。
2. **合成数据用 `pd.Timestamp.now()` 做时间锚**——recency 特征随墙钟秒级漂移（这才是分群漂移的真凶，不是 BLAS）→ 固定锚点 + recency 改"数据最大时间+1 天"的 as-of 锚。
3. **共享 checkpoint 线程跨运行泄漏状态**——run 2 带着 run 1 的 kpi_snapshot 渲染（闸门断言数 6↔9）→ 默认每运行唯一 thread_id，HITL 调用方显式传。
4. agent 模式闸门路由无条件进 approval 节点，被 `interrupt_before` 拦截——**带 checkpointer 的生产环境里 agent 模式从来没产出过报告**（v8 遗留）。路由改为"会在该暂停时才进 approval"。

## 7. Web 层修复（v8→v9 断裂清单）

| # | 问题 | 修复 |
|---|---|---|
| A1 | `/analyze` 不透传 org/user → agent 模式记忆全写 `default` org | `run_workflow` 全参数透传 |
| A2 | SSE 硬连 workflow 图，无视 agent 默认模式 | 按同一模式解析分发，节点事件映射覆盖两种拓扑 |
| A3 | HITL 线程四处不一致（analysis_id vs org:thread），审批永远找不到暂停的 checkpoint | `thread_id_for(org, analysis_id)` 纪律 + Analysis 行持久化 mode（resume 取同拓扑图）+ 首个 Alembic 迁移 |
| A4/A5 | SSE 响应从 raw state 读 `errors`/`analysis_summary`（success 恒 True）；落库 `cost_usd` 恒 0 | 统一走 compat 契约（`metadata.cost_usd`/`kpi_snapshot`） |
| A6 | agent 每次运行重编译图+重连 Postgres | `src/graph/runtime.py` 按模式缓存 |
| A7 | agent 精化轮 off-by-one（`<=` vs `<`） | 统一 `<` |
| A9 | 进程级 CostTracker 并发互相清零 | ContextVar `cost_scope()` 按运行归因，to_thread 传播 |
| 附 | `get_engine` 对 SQLite 传 pool_size（文档里的 sqlite 测试配置从未生效）；web.py 1038 行大杂烩 | pool_size 仅 postgres；拆 `src/web/` 包 |

## 8. 评测策略

- `--router`：确定性路由命中（14/14），CI 安全。
- `--heuristic`：零 key 回归基线——每样本断言 success/路由命中/KPI 非空/报告非空/**两次运行字节一致**。
- `--full`：LLM 评审全链路（需 key）。
- 对账闸门行为由 `test_kpi_gate.py` 18 例钉死（提取/容差/三模式/阈值）。

## 9. Roadmap 的诚实边界

- ad 专家仍是合成数据（已标注 `data_source=synthetic`）——真实广告平台连接器是数据工程活，不在 v9 声称范围内。
- LLM 模式的真机验收矩阵（Ollama 小档/大档 × 双模式 × 闸门）在 v9.1——启发式基线先钉住架构正确性。
- Anthropic 兼容通道未接入；当前覆盖 OpenAI 兼容端点 + Ollama 原生协议。
