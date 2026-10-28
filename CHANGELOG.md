# Changelog

All notable changes to the GrowthPilot Agent project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [9.0.0] - 2026-10-28

### KPI 校验闸门式增长 Agent

v9 的核心原则：确定性工具计算事实，LLM 只负责解读与组织语言，出口处的
KPI 校验闸门对报告中的数字断言逐条追溯，无法回溯到工具产出的内容被拦截。
完整设计文档见 `docs/DESIGN-v9.md`。

### Added

- **对账闸门**（`src/core/kpi_gate.py`）：报告中 KPI 语境数字断言逐条与
  `kpi_snapshot` ∪ L1 台账对账（显示精度容差）；`on` 模式违规即降级确定性
  模板+对账附录，`annotate` 附差异表，`off` 跳过。两种模式共用，最终节点，
  记忆回写移至闸门后。
- **LLM 双轨通道**：`GPA_LLM_BASE_URL` 自动识别本地（Ollama 原生 via
  langchain-ollama / vLLM OpenAI 兼容）与远端（智谱/DeepSeek/任意兼容），
  仅通过 base_url 切换；本地通道无需 key。角色预设（温度+可选模型覆盖）替代 model_tiers。
  `create_llm` 返回 None 是降级哨兵。
- **Agent 四支柱**：plan 节点（子目标清单+只删不造校验+☐/☑/☒ 状态）；
  启发式大脑（llm=None 时按 pending 子目标合成 tool_calls，同一张图）；
  步数硬上限与弱工具调用翻转（`chief_tools:weak`）；步级四元组观测。
- **三层记忆**（`src/memory/store.py`，SQLite+FTS5）：L0 会话（近轮全量+
  更早结构化压缩）、L1 事实台账（写闸 success∧validated∧kpis，读不豁免）、
  L2 别名（组织词汇驱动路由）。遗留 JSON 自动迁移。新增
  GET/PUT /api/v1/memory/aliases 与 GET /api/v1/memory/facts。
- **观测三档**：`gpa analyze -v`（节点级 trace+耗时+降级标记，stderr）、
  `--json`（整态+诚实退出码）、产品态成功静默。
- **评测三模式**：`--router`（14/14 确定性）、`--heuristic`（零 key 回归
  基线，含字节一致断言）、`--full`（LLM 评审）。
- **chat L0 会话**：持久会话、/new、/facts；图内 assistant 轮写入过闸报告。
- **ACI 工具规格**：用途/何时调用/参数/失败模式进 chief 提示词。
- 首个 Alembic 迁移（0001: analyses.mode）。

### Fixed

- **A1-A9 全量**（v8 agent 重构后 Web 层断裂）：多租户透传、SSE 模式分发、
  HITL 线程纪律（agent 模式生产环境从未产出过报告——闸门路由无条件进被
  interrupt 的 approval 节点）、响应/落库读错 key、图编译缓存、精化 off-by-one、
  OSError 级 import 保护、按运行成本归因（ContextVar cost_scope）。
- **确定性根因三连**：LightGBM 多线程训练不确定、合成数据墙钟时间锚导致
  recency 特征漂移、共享 checkpoint 线程跨运行泄漏状态——字节一致不变式
  全部抓住并修复。
- `get_engine` 对 SQLite 传 pool_size（文档里的 sqlite 配置从未生效）。
- 求解器链 HiGHS→CBC→贪心（pulp 3.x 自带 CBC 二进制 ARM Mac 不可用）；
  scipy>=1.16（1.15.3 `_propack` dylib 被新版 macOS dyld 拒绝）。
- web.py（1038 行）拆为 `src/web/` 包（app/deps/routers）。

### Removed

- 死依赖：shap、dowhy（README 曾宣称、代码从未 import）、matplotlib、
  seaborn、chromadb；死代码：ToolRegistry（无人按名解析）、ContextManager
  （refinement_feedback 死链）、Visualizer、KpiSnapshot、重复 CostTracker、
  CLI 桩命令（experiment/attribution/seasonal）。
- 配置：GPA_LLM_PROVIDER / GPA_MODEL_TIERS / GPA_FALLBACK_PROVIDER /
  GPA_EXPERT_MODEL_TIER / GPA_MEMORY_BASE_PATH（迁移表见 README）。

---

## [8.0.0] - 2026-10-16

### Agent-led refactor — Chief Agent tool-calling loop as the primary paradigm

The orchestration paradigm flips from workflow-led (fixed DAG, bounded LLM
decision points) to classic **agent-led**: the LLM dynamically decides which
expert tools to invoke, in which order, whether results suffice, and when to
produce the final report. The fixed DAG remains available as `--mode workflow`
for deterministic degradation.

### Added

- **`src/agent/` package** — the agent-led mode:
  - `state.py`: `AgentState` extends `AnalysisState` with a LangGraph
    `messages` channel (`add_messages` reducer) for the full conversation
    (tool calls, tool results, feedback).
  - `tools.py`: Agent-as-Tool — the five domain experts plus `search_memory`
    rendered as tool specs (Pydantic arg schemas + state-aware handlers).
    `bind_tools` gets the schemas; execution dispatches with graph-state
    access. The LLM receives compact digests (kpis + summary); full validated
    results merge into state via the reducer.
  - `chief.py`: `chief_agent_node` (power-tier LLM + bound tools), parallel
    `agent_tools_node`, `evaluate_gate_node` (LLM-as-judge with bounded
    feedback re-entry), approval and report nodes (output guardrail +
    per-run cost + memory write-back closing the memory loop).
  - `graph.py`: `guard → agent ⇄ tools → evaluate_gate → approval → report`
    compiled with checkpointer and `interrupt_before=["approval"]` (HITL
    preserved). LLM unavailability fails cleanly with an actionable message
    suggesting `--mode workflow`.
- **`GPA_ORCHESTRATION_MODE`** setting (`agent` default) and CLI
  `gpa analyze --mode agent|workflow`.
- 17 new tests (`tests/test_agent/`): tool schemas/dispatch, loop routing,
  quality-gate feedback, failed-expert error recording, mocked end-to-end
  loop, mode dispatch (469 total).

### Fixed

- `AsyncPostgresSaver.from_conn_string` returns an async context manager —
  calling `.setup()` on it crashed; it is now entered explicitly with a
  module-level reference keeping it alive (was silently degrading to
  MemorySaver in production mode).

### Changed

- README/CLAUDE.md rewritten for the agent-led default with the workflow DAG
  documented as the fallback mode; version bumped to 8.0.0.

---

## [7.0.0] - 2026-10-14

### Integrity release — 让每个宣称的机制真实生效

A full pass over the codebase ensuring every advertised feature is actually wired
into the execution path, all tests green, lint/format clean, and documentation
matches reality.

### Fixed

- **Hooks lifecycle now executes**: `execute_node`/`refine_node` invoke experts
  through `BaseAgent.run()` (previously a bare `analyze()` call bypassed
  tracing/logging/metrics hooks entirely).
- **Circuit breakers are process-shared** per `(provider, model, endpoint)` —
  previously every call site owned a private breaker, so a provider-wide outage
  could never trip a common circuit.
- **Self-fallback detection**: a fallback identical to the primary (same
  provider/model/endpoint) is now detected, rejected with a warning, and
  resilience degrades honestly to primary-only. `fallback_api_key` /
  `fallback_base_url` settings added; default fallback provider is now
  DeepSeek (cross-vendor).
- **Per-run cost attribution**: the global `CostTracker` is reset at the start
  of every `run_analysis()`; `token_usage` reports a real input/output split
  (was `output_tokens: 0` and process-lifetime accumulation).
- **`data_path` flows through the graph**: CLI `--data` / API `data_path` used
  to be silently dropped by `run_workflow`; now carried in `AnalysisState` into
  expert params (prospect & retention load real CSV/Parquet).
- **KPI aggregation standardised**: every expert returns a `kpis` dict merged
  into `kpi_snapshot` (the old hardcoded keys made `ad_cpa` permanently 0).
- **Synthetic demo data is feature-correlated**: intent labels are generated
  from a linear feature score + sigmoid + Bernoulli noise (~8% base rate) and
  LTV from the same signal with lognormal noise — demo metrics are now
  meaningful instead of random-label artifacts.
- **Retry backoff reads settings** (`retry_min_wait`/`retry_max_wait`), capped.
- **Failed experts skip LLM evaluation** (zero score + `execution_failed`
  marker) and are excluded from refine — no wasted rounds on crashed pipelines.
- `refine_node` now applies the same expert timeout as `execute_node`.
- `CostTracker.total_cost` returns a float (`round(0, 6)` yielded `int`).
- Stale routing test expectation updated (`"execute"` → `"refine"`).

### Added

- **Memory in the main loop**: `plan_node` retrieves per-org memory
  (ChromaDB / TF-IDF fallback) and injects it into the planning prompt and
  expert params as `memory_context`.
- **`gpa eval` command**: `--router` runs a deterministic, offline router
  accuracy eval against the builtin dataset (14/14); `--full` runs the
  pipeline per sample with the multi-dimensional evaluator.
- **Deterministic report fallback**: `report_node` falls back to the template
  `ReportGenerator` (previously dead code) when LLM synthesis fails.
- Circuit-breaker / ResilientLLM test coverage (11 cases, previously none).
- `power` model tier is now actually used (report synthesis); the evaluator
  uses the resilient fast tier.
- Precision-first guardrails: zh+en attack-phrase patterns with
  false-positive regression tests ("act as", "pretend", "summarize above"
  are legitimate business phrasing and now pass).

### Removed

- Dead code: `src/graph/expert_subgraph.py` (superseded by hooks-in-run),
  `BaseAgent._build_prompt_context`, empty `tests/unit/` tree, unused
  `best_type`/`primary_result`/`rng` locals, unused chromadb import.

### Documentation

- README rewritten to match the real architecture (plan→execute→evaluate DAG,
  honest highlights each traceable to code, fallback configuration guidance,
  synthetic-data disclosure).
- CLAUDE.md updated; Makefile targets fixed (`make test` previously ran an
  empty `tests/unit/` directory; now runs the full suite via
  `python -m pytest`).

### Compatibility

- Ruff: N803/N806 ignored (sklearn `X`/`y` convention); `src/evals/dataset.py`
  exempt from E501. `make lint` now checks both `ruff check` and
  `ruff format --check` (parity with CI).

---

## [5.0.0] - 2026-04-21

### Architecture optimization release — 双轨统一、中间件激活、记忆系统

This release focuses on production-grade infrastructure: middleware stack activation,
persistent memory, structured output validation, multi-model tiering, and configuration caching.

### Added

- **Middleware stack activation**: `BaseAgent._invoke_llm()` now routes through a
  composable middleware stack (`ToolErrorHandlingMiddleware` → `RetryMiddleware` →
  `LoggingMiddleware`) for automatic retry, logging, and error normalization.
- **Persistent memory system**: `MemoryManager` with TF-IDF semantic search for
  cross-session business insight accumulation (extract → store → retrieve → inject).
- **Structured output models (Pydantic)**: `AgentResult`, `ProspectResult`,
  `SubsidyResult`, `ConversionResult`, `RetentionResult`, `AdResult`, `KpiSnapshot`,
  and `AnalysisOutput` — providing validation, serialization, and self-documenting
  contracts between agents.
- **Multi-model tiering**: `create_llm()` supports `tier` parameter mapping to
  `fast` (deepseek-chat) / `default` (gpt-4o-mini) / `power` (gpt-4o) for
  cost-quality trade-off.
- **LLM fallback**: `create_llm_with_fallback()` with automatic provider degradation.
- **Configuration caching**: `get_settings()` uses `functools.lru_cache` to avoid
  repeated environment parsing.
- **Tool registry**: `ToolRegistry` for dynamic tool discovery and registration.
- **Event stream**: `AgentEvent` dataclass with status/progress tracking, emitted
  by all workflow nodes.
- **`result_to_state_update()`** bridge function for converting typed models to
  LangGraph state-compatible dicts.

### Changed

- **State schema expanded**: `AgentState` now includes `events`, `memory_context`,
  `kpi_snapshot`, `seasonal_context` fields with proper `Annotated` reducers.
- **BaseAgent**: Constructor auto-registers `TracingHook` and `LoggingHook`;
  `_invoke_llm` uses middleware-wrapped handler instead of raw `llm.ainvoke`.
- **CLI `info` command**: Updated version display and agent registry table.
- **All agents**: Added structured `analysis` field with `LLMAnalysis` model
  containing summary + confidence + raw_response.

### Fixed

- JSON parsing robustness in `BaseAgent._parse_json_response()` — handles
  markdown-wrapped code blocks (`\`\`\`json ... \`\`\``).
- Tenacity graceful degradation when package is not installed.
- Tool import fallback stubs — agents work in demo mode even when optional
  tool packages are missing.

### Removed

- Legacy `_invoke_llm_with_retry` bypass (now handled by `RetryMiddleware`).
- Inline heuristic fallback duplication in `BaseAgent` (consolidated into tools).

---

## [4.0.0] - 2026-04-19

### LangGraph StateGraph refactor — 5 sub-agents + Orchestrator orchestration

Complete architectural overhaul from ad-hoc agent calling to a proper DAG-based
workflow using LangGraph StateGraph.

### Added

- **LangGraph StateGraph workflow** (`src/graph/workflow.py`):
  - `orchestrator_node`: scope detection (pure decision, no sub-agent calls)
  - Parallel branch: `prospect_node`, `subsidy_node`, `ad_node`
  - Sequential: `conversion_node` → `retention_node`
  - `synthesis_node`: LLM aggregation over all sub-agent results
  - `report_gen_node`: final report formatting
  - Skip nodes for conditional routing
- **OrchestratorAgent rewrite**: Split into `detect_scope()` (sync) and
  `synthesize()` (async) entry-points used by workflow nodes.
- **Hook system** (`src/core/hooks.py`): `PreRunHook`, `PostRunHook` abstract
  classes with `TracingHook`, `LoggingHook`, `MetricsHook` implementations.
- **AgentState TypedDict** (`src/core/state.py`): Typed state with `Annotated`
  reducers for `errors` and `metadata` accumulation.
- **LLM factory** (`src/core/llm_factory.py`): Multi-provider support
  (OpenAI / DeepSeek / Ollama) with environment-based configuration.
- **Settings** (`src/core/config.py`): Pydantic Settings with `GPA_` env prefix.
- **CLI interface** (`src/cli.py`): Click + Rich with `analyze`, `experiment`,
  `attribution`, `seasonal`, `kpi`, `chat`, `info` commands.

### Changed

- All agents now extend `BaseAgent` with standard `run()` → `_execute()` pattern.
- Agent results follow a consistent dict format (`{agent_name}_results` key).
- Scope detection uses keyword scoring with `_SCOPE_KEYWORDS` mapping.

### Fixed

- Sub-agent isolation: each agent handles its own errors, failures don't cascade.
- Demo mode: all agents generate synthetic data when no real data path is provided.

### Removed

- Old monolithic `main.py` entry point (replaced by `cli.py`).
- Direct agent-to-agent calling (replaced by LangGraph edges).

---

## [3.0.0] - 2026-04-16

### Ad Agent + A/B experiment platform + multi-touch attribution

### Added

- **AdAgent** (`src/agents/ad.py`): RTA strategy, OCPX bid optimization,
  creative fatigue analysis, audience segmentation with Lookalike expansion.
- **Ad tools** (`src/tools/ad/`):
  - `RTAStrategy`: Real-Time API decision rule builder with win-rate analysis.
  - `BidOptimizer`: eCPC bidding with CPA target calibration.
  - `CreativeAnalyzer`: Creative performance scoring with fatigue detection.
  - `AudienceAnalyzer`: Segment-level LTV and engagement analysis.
- **A/B experiment platform** (`src/tools/common/`):
  - Sample size calculator (power analysis).
  - Statistical tests: t-test, Mann-Whitney U.
  - ICE prioritization scoring.
- **Multi-touch attribution** (`src/tools/conversion/`):
  - First-touch / last-touch / linear / time-decay / U-shaped models.
  - Seasonal analysis for demand forecasting.
- **CLI experiment commands**: `gpa experiment design` and `gpa experiment analyze`.

### Changed

- ConversionAgent: added `Attributor` and `SeasonalAnalyzer` tool integrations.
- Orchestrator: added `ad` to full scope agent list and RTA/ad keywords.

### Fixed

- Data loading safety: path validation and file type restrictions.
- SubsidyAgent: proper binary treatment column inference from `group` or
  `subsidy_amount` columns.

### Removed

- Hardcoded experiment parameters — now calculated dynamically.

---

## [2.0.0] - 2026-04-13

### Subsidy Agent (causal inference) + Retention Agent

### Added

- **SubsidyAgent** (`src/agents/subsidy.py`): Causal inference engine (DoWhy-based
  ATE estimation), price elasticity estimator, budget optimizer (integer programming),
  subsidy allocator with personalized coupon plans.
- **Subsidy tools** (`src/tools/subsidy/`):
  - `CausalInferenceEngine`: Backdoor criterion ATE with confidence intervals.
  - `ElasticityEstimator`: Price elasticity with significance testing.
  - `BudgetOptimizer`: Constrained optimization for coupon allocation.
  - `SubsidyAllocator`: Personalized subsidy plan generation.
- **RetentionAgent** (`src/agents/retention.py`): Churn prediction model,
  nurture planning (7/14/30-day programs), win-back strategies, cohort analysis.
- **Retention tools** (`src/tools/retention/`):
  - `ChurnPredictor`: Gradient boosting churn model with risk segmentation.
  - `CohortAnalyzer`: Retention cohort matrix with inflection point detection.
  - `NurturePlanner`: Multi-stage new user nurturing program designer.
  - `WinbackPlanner`: Prioritized win-back plan by user value tier.
- **Common data loader** (`src/tools/common/data_loader.py`): Unified CSV/Parquet
  loading with sample data generation for demo mode.

### Changed

- Orchestrator: added `subsidy` and `retention` to scope routing logic.
- BaseAgent: added `_build_prompt_context()` helper for consistent state-to-prompt
  context building.

### Fixed

- Missing dependency handling: `try/except ImportError` with stub fallbacks for
  all tool imports.
- Synthetic data generation: all agents produce valid demo data with seeded RNG
  for reproducibility.

### Removed

- Prototype-only prospect analysis — now properly integrated with downstream agents.

---

## [1.0.0] - 2026-04-10

### Initial release — Agent framework + CLI + ProspectAgent

### Added

- **Project scaffolding**: `pyproject.toml`, `Makefile`, `.env.example`,
  `.gitignore`, directory structure (`src/agents/`, `src/core/`, `src/tools/`,
  `docs/`, `tests/`, `data/`, `reports/`).
- **BaseAgent abstract class** (`src/core/base.py`):
  - Standard `run()` / `_execute()` pattern.
  - `_invoke_llm()` with system + user message composition.
  - `_parse_json_response()` for LLM output parsing.
  - `_build_prompt_context()` for state-to-prompt injection.
- **ProspectAgent** (`src/agents/prospect.py`): Feature engineering, LightGBM
  intent model, user scoring + ranking, RFM segmentation, BG/NBD + Gamma-Gamma
  LTV prediction.
- **Prospect tools** (`src/tools/prospect/`):
  - `FeatureEngine`: Behavioral feature matrix builder with sample data generator.
  - `IntentModel`: LightGBM binary classifier with AUC/Accuracy metrics.
  - `UserScorer`: Weighted intent + LTV scoring with ranking.
  - `UserSegmentor`: RFM quantile segmentation.
  - `LVTPredictor`: BG/NBD + Gamma-Gamma lifetime value model.
- **ConversionAgent** (`src/agents/conversion.py`): Basic funnel analysis,
  reach planning, slot allocation, coupon design.
- **Conversion tools** (`src/tools/conversion/`):
  - `ReachPlanner`: Multi-channel reach strategy designer.
  - `FunnelAnalyzer`: Step-by-step conversion analysis with bottleneck detection.
  - `SlotAllocator`: Resource position optimization.
  - `CouponDesigner`: Segment-aware coupon design with budget constraints.
- **Report generator** (`src/report/generator.py`): Markdown report from
  `AgentState` with KPI summary.
- **CLI** (`src/cli.py`): Click-based interface with `analyze`, `info` commands.
- **Documentation**: `docs/DESIGN.md` (architecture overview).

### Changed

- N/A (initial release).

### Fixed

- N/A (initial release).

### Removed

- N/A (initial release).
