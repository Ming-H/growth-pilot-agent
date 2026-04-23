# Changelog

All notable changes to the GrowthPilot Agent project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
