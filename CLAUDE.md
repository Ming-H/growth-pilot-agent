# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GrowthPilot Agent (`gpa`) — a multi-agent user-growth analytics platform (货运/用户增长) built on LangGraph + FastAPI. Python 3.12+, managed with uv. Project docs and prompts are largely Chinese; code/comments are English.

## Commands

```bash
uv sync --group dev        # install deps

# Tests — full suite (what CI runs; use these env vars locally too)
GPA_DEMO_MODE=true GPA_LLM_API_KEY=sk-test-placeholder DB_URL=sqlite+aiosqlite:///./test.db \
  uv run python -m pytest tests/ -v --tb=short

# Single test file / single test
uv run python -m pytest tests/test_graph/test_nodes.py -v
uv run python -m pytest "tests/test_graph/test_nodes.py::TestExecuteNode" -v

# Lint + format (CI enforces both)
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# CLI (entry point: src/cli.py, installed as `gpa`)
uv run gpa analyze --scope full --query "分析本月用户增长" --budget 500000   # agent mode default
uv run gpa analyze --mode workflow --scope full --query "..."              # fixed DAG
uv run gpa eval --router          # offline deterministic router eval (no LLM)

# Web API (FastAPI, http://localhost:8000/docs)
make web

# Docker (uses deploy/Dockerfile)
make docker-up
```

- `make test` / `make test-fast` run the full suite via `python -m pytest` (never call `pytest` directly — a broken venv shim once made it silently fall back to system Python).
- pytest-asyncio runs in **strict** mode: async tests need `@pytest.mark.asyncio`.
- Ruff: line length 100, rules `E,F,I,N,W` (E741/N803/N806 ignored — sklearn `X`/`y` convention); `src/evals/dataset.py` has an E501 exemption (long Chinese sample strings).

## Architecture

### Execution flow (two orchestration modes)

`run_workflow()` (`src/graph/workflow.py`) dispatches on `Settings.orchestration_mode` (default **agent**) and both modes return the same result contract:

**Agent mode (default, v8)** — `src/agent/graph.py::run_agent_analysis`:
```
guard → agent ⇄ tools → evaluate_gate ─┬→ agent (feedback, ≤ max_refinement_rounds)
                                      └→ approval (interrupt) → report → END
```
- `chief_agent_node` (`src/agent/chief.py`): power-tier LLM with `bind_tools`; the LLM decides which expert tools to call, in what order, and when to answer. Messages live in an `add_messages` channel (`src/agent/state.py::AgentState`).
- `agent_tools_node`: executes tool calls in parallel worker threads; experts run via `BaseAgent.run()` (full hook lifecycle). The LLM receives compact digests (kpis + summary); full validated results merge into `expert_results` via the reducer.
- Tools (`src/agent/tools.py`): `run_{prospect,conversion,subsidy,retention,ad}_analysis` + `search_memory`. Specs are (schema, state-aware handler) pairs — `to_langchain_tools` renders schemas; `execute_tool_call_async` dispatches with graph-state access.
- `evaluate_gate_node`: LLM-as-judge on the final answer; low scores inject feedback back into the loop (bounded).
- `agent_report_node`: report = last AIMessage through the output guardrail + memory write-back.
- If the LLM is unavailable the agent loop fails **cleanly** (no honest fallback in agent-led mode) and suggests `--mode workflow`.

**Workflow mode (fallback)** — `src/graph/graph.py::run_analysis`:

```
plan → execute → evaluate ─┬→ refine → evaluate        (quality loop, ≤ max_refinement_rounds)
                           ├→ approval → report → END  (human-in-the-loop, paused via interrupt)
                           └→ report → END
```
LLM autonomy is bounded to four decision points (routing / synthesis / judging / report); routing falls back to keyword matching.

- **State:** `AnalysisState` TypedDict (`src/graph/state.py`). `expert_results` uses a custom `_replace_by_expert` reducer so a refine-round result *replaces* (not appends) the same expert's earlier result; `execution_errors` uses `operator.add`. TypedDict (not Pydantic) is deliberate — LangGraph reducers need `Annotated` support.
- **Nodes:** pure functions in `src/graph/nodes.py` returning partial state updates. Routers (`_route_after_*`) live in `graph.py`.
- **plan_node** (Chief Agent): input guardrail → memory retrieval (per-org, best-effort) → LLM expert selection with keyword `_intent_classify` fallback. If none selected → END.
- **execute_node** runs selected experts concurrently, each with `asyncio.timeout`, invoked through **`BaseAgent.run()`** — the full hook lifecycle (tracing/logging/metrics) and middleware stack execute on this path. Results are Pydantic-validated at the node boundary (`_validate_expert_result`; hook metadata under `_`-prefixed keys is stripped before validation and re-attached).
- **evaluate_node** scores non-failed experts via LLM-as-judge (`src/core/evaluator.py`, fast tier, resilient LLM). Failed experts get a zero score with `execution_failed: True` and are excluded from refine.
- **refine_node** re-runs below-threshold experts with evaluator feedback, also through `run()` with timeout.
- **approval_node** auto-approves below-threshold budgets; otherwise the graph interrupts (checkpoint-persisted) until a human resumes with the same `thread_id`.
- **report_node** synthesizes with the `power` tier; on LLM failure falls back to the deterministic `ReportGenerator` (`src/report/generator.py`); output passes the output guardrail.

### Three-tier agent hierarchy

1. **Chief (plan node)** — LLM routing/synthesis only.
2. **Experts** (`src/agents/{prospect,conversion,subsidy,retention,ad}.py`) — subclasses of `ExpertAgentBase` (`src/core/expert.py`). Each implements `_init_tools()`, `_execute_pipeline()` (deterministic, sync, offloaded via `asyncio.to_thread`), `_build_synthesis_prompt()`. `analyze()` sets an explicit `success` contract and every result carries a standardised `kpis` dict (merged by `workflow.py` into `kpi_snapshot`).
3. **Tools** (`src/tools/{<domain>}/`) — deterministic ML: LightGBM intent/churn, DoWhy causal ATE/CATE, PuLP ILP budget optimization, BG/NBD LTV. Registered via `@register` into `ToolRegistry` (`src/tools/registry.py`).

Design rule: **tools compute, LLM interprets.** Demo-mode synthetic data is *feature-correlated* (linear signal + sigmoid + Bernoulli noise for labels; lognormal for LTV) so metrics are meaningful — keep it that way.

### Resilience & LLM infrastructure

- `ResilientLLM` (`src/core/llm_factory.py`) wraps primary + fallback with a **process-shared** CircuitBreaker registry keyed by `(provider, model, base_url)` — all call sites for one endpoint share one circuit. Self-fallback (same endpoint as primary) is detected and disabled.
- Model tiers `fast`/`default`/`power` resolve from `Settings.model_tiers`; evaluator=fast, plan/experts=default, report=power.
- Retry backoff reads `Settings.retry_min_wait/retry_max_wait`.
- `CostTracker` (`src/core/token_tracker.py`) is a process-global singleton reset at the start of every `run_analysis()` — per-run cost attribution.

### Supporting subsystems

- **Guardrails** (`src/guardrails/`): precision-first injection detection (explicit attack phrases, zh+en; "act as"/"pretend"/"summarize above" deliberately NOT blocked) + output guard.
- **Memory** (`src/memory/manager.py`): per-org persistent memory — ChromaDB vector store with TF-IDF fallback; retrieved in `plan_node` and injected into expert params as `memory_context`.
- **DB/auth** (`src/db/`, `src/auth/`): async SQLAlchemy + PostgreSQL, multi-tenant by `org_id`, JWT auth. Schema is created via `init_db()`; Alembic is wired for production migrations (`alembic/versions/` currently empty).
- **Web API** (`src/web.py`): JWT auth, sync `/analyze`, SSE streaming, rate limiting, analysis persistence, approval/resume endpoints.
- **Evals** (`src/evals/`): builtin dataset + evaluator; exposed via `gpa eval --router` (deterministic) and `gpa eval --full` (LLM pipeline).
- **data_path**: `run_workflow(data_path=...)` flows through state into expert params; prospect and retention experts load real CSV/Parquet, others generate correlated synthetic data.
