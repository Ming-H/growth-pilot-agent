# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GrowthPilot Agent (`gpa`) — a **reconciliation-gated multi-agent user growth platform** (v9) built on LangGraph + FastAPI. Python 3.12+, managed with uv. Docs and prompts are largely Chinese; code/comments are English.

**分工宪法**: deterministic tools compute facts (KPIs); the LLM only organises language; the KPI reconciliation gate (`src/core/kpi_gate.py`) owns "can this leave the building". Design doc: `docs/DESIGN-v9.md`.

## Commands

```bash
uv sync --group dev        # install deps

# Tests — full suite (what CI runs; use these env vars locally too)
GPA_DEMO_MODE=true GPA_LLM_API_KEY=sk-test-placeholder GPA_DB_URL=sqlite+aiosqlite:///./test.db \
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
uv run gpa analyze -v ...    # diagnostic tier: node trace + degradations (stderr)
uv run gpa analyze --json .. # machine tier: full state JSON, honest exit code
uv run gpa eval --router         # deterministic router eval (no LLM)
uv run gpa eval --heuristic      # zero-key regression baseline (CI-safe)

# Web API (FastAPI, http://localhost:8000/docs)
make web

# Docker (uses deploy/Dockerfile)
make docker-up
```

- `make test` / `make test-fast` run the full suite via `python -m pytest`.
- pytest-asyncio runs in **strict** mode: async tests need `@pytest.mark.asyncio`, async fixtures need `@pytest_asyncio.fixture`.
- conftest pins `OMP_NUM_THREADS=1` etc. BEFORE scientific-stack imports — required by the byte-identical determinism invariant. Do not move those lines.
- Ruff: line length 100, rules `E,F,I,N,W` (E741/N803/N806 ignored — sklearn `X`/`y` convention); `src/evals/dataset.py` has an E501 exemption.

## Architecture

### LLM dual-track channel (v9)

`Settings.llm_base_url` is THE channel switch, auto-detected by host (`src/core/config.py: detect_channel`):
- **local** (localhost/127.0.0.1/::1, zero key): Ollama native (`langchain-ollama`) when the URL is Ollama-shaped (:11434 or path contains `/api`), else OpenAI-compatible with `/v1` appended (vLLM/SGLang). `llm_compat_mode` forces the latter.
- **remote** (needs key): any OpenAI-compatible endpoint (Zhipu GLM `paas/v4`/`coding/paas/v4`, DeepSeek, …).

Roles (fast/default/power) are temperature presets + optional per-role model overrides on the ONE channel (`role_temperatures`/`role_models`). Fallback is remote→remote only. **`create_llm` returns `None` when unavailable — the degradation sentinel** driving every deterministic twin below. `tier=` is a legacy alias for `role=`.

### Execution flow (two modes, one reconciliation gate)

`run_workflow()` (`src/graph/workflow.py`) dispatches on mode; both return the compat contract (success/report/kpi_snapshot/degradations/trace/agent_steps/kpi_gate_result/metadata.cost_usd):

**agent mode (default)** — `src/agent/graph.py`:
```
guard → plan → agent ⇄ tools → evaluate ─┬→ agent (feedback, bounded by max_refinement_rounds, `<` comparison)
                                         └→ approval? → report → kpi_gate → END
```
- `agent_plan_node` (chief.py): LLM (role=fast) emits `{subgoals}` → `_validate_subgoals` **only deletes, never creates** (whitelist/dedupe/cap-5). Zero-key → `_intent_classify` prior, marker `plan:rule`. Subgoal checklist (☐/☑/☒) rides in the chief system prompt.
- `chief_agent_node`: LLM brain, or **heuristic brain** when `llm is None` (synthesises tool_calls for pending subgoals — same graph). Step cap `agent_max_steps` (marker `agent_loop:capped`); local-channel models producing no tool calls flip to the heuristic brain (`chief_tools:weak`).
- `agent_tools_node`: parallel expert execution via `BaseAgent.run()`, field-wise digest truncation, subgoal status updates, kpi_snapshot refresh, step quadruples.
- `_route_after_gate` routes to `approval` **only when it will actually pause** (approval_required ∧ budget ≥ threshold) — the graph compiles with `interrupt_before=["approval"]`, so routing there unconditionally pauses EVERY run before its report.
- `kpi_gate_node` is the final node in BOTH modes: reconciliation + post-gate memory write-back.

**workflow mode (fallback)** — `src/graph/graph.py`: fixed `plan → execute → evaluate ⇄ refine → approval? → report → kpi_gate`.

### KPI reconciliation gate (`src/core/kpi_gate.py`)

Pure regex + arithmetic. Extracts numeric claims near KPI-lexicon words; each must match a value in `kpi_snapshot` ∪ L1 ledger with display-precision tolerance (`8%` ≡ `0.08`). Modes: `on` (degrade to deterministic template + appendix), `annotate`, `off`; `kpi_gate_threshold` dials tolerated fraction. Budget and churn high-risk count are legitimate KPIs — they join the snapshot rather than loosening the gate.

### Three-layer memory (`src/memory/store.py`, SQLite + FTS5)

- **L0 sessions**: last-N turns verbatim + older turns as structured extracts — never inherit free text as truth.
- **L1 verified_facts**: write gate = `success ∧ _validated ∧ kpis 非空`; numbers only; read never bypasses the gate (period uses `''` not NULL — SQLite UNIQUE treats NULLs as distinct).
- **L2 aliases**: org vocabulary steering `_intent_classify` (PUT /api/v1/memory/aliases).
- Episodic search: FTS5 (jieba pre-tokenized) + per-term LIKE fallback, RRF(k=60) + recency boost. All aiosqlite, single-writer lock. Legacy v8 `memory_store.json` auto-migrates.

### Experts & tools

5 experts (`src/agents/*`) subclass `ExpertAgentBase`: `_execute_pipeline` (deterministic, threadpool_limits(1)-wrapped) + `_build_synthesis_prompt`. `llm=None` → deterministic `k=v` summary with `degraded` marker — **absence of a brain is a mode, not a failure** (success stays True). ad expert honestly marks `data_source: "synthetic"` (ignores data_path). Import guards catch `(ImportError, OSError)` — broken native libs degrade, never crash the router.

Agent tool specs (`src/agent/tools.py`) are ACI-style: purpose/when-to-call/args/**failure-modes**.

### Web package (`src/web/`)

`app.py` (factory/lifespan/handlers) + `deps.py` (MemoryStore, compiled-graph cache via `src/graph/runtime.py`, `thread_id_for(org, analysis_id)`) + `routers/{auth,analyses,memory}.py`. HITL discipline: runs, `/state`, `/approve`, `/resume` all use `thread_id_for`; the Analysis row persists its `mode` so resume fetches the same graph topology. `run_workflow` threads org_id/user_id/approval_required/thread_id/session_id (multi-tenancy).

### Cost tracking

`cost_scope()` ContextVar per run (`src/core/token_tracker.py`); `asyncio.to_thread` propagates. Records outside a scope are dropped, never mis-attributed.

## Invariants (do not break)

1. **Determinism**: same input + zero-key mode ⇒ byte-identical report (`tests/test_graph/test_zero_key.py`, `gpa eval --heuristic`). Requires: single-threaded numeric env in conftest, LightGBM deterministic params, fixed synthetic-data time anchors, unique per-run thread_ids.
2. **Honesty**: every degradation visible in `state["degradations"]`; `--json` exit code reflects failure/gate-degradation; `llm_available` never faked (demo_mode does not fake a keyless remote channel).
3. **只删不造**: `_validate_subgoals` never adds experts; the KPI gate never loosens for legit numbers — the numbers join the ground truth instead.
