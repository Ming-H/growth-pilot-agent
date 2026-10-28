"""GrowthPilot Agent CLI - Click + Rich command line interface."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.core import __version__
from src.core.config import get_settings

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync Click callbacks."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop (e.g. Jupyter) - create a new thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _print_errors(errors: list[str]) -> None:
    """Print errors as Rich panels."""
    if not errors:
        return
    console.print(
        Panel("\n".join(f"- {e}" for e in errors), title="Warnings", border_style="yellow")
    )


def _print_report(report: str) -> None:
    """Print the markdown report with Rich formatting."""
    console.print(Panel(report, title="GrowthPilot Report", border_style="blue", expand=False))


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def main(verbose: bool) -> None:
    """GrowthPilot Agent - Freight user growth multi-agent system."""
    level = "DEBUG" if verbose else "INFO"
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# analyze command
# ---------------------------------------------------------------------------


@main.command()
@click.option("--data", "-d", default="", help="Path to data file or directory.")
@click.option(
    "--scope",
    "-s",
    default="full",
    help="Analysis scope (full|prospect|conversion|subsidy|retention|ad).",
)
@click.option("--budget", "-b", default=0.0, type=float, help="Available budget.")
@click.option("--query", "-q", default="", help="Natural language query.")
@click.option("--output", "-o", default="", help="Output file path for the report.")
@click.option(
    "--mode",
    "-m",
    "mode",
    default="",
    type=click.Choice(["", "agent", "workflow"]),
    help="Orchestration mode: agent (tool-calling loop, default) or workflow (fixed DAG).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Diagnostic tier: per-node trace with timing + degradation markers to stderr.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Machine tier: full final state JSON on stdout (exit 1 on failure/gate-degraded).",
)
def analyze(
    data: str,
    scope: str,
    budget: float,
    query: str,
    output: str,
    mode: str,
    verbose: bool,
    as_json: bool,
) -> None:
    """Run the growth analysis pipeline (agent-led by default).

    Observability tiers: product (silent on success) / -v diagnostic /
    --json machine state.
    """
    _run_async(_analyze(data, scope, budget, query, output, mode, verbose, as_json))


def _print_diagnostic_tier(result: dict[str, Any]) -> None:
    """-v: per-node trace table, degradation markers, agent step quadruples → stderr."""
    import sys

    trace = result.get("trace") or []
    if trace:
        table = Table(title="Node Trace", show_header=True, header_style="bold magenta")
        table.add_column("Node", style="cyan")
        table.add_column("ms", style="green", justify="right")
        table.add_column("Detail", style="white")
        for t in trace:
            table.add_row(
                str(t.get("node", "")), str(t.get("dur_ms", "")), str(t.get("detail", ""))[:80]
            )
        consoleerr = Console(file=sys.stderr)
        consoleerr.print(table)
    degradations = result.get("degradations") or []
    if degradations:
        consoleerr = Console(file=sys.stderr)
        consoleerr.print(f"[yellow]degradations: {', '.join(degradations)}[/yellow]")
    steps = result.get("agent_steps") or []
    if steps:
        consoleerr = Console(file=sys.stderr)
        for i, s in enumerate(steps, 1):
            thought = str(s.get("thought", ""))[:60].replace("\n", " ")
            consoleerr.print(
                f'agent[{i}] thought="{thought}" action={s.get("action")} '
                f"obs={str(s.get('obs'))[:60]} gate={s.get('gate')}"
            )


async def _analyze(
    data: str,
    scope: str,
    budget: float,
    query: str,
    output: str,
    mode: str = "",
    verbose: bool = False,
    as_json: bool = False,
) -> None:
    import sys

    from src.graph.workflow import run_workflow

    result = await run_workflow(
        query=query or f"分析范围: {scope}",
        data_path=data,
        budget=budget,
        scope=scope,
        mode=mode,
    )

    # ── Machine tier: full state JSON on stdout; honest exit code ──────
    if as_json:
        gate = result.get("kpi_gate_result") or {}
        degraded = "kpi_gate:degraded" in (result.get("degradations") or [])
        payload = {
            "success": result.get("success"),
            "query": result.get("query"),
            "scope": result.get("scope"),
            "report": result.get("report"),
            "kpi_snapshot": result.get("kpi_snapshot"),
            "degradations": result.get("degradations", []),
            "trace": result.get("trace", []),
            "agent_steps": result.get("agent_steps", []),
            "kpi_gate": gate,
            "errors": result.get("errors", []),
            "metadata": result.get("metadata", {}),
        }
        click.echo(json.dumps(payload, ensure_ascii=False, default=str, indent=2))
        if not result.get("success") or degraded:
            sys.exit(1)
        return

    # ── Diagnostic tier (-v): trace/markers to stderr before the report ──
    if verbose:
        _print_diagnostic_tier(result)

    # ── Product tier: report + KPI table, silent on success ────────────
    report = result.get("report", "")
    if report:
        _print_report(report)
    else:
        console.print("[yellow]No report generated.[/yellow]")

    errors = result.get("errors", [])
    _print_errors(errors)

    kpi = result.get("kpi_snapshot", {})
    if kpi:
        _print_kpi_table(kpi)

    if output:
        out_path = Path(output)
        out_path.write_text(report, encoding="utf-8")
        console.print(f"\n[green]Report saved to {out_path}[/green]")


def _print_kpi_table(kpi: dict[str, Any]) -> None:
    """Print KPI snapshot as a Rich table."""
    # Multipliers/counts render as plain numbers; ratios (0-1) as percentages.
    non_ratio_keys = {"expected_roi", "total_users", "ad_impressions", "ad_target_cpa"}

    table = Table(title="KPI Snapshot", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in kpi.items():
        if isinstance(value, float):
            if key in non_ratio_keys:
                value_str = f"{value:,.2f}"
            else:
                value_str = f"{value:.2%}" if 0 <= value <= 1 else f"{value:,.2f}"
        else:
            value_str = str(value)
        table.add_row(key, value_str)

    console.print(table)


# ---------------------------------------------------------------------------
# kpi command
# ---------------------------------------------------------------------------


@main.command()
def kpi() -> None:
    """Show current KPI snapshot."""
    # Sample KPI data for demo mode
    sample_kpi = {
        "total_users": "1,250,000",
        "monthly_active_users": "450,000",
        "new_users_this_month": "32,000",
        "conversion_rate": "8.5%",
        "average_order_value": "¥2,800",
        "customer_lifetime_value": "¥15,600",
        "churn_rate_30d": "12.3%",
        "retention_rate_30d": "87.7%",
        "nps_score": "42",
        "ad_spend": "¥580,000",
        "cac": "¥186",
        "roi": "3.2x",
    }

    table = Table(title="Current KPI Snapshot", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="green", width=20)

    for metric, value in sample_kpi.items():
        table.add_row(metric.replace("_", " ").title(), value)

    console.print(table)

    console.print(
        Panel(
            "[dim]Note: Showing sample KPI data. "
            "Connect to live data source for real metrics.[/dim]",
            border_style="dim",
        )
    )


# ---------------------------------------------------------------------------
# chat command
# ---------------------------------------------------------------------------


@main.command()
@click.option("--scope", "-s", default="full", help="Default analysis scope.")
def chat(scope: str, session_id: str = "", org_id: str = "default") -> None:
    """Interactive chat with L0 session memory.

    Recent turns are inherited verbatim; older turns arrive as structured
    extracts (query head + KPIs) — never free text as truth. The session
    persists in the memory store and survives restarts.
    """
    import uuid as _uuid

    if not session_id:
        # Reuse the most recent session for this org, else start a new one.
        try:
            from src.core.config import get_settings as _gs
            from src.memory import MemoryStore

            _store = MemoryStore(db_path=_gs().memory_db_path)
            session_id = f"chat-{_uuid.uuid4().hex[:8]}"
            _run_async(_store.close())
        except Exception:
            session_id = f"chat-{_uuid.uuid4().hex[:8]}"

    console.print(
        Panel(
            "[bold]GrowthPilot Interactive Mode[/bold]\n\n"
            f"Default scope: {scope}\n"
            f"Session: {session_id} (L0 memory on)\n"
            "Type your query and press Enter.\n"
            "/new — start a fresh session   /facts <词> — query the L1 fact ledger\n"
            "scope <name> — change scope    kpi — KPI snapshot\n"
            "quit / exit — leave",
            title="Welcome",
            border_style="green",
        )
    )

    current_scope = scope
    _chat_state: dict[str, Any] = {"session_id": session_id, "org_id": org_id}

    while True:
        try:
            user_input = console.input("[bold cyan]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        if user_input.lower() == "kpi":
            # Reuse kpi command logic
            _run_async(_kpi_inline())
            continue

        if user_input.lower() == "/new":
            _chat_state["session_id"] = f"chat-{_uuid.uuid4().hex[:8]}"
            console.print(f"[green]New session: {_chat_state['session_id']}[/green]")
            continue

        if user_input.lower().startswith("/facts"):
            _run_async(_chat_facts(user_input[6:].strip(), _chat_state["org_id"]))
            continue

        if user_input.lower().startswith("scope "):
            new_scope = user_input.split(" ", 1)[1].strip()
            valid_scopes = {"full", "inapp", "prospect", "conversion", "subsidy", "retention", "ad"}
            if new_scope in valid_scopes:
                current_scope = new_scope
                console.print(f"[green]Scope changed to: {current_scope}[/green]")
            else:
                console.print(
                    f"[red]Invalid scope. Valid options: {', '.join(sorted(valid_scopes))}[/red]"
                )
            continue

        if user_input.lower() == "help":
            console.print(
                Panel(
                    "Available commands:\n"
                    "- Any natural language query: runs analysis\n"
                    "- 'scope <name>': change analysis scope\n"
                    "- 'kpi': show current KPI snapshot\n"
                    "- 'quit' / 'exit': exit interactive mode\n"
                    "- 'help': show this message",
                    title="Help",
                    border_style="blue",
                )
            )
            continue

        # Run analysis with L0 session continuity
        console.print(f"[dim]Running analysis with scope={current_scope}...[/dim]")
        _run_async(_chat_analyze(user_input, current_scope, _chat_state))


async def _chat_analyze(query: str, scope: str, chat_state: dict[str, Any]) -> None:
    """Run one chat turn with L0 memory: save the user turn, inject the
    session context into the query, run the analysis (the graph writes the
    assistant turn + gated report at the end)."""
    from src.core.config import get_settings
    from src.graph.workflow import run_workflow
    from src.memory import MemoryStore

    try:
        store = MemoryStore(db_path=get_settings().memory_db_path)
        await store.save_turn(
            chat_state["session_id"],
            org_id=chat_state["org_id"],
            role="user",
            content=query,
        )
        loaded = await store.load_session(chat_state["session_id"], last_n_full=2)
        await store.close()

        # Earlier-turn context rides along as a structured prefix — the
        # router/experts see prior scope, the gate still owns the numbers.
        enriched = query
        if loaded["compressed"]:
            enriched = f"{query}\n\n[会话上下文（结构化摘要）]\n{loaded['compressed'][:600]}"

        result = await run_workflow(
            query=enriched,
            scope=scope,
            org_id=chat_state["org_id"],
            session_id=chat_state["session_id"],
        )
        report = result.get("report", "")
        if report:
            _print_report(report)
        errors = result.get("errors", [])
        _print_errors(errors)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")


async def _chat_facts(term: str, org_id: str) -> None:
    """/facts: read the L1 verified-fact ledger (write-gated upstream)."""
    from src.core.config import get_settings
    from src.memory import MemoryStore

    try:
        store = MemoryStore(db_path=get_settings().memory_db_path)
        facts = await store.search_facts(org_id, metric_key=term or None, limit=20)
        await store.close()
        if not facts:
            console.print("[yellow]L1 台账暂无匹配事实[/yellow]")
            return
        table = Table(title=f"L1 Verified Facts ({len(facts)})", show_header=True)
        table.add_column("Domain", style="cyan")
        table.add_column("Metric", style="green")
        table.add_column("Value", justify="right")
        for f in facts:
            table.add_row(f["domain"], f["metric_key"], f"{f['value']:,.4f}")
        console.print(table)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")


async def _kpi_inline() -> None:
    """Print KPI in chat mode."""
    # Same as kpi command but inline
    sample_kpi = {
        "total_users": "1,250,000",
        "conversion_rate": "8.5%",
        "churn_rate_30d": "12.3%",
        "roi": "3.2x",
    }
    table = Table(title="KPI Snapshot", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for m, v in sample_kpi.items():
        table.add_row(m.replace("_", " ").title(), v)
    console.print(table)


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------


@main.command()
def info() -> None:
    """Show system information."""
    settings = get_settings()

    table = Table(title="GrowthPilot System Info", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    info_items = [
        ("Version", __version__),
        ("Orchestration Mode", settings.orchestration_mode),
        ("LLM Channel", settings.channel),
        ("LLM Base URL", settings.llm_base_url),
        ("LLM Model", settings.llm_model),
        ("LLM Available", "Yes" if settings.llm_available else "No (deterministic degradation)"),
        ("API Key Set", "Yes" if settings.llm_api_key else "No"),
        (
            "Fallback",
            f"{settings.fallback_model} @ {settings.fallback_base_url}"
            if settings.fallback_base_url and settings.fallback_model
            else "(none)",
        ),
        ("KPI Gate", settings.kpi_gate),
        ("Demo Mode", str(settings.demo_mode)),
        ("Log Level", settings.log_level),
        ("Data Directory", str(settings.data_dir)),
        ("Output Directory", str(settings.output_dir)),
        ("Python", sys.version.split()[0]),
    ]

    for prop, value in info_items:
        table.add_row(prop, value)

    console.print(table)

    # Agent registry
    agent_table = Table(title="Available Agents", show_header=True, header_style="bold magenta")
    agent_table.add_column("Agent", style="cyan")
    agent_table.add_column("Scope", style="green")
    agent_table.add_column("Description", style="white")

    agents = [
        ("chief (GrowthPilot)", "all", "主控 Agent — 动态编排专家团队"),
        ("prospect_analysis", "prospect", "潜客识别、评分、分群、LTV预测"),
        ("conversion_analysis", "conversion", "漏斗分析、优惠券、触达策略"),
        ("subsidy_analysis", "subsidy", "因果推断、弹性、预算优化"),
        ("retention_analysis", "retention", "流失预测、培育、召回策略"),
        ("ad_analysis", "ad", "RTA策略、出价优化、创意分析"),
    ]
    for name, scope, desc in agents:
        agent_table.add_row(name, scope, desc)

    console.print(agent_table)

    if not settings.llm_api_key:
        console.print(
            Panel(
                "[yellow]No API key configured. Running in demo/mock mode.[/yellow]\n\n"
                "To enable full functionality:\n"
                "1. Set GPA_LLM_API_KEY in .env file\n"
                "2. Or export GPA_LLM_API_KEY=your-key",
                title="Demo Mode",
                border_style="yellow",
            )
        )


# ---------------------------------------------------------------------------
# eval command (offline quality evaluation)
# ---------------------------------------------------------------------------


@main.command("eval")
@click.option(
    "--router",
    "mode",
    flag_value="router",
    default=True,
    help="Offline deterministic router evaluation (no LLM, CI-safe).",
)
@click.option(
    "--heuristic",
    "mode",
    flag_value="heuristic",
    help="Zero-key regression baseline: full pipeline with the deterministic"
    " brain, asserting success/routing/KPIs/byte-identical reports.",
)
@click.option(
    "--full",
    "mode",
    flag_value="full",
    help="Full pipeline evaluation per sample (requires LLM API key).",
)
@click.option("--limit", "-l", default=0, type=int, help="Max samples (0 = all).")
@click.option("--output", "-o", default="", help="Write the markdown report to a file.")
def eval_cli(mode: str, limit: int, output: str) -> None:
    """Evaluate routing / pipeline quality against the built-in eval dataset."""
    if mode == "router":
        _run_router_eval(limit, output)
    elif mode == "heuristic":
        _run_async(_run_heuristic_eval(limit, output))
    else:
        _run_async(_run_full_eval(limit, output))


async def _run_heuristic_eval(limit: int, output: str) -> None:
    """Zero-key regression baseline — deterministic, offline, CI-safe.

    Forces the heuristic brain (no LLM configured) over the builtin
    dataset in workflow mode and asserts per sample: success, routing
    hit, non-empty KPI snapshot, non-empty report, and a byte-identical
    report across two runs (the determinism invariant).
    """
    import os

    os.environ["GPA_LLM_CHANNEL"] = "remote"
    os.environ["GPA_LLM_API_KEY"] = ""
    get_settings.cache_clear()
    from src.core.config import get_settings as _gs

    assert not _gs().llm_available, "heuristic eval requires no LLM"

    from src.evals.dataset import EvalDataset
    from src.graph import nodes as graph_nodes
    from src.graph.workflow import run_workflow

    dataset = EvalDataset.from_builtin()
    samples = dataset.samples[: limit or None]

    table = Table(title="Heuristic Regression Baseline (zero-key, deterministic)")
    table.add_column("Sample", style="cyan", max_width=32)
    table.add_column("Success", justify="center")
    table.add_column("Route", justify="center")
    table.add_column("KPIs", justify="center")
    table.add_column("Report", justify="center")
    table.add_column("Deterministic", justify="center")

    passed = 0
    for s in samples:
        common = dict(query=s.input_query, scope="", budget=0, mode="workflow")
        graph_nodes._clear_expert_cache()
        r1 = await run_workflow(**common)
        graph_nodes._clear_expert_cache()
        r2 = await run_workflow(**common)

        ok_success = r1.get("success") is True
        expected = s.scope if s.scope != "full" else None
        experts = list((r1.get("expert_results") or {}).keys())
        ok_route = (expected in experts) if expected else bool(experts)
        ok_kpis = bool(r1.get("kpi_snapshot"))
        ok_report = bool(r1.get("report"))
        ok_det = r1.get("report") == r2.get("report")
        all_ok = ok_success and ok_route and ok_kpis and ok_report and ok_det
        passed += all_ok

        def _mark(ok: bool) -> str:
            return "[green]✓[/green]" if ok else "[red]✗[/red]"

        table.add_row(
            s.id or s.input_query[:28],
            _mark(ok_success),
            _mark(ok_route),
            _mark(ok_kpis),
            _mark(ok_report),
            _mark(ok_det),
        )

    console.print(table)
    rate = passed / len(samples) if samples else 0.0
    console.print(f"Heuristic baseline: [bold]{rate:.0%}[/bold] ({passed}/{len(samples)})")
    if output:
        Path(output).write_text(
            f"# Heuristic Regression Baseline\n\nPass: {rate:.0%} ({passed}/{len(samples)})\n",
            encoding="utf-8",
        )


def _run_router_eval(limit: int, output: str) -> None:
    """Deterministic router eval: intent-classifier hits vs expected scope."""
    from src.evals.dataset import EvalDataset
    from src.graph.nodes import _intent_classify

    dataset = EvalDataset.from_builtin()
    samples = dataset.samples[: limit or None]

    table = Table(title="Router Evaluation (deterministic, no LLM)", show_header=True)
    table.add_column("Sample", style="cyan", max_width=32)
    table.add_column("Expected", style="green")
    table.add_column("Selected", style="white")
    table.add_column("Hit", justify="center")

    hits = 0
    for s in samples:
        selected = _intent_classify(s.input_query)
        expected = s.scope if s.scope != "full" else None
        if expected is None:
            ok = len(selected) > 0  # full-scope samples: any non-empty routing
        else:
            ok = expected in selected
        hits += ok
        table.add_row(
            s.id or s.input_query[:28],
            s.scope,
            ",".join(selected),
            "[green]✓[/green]" if ok else "[red]✗[/red]",
        )

    rate = hits / len(samples) if samples else 0.0
    console.print(table)
    console.print(f"Router accuracy: [bold]{rate:.0%}[/bold] ({hits}/{len(samples)})")

    if output:
        Path(output).write_text(
            f"# Router Evaluation\n\nAccuracy: {rate:.0%} ({hits}/{len(samples)})\n",
            encoding="utf-8",
        )
        console.print(f"[green]Saved to {output}[/green]")


async def _run_full_eval(limit: int, output: str) -> None:
    """Full pipeline eval: run the graph per sample, score expected keys + judge."""
    import time as _time

    from src.evals.dataset import EvalDataset
    from src.evals.evaluator import AgentEvaluator, EvalResult
    from src.evals.report import EvalReport
    from src.graph.workflow import run_workflow

    dataset = EvalDataset.from_builtin()
    samples = dataset.samples[: limit or None]
    evaluator = AgentEvaluator()

    results: list[EvalResult] = []
    for s in samples:
        start = _time.perf_counter()
        state = await run_workflow(query=s.input_query, scope="", budget=0)
        elapsed = _time.perf_counter() - start

        expert_results = state.get("expert_results", {})
        actual = expert_results.get(s.agent_name, {}) if isinstance(expert_results, dict) else {}
        result = await evaluator.evaluate_agent_run(
            agent_name=s.agent_name,
            input_query=s.input_query,
            actual_output=actual,
            expected_output=s.reference_answer,
            expected_keys=s.expected_keys,
            latency_seconds=elapsed,
        )
        results.append(result)
        console.print(f"[cyan]{s.id}[/cyan] overall={result.overall_score:.2f}")

    report = EvalReport(results)
    markdown = report.to_markdown()
    console.print(Panel(markdown[:1500], title="Evaluation Report"))

    out_path = Path(output) if output else Path("reports/eval_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    console.print(f"[green]Full report saved to {out_path}[/green]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
