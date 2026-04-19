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
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

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
    console.print(Panel("\n".join(f"- {e}" for e in errors), title="Warnings", border_style="yellow"))


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
@click.option("--scope", "-s", default="full", help="Analysis scope (full|prospect|conversion|subsidy|retention|ad|inapp).")
@click.option("--budget", "-b", default=0.0, type=float, help="Available budget.")
@click.option("--query", "-q", default="", help="Natural language query.")
@click.option("--output", "-o", default="", help="Output file path for the report.")
def analyze(data: str, scope: str, budget: float, query: str, output: str) -> None:
    """Run the full growth analysis pipeline."""
    _run_async(_analyze(data, scope, budget, query, output))


async def _analyze(data: str, scope: str, budget: float, query: str, output: str) -> None:
    from src.graph.workflow import run_workflow

    console.print(Panel(
        f"[bold]GrowthPilot Analysis[/bold]\n\n"
        f"Scope: {scope}\n"
        f"Data: {data or '(none)'}\n"
        f"Budget: {budget or '(none)'}\n"
        f"Query: {query or '(none)'}",
        title="Configuration",
        border_style="green",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running workflow...", total=None)

        result = await run_workflow(
            query=query or f"分析范围: {scope}",
            data_path=data,
            budget=budget,
            scope=scope,
        )
        progress.update(task, completed=True)

    # Print report
    report = result.get("report", "")
    if report:
        _print_report(report)
    else:
        console.print("[yellow]No report generated.[/yellow]")

    # Print errors
    errors = result.get("errors", [])
    _print_errors(errors)

    # Print KPI snapshot
    kpi = result.get("kpi_snapshot", {})
    if kpi:
        _print_kpi_table(kpi)

    # Save to file if requested
    if output:
        out_path = Path(output)
        out_path.write_text(report, encoding="utf-8")
        console.print(f"\n[green]Report saved to {out_path}[/green]")


def _print_kpi_table(kpi: dict[str, Any]) -> None:
    """Print KPI snapshot as a Rich table."""
    table = Table(title="KPI Snapshot", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in kpi.items():
        if isinstance(value, float):
            value_str = f"{value:.2%}" if value < 1 else f"{value:.2f}"
        else:
            value_str = str(value)
        table.add_row(key, value_str)

    console.print(table)


# ---------------------------------------------------------------------------
# experiment command
# ---------------------------------------------------------------------------


@main.group()
def experiment() -> None:
    """Experiment design and analysis commands."""
    pass


@experiment.command("design")
@click.option("--data", "-d", default="", help="Path to experiment data.")
@click.option("--metric", "-m", default="conversion_rate", help="Primary metric to optimize.")
@click.option("--budget", "-b", default=0.0, type=float, help="Experiment budget.")
def experiment_design(data: str, metric: str, budget: float) -> None:
    """Design an A/B test experiment."""
    console.print(Panel(
        f"[bold]Experiment Design[/bold]\n\n"
        f"Data: {data or '(none)'}\n"
        f"Primary metric: {metric}\n"
        f"Budget: {budget}",
        border_style="blue",
    ))

    # Generate a sample experiment design
    design = _generate_experiment_design(metric, budget)
    console.print_json(json.dumps(design, indent=2, ensure_ascii=False))


def _generate_experiment_design(metric: str, budget: float) -> dict[str, Any]:
    """Generate a sample experiment design (placeholder)."""
    return {
        "experiment_name": f"{metric}_optimization_test",
        "hypothesis": f"Optimizing {metric} will improve overall conversion",
        "primary_metric": metric,
        "variants": [
            {"name": "control", "description": "Current strategy"},
            {"name": "treatment_a", "description": "Optimized targeting"},
            {"name": "treatment_b", "description": "Enhanced incentive"},
        ],
        "sample_size": 10000,
        "duration_days": 14,
        "significance_level": 0.05,
        "power": 0.8,
        "budget": budget,
        "status": "designed",
    }


@experiment.command("analyze")
@click.option("--data", "-d", default="", help="Path to experiment results data.")
@click.option("--metric", "-m", default="conversion_rate", help="Primary metric.")
def experiment_analyze(data: str, metric: str) -> None:
    """Analyze experiment results."""
    console.print(Panel(
        f"[bold]Experiment Analysis[/bold]\n\n"
        f"Data: {data or '(none)'}\n"
        f"Primary metric: {metric}",
        border_style="blue",
    ))

    # Placeholder analysis result
    result = {
        "experiment": metric,
        "winner": "treatment_a",
        "lift": 0.12,
        "p_value": 0.003,
        "confidence": 0.95,
        "recommendation": "Roll out treatment_a as the new default",
    }
    console.print_json(json.dumps(result, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# attribution command
# ---------------------------------------------------------------------------


@main.command()
@click.option("--model", "-m", default="last_touch", help="Attribution model (last_touch|first_touch|linear|shapley).")
@click.option("--data", "-d", default="", help="Path to attribution data.")
def attribution(model: str, data: str) -> None:
    """Run attribution analysis."""
    console.print(Panel(
        f"[bold]Attribution Analysis[/bold]\n\n"
        f"Model: {model}\n"
        f"Data: {data or '(none)'}",
        border_style="green",
    ))

    # Placeholder attribution result
    table = Table(title=f"Attribution Results ({model})", show_header=True, header_style="bold magenta")
    table.add_column("Channel", style="cyan")
    table.add_column("Attribution Weight", style="green")
    table.add_column("Revenue Contribution", style="yellow")

    channels = [
        ("Search Ads", "35%", "¥125,000"),
        ("Social Ads", "25%", "¥89,000"),
        ("Direct", "20%", "¥71,000"),
        ("Referral", "12%", "¥43,000"),
        ("Email", "8%", "¥29,000"),
    ]
    for ch, weight, revenue in channels:
        table.add_row(ch, weight, revenue)

    console.print(table)


# ---------------------------------------------------------------------------
# seasonal command
# ---------------------------------------------------------------------------


@main.command()
@click.option("--forecast", "-f", is_flag=True, help="Show seasonal forecast.")
@click.option("--calendar", "-c", is_flag=True, help="Show seasonal calendar events.")
def seasonal(forecast: bool, calendar: bool) -> None:
    """Show seasonal context and forecasts."""
    if forecast or (not forecast and not calendar):
        table = Table(title="Seasonal Forecast", show_header=True, header_style="bold magenta")
        table.add_column("Period", style="cyan")
        table.add_column("Factor", style="green")
        table.add_column("Impact", style="yellow")
        table.add_column("Recommendation", style="white")

        rows = [
            ("Q2 Week 1-2", "Freight Peak", "+25% demand", "Increase ad budget by 20%"),
            ("Q2 Week 3-4", "Normal", "Baseline", "Maintain current strategy"),
            ("Q3 Week 1-2", "Mid-Year Sale", "+15% demand", "Prepare promotional campaigns"),
            ("Q3 Week 3-4", "Summer Dip", "-10% demand", "Focus on retention"),
            ("Q4 Week 1-2", "Double 11", "+40% demand", "Maximize acquisition budget"),
            ("Q4 Week 3-4", "Year-End Rush", "+30% demand", "Conversion optimization"),
        ]
        for period, factor, impact, rec in rows:
            table.add_row(period, factor, impact, rec)

        console.print(table)

    if calendar:
        console.print(Panel(
            "[bold]Upcoming Calendar Events[/bold]\n\n"
            "- Double 11 (Nov 11): Major e-commerce event\n"
            "- Mid-Year Sale (Jun 18): 618 promotion\n"
            "- Spring Festival (Jan/Feb): Logistics slowdown\n"
            "- National Day (Oct 1-7): Holiday shipping peak\n"
            "- Year-End Clearance (Dec): Inventory movement",
            title="Seasonal Calendar",
            border_style="green",
        ))


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

    console.print(Panel(
        "[dim]Note: Showing sample KPI data. Connect to live data source for real metrics.[/dim]",
        border_style="dim",
    ))


# ---------------------------------------------------------------------------
# chat command
# ---------------------------------------------------------------------------


@main.command()
@click.option("--scope", "-s", default="full", help="Default analysis scope.")
def chat(scope: str) -> None:
    """Interactive chat mode for growth analysis queries."""
    console.print(Panel(
        "[bold]GrowthPilot Interactive Mode[/bold]\n\n"
        f"Default scope: {scope}\n"
        "Type your query and press Enter.\n"
        "Type 'quit' or 'exit' to stop.\n"
        "Type 'scope <name>' to change scope.\n"
        "Type 'kpi' to see current KPI snapshot.",
        title="Welcome",
        border_style="green",
    ))

    current_scope = scope

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

        if user_input.lower().startswith("scope "):
            new_scope = user_input.split(" ", 1)[1].strip()
            valid_scopes = {"full", "inapp", "prospect", "conversion", "subsidy", "retention", "ad"}
            if new_scope in valid_scopes:
                current_scope = new_scope
                console.print(f"[green]Scope changed to: {current_scope}[/green]")
            else:
                console.print(f"[red]Invalid scope. Valid options: {', '.join(sorted(valid_scopes))}[/red]")
            continue

        if user_input.lower() == "help":
            console.print(Panel(
                "Available commands:\n"
                "- Any natural language query: runs analysis\n"
                "- 'scope <name>': change analysis scope\n"
                "- 'kpi': show current KPI snapshot\n"
                "- 'quit' / 'exit': exit interactive mode\n"
                "- 'help': show this message",
                title="Help",
                border_style="blue",
            ))
            continue

        # Run analysis
        console.print(f"[dim]Running analysis with scope={current_scope}...[/dim]")
        _run_async(_chat_analyze(user_input, current_scope))


async def _chat_analyze(query: str, scope: str) -> None:
    """Run a single analysis from chat mode."""
    from src.graph.workflow import run_workflow

    try:
        result = await run_workflow(query=query, scope=scope)
        report = result.get("report", "")
        if report:
            _print_report(report)
        errors = result.get("errors", [])
        _print_errors(errors)
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
        ("Version", "4.0.0"),
        ("LLM Provider", settings.llm_provider),
        ("LLM Model", settings.llm_model),
        ("API Key Set", "Yes" if settings.llm_api_key else "No (demo mode)"),
        ("Fallback Provider", settings.fallback_provider),
        ("Fallback Model", settings.fallback_model),
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
        ("orchestrator", "all", "Orchestrates sub-agents based on query"),
        ("prospect", "prospect", "User scoring, segmentation, LTV prediction"),
        ("conversion", "conversion", "Funnel analysis, coupons, attribution"),
        ("subsidy", "subsidy", "Causal inference, elasticity, budget optimization"),
        ("retention", "retention", "Churn prediction, nurture, winback"),
        ("ad", "ad", "RTA strategy, bid optimization, creative analysis"),
    ]
    for name, scope, desc in agents:
        agent_table.add_row(name, scope, desc)

    console.print(agent_table)

    if not settings.llm_api_key:
        console.print(Panel(
            "[yellow]No API key configured. Running in demo/mock mode.[/yellow]\n\n"
            "To enable full functionality:\n"
            "1. Set GPA_LLM_API_KEY in .env file\n"
            "2. Or export GPA_LLM_API_KEY=your-key",
            title="Demo Mode",
            border_style="yellow",
        ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
