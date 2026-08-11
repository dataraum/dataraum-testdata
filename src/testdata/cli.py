"""Typer CLI for test data generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import typer

from testdata.entropy.strategies import load_all_strategies
from testdata.export import ExportFormat
from testdata.scenarios.runner import discover_scenarios, run_scenario

app = typer.Typer(
    name="testdata",
    help="Synthetic test data generator with known entropy injections.",
    no_args_is_help=True,
)


@app.command()
def generate(
    scenario: str = typer.Option("month-end-close", help="Scenario to generate"),
    strategy: str = typer.Option(None, help="Injection strategy (default: from scenario YAML)"),
    output: Path = typer.Option(..., help="Output directory"),
    seed: int = typer.Option(None, help="Random seed (default: from scenario YAML)"),
    months: int = typer.Option(None, help="Number of months (default: from scenario YAML)"),
    fmt: str = typer.Option("csv", "--format", help="Export format: csv, parquet, json, jsonl, both"),
    lever: str = typer.Option(
        None,
        help='Constructed intervention as JSON, e.g. \'{"type":"rate","driver":"price","period_k":6,"factor":1.15}\'',
    ),
    levers: str = typer.Option(
        None, help="Several interventions as a JSON list — writes the measured interaction too"
    ),
    trend_price: float = typer.Option(0.0, help="Annual price drift, e.g. 0.03 — a control, not an event"),
    trend_volume: float = typer.Option(0.0, help="Annual volume drift, e.g. 0.05"),
    merchants: int = typer.Option(0, help="Size of the high-cardinality Zipfian payer dimension (0 = off)"),
    merchant_exponent: float = typer.Option(1.05, help="Zipf exponent for that dimension"),
) -> None:
    """Generate synthetic test data with entropy injections."""
    scenarios = discover_scenarios()
    if scenario not in scenarios:
        typer.echo(f"Unknown scenario: {scenario!r}. Available: {list(scenarios.keys())}")
        raise typer.Exit(1)

    # The lever spec crosses this boundary as JSON because that is what it is: a plain
    # nested mapping the generator validates. Parsed here so a malformed one fails at
    # the command line rather than halfway through a generation.
    try:
        lever_spec = json.loads(lever) if lever else None
        lever_list = json.loads(levers) if levers else None
    except json.JSONDecodeError as exc:
        typer.echo(f"Could not parse the lever spec as JSON: {exc}")
        raise typer.Exit(1) from exc
    trend = {k: v for k, v in (("price", trend_price), ("volume", trend_volume)) if v}

    if fmt not in ("csv", "parquet", "json", "jsonl", "both"):
        typer.echo(f"Unknown format: {fmt!r}. Available: csv, parquet, json, jsonl, both")
        raise typer.Exit(1)

    # Resolve effective values for display (CLI override → YAML default)
    config = scenarios[scenario]
    eff_seed = seed if seed is not None else config.seed
    eff_months = months if months is not None else config.months
    eff_strategy = strategy if strategy is not None else config.strategy

    typer.echo(f"Generating scenario={scenario!r} strategy={eff_strategy!r} seed={eff_seed} months={eff_months}")
    typer.echo(f"Output: {output} (format={fmt})")

    result = run_scenario(
        scenario,
        strategy_name=strategy,
        seed=seed,
        months=months,
        output_dir=output,
        fmt=cast(ExportFormat, fmt),
        lever=lever_spec,
        levers=lever_list,
        trend=trend or None,
        merchants=merchants,
        merchant_exponent=merchant_exponent,
    )

    registry = result["registry"]
    dataframes = result["dataframes"]

    # The identity, not the path, is what a consumer pins — so print it where the
    # person running the generator will see it.
    typer.echo(f"Corpus: {result['identity'].describe()}")

    typer.echo(f"\nGenerated {sum(len(df) for df in dataframes.values())} total rows across {len(dataframes)} tables:")
    for name, df in dataframes.items():
        typer.echo(f"  {name}: {len(df)} rows, {len(df.columns)} columns")

    typer.echo(f"\nEntropy injections: {len(registry)}")
    if len(registry) > 0:
        summary = registry.summary()
        typer.echo(f"  By layer: {summary.get('by_layer', {})}")
        typer.echo(f"  By defect: {summary.get('by_defect', {})}")

    typer.echo(f"\nFiles written to: {output}")


@app.command()
def list_scenarios() -> None:
    """List available scenarios."""
    for name, config in discover_scenarios().items():
        typer.echo(f"  {name}: {config.description}")


@app.command()
def describe(
    scenario: str = typer.Option("month-end-close", help="Scenario to describe"),
) -> None:
    """Describe a scenario's configuration."""
    scenarios = discover_scenarios()
    if scenario not in scenarios:
        typer.echo(f"Unknown scenario: {scenario!r}. Available: {list(scenarios.keys())}")
        raise typer.Exit(1)

    config = scenarios[scenario]
    typer.echo(f"Scenario: {config.name}")
    typer.echo(f"Description: {config.description}")
    typer.echo(f"Tables: {', '.join(config.tables)}")
    typer.echo("\nDefaults:")
    typer.echo(f"  strategy: {config.strategy}")
    typer.echo(f"  seed: {config.seed}")
    typer.echo(f"  months: {config.months}")
    typer.echo(f"  fiscal_start: {config.fiscal_start}")
    typer.echo(f"  normalization: {config.normalization}")
    typer.echo(f"  scale_profile: {config.scale_profile}")
    typer.echo("\nAvailable strategies: " + ", ".join(sorted(load_all_strategies())))


if __name__ == "__main__":
    app()
