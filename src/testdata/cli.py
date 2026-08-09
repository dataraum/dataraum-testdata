"""Typer CLI for test data generation."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import typer

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
) -> None:
    """Generate synthetic test data with entropy injections."""
    scenarios = discover_scenarios()
    if scenario not in scenarios:
        typer.echo(f"Unknown scenario: {scenario!r}. Available: {list(scenarios.keys())}")
        raise typer.Exit(1)

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
    typer.echo("\nAvailable strategies: clean, low, medium, high")


if __name__ == "__main__":
    app()
