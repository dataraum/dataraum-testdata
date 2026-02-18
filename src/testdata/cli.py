"""Typer CLI for test data generation."""

from __future__ import annotations

from pathlib import Path

import typer

from testdata.scenarios.month_end_close import SCENARIOS, run_scenario

app = typer.Typer(
    name="testdata",
    help="Synthetic test data generator with known entropy injections.",
    no_args_is_help=True,
)


@app.command()
def generate(
    scenario: str = typer.Option("month-end-close", help="Scenario to generate"),
    strategy: str = typer.Option("medium", help="Injection strategy: clean, low, medium, high"),
    output: Path = typer.Option(..., help="Output directory for CSV and YAML files"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    months: int = typer.Option(12, help="Number of months in fiscal year"),
) -> None:
    """Generate synthetic test data with entropy injections."""
    if scenario not in SCENARIOS:
        typer.echo(f"Unknown scenario: {scenario!r}. Available: {list(SCENARIOS.keys())}")
        raise typer.Exit(1)

    typer.echo(f"Generating scenario={scenario!r} strategy={strategy!r} seed={seed} months={months}")
    typer.echo(f"Output: {output}")

    result = run_scenario(
        strategy_name=strategy,
        seed=seed,
        months=months,
        output_dir=output,
    )

    registry = result["registry"]
    dataframes = result["dataframes"]

    typer.echo(f"\nGenerated {sum(len(df) for df in dataframes.values())} total rows across {len(dataframes)} tables:")
    for name, df in dataframes.items():
        typer.echo(f"  {name}: {len(df)} rows, {len(df.columns)} columns")

    typer.echo(f"\nEntropy injections: {len(registry)}")
    if len(registry) > 0:
        summary = registry.summary()
        typer.echo(f"  By layer: {summary.get('by_layer', {})}")
        typer.echo(f"  By detector: {summary.get('by_detector', {})}")

    typer.echo(f"\nFiles written to: {output}")


@app.command()
def list_scenarios() -> None:
    """List available scenarios."""
    for name, info in SCENARIOS.items():
        typer.echo(f"  {name}: {info['description']}")


@app.command()
def describe(
    scenario: str = typer.Option("month-end-close", help="Scenario to describe"),
) -> None:
    """Describe a scenario and its tables."""
    if scenario not in SCENARIOS:
        typer.echo(f"Unknown scenario: {scenario!r}. Available: {list(SCENARIOS.keys())}")
        raise typer.Exit(1)

    info = SCENARIOS[scenario]
    typer.echo(f"Scenario: {info['name']}")
    typer.echo(f"Description: {info['description']}")
    typer.echo(f"Default strategy: {info['default_strategy']}")
    tables: list[str] = info["tables"]  # type: ignore[assignment]
    typer.echo(f"Tables: {', '.join(tables)}")
    typer.echo("\nAvailable strategies: clean, low, medium, high")


if __name__ == "__main__":
    app()
