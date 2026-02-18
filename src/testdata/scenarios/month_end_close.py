"""Month-end-close scenario: generates finance data and applies entropy injections."""

from __future__ import annotations

import inspect
import random
from pathlib import Path

import polars as pl
import yaml

from testdata.canonical.finance.generators import generate_finance_dataset
from testdata.config import get_config_dir
from testdata.entropy import injectors
from testdata.entropy.registry import InjectionRegistry
from testdata.entropy.strategies import InjectionSpec, get_strategy
from testdata.export import dataset_to_dataframes, export_dataframes


SCENARIO_NAME = "month-end-close"
SCENARIO_DESCRIPTION = "Month-end close for a mid-size company. 12-month fiscal year with ~10K rows across 8 tables."


def _load_scenario_defaults() -> dict:
    """Load defaults from the scenario YAML config."""
    path = get_config_dir() / "scenarios" / "month_end_close.yaml"
    with open(path) as f:
        return yaml.safe_load(f).get("defaults", {})


def _apply_injection(
    spec: InjectionSpec,
    dataframes: dict[str, pl.DataFrame],
    registry: InjectionRegistry,
    rng: random.Random,
) -> None:
    """Apply a single injection spec to the appropriate DataFrame."""
    df = dataframes[spec.table]
    fn = getattr(injectors, spec.injector)

    # Build all possible kwargs, then filter to what the function accepts
    all_kwargs = dict(spec.kwargs)
    all_kwargs["registry"] = registry
    all_kwargs["table_name"] = spec.table
    all_kwargs["rng"] = rng

    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys()) - {"df"}
    kwargs = {k: v for k, v in all_kwargs.items() if k in accepted}

    dataframes[spec.table] = fn(df=df, **kwargs)


def run_scenario(
    strategy_name: str | None = None,
    seed: int | None = None,
    months: int | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Generate finance data, apply entropy injections, and export.

    Parameters fall back to ``config/scenarios/month_end_close.yaml`` defaults
    when not provided.  CLI flags override these.

    Args:
        strategy_name: Which injection strategy to use (clean/low/medium/high).
        seed: Random seed for deterministic generation.
        months: Number of months in the fiscal year.
        output_dir: Where to write CSV + YAML files. If None, returns data without writing.

    Returns:
        Dict with 'dataframes', 'registry', and 'dataset' keys.
    """
    defaults = _load_scenario_defaults()
    if strategy_name is None:
        strategy_name = defaults.get("strategy", "medium")
    if seed is None:
        seed = defaults.get("seed", 42)
    if months is None:
        months = defaults.get("months", 12)

    strategy = get_strategy(strategy_name)
    rng = random.Random(seed + 1000)  # Offset seed so injections differ from generation

    # Step 1: Generate clean data
    dataset = generate_finance_dataset(seed=seed, months=months)

    # Step 2: Convert to DataFrames
    dataframes = dataset_to_dataframes(dataset)

    # Step 3: Apply injections
    registry = InjectionRegistry()
    for spec in strategy.injections:
        _apply_injection(spec, dataframes, registry, rng)

    # Step 4: Export if output_dir provided
    if output_dir is not None:
        generation_params = {
            "scenario": SCENARIO_NAME,
            "strategy": strategy_name,
            "seed": seed,
            "months": months,
            "injection_count": len(registry),
        }
        export_dataframes(
            dataframes=dataframes,
            output_dir=output_dir,
            entropy_records=registry.export_dicts(),
            generation_params=generation_params,
        )

    return {
        "dataframes": dataframes,
        "registry": registry,
        "dataset": dataset,
    }


SCENARIOS = {
    SCENARIO_NAME: {
        "name": SCENARIO_NAME,
        "description": SCENARIO_DESCRIPTION,
        "tables": [
            "chart_of_accounts", "journal_entries", "journal_lines",
            "invoices", "payments", "bank_transactions", "fx_rates", "trial_balance",
        ],
        "default_strategy": "medium",
        "run": run_scenario,
    },
}
