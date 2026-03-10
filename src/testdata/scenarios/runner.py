"""Shared scenario runner — orchestrates generation, injection, and export.

Scenarios are defined entirely by YAML config files in ``config/scenarios/``.
This module provides the single ``run_scenario()`` entry point that:

1. Loads the scenario YAML (single source of truth for all defaults)
2. Applies CLI overrides for seed/months/strategy
3. Generates clean data → applies injections → normalizes → exports
"""

from __future__ import annotations

import inspect
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
import yaml

from testdata.canonical.finance.generators import generate_finance_dataset
from testdata.config import get_config_dir
from testdata.entropy import injectors
from testdata.entropy.registry import InjectionRegistry
from testdata.entropy.strategies import InjectionSpec, get_strategy
from testdata.export import ExportFormat, dataset_to_dataframes, export_dataframes
from testdata.ground_truth import (
    calculate_ground_truth,
    estimate_injection_impact,
    export_ground_truth,
)
from testdata.schema_transforms import (
    apply_column_style,
    apply_key_strategy,
    apply_normalization,
)


@dataclass
class SourceConfig:
    """A single data source within a multi-source scenario."""

    name: str
    description: str
    tables: list[str]
    column_style: str  # snake_case, camelCase, PascalCase, legacy
    key_strategy: str  # surrogate, natural, uuid, composite
    format: str  # csv, parquet


@dataclass
class ScenarioConfig:
    """Parsed scenario configuration from YAML."""

    name: str
    description: str
    tables: list[str]
    # Defaults — the single source of truth
    seed: int
    months: int
    strategy: str
    # Generator parameters
    normalization: str
    fiscal_start: date
    generator_kwargs: dict
    # Multi-source (None for single-source scenarios)
    sources: list[SourceConfig] | None = None


def load_scenario_config(scenario_name: str) -> ScenarioConfig:
    """Load and parse a scenario YAML by name.

    Raises FileNotFoundError if the YAML doesn't exist.
    Raises KeyError if required fields are missing.
    """
    yaml_name = scenario_name.replace("-", "_") + ".yaml"
    path = get_config_dir() / "scenarios" / yaml_name
    with open(path) as f:
        raw = yaml.safe_load(f)

    defaults = raw["defaults"]
    gen = raw.get("generator", {})

    # Extract generator kwargs (everything except normalization and fiscal_start)
    # Note: journal_entries_per_month, journal_entries_stddev, and
    # bank_transactions_count are legacy params ignored by event-driven generator
    gen_kwargs: dict = {}
    for key in (
        "invoices_count",
        "q4_seasonal_boost",
    ):
        if key in gen:
            gen_kwargs[key] = gen[key]

    # Parse multi-source definitions
    sources: list[SourceConfig] | None = None
    if "sources" in raw:
        sources = []
        for src_name, src_cfg in raw["sources"].items():
            sources.append(SourceConfig(
                name=src_name,
                description=src_cfg.get("description", ""),
                tables=src_cfg["tables"],
                column_style=src_cfg.get("column_style", "snake_case"),
                key_strategy=src_cfg.get("key_strategy", "surrogate"),
                format=src_cfg.get("format", "csv"),
            ))

    return ScenarioConfig(
        name=raw["name"],
        description=raw["description"].strip(),
        tables=raw["tables"],
        seed=defaults["seed"],
        months=defaults["months"],
        strategy=defaults["strategy"],
        normalization=gen.get("normalization", "full"),
        fiscal_start=date.fromisoformat(gen.get("fiscal_start", "2025-01-01")),
        generator_kwargs=gen_kwargs,
        sources=sources,
    )


def _apply_injection(
    spec: InjectionSpec,
    dataframes: dict[str, pl.DataFrame],
    registry: InjectionRegistry,
    rng: random.Random,
) -> None:
    """Apply a single injection spec to the appropriate DataFrame."""
    df = dataframes[spec.table]
    fn = getattr(injectors, spec.injector)

    all_kwargs = dict(spec.kwargs)
    all_kwargs["registry"] = registry
    all_kwargs["table_name"] = spec.table
    all_kwargs["rng"] = rng

    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys()) - {"df"}
    kwargs = {k: v for k, v in all_kwargs.items() if k in accepted}

    dataframes[spec.table] = fn(df=df, **kwargs)


def run_scenario(
    scenario_name: str,
    *,
    strategy_name: str | None = None,
    seed: int | None = None,
    months: int | None = None,
    output_dir: Path | None = None,
    fmt: ExportFormat = "csv",
) -> dict:
    """Generate data for a named scenario, apply entropy, and export.

    CLI overrides (strategy_name, seed, months) replace scenario YAML defaults
    when provided. When ``None``, the YAML default is used.

    Args:
        scenario_name: Which scenario YAML to load (e.g. "month-end-close").
        strategy_name: Override injection strategy.
        seed: Override random seed.
        months: Override month count.
        output_dir: Where to write output. If None, returns data only.
        fmt: Export format — "csv", "parquet", or "both".

    Returns:
        Dict with 'dataframes', 'registry', 'dataset', and 'config' keys.
    """
    config = load_scenario_config(scenario_name)

    # Resolve: CLI override → YAML default (no hardcoded fallbacks)
    seed = seed if seed is not None else config.seed
    months = months if months is not None else config.months
    strategy_name = strategy_name if strategy_name is not None else config.strategy

    strategy = get_strategy(strategy_name)
    rng = random.Random(seed + 1000)  # Offset so injections differ from generation

    # Step 1: Generate clean data
    dataset = generate_finance_dataset(
        seed=seed,
        months=months,
        fiscal_start=config.fiscal_start,
        **config.generator_kwargs,
    )

    # Step 2: Compute ground truth from clean data (before injection)
    ground_truth = calculate_ground_truth(
        dataset,
        seed=seed,
        strategy=strategy_name,
        fiscal_start=config.fiscal_start,
        months=months,
    )

    # Step 3: Convert to DataFrames
    dataframes = dataset_to_dataframes(dataset)

    # Step 4: Apply injections
    registry = InjectionRegistry()
    for spec in strategy.injections:
        _apply_injection(spec, dataframes, registry, rng)

    # Step 5: Estimate injection impact on ground truth metrics
    if len(registry) > 0:
        ground_truth.injection_impact = estimate_injection_impact(
            registry.export_dicts()
        )

    # Step 6: Apply normalization
    dataframes, table_mapping = apply_normalization(dataframes, config.normalization)
    if table_mapping:
        registry.remap_tables(table_mapping)

    # Step 7: Export
    if output_dir is not None:
        generation_params = {
            "scenario": scenario_name,
            "strategy": strategy_name,
            "seed": seed,
            "months": months,
            "normalization": config.normalization,
            "injection_count": len(registry),
        }
        if config.sources:
            _export_multi_source(
                dataframes=dataframes,
                sources=config.sources,
                output_dir=output_dir,
                seed=seed,
                entropy_records=registry.export_dicts(),
                generation_params=generation_params,
            )
        else:
            export_dataframes(
                dataframes=dataframes,
                output_dir=output_dir,
                entropy_records=registry.export_dicts(),
                generation_params=generation_params,
                fmt=fmt,
            )
        export_ground_truth(ground_truth, output_dir)

    return {
        "dataframes": dataframes,
        "registry": registry,
        "dataset": dataset,
        "config": config,
        "ground_truth": ground_truth,
    }


def _export_multi_source(
    dataframes: dict[str, pl.DataFrame],
    sources: list[SourceConfig],
    output_dir: Path,
    seed: int,
    entropy_records: list[dict],
    generation_params: dict,
) -> None:
    """Export data split across multiple source directories.

    Each source gets its own subdirectory with per-source column/key transforms.
    A top-level ``sources.yaml`` indexes all sources.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    source_index: list[dict] = []

    for src in sources:
        # Extract this source's tables
        src_dfs = {t: dataframes[t] for t in src.tables if t in dataframes}

        # Apply key strategy (before column rename so key columns keep canonical names)
        if src.key_strategy != "surrogate":
            src_dfs = apply_key_strategy(src_dfs, src.key_strategy, seed=seed)

        # Apply column style
        if src.column_style != "snake_case":
            src_dfs = apply_column_style(src_dfs, src.column_style)

        # Export to source subdirectory
        src_dir = output_dir / src.name
        src_params = {**generation_params, "source": src.name, "column_style": src.column_style}
        export_dataframes(
            dataframes=src_dfs,
            output_dir=src_dir,
            entropy_records=None,  # Entropy map goes at top level
            generation_params=src_params,
            fmt=src.format,
        )

        source_index.append({
            "name": src.name,
            "description": src.description,
            "directory": src.name,
            "tables": list(src_dfs.keys()),
            "column_style": src.column_style,
            "key_strategy": src.key_strategy,
            "format": src.format,
        })

    # Write top-level sources.yaml
    with open(output_dir / "sources.yaml", "w") as f:
        yaml.dump(
            {"sources": source_index, "generation": generation_params},
            f, default_flow_style=False, sort_keys=False,
        )

    # Write top-level entropy map
    entropy_data = {
        "injections": entropy_records or [],
        "total_injections": len(entropy_records) if entropy_records else 0,
    }
    with open(output_dir / "entropy_map.yaml", "w") as f:
        yaml.dump(entropy_data, f, default_flow_style=False, sort_keys=False)


def discover_scenarios() -> dict[str, ScenarioConfig]:
    """Discover all scenario YAML files and return their configs."""
    scenario_dir = get_config_dir() / "scenarios"
    scenarios: dict[str, ScenarioConfig] = {}
    for path in sorted(scenario_dir.glob("*.yaml")):
        with open(path) as f:
            raw = yaml.safe_load(f)
        name = raw["name"]
        scenarios[name] = load_scenario_config(name)
    return scenarios
