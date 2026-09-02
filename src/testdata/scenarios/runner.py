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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import polars as pl
import yaml

from testdata.canonical.finance.generators import (
    Lever,
    MixSolution,
    generate_finance_dataset,
    mix_outcome,
)
from testdata.config import get_config_dir
from testdata.entropy import injectors
from testdata.entropy.families import REL_CHILD_TABLE
from testdata.entropy.registry import InjectionRegistry
from testdata.entropy.strategies import InjectionSpec, get_strategy, load_strategy
from testdata.export import ExportFormat, dataset_to_dataframes, export_dataframes
from testdata.ground_truth import (
    GroundTruth,
    calculate_ground_truth,
    estimate_injection_impact,
    export_ground_truth,
)
from testdata.identity import CorpusIdentity
from testdata.metadata_truth import export_metadata_truth
from testdata.scale import DEFAULT_PROFILE
from testdata.schema_transforms import (
    ColumnStyle,
    KeyStrategy,
    NormalizationLevel,
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
    column_style: ColumnStyle
    key_strategy: KeyStrategy
    format: ExportFormat


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
    normalization: NormalizationLevel
    scale_profile: str
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

    # Extract generator kwargs (everything except normalization and fiscal_start).
    # An ALLOWLIST, not a passthrough: a scenario YAML naming a key the generator does
    # not take is dropped here rather than reaching it. `generate_finance_dataset` used
    # to end in `**_kwargs`, which meant a misspelled argument — `scale_profile=` for
    # `profile=` — silently generated the default firm instead of raising.
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
            sources.append(
                SourceConfig(
                    name=src_name,
                    description=src_cfg.get("description", ""),
                    tables=src_cfg["tables"],
                    column_style=src_cfg.get("column_style", "snake_case"),
                    key_strategy=src_cfg.get("key_strategy", "surrogate"),
                    format=src_cfg.get("format", "csv"),
                )
            )

    return ScenarioConfig(
        name=raw["name"],
        description=raw["description"].strip(),
        tables=raw["tables"],
        seed=defaults["seed"],
        months=defaults["months"],
        strategy=defaults["strategy"],
        normalization=gen.get("normalization", "full"),
        scale_profile=gen.get("scale_profile", DEFAULT_PROFILE),
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
    # The run's full table mapping — passed only to injectors that declare it, so a
    # family can emit or fill companion tables (stock/flow probe events; the
    # relationship_pairs parent and child); the signature filter below keeps
    # single-table injectors untouched.
    all_kwargs["dataframes"] = dataframes

    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys()) - {"df"}
    kwargs = {k: v for k, v in all_kwargs.items() if k in accepted}

    recorded_before = len(registry)
    dataframes[spec.table] = fn(df=df, **kwargs)

    # Stamp the strategy's consumer hint onto EVERY record this injection produced.
    # Labelling only the last record silently mislabels multi-record injectors (one
    # record per probe column or relationship pair) and can clobber an unrelated
    # injection's record when an injector recorded nothing.
    if spec.consumer_hint is not None:
        for injection in registry.injections_since(recorded_before):
            injection.consumer_hint = spec.consumer_hint


def run_scenario(
    scenario_name: str,
    *,
    strategy_name: str | None = None,
    strategy_file: Path | None = None,
    seed: int | None = None,
    months: int | None = None,
    output_dir: Path | None = None,
    truth_dir: Path | None = None,
    fmt: ExportFormat = "csv",
    lever: dict | None = None,
    levers: list[dict] | None = None,
    trend: dict | None = None,
    merchants: int = 0,
    merchant_exponent: float = 1.05,
) -> dict:
    """Generate data for a named scenario, apply entropy, and export.

    CLI overrides (strategy_name, seed, months) replace scenario YAML defaults
    when provided. When ``None``, the YAML default is used.

    ``lever`` applies a constructed intervention to the generating
    process itself — e.g. ``{"type": "price_level", "period_k": 36,
    "factor": 1.15}``. Recorded in ``intervention.yaml`` next to the data;
    a same-seed run without the lever is the exact counterfactual baseline.

    ``trend`` drifts prices and/or volumes a few percent a year — ``{"price": 0.03,
    "volume": 0.05}``. It is the CONTROL for a lever: everything rises and nothing
    happened. Recorded in the corpus stamp rather than intervention.yaml, because
    there is no activation period and nothing to attribute. Absent, the stamp and the
    digest are byte-identical to what they were before it existed.

    ``merchants`` turns on the high-cardinality Zipfian payer dimension at that pool
    size — a ``merchants`` table plus ``bank_transactions.merchant_id``, with the
    realised frequency distribution published under ``dimensions`` in ground_truth.
    0 (the default) means neither the table nor the column exists.

    ``levers`` applies several at once — the interaction pair. Their combined effect
    is NOT the sum of the singles, so the export additionally carries a measured
    ``interaction`` block built from the generated full factorial. Mutually exclusive
    with ``lever``; a single-lever run's files and corpus id are untouched by its
    existence.

    Args:
        scenario_name: Which scenario YAML to load (e.g. "month-end-close").
        strategy_name: Override injection strategy by name.
        strategy_file: Override injection strategy from arbitrary YAML path.
            Takes precedence over strategy_name.
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

    if strategy_file is not None:
        strategy = load_strategy(strategy_file)
        strategy_name = strategy.name
    else:
        strategy_name = strategy_name if strategy_name is not None else config.strategy
        strategy = get_strategy(strategy_name)
    rng = random.Random(seed + 1000)  # Offset so injections differ from generation

    # Generate probe-table grains only when a strategy injects into them, so other
    # strategies (the baseline) are untouched.
    probe_series = 15 if any(s.table == "measure_probes" for s in strategy.injections) else 0
    formula_probe_rows = 300 if any(s.table == "formula_probes" for s in strategy.injections) else 0
    # Same gate for the relationship probe grains: parent ids + child
    # rows exist only when a strategy targets the child probe table.
    needs_relationship_probes = any(s.table == REL_CHILD_TABLE for s in strategy.injections)
    # Same gate for the role-playing-FK shape: dimension + both
    # fact grains exist only when a strategy targets the role-play fact.
    needs_roleplay = any(s.table == "orders" for s in strategy.injections)

    if lever is not None and levers:
        raise ValueError("pass `lever` or `levers`, not both")
    lever_dicts = [dict(spec) for spec in (levers if levers else ([lever] if lever is not None else []))]
    lever_specs = tuple(Lever(**spec) for spec in lever_dicts)

    # The corpus is a function of these parameters (§6). Built before generation
    # rather than at export, so it is knowable without writing anything — a consumer
    # can pin the id and only then decide it needs the bytes.
    identity = CorpusIdentity(
        scenario=scenario_name,
        strategy=strategy_name,
        seed=seed,
        months=months,
        fiscal_start=config.fiscal_start.isoformat(),
        profile=config.scale_profile,
        normalization=config.normalization,
        lever=_lever_payload(lever_dicts),
        trend=dict(trend) if trend else None,
        merchants=merchants,
    )

    # Step 1: Generate clean data
    dataset = generate_finance_dataset(
        seed=seed,
        months=months,
        fiscal_start=config.fiscal_start,
        probe_series=probe_series,
        formula_probe_rows=formula_probe_rows,
        relation_parents=300 if needs_relationship_probes else 0,
        relation_children=1200 if needs_relationship_probes else 0,
        roleplay_addresses=60 if needs_roleplay else 0,
        roleplay_orders=400 if needs_roleplay else 0,
        roleplay_deliveries=700 if needs_roleplay else 0,
        levers=lever_specs,
        trend=trend,
        merchants=merchants,
        merchant_exponent=merchant_exponent,
        profile=config.scale_profile,
        **config.generator_kwargs,
    )

    # Step 2: Compute ground truth from clean data (before injection)
    ground_truth = calculate_ground_truth(
        dataset,
        fiscal_start=config.fiscal_start,
        months=months,
        merchant_exponent=merchant_exponent,
    )

    # Step 3: Convert to DataFrames
    dataframes = dataset_to_dataframes(dataset)

    # Step 4: Apply injections
    registry = InjectionRegistry()
    for spec in strategy.injections:
        _apply_injection(spec, dataframes, registry, rng)

    # Step 5: Estimate injection impact on ground truth metrics
    if len(registry) > 0:
        ground_truth.injection_impact = estimate_injection_impact(registry.export_dicts())

    # Step 6: Apply normalization
    dataframes, table_mapping = apply_normalization(dataframes, config.normalization)
    if table_mapping:
        registry.remap_tables(table_mapping)

    # Step 7: Export
    if output_dir is not None:
        # The answer key defaults to a sibling, never the corpus dir:
        # whatever mounts or serves the data cannot reach the truth.
        if truth_dir is None:
            truth_dir = output_dir.parent / (output_dir.name + "-truth")
        truth_dir.mkdir(parents=True, exist_ok=True)
        # Run facts that may travel with the data: none for a single
        # source; a multi-source export names its source. The injection
        # count is answer-key material — entropy_map.yaml carries it as
        # total_injections — and stays out of anything the corpus dir
        # holds (ruled 2026-09-02).
        run_facts: dict = {}
        if config.sources:
            _export_multi_source(
                dataframes=dataframes,
                sources=config.sources,
                output_dir=output_dir,
                truth_dir=truth_dir,
                seed=seed,
                entropy_records=registry.export_dicts(),
                identity=identity,
                run_facts=run_facts,
            )
        else:
            export_dataframes(
                dataframes=dataframes,
                output_dir=output_dir,
                entropy_records=registry.export_dicts(),
                identity=identity,
                run_facts=run_facts,
                fmt=fmt,
                truth_dir=truth_dir,
            )
        export_ground_truth(ground_truth, truth_dir, identity)
        # Agent-layer ground truth — top-level like entropy_map/ground_truth,
        # table names remapped to this run's normalization, canonical (snake) columns.
        # ``level`` drives the folded-dimension truth for denormalized shapes.
        export_metadata_truth(
            truth_dir,
            table_mapping=table_mapping,
            level=config.normalization,
            # post-injection frames drive the data-derived measured_in.cross_unit flags
            dataframes=dataframes,
            identity=identity,
        )
        if lever_specs:
            records = []
            for lever_item in lever_specs:
                # A mix lever's truth is its solved pair and the share it actually
                # landed on — measured off the corpus that shipped, not the request.
                mix = share = None
                if lever_item.effective_driver == "share":
                    mix, share = mix_outcome(
                        lever_item, dataset,
                        seed=seed, months=months, fiscal_start=config.fiscal_start,
                        profile=config.scale_profile,
                        **{k: v for k, v in config.generator_kwargs.items() if k == "q4_seasonal_boost"},
                    )
                records.append(
                    _export_intervention(
                        lever_item, fiscal_start=config.fiscal_start, months=months,
                        mix=mix, realised_share=share,
                    )
                )
            # The factorial is generated, so its cost is 2^k - 1 extra corpora. Bounded
            # rather than refused: a pair is the case the truth exists for, and beyond
            # three levers the cost stops being worth an answer nobody asked for.
            interaction = None
            if 2 <= len(lever_specs) <= 3:
                interaction = _interaction_truth(
                    lever_specs, lever_dicts, ground_truth, identity,
                    seed=seed, months=months, fiscal_start=config.fiscal_start,
                    profile=config.scale_profile, generator_kwargs=config.generator_kwargs,
                    trend=trend,
                )
            elif len(lever_specs) > 3:
                interaction = {
                    "measured": False,
                    "reason": (
                        f"{len(lever_specs)} levers need {2 ** len(lever_specs)} corpora for a full "
                        "factorial; generate the subsets yourself if you need the term"
                    ),
                }
            _write_intervention(records, truth_dir, identity=identity, interaction=interaction)

    return {
        "dataframes": dataframes,
        "registry": registry,
        "dataset": dataset,
        "config": config,
        "ground_truth": ground_truth,
        "identity": identity,
    }


def _lever_payload(lever_dicts: Sequence[dict]) -> dict | None:
    """The identity's ``lever`` field for a set of levers.

    A single lever serialises exactly as it always has — the flat spec dict — so an
    existing run's corpus id is bit-for-bit what it was. Several go under one key
    INSIDE that field rather than as a new top-level one, for the same reason: the
    digest's shape must not change for runs that do not use the new parameter.
    """
    if not lever_dicts:
        return None
    if len(lever_dicts) == 1:
        return dict(lever_dicts[0])
    return {"levers": [dict(spec) for spec in lever_dicts]}


_LEVER_LABELS = "ABCDEFGH"


def _subset_label(labels: Sequence[str]) -> str:
    return "+".join(labels) if labels else "baseline"


def _annual_numbers(truth: GroundTruth) -> dict[str, float]:
    """The annual metrics as plain floats — the surface an interaction is measured on."""
    annual = truth.annual
    return {
        name: float(getattr(annual, name))
        for name in type(annual).model_fields
        if isinstance(getattr(annual, name), int | float | Decimal)
    }


def _monthly_numbers(truth: GroundTruth) -> dict[str, list[float]]:
    """Each monthly metric as a series, in period order.

    Annual figures are not enough to see an interaction, and for some pairs they show
    NONE. A collection-lag lever moves when cash lands, not whether: a receipt is
    clamped to the fiscal end, so by 31 December the levered and baseline corpora hold
    the same cash and the same AR, and every annual metric agrees to the cent. The
    lever's entire footprint — and therefore its whole interaction with a price
    lever — is inside the year. Recording only the annual block would publish
    "interaction: 0.00" for a pair whose interaction is millions in September.
    """
    if not truth.monthly:
        return {}
    names = [
        name
        for name in type(truth.monthly[0]).model_fields
        if isinstance(getattr(truth.monthly[0], name), int | float | Decimal)
    ]
    return {name: [float(getattr(row, name)) for row in truth.monthly] for name in names}


def _interaction_truth(
    levers: Sequence[Lever],
    lever_dicts: Sequence[dict],
    combined: GroundTruth,
    identity: CorpusIdentity | None,
    *,
    seed: int,
    months: int,
    fiscal_start: date,
    profile: str,
    generator_kwargs: dict,
    trend: dict | None = None,
) -> dict:
    """The measured non-additivity of a lever set — the whole point of running two.

    Two levers are not two facts. Their combined effect is the sum of the singles only
    if they act on disjoint quantities, and a price change and a payment-terms change
    do not: the terms change delays cash that the price change already made bigger. A
    consumer that fits main effects and stops is wrong by exactly the amount recorded
    here, and until it is recorded nothing on disk says by how much.

    Measured, not derived: the full factorial is GENERATED — every subset of the
    levers run as its own same-seed corpus — and the highest-order interaction is
    ``sum over subsets S of (-1)^(k-|S|) * y_S``. The full-set corpus is the run that
    is already happening, so a pair costs three extra generations rather than four.
    """
    labels = _LEVER_LABELS[: len(levers)]
    subsets: list[tuple[int, ...]] = [()]
    for i in range(len(levers)):
        subsets = subsets + [s + (i,) for s in subsets]
    full = tuple(range(len(levers)))

    values: dict[tuple[int, ...], dict[str, float]] = {full: _annual_numbers(combined)}
    series: dict[tuple[int, ...], dict[str, list[float]]] = {full: _monthly_numbers(combined)}
    corpus_ids: dict[str, str] = {}
    for subset in subsets:
        name = _subset_label([labels[i] for i in subset])
        if identity is not None:
            corpus_ids[name] = replace(
                identity, lever=_lever_payload([lever_dicts[i] for i in subset])
            ).corpus_id
        if subset == full:
            continue
        corpus = generate_finance_dataset(
            seed=seed, months=months, fiscal_start=fiscal_start, profile=profile,
            levers=[levers[i] for i in subset], trend=trend, **generator_kwargs,
        )
        truth = calculate_ground_truth(corpus, fiscal_start=fiscal_start, months=months)
        values[subset] = _annual_numbers(truth)
        series[subset] = _monthly_numbers(truth)

    def alternating(read) -> float:
        """Σ over subsets of (−1)^(k−|S|)·y_S — the highest-order interaction term."""
        return sum((-1) ** (len(levers) - len(s)) * read(s) for s in subsets)

    base = values[()]
    metrics: dict[str, dict[str, float | None]] = {}
    for name, baseline in base.items():
        deltas = {
            _subset_label([labels[i] for i in s]): round(values[s][name] - baseline, 2)
            for s in subsets
            if s
        }
        interaction = alternating(lambda s, _n=name: values[s][_n])
        additive = sum(deltas[labels[i]] for i in range(len(levers)))
        joint = deltas[_subset_label(list(labels))]
        metrics[name] = {
            "baseline": round(baseline, 2),
            **deltas,
            "additive_prediction": round(additive, 2),
            "interaction": round(interaction, 2),
            "interaction_share_of_combined": round(interaction / joint, 6) if joint else None,
        }

    periods = [row.period for row in combined.monthly]
    monthly: dict[str, dict] = {}
    for name in series[()]:
        terms = [
            round(alternating(lambda s, _n=name, _t=t: series[s][_n][_t]), 2) for t in range(len(periods))
        ]
        if not any(abs(term) > 0.005 for term in terms):
            continue  # this metric is additive in these levers, and silence says so
        peak = max(range(len(terms)), key=lambda t: abs(terms[t]))
        monthly[name] = {
            "periods": periods,
            "interaction": terms,
            "peak_period": periods[peak],
            "peak_interaction": terms[peak],
        }

    return {
        "levers": [
            {
                "label": labels[i],
                "type": lever.type,
                "driver": lever.effective_driver,
                "period_k": lever.period_k,
                "factor": dict(lever.factor) if isinstance(lever.factor, Mapping) else lever.factor,
                "scope": {dim: list(members) for dim, members in lever.scope.items()} if lever.scope else None,
            }
            for i, lever in enumerate(levers)
        ],
        "corpus_ids": corpus_ids,
        "definition": (
            "Every entry under `metrics` is an ANNUAL metric. `baseline` is its value with no "
            "lever; each label (and each combination) is that subset's delta from baseline, "
            "measured on a generated same-seed corpus, not estimated. `additive_prediction` is "
            "the sum of the single-lever deltas — what a main-effects model predicts. "
            "`interaction` is the alternating sum over all subsets, i.e. combined minus "
            "additive_prediction for a pair, and it is exactly the error that model makes. "
            "`monthly` carries the same term per period for every metric where it is "
            "non-zero somewhere; a metric absent from it is additive in these levers at "
            "every period. Read `monthly` before concluding from `metrics`: a lever that "
            "moves timing rather than totals can interact by millions inside the year and "
            "show 0.00 at year end."
        ),
        "metrics": metrics,
        "monthly": monthly,
    }


def _export_intervention(
    lever: Lever,
    *,
    fiscal_start: date | None,
    months: int,
    mix: MixSolution | None = None,
    realised_share: float | None = None,
) -> dict:
    """One lever's ground-truth record — the spec plus the analytic effect statement.

    Analogous to entropy_map.yaml for injections. The numeric per-period true
    effect is obtained by the consumer via the exact same-seed counterfactual
    pair (run the identical scenario without ``lever``).
    """
    start = fiscal_start if fiscal_start is not None else date(2025, 1, 1)
    activation = date(start.year + (start.month - 1 + lever.period_k) // 12, (start.month - 1 + lever.period_k) % 12 + 1, 1)
    # The effect statement is DRIVER-SPECIFIC. Emitting the price-lever wording for a
    # volume lever would put a wrong ground truth on disk — the exact failure this
    # file exists to prevent.
    driver = lever.effective_driver
    if driver == "share":
        affected = {
            "direct": "order COUNT per customer per month for months >= period_k, scaled UP inside the scope and DOWN outside it, by the two solved factors",
            "propagated": "revenue, cost of sale, AR invoices and receipts for the orders each side gains or loses; inventory replenishment; trial_balance and balance_sheet",
            "unaffected": "within-member rates — unit price, order size, discount, payment behaviour are drawn exactly as in the baseline; the expenditure cycle, operating events, fx_rates",
        }
        analytic_effect = (
            "composition moves, total activity does not. The scoped members' share of order "
            "count goes from `mix.baseline_share` to `mix.target_share`; the complement is "
            "scaled by the DERIVED `mix.complement_factor`, which solves "
            "s0*ft + (1-s0)*fc = 1, so expected total order count is unchanged. Any move in "
            "an aggregate metric is therefore compositional: it comes from the members' "
            "differing within-member rates, not from more or less business. Realised shares "
            "differ slightly from the target because an order count is a rounded draw — "
            "`mix.realised_share` is what the corpus actually holds."
        )
    elif driver == "collection_lag":
        affected = {
            "direct": "receipts.date and the cash/bank entries derived from it, for in-scope sales in months >= period_k",
            "propagated": "AR ageing and closing AR balance, DSO and the cash conversion cycle, bank_transactions dates, trial_balance and balance_sheet",
            "unaffected": "WHICH sales are collected and for how much (decided before the lag is drawn), every order, line, AR invoice amount and the whole expenditure cycle",
        }
        analytic_effect = (
            "the drawn 5-45 day lag is scaled by `factor` for in-scope sales; the same set of "
            "sales is collected in both runs, so the pair differs only in when cash lands. "
            "CAVEAT, by construction: a receipt is clamped to the fiscal end, so late-year "
            "sales absorb part of a lag increase rather than moving by the full factor. DSO "
            "moves by LESS than `factor` near the year boundary and the effect is not linear "
            "in it — compare the pair rather than assuming proportionality."
        )
    elif driver == "frequency":
        affected = {
            "direct": "order COUNT per customer per month for months >= period_k; the added orders carry their own lines, revenue, cost of sale and AR invoice",
            "propagated": "revenue and COGS postings for the added orders, their receipts/bank inflows, inventory replenishment (sized off monthly COGS), trial_balance and balance_sheet",
            "unaffected": "every pre-existing order and order line (byte-identical), the expenditure cycle (invoices, payments, AP), operating events, fx_rates",
        }
        analytic_effect = (
            "orders of the same-seed baseline are a strict SUBSET of the levered run's: every "
            "pre-existing order, line, AR invoice and receipt is byte-identical, because order i "
            "of a (customer, month) draws from its own identity-keyed stream. The difference "
            "between the two corpora IS the added volume, so its revenue, cost of sale and DB1 "
            "contribution are computable to the cent by set difference. Order counts scale by "
            "`factor` in expectation, not exactly (the count is a rounded draw)."
        )
    else:
        affected = {
            "direct": "sales_order_lines.unit_price and line_amount for orders in months >= period_k",
            "propagated": "the revenue-account credits, AR debits and AR invoice amounts derived from those lines; cash receipts / bank inflows for levered sales (5-45d collection lag); trial_balance and balance_sheet lines derived from them",
            "unaffected": "order and line COUNTS, units, the discount off list, product standard cost and therefore cost of sale; the expenditure cycle (invoices, payments, AP), operating events, fx_rates",
        }
        analytic_effect = (
            "realised unit price for months >= period_k scales by exactly `factor` vs the "
            "same-seed baseline (RNG stream is identical; scaling is applied after every draw, "
            "including the discount). Revenue-account activity follows, and so do the "
            "entity-grain figures — db1_by_customer and db1_by_product_group are computed off "
            "the same lines. Receipts follow with the collection lag. Cost of sale is unchanged, "
            "so contribution margin moves by the full price delta."
        )

    # A scope narrows every sentence above, so it is stated rather than left to be
    # inferred from `corpus.lever`. An intervention record that reads as table-wide
    # when one segment moved is the wrong ground truth, which is what this file
    # exists to prevent.
    if lever.scope:
        scope_text = ", ".join(f"{dim}={list(members)}" for dim, members in sorted(lever.scope.items()))
        affected = {
            **affected,
            "scope": f"restricted to {scope_text} (several dimensions intersect); entities outside it are byte-identical to the baseline",
        }
        analytic_effect = (
            f"{analytic_effect} SCOPED: everything above holds WITHIN {scope_text} and nowhere "
            "else. Outside the scope the two corpora are byte-identical, so the aggregate moves "
            "while exactly one named slice moves under it — the slice is the answer key."
        )

    # A per-member factor changes what the aggregate delta MEANS. One number is
    # consistent with infinitely many per-member stories, so an attribution claim can
    # only be graded against the story that was actually run — which is this map.
    heterogeneous = isinstance(lever.factor, Mapping)
    if heterogeneous:
        assert isinstance(lever.factor, Mapping)  # narrowed above; mypy needs it said
        dim = lever.factor_dimension
        spread = ", ".join(f"{member}={value}" for member, value in sorted(lever.factor.items()))
        affected = {
            **affected,
            "heterogeneity": f"each {dim} moves by its OWN factor ({spread}), not by a slice-wide one",
        }
        analytic_effect = (
            f"{analytic_effect} HETEROGENEOUS: the factor is per-{dim} ({spread}), so the "
            "aggregate delta is the member-weighted mix of these and matches no single one of "
            "them. Recovering the aggregate proves nothing about attribution; the per-member "
            "figures are the answer key, and a claim that names the wrong member as the driver "
            "is wrong even when its total is right."
        )

    record: dict = {
        "type": lever.type,
        "driver": driver,
        "period_k": lever.period_k,
        "activation_period": activation.strftime("%Y-%m"),
        "scope": {dim: list(members) for dim, members in lever.scope.items()} if lever.scope else None,
        "months_total": months,
    }
    # A mix lever has no single factor — it has a solved pair and a share it actually
    # landed on, which is not the share that was asked for (an order count is a
    # rounded draw). Publishing only the request would put an unachieved number on
    # disk as though it were the truth.
    if mix is not None:
        record["mix"] = {
            "baseline_share": round(mix.baseline_share, 6),
            "target_share": round(mix.target_share, 6),
            "realised_share": round(realised_share, 6) if realised_share is not None else None,
            "target_factor": round(mix.target_factor, 6),
            "complement_factor": round(mix.complement_factor, 6),
        }
    elif heterogeneous:
        assert isinstance(lever.factor, Mapping)
        record["factor"] = {str(member): value for member, value in lever.factor.items()}
        record["factor_dimension"] = lever.factor_dimension
    else:
        record["factor"] = lever.factor
    record["affected"] = affected
    record["analytic_effect"] = analytic_effect
    record["counterfactual"] = "re-run the identical scenario (same seed/months/strategy) without `lever`"
    return record


def _write_intervention(
    records: Sequence[dict],
    output_dir: Path,
    *,
    identity: CorpusIdentity | None = None,
    interaction: dict | None = None,
) -> None:
    """Write intervention.yaml — one record per lever, plus their interaction.

    A single lever keeps the shape it has always had (``intervention:``, one mapping),
    because consumers bind to it. Several become ``interventions:``, a list of exactly
    those same records, so nothing has to be re-learned to read one of them — and the
    ``interaction`` block, without which a multi-lever corpus would publish two truths
    and leave the one that matters unstated.

    Either way the record carries the counterfactual's corpus id alongside its own: the
    instruction "re-run without the lever" is only actionable if the baseline can be
    named, and a pair compared across a generator change is not a counterfactual.
    """
    payload: dict = {}
    if identity is not None:
        payload["corpus"] = identity.as_dict()
        payload["counterfactual_corpus_id"] = identity.baseline().corpus_id
    if len(records) == 1:
        payload["intervention"] = records[0]
    else:
        payload["interventions"] = list(records)
    if interaction is not None:
        payload["interaction"] = interaction
    with (output_dir / "intervention.yaml").open("w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def _export_multi_source(
    dataframes: dict[str, pl.DataFrame],
    sources: list[SourceConfig],
    output_dir: Path,
    seed: int,
    entropy_records: list[dict],
    identity: CorpusIdentity,
    run_facts: dict,
    truth_dir: Path | None = None,
) -> None:
    """Export data split across multiple source directories.

    Each source gets its own subdirectory with per-source column/key transforms.
    A top-level ``sources.yaml`` indexes all sources.

    Every source carries the *same* corpus id: they are one corpus exported through
    different conventions, and a reconciliation scenario whose sources could not be
    shown to share an origin would be missing its own premise.
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
        export_dataframes(
            dataframes=src_dfs,
            output_dir=src_dir,
            entropy_records=None,  # Entropy map goes at top level
            identity=identity,
            run_facts={**run_facts, "source": src.name, "column_style": src.column_style},
            fmt=src.format,
            truth_dir=(truth_dir / src.name) if truth_dir else None,
        )

        source_index.append(
            {
                "name": src.name,
                "description": src.description,
                "directory": src.name,
                "tables": list(src_dfs.keys()),
                "column_style": src.column_style,
                "key_strategy": src.key_strategy,
                "format": src.format,
            }
        )

    # Write top-level sources.yaml
    index: dict = {"corpus": identity.as_dict()}
    if run_facts:
        index["run"] = run_facts
    index["sources"] = source_index
    with open(output_dir / "sources.yaml", "w") as f:
        yaml.dump(index, f, default_flow_style=False, sort_keys=False)

    # Write top-level entropy map — answer-key material, so it rides
    # with the truth when a truth_dir stands.
    entropy_data = {
        "corpus": identity.as_dict(),
        "injections": entropy_records or [],
        "total_injections": len(entropy_records) if entropy_records else 0,
    }
    if truth_dir is not None:
        truth_dir.mkdir(parents=True, exist_ok=True)
    with open((truth_dir or output_dir) / "entropy_map.yaml", "w") as f:
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
