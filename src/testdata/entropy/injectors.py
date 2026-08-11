"""Entropy injection functions — one per detector type.

Each injector mutates a Polars DataFrame in-place (returns modified copy)
and records what it did in the InjectionRegistry.
"""

from __future__ import annotations

import calendar
import random

import polars as pl

from .families import (
    FormulaDivergenceFamilyParams,
    ProbeColumn,
    REL_PARENT_TABLE,
    NullTokenFamilyParams,
    RelationshipFamilyParams,
    RelationshipPairSpec,
    SliceConditionalNullFamilyParams,
    StockFlowFamilyParams,
    apply_operation,
    mint_decoy,
    sample_formula_divergence_family,
    sample_mixed_units_family,
    sample_null_token_family,
    sample_relationship_family,
    sample_slice_conditional_null_family,
    sample_stock_flow_family,
    stock_flow_events_column,
)
from .registry import EntropyInjection, InjectionRegistry
from .scoping import scope_parameters, slice_rows


def _safe_series(name: str, values: list, existing_dtype: pl.DataType) -> pl.Series:
    """Create a Polars Series, coercing values to match existing column dtype.

    When a column has been converted to Utf8 by corrupt_types, subsequent
    injectors may produce float values. This helper converts them to strings
    to avoid mixed-type errors.
    """
    if existing_dtype == pl.Utf8:
        values = [str(v) if v is not None and not isinstance(v, str) else v for v in values]
    return pl.Series(name, values)


def corrupt_types(
    df: pl.DataFrame,
    col: str,
    ratio: float,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    severity: str = "medium",
) -> pl.DataFrame:
    """Replace numeric/date values with unparseable strings."""
    n = len(df)
    count = max(1, int(n * ratio))
    indices = rng.sample(range(n), min(count, n))
    garbage = ["N/A", "#ERR", "---", "null", "TBD", "", "see note", "??", "PENDING"]

    # The set is built ONCE. Spelling this `i in set(indices)` rebuilds it per row,
    # which is O(n x count) and invisible until the corpus grows: at 158k journal
    # lines it turned a 0.4s injection into 63s.
    chosen = set(indices)
    mask = pl.Series("mask", [i in chosen for i in range(n)])
    replacements = pl.Series("repl", [rng.choice(garbage) for _ in range(n)])
    new_col = pl.when(mask).then(replacements).otherwise(df[col].cast(pl.Utf8)).alias(col)
    df = df.with_columns(new_col)

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("TYPE"),
            target_file=f"{table_name}.csv",
            target_column=col,
            target_rows=sorted(indices),
            layer="value",
            defect="type_fidelity",
            defect_detail="type_corruption",
            injection_type="corrupt_type",
            parameters={"ratio": ratio, "garbage_values": garbage},
            severity=severity,
        )
    )
    return df


def inject_null_tokens(
    df: pl.DataFrame,
    col: str,
    ratio: float,
    tokens: list[str],
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    severity: str = "medium",
) -> pl.DataFrame:
    """Replace numeric values with null-marker sentinel TOKENS from a parameterized pool.

    Unlike ``corrupt_types``' fixed garbage list, the token pool is supplied by the
    strategy — so a consumer asserts recall on a *family* of variations (different
    tokens / rates / seeds), never a memorized fixture. The tokens fail
    the column's numeric cast → quarantine → the ``null_semantics`` adjudication
    pools quarantine / type / vocabulary witnesses per rejected token (is-null vs
    is-value). Use tokens NOT in the base null vocabulary so the vocabulary witness
    dissents and conflict fires (the novel-sentinel case); a ``null_value`` teach
    of these tokens then closes the loop (conflict → 0).
    """
    n = len(df)
    count = max(1, int(n * ratio))
    indices = rng.sample(range(n), min(count, n))

    # The set is built ONCE. Spelling this `i in set(indices)` rebuilds it per row,
    # which is O(n x count) and invisible until the corpus grows: at 158k journal
    # lines it turned a 0.4s injection into 63s.
    chosen = set(indices)
    mask = pl.Series("mask", [i in chosen for i in range(n)])
    replacements = pl.Series("repl", [rng.choice(tokens) for _ in range(n)])
    new_col = pl.when(mask).then(replacements).otherwise(df[col].cast(pl.Utf8)).alias(col)
    df = df.with_columns(new_col)

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("NULLTOK"),
            target_file=f"{table_name}.csv",
            target_column=col,
            target_rows=sorted(indices),
            layer="value",
            defect="null_encoding",
            defect_detail="null_marker",
            injection_type="inject_null_tokens",
            parameters={"ratio": ratio, "tokens": list(tokens)},
            severity=severity,
        )
    )
    return df


def inject_null_token_family(
    df: pl.DataFrame,
    col: str,
    seed: int,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,  # accepted for dispatch uniformity; the family uses its OWN seed
    n_markers: list[int] | None = None,
    marker_ratio: list[float] | None = None,
    decoy_ratio: list[float] | None = None,
    vocab_coverage: list[float] | None = None,
    decoy_style: str | None = None,
    severity: str = "medium",
) -> pl.DataFrame:
    """Inject a SAMPLED null_tokens family — two labelled classes.

    The generative successor to ``inject_null_tokens``: instead of a fixed token
    list, a recorded ``seed`` samples a family instance (markers + decoys + rates;
    see :mod:`testdata.entropy.families`). Different seeds → different surface,
    same semantics; the recorded seed reproduces exactly (AC1) and is independent
    of injection order (its own RNG, not the shared stream).

    Injects two ground-truth-labelled classes into ``col``:

    * **markers** (``is-null``) — a small set of sentinel shapes, each repeated
      across many rows so they cluster in the quarantine.
    * **decoys** (``is-value``) — genuine amounts (locale/currency/unit/annotated),
      distinct per row so they smear.

    Both fail the numeric cast → quarantine → the ``null_semantics`` adjudication
    pools its witnesses per token. The recorded ``parameters`` carry the marker
    set, the minted decoys, and their row sets — the labels the calibration rig
    scores each witness against.
    """
    overrides: dict[str, object] = {}
    if n_markers is not None:
        overrides["n_markers"] = tuple(n_markers)
    if marker_ratio is not None:
        overrides["marker_ratio"] = tuple(marker_ratio)
    if decoy_ratio is not None:
        overrides["decoy_ratio"] = tuple(decoy_ratio)
    if vocab_coverage is not None:
        overrides["vocab_coverage"] = tuple(vocab_coverage)
    if decoy_style is not None:
        overrides["decoy_style"] = decoy_style
    sample = sample_null_token_family(seed, NullTokenFamilyParams(**overrides))  # type: ignore[arg-type]

    n = len(df)
    place = random.Random(f"null_token_family:{col}:{seed}")  # seed-derived, order-independent
    marker_count = max(1, int(n * sample.marker_ratio))
    decoy_count = int(n * sample.decoy_ratio)
    chosen = place.sample(range(n), min(marker_count + decoy_count, n))
    marker_rows = chosen[:marker_count]
    decoy_rows = chosen[marker_count:]

    new_values = df[col].cast(pl.Utf8).to_list()
    for i in marker_rows:
        new_values[i] = place.choice(sample.markers)
    minted_decoys: list[str] = []
    for i in decoy_rows:
        decoy = mint_decoy(place, sample.decoy_style)
        minted_decoys.append(decoy)
        new_values[i] = decoy
    df = df.with_columns(pl.Series(col, new_values))

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("NULLFAM"),
            target_file=f"{table_name}.csv",
            target_column=col,
            target_rows=sorted(marker_rows + decoy_rows),
            layer="value",
            defect="null_encoding",
            defect_detail="null_marker",
            injection_type="inject_null_token_family",
            parameters={
                "family": "null_tokens",
                "seed": seed,
                "markers": list(sample.markers),
                "in_vocab_markers": list(sample.in_vocab_markers),
                "vocab_coverage": sample.vocab_coverage,
                "decoy_style": sample.decoy_style,
                "decoys": sorted(set(minted_decoys)),
                "marker_ratio": sample.marker_ratio,
                "decoy_ratio": sample.decoy_ratio,
                "marker_rows": sorted(marker_rows),
                "decoy_rows": sorted(decoy_rows),
            },
            severity=severity,
        )
    )
    return df


def inject_scale_mix(
    df: pl.DataFrame,
    col: str,
    seed: int,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,  # accepted for dispatch uniformity; the family uses its OWN seed
    severity: str = "medium",
) -> pl.DataFrame:
    """Push a SAMPLED fraction of a numeric column to a different SCALE.

    The generative mixed-units family: a recorded ``seed`` samples a scale factor (a
    power of ten — kEUR among EUR) and a mix ratio, then multiplies that fraction of
    the column's values by the factor. The values now span two log-magnitude modes
    under one (still-declared) unit — the conflict ``unit_consistency`` adjudicates.
    Deliberately a SCALE mix, not a ×1.1 currency mix (undetectable from values).
    """
    sample = sample_mixed_units_family(seed)
    n = len(df)
    place = random.Random(f"scale_mix:{col}:{seed}")  # seed-derived, order-independent
    count = max(1, int(n * sample.mix_ratio))
    rows = place.sample(range(n), min(count, n))

    values = df[col].to_list()
    for i in rows:
        if values[i] is not None:
            values[i] = round(float(values[i]) * sample.scale_factor, 2)
    df = df.with_columns(_safe_series(col, values, df[col].dtype))

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("SCALEMIX"),
            target_file=f"{table_name}.csv",
            target_column=col,
            target_rows=sorted(rows),
            layer="semantic",
            defect="unit_consistency",
            defect_detail="unit_consistency",
            injection_type="inject_scale_mix",
            parameters={
                "seed": seed,
                "scale_factor": sample.scale_factor,
                "mix_ratio": sample.mix_ratio,
            },
            severity=severity,
        )
    )
    return df


def introduce_nulls(
    df: pl.DataFrame,
    col: str,
    ratio: float,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    severity: str = "medium",
) -> pl.DataFrame:
    """Set values to null at the given ratio."""
    n = len(df)
    count = max(1, int(n * ratio))
    indices = rng.sample(range(n), min(count, n))

    # The set is built ONCE. Spelling this `i in set(indices)` rebuilds it per row,
    # which is O(n x count) and invisible until the corpus grows: at 158k journal
    # lines it turned a 0.4s injection into 63s.
    chosen = set(indices)
    mask = pl.Series("mask", [i in chosen for i in range(n)])
    new_col = pl.when(mask).then(pl.lit(None)).otherwise(df[col]).alias(col)
    df = df.with_columns(new_col)

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("NULL"),
            target_file=f"{table_name}.csv",
            target_column=col,
            target_rows=sorted(indices),
            layer="value",
            defect="completeness",
            defect_detail="null_injection",
            injection_type="introduce_nulls",
            parameters={"ratio": ratio},
            severity=severity,
        )
    )
    return df


def inject_slice_conditional_null(
    df: pl.DataFrame,
    col: str,
    slice_col: str,
    seed: int,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,  # accepted for dispatch uniformity; the family uses its OWN seed
    n_affected: list[int] | None = None,
    conditional_rate: list[float] | None = None,
    base_rate: list[float] | None = None,
    min_affected_mass: float | None = None,
    severity: str = "medium",
) -> pl.DataFrame:
    """Inject a SAMPLED slice_conditional_null family — concentrated missingness.

    Nulls in ``col`` CONCENTRATED in 1-2 slices of the categorical ``slice_col`` (a measure
    a cost center failed to import), with a low base rate everywhere else — so the overall
    null_ratio stays mild and only the concentration statistic (bias-corrected Cramér's V of
    ``col IS NULL`` × ``slice_col``) fires. A recorded ``seed`` samples the missingness shape
    (see :mod:`testdata.entropy.families`); the concrete affected slice VALUES are resolved
    here from the data's per-slice masses (seeded, order-independent) so the recorded
    ``parameters`` carry the ``(col, slice_col)`` pair + the affected values — the label the
    ``slice_conditional_null`` detector's recall asserts and a ``document_business_rule``
    teach on that pair closes.
    """
    overrides: dict[str, object] = {}
    if n_affected is not None:
        overrides["n_affected"] = tuple(n_affected)
    if conditional_rate is not None:
        overrides["conditional_rate"] = tuple(conditional_rate)
    if base_rate is not None:
        overrides["base_rate"] = tuple(base_rate)
    if min_affected_mass is not None:
        overrides["min_affected_mass"] = min_affected_mass
    sample = sample_slice_conditional_null_family(
        seed, SliceConditionalNullFamilyParams(**overrides)  # type: ignore[arg-type]
    )

    place = random.Random(f"slice_conditional_null:{col}:{slice_col}:{seed}")
    slice_values = df[slice_col].to_list()
    n = len(df)

    # Per-slice row mass over LABELLED rows (a null slice is no slice — the detector
    # conditions only on rows that carry a label, so they can never be "affected").
    labelled = [i for i in range(n) if slice_values[i] is not None]
    counts: dict[object, int] = {}
    for i in labelled:
        counts[slice_values[i]] = counts.get(slice_values[i], 0) + 1
    # Stable, seed-shuffled candidate order so the choice is deterministic but not
    # alphabetical (no bias toward any particular center across seeds).
    candidates = sorted(counts, key=lambda v: str(v))
    place.shuffle(candidates)

    affected: list[object] = []
    for _ in range(50):
        pick = place.sample(candidates, min(sample.n_affected, len(candidates)))
        if sum(counts[v] for v in pick) / n >= sample.min_affected_mass:
            affected = pick
            break
    if not affected:  # fall back to the largest slices — guarantees the mass floor
        affected = sorted(counts, key=lambda v: counts[v], reverse=True)[: sample.n_affected]
    affected_set = set(affected)

    values = df[col].to_list()
    nulled: list[int] = []
    for i in range(n):
        sv = slice_values[i]
        rate = sample.conditional_rate if sv in affected_set else sample.base_rate
        if values[i] is not None and place.random() < rate:
            values[i] = None
            nulled.append(i)
    df = df.with_columns(_safe_series(col, values, df[col].dtype))

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("SLICENULL"),
            target_file=f"{table_name}.csv",
            target_column=col,
            target_rows=sorted(nulled),
            layer="value",
            defect="completeness",
            defect_detail="slice_conditional_null",
            injection_type="inject_slice_conditional_null",
            parameters={
                "family": "slice_conditional_null",
                "seed": seed,
                "slice_column": slice_col,
                "affected_slices": sorted(str(v) for v in affected),
                "n_affected": sample.n_affected,
                "conditional_rate": sample.conditional_rate,
                "base_rate": sample.base_rate,
                "min_affected_mass": sample.min_affected_mass,
                "overall_null_ratio": round(len(nulled) / n, 4) if n else 0.0,
            },
            severity=severity,
        )
    )
    return df


def inject_outliers(
    df: pl.DataFrame,
    col: str,
    ratio: float,
    factor: float,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    severity: str = "medium",
) -> pl.DataFrame:
    """Insert values beyond IQR fences (multiply by factor)."""
    n = len(df)
    count = max(1, int(n * ratio))
    indices = rng.sample(range(n), min(count, n))

    values = df[col].to_list()
    actual_affected = []
    for i in indices:
        if values[i] is not None:
            try:
                val = float(values[i])
            except ValueError, TypeError:
                continue
            values[i] = val * factor * rng.choice([1, -1]) if val != 0 else factor * 1000
            actual_affected.append(i)

    df = df.with_columns(_safe_series(col, values, df[col].dtype))

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("OUTLIER"),
            target_file=f"{table_name}.csv",
            target_column=col,
            target_rows=sorted(actual_affected),
            layer="value",
            defect="distribution",
            defect_detail="outlier_injection",
            injection_type="inject_outliers",
            parameters={"ratio": ratio, "factor": factor},
            severity=severity,
        )
    )
    return df


def break_benford(
    df: pl.DataFrame,
    col: str,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    method: str = "round_numbers",
    severity: str = "medium",
) -> pl.DataFrame:
    """Replace amount distribution with non-Benford pattern."""
    n = len(df)
    values = df[col].to_list()
    affected = []

    for i in range(n):
        if values[i] is None:
            continue
        val = abs(float(values[i]))
        if val < 1:
            continue

        if method == "round_numbers":
            # Replace with round numbers (100, 500, 1000, 5000, etc.)
            if rng.random() < 0.6:
                magnitude = 10 ** rng.randint(2, 4)
                multiplier = rng.choice([1, 2, 5, 10])
                new_val = float(magnitude * multiplier)
                sign = 1 if float(values[i]) >= 0 else -1
                values[i] = new_val * sign
                affected.append(i)
        elif method == "uniform":
            if rng.random() < 0.6:
                sign = 1 if float(values[i]) >= 0 else -1
                values[i] = round(rng.uniform(100, 50000), 2) * sign
                affected.append(i)

    df = df.with_columns(_safe_series(col, values, df[col].dtype))

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("BENFORD"),
            target_file=f"{table_name}.csv",
            target_column=col,
            target_rows=sorted(affected),
            layer="value",
            defect="distribution",
            defect_detail="benford_violation",
            injection_type="break_benford",
            parameters={"method": method},
            severity=severity,
        )
    )
    return df


def obscure_column_names(
    df: pl.DataFrame,
    mapping: dict[str, str],
    registry: InjectionRegistry,
    table_name: str,
    severity: str = "medium",
) -> pl.DataFrame:
    """Rename columns to cryptic names."""
    df = df.rename(mapping)

    for original, obscured in mapping.items():
        registry.record(
            EntropyInjection(
                injection_id=registry.next_id("NAME"),
                target_file=f"{table_name}.csv",
                target_column=obscured,
                target_rows=[],  # All rows affected (schema change)
                layer="semantic",
                defect="business_meaning",
                defect_detail="column_name_obscured",
                injection_type="obscure_column_names",
                parameters={"original": original, "obscured": obscured},
                severity=severity,
            )
        )
    return df


def mix_units(
    df: pl.DataFrame,
    col: str,
    alt_currency: str,
    ratio: float,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    fx_rate: float = 1.1,
    severity: str = "medium",
    unit_col: str | None = None,
    dataframes: dict[str, pl.DataFrame] | None = None,
    scope: dict[str, list[str]] | None = None,
    periods: list[str] | None = None,
) -> pl.DataFrame:
    """Mix currencies — convert some values at FX rate, declared or undeclared.

    Two variants of one family:

    * **undeclared** (default, ``unit_col=None``): values converted, the unit column
      untouched — the mixing is invisible to any reader of the DECLARED unit surface
      (the original shape; a unit-column gate must stay silent on it).
    * **declared** (``unit_col`` given): the converted rows also write
      ``alt_currency`` into ``unit_col`` — a genuinely multi-currency table (the same
      economic amounts recorded in another currency, honestly declared). This is the
      shape the cross-unit drill gate must flag, and the data-derived
      ``measured_in.cross_unit`` truth flips True from it.

    ``scope`` and ``periods`` confine it to a slice — the **unit-mix artifact**: a
    named slice of a named period whose measure is silently scaled by a constant. It
    is one half of the confusable pair, the other being a price ``rate`` lever on the
    same slice. The two are meant to be separable only by reasoning about whether the
    business changed, so this records the analytic effect on the column total
    (below), which is what lets the pair be tuned to the same apparent magnitude.
    """
    n = len(df)
    in_scope = (
        slice_rows(df, table_name, dataframes, scope=scope, periods=periods)
        if (scope or periods) and dataframes is not None
        else list(range(n))
    )
    count = max(1, int(len(in_scope) * ratio)) if in_scope else 0
    indices = rng.sample(in_scope, min(count, len(in_scope)))

    values = df[col].to_list()
    original = [float(v) if v is not None else 0.0 for v in values]
    converted: list[int] = []
    for i in indices:
        if values[i] is not None:
            values[i] = round(float(values[i]) * fx_rate, 2)
            converted.append(i)

    df = df.with_columns(_safe_series(col, values, df[col].dtype))

    if unit_col is not None:
        units = df[unit_col].to_list()
        for i in converted:
            units[i] = alt_currency
        df = df.with_columns(_safe_series(unit_col, units, df[unit_col].dtype))

    # The analytic effect on the column total. Exact, not estimated: the artifact is
    # a constant scaling of a known set of rows, so the delta is closed-form. This is
    # the number the confusable pair is tuned on — ground truth is computed BEFORE
    # injection by design, so a baseline/injected pair yields no metric delta and
    # there would otherwise be nothing to match a lever's magnitude against.
    affected_mass = sum(original[i] for i in converted)
    column_total = sum(original)
    delta = affected_mass * (fx_rate - 1.0)

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("UNIT"),
            target_file=f"{table_name}.csv",
            target_column=col,
            target_rows=sorted(indices),
            layer="semantic",
            defect="unit_consistency",
            defect_detail="declared_mixed_currency" if unit_col else "mixed_currency",
            injection_type="mix_units",
            parameters={
                "alt_currency": alt_currency,
                "ratio": ratio,
                "fx_rate": fx_rate,
                "unit_col": unit_col,
                **scope_parameters(scope, periods, len(in_scope), n),
                "affected_mass": round(affected_mass, 2),
                "column_total_before": round(column_total, 2),
                "apparent_delta": round(delta, 2),
                "apparent_delta_pct": round(delta / column_total * 100, 4) if column_total else 0.0,
            },
            severity=severity,
        )
    )
    return df


def inject_role_playing_fks(
    df: pl.DataFrame,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    dataframes: dict[str, pl.DataFrame],
    seed: int | None = None,
    same_addr_ratio: float = 0.3,
    severity: str = "low",
) -> pl.DataFrame:
    """Fill the role-playing-FK columns on the role-play probe shape.

    A LABELLED-CLEAN structural family, not a defect: one dimension (addresses),
    one fact carrying TWO FK roles to it (orders.bill_to_addr / ship_to_addr —
    also the two-FKs-one-table-pair shape), and a second fact sharing the
    ship_to role under a DIFFERENT column name (deliveries.delivery_addr — the
    judge's conform path). Role consistency is BY CONSTRUCTION: every delivery's
    delivery_addr equals its parent order's ship_to_addr, so the lineage witness
    has a data-level same-role signal, and bill_to must never pair with it.

    ``same_addr_ratio`` of orders bill and ship to the SAME address (realistic
    overlap that a naive value-overlap pairing would seize on — the trap leg).
    Records one ``stratum="genuine_clean"`` registry entry per role column
    (precision material, never a recall target — the calibration-negative
    convention). Truth: the export's conditional ``fk_roles`` + relationships
    sections (metadata_truth), emitted only when the shape is present.
    """
    if table_name != "orders":
        raise ValueError("inject_role_playing_fks targets the 'orders' probe table")
    local_rng = random.Random(seed) if seed is not None else rng
    address_ids = dataframes["addresses"]["address_id"].to_list()

    ship_to = [local_rng.choice(address_ids) for _ in range(len(df))]
    bill_to = [
        ship if local_rng.random() < same_addr_ratio else local_rng.choice(address_ids)
        for ship in ship_to
    ]
    df = df.with_columns(
        pl.Series("bill_to_addr", bill_to),
        pl.Series("ship_to_addr", ship_to),
    )

    # deliveries.delivery_addr = the parent order's ship_to (same role, different name)
    ship_by_order = dict(zip(df["order_id"].to_list(), ship_to, strict=True))
    deliveries = dataframes["deliveries"]
    delivery_addr = [ship_by_order[oid] for oid in deliveries["order_id"].to_list()]
    dataframes["deliveries"] = deliveries.with_columns(pl.Series("delivery_addr", delivery_addr))

    roles = {
        ("orders", "bill_to_addr"): "bill_to",
        ("orders", "ship_to_addr"): "ship_to",
        ("deliveries", "delivery_addr"): "ship_to",
    }
    for (tbl, col), role in roles.items():
        registry.record(
            EntropyInjection(
                injection_id=registry.next_id("ROLEFK"),
                target_file=f"{tbl}.csv",
                target_column=col,
                target_rows=[],  # labelled-clean: nothing corrupted
                layer="structural",
                defect="relationships",
                defect_detail="role_playing_fks",
                injection_type="inject_role_playing_fks",
                parameters={
                    "family": "role_playing_fks",
                    "stratum": "genuine_clean",
                    "seed": seed,
                    "fk_role": role,
                    "referenced_table": "addresses" if col != "order_id" else "orders",
                    "same_addr_ratio": same_addr_ratio,
                    "n_addresses": len(address_ids),
                    "n_orders": len(df),
                    "n_deliveries": len(dataframes["deliveries"]),
                },
                severity=severity,
            )
        )
    return df


def corrupt_dates(
    df: pl.DataFrame,
    col: str,
    formats: list[str],
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    severity: str = "medium",
) -> pl.DataFrame:
    """Store dates as ambiguous string formats (e.g., MM/DD/YYYY vs DD/MM/YYYY)."""
    from datetime import date as date_type

    n = len(df)
    values = df[col].cast(pl.Utf8).to_list()
    affected = []

    for i in range(n):
        if values[i] is None:
            continue
        try:
            # Parse the ISO date
            d = date_type.fromisoformat(values[i])
        except ValueError, TypeError:
            continue

        fmt = rng.choice(formats)
        if fmt == "MM/DD/YYYY":
            values[i] = f"{d.month:02d}/{d.day:02d}/{d.year}"
        elif fmt == "DD/MM/YYYY":
            values[i] = f"{d.day:02d}/{d.month:02d}/{d.year}"
        elif fmt == "DD-Mon-YY":
            values[i] = d.strftime("%d-%b-%y")
        elif fmt == "YYYYMMDD":
            values[i] = d.strftime("%Y%m%d")
        elif fmt == "Mon DD, YYYY":
            values[i] = d.strftime("%b %d, %Y")
        elif fmt == "epoch":
            import calendar

            values[i] = str(calendar.timegm(d.timetuple()))
        else:
            values[i] = d.isoformat()
        affected.append(i)

    df = df.with_columns(_safe_series(col, values, df[col].dtype))

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("DATE"),
            target_file=f"{table_name}.csv",
            target_column=col,
            target_rows=sorted(affected),
            layer="structural",
            defect="format_consistency",
            defect_detail="ambiguous_dates",
            injection_type="corrupt_dates",
            parameters={"formats": formats},
            severity=severity,
        )
    )
    return df


_FK_VARIANTS = ("distinct_orphan", "repeated_orphan", "shuffled")

_FK_DETAIL = {
    "distinct_orphan": "orphaned_foreign_keys",
    "repeated_orphan": "repeated_orphan_foreign_key",
    "shuffled": "shuffled_foreign_keys",
}


def _banded_rows(rows: list[int], df: pl.DataFrame, within_col: str | None, bands: int) -> list[list[int]]:
    """Split rows into ``bands`` groups of similar ``within_col`` magnitude.

    A shuffle confined to a band keeps the swapped rows' amounts comparable, so the
    amount disagreement it creates is small rather than arbitrary. It does not
    ELIMINATE that disagreement — for that, see :func:`_exact_groups`.
    """
    if within_col is None or bands <= 1:
        return [rows]
    values = df[within_col].to_list()

    def magnitude(i: int) -> tuple[int, float]:
        try:
            return (0, float(values[i]))
        except (TypeError, ValueError):  # unparseable sorts last, in its own band
            return (1, 0.0)

    ordered = sorted(rows, key=magnitude)
    size = max(1, -(-len(ordered) // bands))  # ceil, so `bands` groups at most
    return [ordered[i : i + size] for i in range(0, len(ordered), size)]


def _exact_groups(
    df: pl.DataFrame, in_scope: list[int], within_col: str, target: int, rng: random.Random
) -> tuple[list[list[int]], int]:
    """Groups of rows sharing an EXACT ``within_col`` value, until ``target`` rows are covered.

    Rows that swap keys with a row owing the identical amount leave every amount
    agreement intact to the cent — the reconciliation still balances, the payment
    still equals its invoice, and no numeric check anywhere sees anything. Only the
    non-numeric consequences of the pairing remain: the payment now sits against
    another vendor's invoice, on a date that no longer follows from it.

    Whole groups are taken rather than sampled rows, because a sample would mostly
    land on values that occur once and those rows have nobody to trade with. Returns
    the taken groups and the ELIGIBLE row count — the real denominator, since only
    rows with a peer can take part, and a consumer comparing ``ratio`` against the
    table would otherwise be comparing against the wrong population.
    """
    values = df[within_col].to_list()
    by_value: dict[str, list[int]] = {}
    for i in in_scope:
        by_value.setdefault(str(values[i]), []).append(i)

    peers = sorted((rows for rows in by_value.values() if len(rows) > 1), key=lambda rows: rows[0])
    eligible = sum(len(rows) for rows in peers)
    rng.shuffle(peers)

    taken: list[list[int]] = []
    covered = 0
    for rows in peers:
        if covered >= target:
            break
        taken.append(rows)
        covered += len(rows)
    return taken, eligible


def _derange(group: list[int], rng: random.Random) -> list[int]:
    """A random cyclic permutation of ``group`` — no element keeps its position.

    Sattolo's algorithm: guaranteed a derangement in one pass, so every sampled row
    genuinely receives another row's key rather than rejection-sampling a shuffle.
    """
    perm = list(group)
    for i in range(len(perm) - 1, 0, -1):
        j = rng.randrange(i)  # strictly below i — this is what makes it cyclic
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def break_referential_integrity(
    df: pl.DataFrame,
    fk_col: str,
    ratio: float,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    severity: str = "high",
    variant: str = "distinct_orphan",
    within_col: str | None = None,
    bands: int | None = 4,
    dataframes: dict[str, pl.DataFrame] | None = None,
    scope: dict[str, list[str]] | None = None,
    periods: list[str] | None = None,
) -> pl.DataFrame:
    """Break the child→parent relationship, in one of three shapes.

    All three break the same thing — a row points at a parent it should not — but they
    look different enough on the surface that a detector can pass one and fail the
    others. The point of having all three is to make "does the FK resolve" the only
    test that catches every one:

    * ``distinct_orphan`` (default, the original) — each broken row gets its OWN
      invented id. The orphan values are unique, so they also look like a burst of
      rare categories, and a cardinality or frequency heuristic catches them without
      ever joining.
    * ``repeated_orphan`` — ONE invented id reused across every broken row. Now the
      orphan is a *common* value, the single most frequent key in the column. Every
      rare-value heuristic is silent on it and only the join says anything.
    * ``shuffled`` — no invented ids at all: the sampled rows trade keys among
      themselves. Every value still exists in the parent, the column's value
      distribution is untouched, and the FK "resolves" on every row — it just resolves
      to the wrong parent. Nothing short of checking what the pairing MEANS finds it.

    ``within_col`` decides how much of the amount surface the shuffle disturbs, which
    matters because shuffling ``payments.invoice_id`` blind also breaks payment↔invoice
    amount agreement — and then a detector catches it on the amounts, never looking at
    the relationship at all. The same shortcut in a new coat. Three settings:

    * ``within_col=None`` — free shuffle. Amounts disagree arbitrarily.
    * ``within_col`` + ``bands=N`` — swap inside a magnitude band. Disagreements are
      small but present; in a corpus where payments settle their invoices exactly,
      "small but present" is still an exact-match failure.
    * ``within_col`` + ``bands=None`` — swap only among rows owing the IDENTICAL
      amount. Every amount check passes to the cent and the relationship is the one
      and only thing that is wrong. This is the honest hard case, and ``ratio`` then
      counts against the rows that HAVE a peer (recorded as ``eligible_rows``), since
      no other row can take part.

    ``scope``/``periods`` confine the breakage to a named slice, the shape a single
    bad feed produces. Absent, the whole table is in play and the record is exactly
    what it always was.

    The shuffled variant reuses values already in the column, so it assumes the column
    still resolves when it runs — order it before any orphan-injecting spec on the
    same column, not after.
    """
    if variant not in _FK_VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (supported: {list(_FK_VARIANTS)})")

    n = len(df)
    in_scope = (
        slice_rows(df, table_name, dataframes, scope=scope, periods=periods)
        if (scope or periods) and dataframes is not None
        else list(range(n))
    )
    count = max(1, int(len(in_scope) * ratio)) if in_scope else 0
    indices = rng.sample(in_scope, min(count, len(in_scope)))

    values = df[fk_col].to_list()
    original = list(values)
    detail: dict[str, object] = {}

    if variant == "distinct_orphan":
        for i in indices:
            values[i] = f"ORPHAN-{rng.randint(900000, 999999)}"
        changed = list(indices)
    elif variant == "repeated_orphan":
        orphan = f"ORPHAN-{rng.randint(900000, 999999)}"  # drawn ONCE, that is the point
        for i in indices:
            values[i] = orphan
        changed = list(indices)
        detail = {"orphan_value": orphan, "distinct_orphans": 1}
    else:
        eligible: int | None = None
        if within_col is not None and bands is None:
            groups, eligible = _exact_groups(df, in_scope, within_col, count, rng)
        else:
            groups = _banded_rows(sorted(indices), df, within_col, bands or 1)

        changed = []
        for group in groups:
            if len(group) < 2:  # a lone row has nobody to trade with
                continue
            for row, source in zip(group, _derange(group, rng), strict=True):
                # Deranged POSITIONS still coincide in VALUE when two rows already
                # pointed at the same parent — those rows are unchanged, so they are
                # not in the answer key.
                if original[source] != original[row]:
                    values[row] = original[source]
                    changed.append(row)
        detail = {
            "within_col": within_col,
            "grouping": "exact_value" if eligible is not None else ("band" if within_col else "free"),
            "bands": bands if (within_col and bands) else None,
            # The defining property, and the reason this variant exists: no heuristic
            # over the column's own values can see it.
            "all_values_exist_in_parent": True,
            "correct_values": [str(original[i]) for i in sorted(changed)],
        }
        if eligible is not None:
            detail["eligible_rows"] = eligible

    df = df.with_columns(pl.Series(fk_col, values))

    parameters: dict[str, object] = {"ratio": ratio}
    if variant != "distinct_orphan":
        parameters["variant"] = variant
    parameters.update(detail)
    parameters.update(scope_parameters(scope, periods, len(in_scope), n))

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("FK"),
            target_file=f"{table_name}.csv",
            target_column=fk_col,
            target_rows=sorted(changed),
            layer="structural",
            defect="referential_integrity",
            defect_detail=_FK_DETAIL[variant],
            injection_type="break_referential_integrity",
            parameters=parameters,
            severity=severity,
        )
    )
    return df


def add_duplicate_fk_paths(
    df: pl.DataFrame,
    existing_fk_col: str,
    new_col_name: str,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    noise_ratio: float = 0.05,
    severity: str = "medium",
    dataframes: dict[str, pl.DataFrame] | None = None,
    scope: dict[str, list[str]] | None = None,
    periods: list[str] | None = None,
) -> pl.DataFrame:
    """Add a redundant FK column with slight noise.

    ``scope``/``periods`` confine the DIVERGENCE, not the column: the second path
    exists on every row (an alternative join path that appeared on half a table would
    be a different, more obvious defect), and only in-scope rows disagree with it. So
    a consumer that picks the wrong path gets a wrong answer for exactly one named
    slice, which is the shape a metric's lineage actually breaks in.
    """
    values = df[existing_fk_col].to_list()
    n = len(values)
    new_values = list(values)  # copy

    in_scope = (
        slice_rows(df, table_name, dataframes, scope=scope, periods=periods)
        if (scope or periods) and dataframes is not None
        else list(range(n))
    )
    noise_count = max(1, int(len(in_scope) * noise_ratio)) if in_scope else 0
    noise_indices = rng.sample(in_scope, min(noise_count, len(in_scope)))
    for i in noise_indices:
        new_values[i] = f"ALT-{rng.randint(100000, 999999)}"

    df = df.with_columns(pl.Series(new_col_name, new_values))

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("DUPFK"),
            target_file=f"{table_name}.csv",
            target_column=new_col_name,
            target_rows=sorted(noise_indices),
            layer="structural",
            defect="join_paths",
            defect_detail="duplicate_fk_path",
            injection_type="add_duplicate_fk_paths",
            parameters={
                "source_col": existing_fk_col,
                "noise_ratio": noise_ratio,
                **scope_parameters(scope, periods, len(in_scope), n),
            },
            severity=severity,
        )
    )
    return df


def drop_slice_rows(
    df: pl.DataFrame,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    dataframes: dict[str, pl.DataFrame] | None = None,
    scope: dict[str, list[str]] | None = None,
    periods: list[str] | None = None,
    ratio: float = 1.0,
    severity: str = "high",
) -> pl.DataFrame:
    """A delivery gap — a slice's rows simply absent for a period.

    The defect a late or failed feed actually produces: not corrupted values but
    missing ones, with nothing on the row to say so. A metric computed over the
    period is wrong and internally consistent, which is what makes it hard.

    **This is the one injector that changes the row count, and that changes what its
    truth record means.** Every other injector mutates a column in place, so
    ``target_rows`` are positions a consumer can look up in the exported file. Here
    the rows are gone: positional indices would point at whichever rows shifted up
    into them, naming innocent rows as the defect. So ``target_rows`` is empty and
    the record carries ``removed_keys`` — the identifying values that are no longer
    there — plus the row count before and after. A key survives deletion; an index
    does not.

    ``ratio`` below 1.0 thins the slice rather than removing it, for the partial-feed
    case; the default is the whole slice.
    """
    n = df.height
    in_scope = (
        slice_rows(df, table_name, dataframes, scope=scope, periods=periods)
        if (scope or periods) and dataframes is not None
        else list(range(n))
    )
    if not in_scope:
        raise ValueError(f"drop_slice_rows matched no rows in {table_name!r}: nothing to remove")

    count = len(in_scope) if ratio >= 1.0 else max(1, int(len(in_scope) * ratio))
    removed = sorted(rng.sample(in_scope, min(count, len(in_scope))))

    key_col = df.columns[0]  # the identifying column by table convention
    removed_keys = [df[key_col][i] for i in removed]

    keep = [i for i in range(n) if i not in set(removed)]
    df = df[keep]

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("GAP"),
            target_file=f"{table_name}.csv",
            target_column=key_col,
            # Deliberately empty: see the docstring. The rows this names are gone, so
            # an index would point at an innocent survivor.
            target_rows=[],
            layer="structural",
            defect="completeness",
            defect_detail="delivery_gap",
            injection_type="drop_slice_rows",
            parameters={
                "key_column": key_col,
                "removed_keys": [str(k) for k in removed_keys],
                "removed_count": len(removed),
                "rows_before": n,
                "rows_after": df.height,
                "ratio": ratio,
                **scope_parameters(scope, periods, len(in_scope), n),
            },
            severity=severity,
        )
    )
    return df


def drift_formula(
    df: pl.DataFrame,
    derived_col: str,
    source_cols: list[str],
    error_ratio: float,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    severity: str = "medium",
) -> pl.DataFrame:
    """Introduce rounding/computation errors in derived columns."""
    n = len(df)
    count = max(1, int(n * error_ratio))
    indices = rng.sample(range(n), min(count, n))

    values = df[derived_col].to_list()
    for i in indices:
        if values[i] is not None:
            val = float(values[i])
            # Small rounding error (±0.01 to ±1.0)
            error = rng.uniform(0.01, 1.0) * rng.choice([1, -1])
            values[i] = round(val + error, 2)

    df = df.with_columns(_safe_series(derived_col, values, df[derived_col].dtype))

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("DRIFT"),
            target_file=f"{table_name}.csv",
            target_column=derived_col,
            target_rows=sorted(indices),
            layer="computational",
            defect="derived_consistency",
            defect_detail="formula_drift",
            injection_type="drift_formula",
            parameters={"source_cols": source_cols, "error_ratio": error_ratio},
            severity=severity,
        )
    )
    return df


def inject_temporal_drift(
    df: pl.DataFrame,
    value_col: str,
    time_col: str,
    shift_date: str,
    shift_factor: float,
    registry: InjectionRegistry,
    table_name: str,
    severity: str = "medium",
) -> pl.DataFrame:
    """Shift distribution of a value column after a certain date."""
    from datetime import date as date_type

    cutoff = date_type.fromisoformat(shift_date)
    time_values = df[time_col].cast(pl.Utf8).to_list()
    val_values = df[value_col].to_list()
    affected = []

    for i in range(len(df)):
        if time_values[i] is None or val_values[i] is None:
            continue
        try:
            d = date_type.fromisoformat(time_values[i])
        except ValueError, TypeError:
            continue
        if d >= cutoff:
            val_values[i] = round(float(val_values[i]) * shift_factor, 2)
            affected.append(i)

    df = df.with_columns(_safe_series(value_col, val_values, df[value_col].dtype))

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("TDRIFT"),
            target_file=f"{table_name}.csv",
            target_column=value_col,
            target_rows=sorted(affected),
            layer="value",
            defect="temporal_stability",
            defect_detail="distribution_shift",
            injection_type="inject_temporal_drift",
            parameters={"shift_date": shift_date, "shift_factor": shift_factor},
            severity=severity,
        )
    )
    return df


# --- Cross-table relationship injectors ---
# These corrupt ONE side of a cross-table relationship.
# The other table retains the original value, creating a detectable mismatch.


def break_gl_invoice_match(
    df: pl.DataFrame,
    col: str,
    ratio: float,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    factor_range: tuple[float, float] = (0.8, 1.3),
    severity: str = "medium",
) -> pl.DataFrame:
    """Scale invoice amounts so they no longer match corresponding GL entries.

    Operates on the invoices table. The matching journal entry (DR Expense,
    CR AP) retains the original amount, creating a cross-table mismatch
    that a relationship detector should catch.
    """
    n = len(df)
    count = max(1, int(n * ratio))
    indices = rng.sample(range(n), min(count, n))

    values = df[col].to_list()
    actual_affected = []
    for i in indices:
        if values[i] is not None:
            try:
                val = float(values[i])
            except ValueError, TypeError:
                continue
            factor = rng.uniform(*factor_range)
            # Avoid factor ≈ 1.0 (undetectable)
            if abs(factor - 1.0) < 0.05:
                factor = factor_range[1]
            values[i] = round(val * factor, 2)
            actual_affected.append(i)

    df = df.with_columns(_safe_series(col, values, df[col].dtype))

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("XMATCH"),
            target_file=f"{table_name}.csv",
            target_column=col,
            target_rows=sorted(actual_affected),
            layer="structural",
            defect="cross_table_consistency",
            defect_detail="gl_invoice_mismatch",
            injection_type="break_gl_invoice_match",
            parameters={"ratio": ratio, "factor_range": list(factor_range)},
            severity=severity,
        )
    )
    return df


def break_payment_bank_match(
    df: pl.DataFrame,
    col: str,
    ratio: float,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    factor_range: tuple[float, float] = (0.9, 1.2),
    severity: str = "medium",
) -> pl.DataFrame:
    """Scale payment amounts so they no longer match corresponding bank transactions.

    Operates on the payments table. The matching bank transaction retains
    the original amount, creating a cross-table mismatch.
    """
    n = len(df)
    count = max(1, int(n * ratio))
    indices = rng.sample(range(n), min(count, n))

    values = df[col].to_list()
    actual_affected = []
    for i in indices:
        if values[i] is not None:
            try:
                val = float(values[i])
            except ValueError, TypeError:
                continue
            factor = rng.uniform(*factor_range)
            if abs(factor - 1.0) < 0.03:
                factor = factor_range[1]
            values[i] = round(val * factor, 2)
            actual_affected.append(i)

    df = df.with_columns(_safe_series(col, values, df[col].dtype))

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("XMATCH"),
            target_file=f"{table_name}.csv",
            target_column=col,
            target_rows=sorted(actual_affected),
            layer="structural",
            defect="cross_table_consistency",
            defect_detail="payment_bank_mismatch",
            injection_type="break_payment_bank_match",
            parameters={"ratio": ratio, "factor_range": list(factor_range)},
            severity=severity,
        )
    )
    return df


def break_trial_balance(
    df: pl.DataFrame,
    col: str,
    ratio: float,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    error_range: tuple[float, float] = (0.01, 5.0),
    severity: str = "medium",
) -> pl.DataFrame:
    """Introduce errors in trial balance so it no longer matches cumulative GL.

    Unlike drift_formula (which adds small rounding errors), this creates
    meaningful discrepancies that indicate the TB was exported from a
    different source or aggregated incorrectly.
    """
    n = len(df)
    count = max(1, int(n * ratio))
    indices = rng.sample(range(n), min(count, n))

    values = df[col].to_list()
    actual_affected = []
    for i in indices:
        if values[i] is not None:
            try:
                val = float(values[i])
            except ValueError, TypeError:
                continue
            # Percentage error (±error_range)
            error_pct = rng.uniform(*error_range) * rng.choice([1, -1]) / 100.0
            values[i] = round(val * (1 + error_pct), 2)
            actual_affected.append(i)

    df = df.with_columns(_safe_series(col, values, df[col].dtype))

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("XBAL"),
            target_file=f"{table_name}.csv",
            target_column=col,
            target_rows=sorted(actual_affected),
            layer="computational",
            defect="cross_table_consistency",
            defect_detail="trial_balance_gl_mismatch",
            injection_type="break_trial_balance",
            parameters={"ratio": ratio, "error_range": list(error_range)},
            severity=severity,
        )
    )
    return df


def create_mutual_exclusivity(
    df: pl.DataFrame,
    col_a: str,
    col_b: str,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    severity: str = "low",
) -> pl.DataFrame:
    """Ensure two columns are never both populated (mutual exclusivity pattern).

    This is a natural pattern in double-entry bookkeeping (debit/credit),
    but we enforce it strictly to create a detectable dimensional entropy signal.
    """
    n = len(df)
    a_vals = df[col_a].to_list()
    b_vals = df[col_b].to_list()
    a_is_str = df[col_a].dtype == pl.Utf8
    b_is_str = df[col_b].dtype == pl.Utf8
    affected = []

    zero_a = "0.0" if a_is_str else 0.0
    zero_b = "0.0" if b_is_str else 0.0

    for i in range(n):
        try:
            a_nonzero = a_vals[i] is not None and float(a_vals[i]) != 0
            b_nonzero = b_vals[i] is not None and float(b_vals[i]) != 0
        except ValueError, TypeError:
            continue
        if a_nonzero and b_nonzero:
            # Already both populated — zero one out
            if rng.random() < 0.5:
                b_vals[i] = zero_b
            else:
                a_vals[i] = zero_a
            affected.append(i)

    df = df.with_columns(
        [
            _safe_series(col_a, a_vals, df[col_a].dtype),
            _safe_series(col_b, b_vals, df[col_b].dtype),
        ]
    )

    registry.record(
        EntropyInjection(
            injection_id=registry.next_id("MUTEX"),
            target_file=f"{table_name}.csv",
            target_column=f"{col_a}/{col_b}",
            target_rows=sorted(affected),
            layer="structural",
            defect="dimensional_structure",
            defect_detail="mutual_exclusivity",
            injection_type="create_mutual_exclusivity",
            parameters={"col_a": col_a, "col_b": col_b},
            severity=severity,
        )
    )
    return df


def _stock_values(series: list[str], rng: random.Random) -> tuple[list[float], list[float]]:
    """A carried-forward LEVEL: per series, a running cumulative across its ordered rows.

    Relies on the probe table being in ``(series, period)`` row order (the skeleton
    generator guarantees it), so each series accumulates in period order — the
    structural signature of a stock that must not be summed across periods.

    Returns ``(levels, movements)`` aligned per row: ``movements[i]`` is the level's
    net change from the previous period (from the drawn opening for a series' first
    row) — the per-cell target an events backing must sum to.
    """
    running: dict[str, float] = {}
    levels: list[float] = []
    movements: list[float] = []
    for sid in series:
        # The opening draw happens on EVERY row (dict.get evaluates its default), as it
        # always has — preserved so recorded seeds reproduce the exact value surface.
        prev = running.get(sid, round(rng.uniform(500, 5000), 2))  # opening level
        level = round(prev + rng.uniform(-200, 300), 2)  # carry forward + small net movement
        running[sid] = level
        levels.append(level)
        movements.append(round(level - prev, 2))
    return levels, movements


def _flow_values(n: int, rng: random.Random) -> list[float]:
    """A per-period MOVEMENT: independent positive per-row amounts (summable across periods)."""
    return [round(rng.uniform(10, 2000), 2) for _ in range(n)]


# The events/movements table emitted alongside the probe table when a strategy backs
# stock columns. One numeric column per backed stock; per-(series, period)
# sums reconcile to the stock's deltas — the structural_reconciliation witness's input.
PROBE_EVENTS_TABLE = "probe_events"


def _split_amount(total: float, k: int, rng: random.Random) -> list[float]:
    """Split ``total`` into ``k`` cent-rounded parts that sum to it exactly.

    The first ``k − 1`` parts are random fractions of the total; the last absorbs the
    rounding remainder, so ``sum(parts) == total`` to the cent (the reconciliation
    identity the backed events must satisfy by construction).
    """
    weights = [rng.random() for _ in range(k)]
    scale = sum(weights)
    parts = [round(total * w / scale, 2) for w in weights[:-1]]
    parts.append(round(total - sum(parts), 2))
    return parts


def _broken_series_deviation(
    cells_by_series: dict[str, list[int]],
    movements: list[float],
    col: ProbeColumn,
    rng: random.Random,
) -> dict[str, float]:
    """The per-series perturbation magnitude for one BROKEN backed column.

    A ``break_ratio`` fraction of series (at least one) is broken; each broken series
    gets a constant per-cell deviation of ``break_magnitude × mean |movement|`` — so the
    engine's per-entity stock residual lands ≈ ``break_magnitude``, the knob the rig
    sweeps across the abstain gate. Intact series keep deviation 0.0 (they reconcile).
    """
    series_ids = sorted(cells_by_series)
    n_broken = max(1, round(len(series_ids) * col.break_ratio))
    broken = rng.sample(series_ids, min(n_broken, len(series_ids)))
    out: dict[str, float] = {}
    for sid in broken:
        rows = cells_by_series[sid]
        scale = sum(abs(movements[i]) for i in rows) / len(rows)
        out[sid] = round(col.break_magnitude * scale, 2)
    return out


def _period_days(period: str) -> int:
    """Days in a ``YYYY-MM`` period label (the probe skeleton's grain)."""
    try:
        year_s, month_s = period.split("-")
        return calendar.monthrange(int(year_s), int(month_s))[1]
    except (ValueError, IndexError) as e:
        raise ValueError(f"stock_flow events backing needs 'YYYY-MM' period labels, got {period!r}") from e


def _events_frame(
    series: list[str],
    periods: list[str],
    backed: list[ProbeColumn],
    movements_by_col: dict[str, list[float]],
    events_per_cell: tuple[int, int],
    rng: random.Random,
) -> pl.DataFrame:
    """Build the probe_events table for the backed stock columns.

    One row per event: ``event_id``, ``series_id`` (the shared slice dimension),
    ``event_date`` (an ISO date inside the cell's period — the table's own time axis),
    and one movements column per backed stock. Per (series, period) cell, each backed
    column's events sum EXACTLY to that cell's stock delta — except the broken series
    of a broken column, whose cell sums are off by the sampled deviation (random sign
    per cell, so no signed convention can absorb the break).
    """
    cells_by_series: dict[str, list[int]] = {}
    for i, sid in enumerate(series):
        cells_by_series.setdefault(sid, []).append(i)

    # Per broken column: series_id → constant per-cell deviation magnitude.
    deviations: dict[str, dict[str, float]] = {
        col.name: _broken_series_deviation(cells_by_series, movements_by_col[col.name], col, rng)
        for col in backed
        if col.broken
    }

    ids: list[str] = []
    sids: list[str] = []
    dates: list[str] = []
    amounts: dict[str, list[float]] = {stock_flow_events_column(c.name): [] for c in backed}

    counter = 0
    for i, (sid, period) in enumerate(zip(series, periods, strict=True)):
        k = rng.randint(*events_per_cell)
        last_day = _period_days(period)
        for _ in range(k):
            counter += 1
            ids.append(f"EV{counter:06d}")
            sids.append(sid)
            dates.append(f"{period}-{rng.randint(1, last_day):02d}")
        for col in backed:
            target = movements_by_col[col.name][i]
            dev = deviations.get(col.name, {}).get(sid, 0.0)
            if dev:
                target = round(target + rng.choice((-1.0, 1.0)) * dev, 2)
            amounts[stock_flow_events_column(col.name)].extend(_split_amount(target, k, rng))

    return pl.DataFrame(
        {
            "event_id": ids,
            "series_id": sids,
            "event_date": dates,
            **amounts,
        }
    )


def inject_stock_flow_probes(
    df: pl.DataFrame,
    seed: int,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,  # accepted for dispatch uniformity; the family uses its OWN seed
    n_columns: list[int] | None = None,
    stock_fraction: list[float] | None = None,
    ambiguity: list[float] | None = None,
    backed_fraction: list[float] | None = None,
    broken_fraction: list[float] | None = None,
    break_ratio: list[float] | None = None,
    break_magnitude: list[float] | None = None,
    events_per_cell: list[int] | None = None,
    severity: str = "medium",
    dataframes: dict[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Add SAMPLED clearly-named stock/flow measure columns to the probe table.

    Each sampled ``ProbeColumn`` becomes one measure column: a STOCK is a carried-forward
    level (a per-series running cumulative across periods), a FLOW is a per-period
    movement (independent per-row amounts). The clear name carries the stock/flow signal
    — the LLM's job is to read the behaviour from it — and each column is recorded with
    its ``is_stock`` label, the ground truth the temporal_behavior ``llm_claim`` witness
    is scored against. Surface varies by ``seed``; the recorded seed
    reproduces exactly and is independent of injection order (its own RNG).

    EVENTS BACKING: when the family samples ``backed`` stock columns
    (``backed_fraction`` > 0), a ``probe_events`` movements table is emitted alongside
    the probe table into ``dataframes`` — per (series, period) cell, each backed
    column's events sum to the stock's delta (opening + Σ events = closing), except the
    sampled broken strata. That identity is what the temporal_behavior
    ``structural_reconciliation`` witness reads; the registry parameters (``backed``,
    ``reconciles``, ``break_ratio``, ``break_magnitude``, ``events_column``) are the
    rig's ground truth for it. Backing changes the events table only — measure values
    and names are identical with backing on or off (orthogonal axes).
    """
    overrides: dict[str, object] = {}
    if n_columns is not None:
        overrides["n_columns"] = tuple(n_columns)
    if stock_fraction is not None:
        overrides["stock_fraction"] = tuple(stock_fraction)
    if ambiguity is not None:
        overrides["ambiguity"] = tuple(ambiguity)
    if backed_fraction is not None:
        overrides["backed_fraction"] = tuple(backed_fraction)
    if broken_fraction is not None:
        overrides["broken_fraction"] = tuple(broken_fraction)
    if break_ratio is not None:
        overrides["break_ratio"] = tuple(break_ratio)
    if break_magnitude is not None:
        overrides["break_magnitude"] = tuple(break_magnitude)
    if events_per_cell is not None:
        overrides["events_per_cell"] = tuple(events_per_cell)
    params = StockFlowFamilyParams(**overrides)  # type: ignore[arg-type]
    sample = sample_stock_flow_family(seed, params)

    n = len(df)
    series = df["series_id"].to_list() if "series_id" in df.columns else ["S000"] * n
    place = random.Random(f"stock_flow_values:{seed}")  # seed-derived, order-independent

    additions: list[pl.Series] = []
    movements_by_col: dict[str, list[float]] = {}
    for col in sample.columns:
        if col.is_stock:
            values, movements = _stock_values(series, place)
            if col.backed:
                movements_by_col[col.name] = movements
        else:
            values = _flow_values(n, place)
        additions.append(pl.Series(col.name, values))
        registry.record(
            EntropyInjection(
                injection_id=registry.next_id("STOCKFLOW"),
                target_file=f"{table_name}.csv",
                target_column=col.name,
                target_rows=list(range(n)),
                layer="semantic",
                defect="measure_behavior",
                defect_detail="temporal_behavior",
                injection_type="inject_stock_flow_probes",
                parameters={
                    "family": "stock_flow",
                    "seed": seed,
                    "name": col.name,
                    "is_stock": col.is_stock,
                    "true_behavior": "stock" if col.is_stock else "flow",
                    "ambiguous": col.ambiguous,  # hard (conflicting-cue) vs clear regime
                    # Events backing — the structural_reconciliation rig's
                    # ground truth: confirm on backed+reconciling, degrade/abstain on
                    # broken, abstain on unbacked.
                    "backed": col.backed,
                    "reconciles": col.backed and not col.broken,
                    "break_ratio": col.break_ratio,
                    "break_magnitude": col.break_magnitude,
                    "events_table": PROBE_EVENTS_TABLE if col.backed else None,
                    "events_column": stock_flow_events_column(col.name) if col.backed else None,
                },
                severity=severity,
            )
        )

    backed = [c for c in sample.columns if c.backed]
    if backed:
        if dataframes is None:
            raise ValueError(
                "stock_flow events backing sampled but no dataframes dict was passed — "
                "the injector emits probe_events alongside the probe table and needs "
                "the run's table mapping to do it (the scenario runner provides it)."
            )
        if PROBE_EVENTS_TABLE in dataframes:
            raise ValueError(f"{PROBE_EVENTS_TABLE} already exists — one events-backed injection per run")
        if "period" not in df.columns:
            raise ValueError("stock_flow events backing needs the probe skeleton's 'period' column")
        events_rng = random.Random(f"stock_flow_events:{seed}")  # seed-derived, order-independent
        dataframes[PROBE_EVENTS_TABLE] = _events_frame(
            series=series,
            periods=df["period"].to_list(),
            backed=backed,
            movements_by_col=movements_by_col,
            events_per_cell=params.events_per_cell,
            rng=events_rng,
        )

    return df.with_columns(*additions)


def inject_formula_divergence(
    df: pl.DataFrame,
    seed: int,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,  # accepted for dispatch uniformity; the family uses its OWN seed
    n_groups: list[int] | None = None,
    agree_fraction: list[float] | None = None,
    wholesale_fraction: list[float] | None = None,
    divergence_ratio: list[float] | None = None,
    scaled_fraction: list[float] | None = None,
    scaled_rate: list[float] | None = None,
    severity: str = "medium",
) -> pl.DataFrame:
    """Add SAMPLED formula groups whose target NAME advertises one formula.

    The generative successor to ``drift_formula``: each sampled
    :class:`~testdata.entropy.families.FormulaProbeGroup` becomes three numeric columns
    on the probe table — two sources and a target named for a formula over them. In the
    **agree** stratum the values follow the named formula (the true-negative class); in
    **wholesale** every row follows a different sampled truth (an in-space op-swap, or
    the labelled out-of-space scaled kind); in **partial** a sampled fraction of rows
    does. Values stay NUMERIC throughout — the divergence is a different formula, never
    an unparseable token — and targets are computed from the 2-dp-rounded sources, so a
    clean row sits within the engine's 0.01 absolute grading tolerance while a
    divergent row violates it by construction.

    Each TARGET column is recorded with the labels the calibration rig scores the
    derived_value witnesses against: ``named_formula`` (what the name advertises — the
    ``llm_hypothesis`` ground truth), ``actual_formula`` (what the divergent values
    follow), ``divergence_mode`` / ``divergence_ratio``, and ``discoverable`` (whether
    the truth lies in the engine's binary-op discovery language). Source columns are
    unlabelled scaffolding. Surface varies by ``seed``; the recorded seed reproduces
    exactly and is independent of injection order (its own RNG).
    """
    overrides: dict[str, object] = {}
    if n_groups is not None:
        overrides["n_groups"] = tuple(n_groups)
    if agree_fraction is not None:
        overrides["agree_fraction"] = tuple(agree_fraction)
    if wholesale_fraction is not None:
        overrides["wholesale_fraction"] = tuple(wholesale_fraction)
    if divergence_ratio is not None:
        overrides["divergence_ratio"] = tuple(divergence_ratio)
    if scaled_fraction is not None:
        overrides["scaled_fraction"] = tuple(scaled_fraction)
    if scaled_rate is not None:
        overrides["scaled_rate"] = tuple(scaled_rate)
    sample = sample_formula_divergence_family(seed, FormulaDivergenceFamilyParams(**overrides))  # type: ignore[arg-type]

    n = len(df)
    place = random.Random(f"formula_divergence_values:{seed}")  # seed-derived, order-independent
    additions: list[pl.Series] = []
    for group in sample.groups:
        a_vals = [round(place.uniform(*group.a_range), 2) for _ in range(n)]
        b_vals = [round(place.uniform(*group.b_range), 2) for _ in range(n)]
        if group.mode == "wholesale":
            divergent = list(range(n))
        elif group.mode == "partial":
            divergent = sorted(place.sample(range(n), max(1, min(n, round(n * group.divergence_ratio)))))
        else:
            divergent = []
        divergent_set = set(divergent)

        values: list[float] = []
        for i in range(n):
            named = apply_operation(group.named_op, a_vals[i], b_vals[i])
            if i not in divergent_set:
                values.append(round(named, 2))
            elif group.factor is not None:
                values.append(round(named * group.factor, 2))
            else:
                values.append(round(apply_operation(group.actual_op, a_vals[i], b_vals[i]), 2))

        additions.extend(
            (
                pl.Series(group.source_a, a_vals),
                pl.Series(group.source_b, b_vals),
                pl.Series(group.target, values),
            )
        )
        registry.record(
            EntropyInjection(
                injection_id=registry.next_id("FORMDIV"),
                target_file=f"{table_name}.csv",
                target_column=group.target,
                target_rows=divergent,
                layer="computational",
                defect="derived_consistency",
                defect_detail="formula_divergence",
                injection_type="inject_formula_divergence",
                parameters={
                    "family": "formula_divergence",
                    "seed": seed,
                    "theme": group.theme,
                    "named_formula": group.named_formula,
                    "named_op": group.named_op,
                    "actual_formula": group.actual_formula,
                    "actual_op": group.actual_op,
                    "factor": group.factor,
                    "divergence_mode": group.mode,
                    "divergence_ratio": group.divergence_ratio,
                    "discoverable": group.discoverable,
                    "source_columns": [group.source_a, group.source_b],
                },
                severity=severity,
            )
        )
    return df.with_columns(*additions)


def _genuine_pair_values(
    spec: RelationshipPairSpec, n_parent: int, n_child: int, place: random.Random
) -> tuple[list[str], list[str], list[int], float]:
    """Build one genuine (clean or orphan-broken) FK pair's value columns.

    The parent column is a unique id permutation (the pool); the child column
    references a strict subset of it (``referenced_fraction``), with
    ``orphan_ratio`` of child ROWS re-pointed at a small out-of-pool orphan id
    set. Each referenced/orphan id is force-placed at least once, so the
    distinct-value sets are exactly the designed sets.

    Returns (parent_values, child_values, orphan_rows, expected_join_confidence).
    """
    pool = [f"{spec.pool_prefix}-{i:04d}" for i in range(n_parent)]
    parent_values = place.sample(pool, n_parent)  # permutation → a unique key column

    assert spec.referenced_fraction is not None
    n_ref = min(max(2, round(spec.referenced_fraction * n_parent)), n_parent - 1)
    referenced = place.sample(pool, n_ref)

    orphan_count = max(1, int(n_child * spec.orphan_ratio)) if spec.orphan_ratio > 0 else 0
    opool_size = 0
    if orphan_count:
        opool_size = max(1, round((spec.orphan_pool_fraction or 0.0) * n_parent))
        opool_size = min(opool_size, orphan_count)
    # Orphan ids share the pool's shape but live in a disjoint range — deleted
    # parents, not garbage tokens.
    orphan_pool = [f"{spec.pool_prefix}-{9000 + i:04d}" for i in range(opool_size)]

    orphan_rows = sorted(place.sample(range(n_child), orphan_count)) if orphan_count else []
    child_values: list[str | None] = [None] * n_child
    for idx, row in enumerate(orphan_rows):
        child_values[row] = orphan_pool[idx] if idx < opool_size else place.choice(orphan_pool)

    non_orphan_rows = [r for r in range(n_child) if child_values[r] is None]
    if len(non_orphan_rows) < n_ref:
        raise ValueError(
            f"relationship_pairs: pair {spec.child_column} needs {n_ref} non-orphan child rows "
            f"to reference the parent subset but only {len(non_orphan_rows)} remain — grow the "
            "child probe table or lower orphan_ratio/referenced_fraction."
        )
    place.shuffle(non_orphan_rows)
    for idx, row in enumerate(non_orphan_rows):
        child_values[row] = referenced[idx] if idx < n_ref else place.choice(referenced)

    # The exact statistic the data witness will read (joins.py): intersection of
    # distinct sets = the referenced subset; union = pool + orphan ids. A clean
    # pair (child ⊆ parent, >10 distinct) trips the full-containment flag → 1.0.
    jaccard = n_ref / (n_parent + opool_size)
    expected = 1.0 if opool_size == 0 and min(n_parent, n_ref) > 10 else jaccard
    return parent_values, [v for v in child_values if v is not None], orphan_rows, expected


def _spurious_pair_values(
    spec: RelationshipPairSpec, n_parent: int, n_child: int, place: random.Random
) -> tuple[list[str], list[str], float]:
    """Build one spurious pair: two code columns sharing a value POOL, no FK.

    Both sides draw ``pool_size`` distinct codes from one namespace with a
    designed intersection sized for ``designed_jaccard`` — overlap by shared
    vocabulary, never by copying. The intersection stays strictly below both
    set sizes so the full-containment flag can never fire.

    Returns (parent_values, child_values, actual_jaccard).
    """
    assert spec.pool_size is not None and spec.designed_jaccard is not None
    m = spec.pool_size
    if m > n_parent or m > n_child:
        raise ValueError(
            f"relationship_pairs: spurious pool_size {m} exceeds a probe table's rows "
            f"({n_parent} parent / {n_child} child) — every code must appear at least once."
        )
    # jaccard j = i / (2m − i)  →  i = 2mj / (1 + j); clamp keeps containment partial.
    intersection = min(round(2 * m * spec.designed_jaccard / (1 + spec.designed_jaccard)), m - 1)
    codes = [f"{spec.pool_prefix}-{k:04d}" for k in range(2 * m - intersection)]
    set_parent = codes[:m]  # shared ∪ parent-only
    set_child = codes[:intersection] + codes[m:]  # shared ∪ child-only

    parent_values = set_parent + [place.choice(set_parent) for _ in range(n_parent - m)]
    place.shuffle(parent_values)
    child_values = set_child + [place.choice(set_child) for _ in range(n_child - m)]
    place.shuffle(child_values)
    return parent_values, child_values, intersection / (2 * m - intersection)


def inject_relationship_pairs(
    df: pl.DataFrame,
    seed: int,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,  # accepted for dispatch uniformity; the family uses its OWN seed
    dataframes: dict[str, pl.DataFrame],
    n_clean: list[int] | None = None,
    n_broken: list[int] | None = None,
    n_spurious: list[int] | None = None,
    referenced_fraction: list[float] | None = None,
    orphan_ratio: list[float] | None = None,
    orphan_pool_fraction: list[float] | None = None,
    spurious_jaccard: list[float] | None = None,
    spurious_pool_size: list[int] | None = None,
    severity: str = "medium",
) -> pl.DataFrame:
    """Fill the relationship probe tables with SAMPLED labelled pairs.

    The generative successor to the fixed ``break_referential_integrity``: a
    recorded ``seed`` samples directed column pairs across three strata
    (genuine_clean / genuine_broken / spurious_overlap; see
    :mod:`testdata.entropy.families`). ``df`` is the CHILD probe table
    (:data:`REL_CHILD_TABLE`, the strategy's injection target); the PARENT probe
    table is filled in place through the runner's ``dataframes`` pass-through.
    Each pair becomes one parent column + one child column and ONE registry
    record whose ``parameters`` carry the pair identity, label, stratum, and the
    designed statistics — the ground truth the calibration rig scores the
    relationship_discovery witnesses against.
    """
    overrides: dict[str, object] = {}
    if n_clean is not None:
        overrides["n_clean"] = tuple(n_clean)
    if n_broken is not None:
        overrides["n_broken"] = tuple(n_broken)
    if n_spurious is not None:
        overrides["n_spurious"] = tuple(n_spurious)
    if referenced_fraction is not None:
        overrides["referenced_fraction"] = tuple(referenced_fraction)
    if orphan_ratio is not None:
        overrides["orphan_ratio"] = tuple(orphan_ratio)
    if orphan_pool_fraction is not None:
        overrides["orphan_pool_fraction"] = tuple(orphan_pool_fraction)
    if spurious_jaccard is not None:
        overrides["spurious_jaccard"] = tuple(spurious_jaccard)
    if spurious_pool_size is not None:
        overrides["spurious_pool_size"] = tuple(spurious_pool_size)
    sample = sample_relationship_family(seed, RelationshipFamilyParams(**overrides))  # type: ignore[arg-type]

    parent = dataframes.get(REL_PARENT_TABLE)
    if parent is None or parent.is_empty() or df.is_empty():
        raise ValueError(
            f"relationship_pairs: needs the '{REL_PARENT_TABLE}' parent probe table and a non-empty "
            f"'{table_name}' child — the scenario runner generates both grains when a strategy "
            f"injects into '{table_name}'."
        )
    n_parent, n_child = len(parent), len(df)
    place = random.Random(f"relationship_pairs_values:{seed}")  # seed-derived, order-independent

    parent_cols: list[pl.Series] = []
    child_cols: list[pl.Series] = []
    for spec in sample.pairs:
        orphan_rows: list[int] = []
        if spec.label == "genuine":
            p_vals, c_vals, orphan_rows, expected = _genuine_pair_values(spec, n_parent, n_child, place)
        else:
            p_vals, c_vals, actual_jaccard = _spurious_pair_values(spec, n_parent, n_child, place)
            expected = actual_jaccard
        parent_cols.append(pl.Series(spec.parent_column, p_vals))
        child_cols.append(pl.Series(spec.child_column, c_vals))

        orphan_count = len(orphan_rows)
        registry.record(
            EntropyInjection(
                injection_id=registry.next_id("RELPAIR"),
                target_file=f"{table_name}.csv",
                target_column=spec.child_column,
                target_rows=orphan_rows,  # the corrupted units; empty for clean/spurious
                layer="structural",
                defect="relationships",
                defect_detail="relationship_discovery",
                injection_type="inject_relationship_pairs",
                parameters={
                    "family": "relationship_pairs",
                    "seed": seed,
                    "pair": f"{table_name}.{spec.child_column} -> {REL_PARENT_TABLE}.{spec.parent_column}",
                    "child_table": table_name,
                    "child_column": spec.child_column,
                    "parent_table": REL_PARENT_TABLE,
                    "parent_column": spec.parent_column,
                    "label": spec.label,
                    "stratum": spec.stratum,
                    "pool_prefix": spec.pool_prefix,
                    "referenced_fraction": spec.referenced_fraction,
                    "orphan_ratio": spec.orphan_ratio,
                    "orphan_count": orphan_count,
                    "actual_orphan_fraction": round(orphan_count / n_child, 6),
                    "designed_jaccard": spec.designed_jaccard,
                    "expected_join_confidence": round(expected, 4),
                },
                severity=severity,
            )
        )

    dataframes[REL_PARENT_TABLE] = parent.with_columns(parent_cols)
    return df.with_columns(child_cols)


def inject_driver_effect(
    df: pl.DataFrame,
    col: str,
    slice_col: str,
    factors: list[float],
    seed: int,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,  # accepted for dispatch uniformity; the family uses its OWN seed
    severity: str = "medium",
) -> pl.DataFrame:
    """Make ``slice_col`` a KNOWN driver of ``col`` — a factor LADDER across values.

    The driver-discovery gate (``dataraum.analysis.drivers``) ranks a categorical
    dimension by the permutation-gated between-group variance reduction of a measure, and
    reports each group's interesting-slice effect = ``group_mean / baseline − 1``. This
    family concentrates a driver in ``slice_col``: it picks ``len(factors)`` distinct
    labelled values (seeded, order-independent) and multiplies ``col`` by an ascending
    factor per value — a higher factor is a stronger driver (a larger positive slice
    effect), the ORDERING the eval oracle asserts. Any remaining value keeps its natural
    scale — the in-run reference the reference-vs-injected contrast reads.

    ONE-SIDED by design (``col`` = debit, ZERO on credit lines): scaling a group's rows by
    ``f`` multiplies that group's MEAN of ``col`` by exactly ``f`` (the zeros are
    scale-invariant), so the group mean SHIFTS and the variance-reduction statistic reads
    it. A BALANCED measure (net_amount, ≈0 mean per cost_center) is invisible to a
    mean-based statistic — the /ground wall this family was gated against. Because ``col`` is one-sided, the same scale also shifts the
    group's mean of a derived net measure (net = debit − credit; the debit lines go from
    ``+x`` to ``+f·x`` while credit lines are untouched), so the signal survives whichever
    additive measure the engine ranks. Row-grain and ICC-safe: ``slice_col`` (cost_center)
    is not an entity key (η²≈0 on clean), so the row permutation null is valid.

    Records ONE registry entry per (value, factor): ``dimension`` (slice_col), ``value``,
    ``measure`` (col), and ``factor`` — the ground truth the driver oracle reads to assert
    ``slice_col`` ∈ ranked_dimensions under injection and the interesting-slice effects
    ordered by factor (never a point-value threshold; eval grammar).
    """
    if col not in df.columns or slice_col not in df.columns:
        raise ValueError(
            f"inject_driver_effect: {table_name!r} needs both {col!r} and {slice_col!r}"
        )

    place = random.Random(f"driver_effect:{table_name}:{col}:{slice_col}:{seed}")
    slice_values = df[slice_col].to_list()

    # Distinct labelled values with support (a null slice is no slice — the driver
    # conditions only on rows that carry a label). Seed-shuffled so the factor→value
    # assignment is generative (no memorized center), the LADDER the invariant.
    counts: dict[object, int] = {}
    for v in slice_values:
        if v is not None:
            counts[v] = counts.get(v, 0) + 1
    candidates = sorted(counts, key=lambda v: str(v))
    place.shuffle(candidates)

    ladder = sorted(factors)  # ascending — the ordering the oracle grades effects against
    if len(candidates) < len(ladder):
        raise ValueError(
            f"inject_driver_effect: {slice_col!r} has {len(candidates)} labelled value(s) but "
            f"the factor ladder needs {len(ladder)} — shorten `factors` or widen the dimension."
        )
    assignment = dict(zip(candidates[: len(ladder)], ladder, strict=True))  # value → factor

    values = df[col].to_list()
    # Only rows whose value MATERIALLY moved (nonzero pre-scale) are recorded affected —
    # the zeros multiply to zero (scale-invariant) and carry no signal.
    affected_by_value: dict[object, list[int]] = {v: [] for v in assignment}
    for i in range(len(df)):
        factor = assignment.get(slice_values[i])
        if factor is None or values[i] is None:
            continue
        original = float(values[i])
        values[i] = round(original * factor, 2)
        if original != 0.0:
            affected_by_value[slice_values[i]].append(i)
    df = df.with_columns(_safe_series(col, values, df[col].dtype))

    for value, factor in sorted(assignment.items(), key=lambda kv: kv[1]):
        rows = affected_by_value[value]
        registry.record(
            EntropyInjection(
                injection_id=registry.next_id("DRIVER"),
                target_file=f"{table_name}.csv",
                target_column=col,
                target_rows=sorted(rows),
                layer="value",
                defect="driver_effect",
                defect_detail="slice_scale",
                injection_type="inject_driver_effect",
                parameters={
                    "family": "driver_effect",
                    "seed": seed,
                    "slice_column": slice_col,
                    "value": str(value),
                    "measure": col,
                    "factor": factor,
                    "n_scaled": len(rows),
                },
                severity=severity,
            )
        )
    return df
