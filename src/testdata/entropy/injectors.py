"""Entropy injection functions — one per detector type.

Each injector mutates a Polars DataFrame in-place (returns modified copy)
and records what it did in the InjectionRegistry.
"""

from __future__ import annotations

import random

import polars as pl

from .registry import EntropyInjection, InjectionRegistry


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

    mask = pl.Series("mask", [i in set(indices) for i in range(n)])
    replacements = pl.Series("repl", [rng.choice(garbage) for _ in range(n)])
    new_col = pl.when(mask).then(replacements).otherwise(df[col].cast(pl.Utf8)).alias(col)
    df = df.with_columns(new_col)

    registry.record(EntropyInjection(
        injection_id=registry.next_id("TYPE"),
        target_file=f"{table_name}.csv",
        target_column=col,
        target_rows=sorted(indices),
        layer="value",
        dimension="type_fidelity",
        sub_dimension="type_corruption",
        detector_id="type_fidelity",
        injection_type="corrupt_type",
        parameters={"ratio": ratio, "garbage_values": garbage},
        severity=severity,
    ))
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

    mask = pl.Series("mask", [i in set(indices) for i in range(n)])
    new_col = pl.when(mask).then(pl.lit(None)).otherwise(df[col]).alias(col)
    df = df.with_columns(new_col)

    registry.record(EntropyInjection(
        injection_id=registry.next_id("NULL"),
        target_file=f"{table_name}.csv",
        target_column=col,
        target_rows=sorted(indices),
        layer="value",
        dimension="completeness",
        sub_dimension="null_injection",
        detector_id="null_ratio",
        injection_type="introduce_nulls",
        parameters={"ratio": ratio},
        severity=severity,
    ))
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
            except (ValueError, TypeError):
                continue
            values[i] = val * factor * rng.choice([1, -1]) if val != 0 else factor * 1000
            actual_affected.append(i)

    df = df.with_columns(_safe_series(col, values, df[col].dtype))

    registry.record(EntropyInjection(
        injection_id=registry.next_id("OUTLIER"),
        target_file=f"{table_name}.csv",
        target_column=col,
        target_rows=sorted(actual_affected),
        layer="value",
        dimension="distribution",
        sub_dimension="outlier_injection",
        detector_id="outlier_rate",
        injection_type="inject_outliers",
        parameters={"ratio": ratio, "factor": factor},
        severity=severity,
    ))
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

    registry.record(EntropyInjection(
        injection_id=registry.next_id("BENFORD"),
        target_file=f"{table_name}.csv",
        target_column=col,
        target_rows=sorted(affected),
        layer="value",
        dimension="distribution",
        sub_dimension="benford_violation",
        detector_id="benford",
        injection_type="break_benford",
        parameters={"method": method},
        severity=severity,
    ))
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
        registry.record(EntropyInjection(
            injection_id=registry.next_id("NAME"),
            target_file=f"{table_name}.csv",
            target_column=obscured,
            target_rows=[],  # All rows affected (schema change)
            layer="semantic",
            dimension="business_meaning",
            sub_dimension="column_name_obscured",
            detector_id="business_meaning",
            injection_type="obscure_column_names",
            parameters={"original": original, "obscured": obscured},
            severity=severity,
        ))
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
) -> pl.DataFrame:
    """Mix currencies without declaration — convert some values at FX rate."""
    n = len(df)
    count = max(1, int(n * ratio))
    indices = rng.sample(range(n), min(count, n))

    values = df[col].to_list()
    for i in indices:
        if values[i] is not None:
            values[i] = round(float(values[i]) * fx_rate, 2)

    df = df.with_columns(_safe_series(col, values, df[col].dtype))

    registry.record(EntropyInjection(
        injection_id=registry.next_id("UNIT"),
        target_file=f"{table_name}.csv",
        target_column=col,
        target_rows=sorted(indices),
        layer="semantic",
        dimension="unit_consistency",
        sub_dimension="mixed_currency",
        detector_id="unit_entropy",
        injection_type="mix_units",
        parameters={"alt_currency": alt_currency, "ratio": ratio, "fx_rate": fx_rate},
        severity=severity,
    ))
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
        except (ValueError, TypeError):
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

    registry.record(EntropyInjection(
        injection_id=registry.next_id("DATE"),
        target_file=f"{table_name}.csv",
        target_column=col,
        target_rows=sorted(affected),
        layer="structural",
        dimension="format_consistency",
        sub_dimension="ambiguous_dates",
        detector_id="temporal_entropy",
        injection_type="corrupt_dates",
        parameters={"formats": formats},
        severity=severity,
    ))
    return df


def break_referential_integrity(
    df: pl.DataFrame,
    fk_col: str,
    ratio: float,
    registry: InjectionRegistry,
    table_name: str,
    rng: random.Random,
    severity: str = "high",
) -> pl.DataFrame:
    """Replace FK values with non-existent IDs."""
    n = len(df)
    count = max(1, int(n * ratio))
    indices = rng.sample(range(n), min(count, n))

    values = df[fk_col].to_list()
    for i in indices:
        values[i] = f"ORPHAN-{rng.randint(900000, 999999)}"

    df = df.with_columns(pl.Series(fk_col, values))

    registry.record(EntropyInjection(
        injection_id=registry.next_id("FK"),
        target_file=f"{table_name}.csv",
        target_column=fk_col,
        target_rows=sorted(indices),
        layer="structural",
        dimension="referential_integrity",
        sub_dimension="orphaned_foreign_keys",
        detector_id="relationship_entropy",
        injection_type="break_referential_integrity",
        parameters={"ratio": ratio},
        severity=severity,
    ))
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
) -> pl.DataFrame:
    """Add a redundant FK column with slight noise."""
    values = df[existing_fk_col].to_list()
    n = len(values)
    new_values = list(values)  # copy

    # Add noise to some values
    noise_count = max(1, int(n * noise_ratio))
    noise_indices = rng.sample(range(n), min(noise_count, n))
    for i in noise_indices:
        new_values[i] = f"ALT-{rng.randint(100000, 999999)}"

    df = df.with_columns(pl.Series(new_col_name, new_values))

    registry.record(EntropyInjection(
        injection_id=registry.next_id("DUPFK"),
        target_file=f"{table_name}.csv",
        target_column=new_col_name,
        target_rows=sorted(noise_indices),
        layer="structural",
        dimension="join_paths",
        sub_dimension="duplicate_fk_path",
        detector_id="join_path_determinism",
        injection_type="add_duplicate_fk_paths",
        parameters={"source_col": existing_fk_col, "noise_ratio": noise_ratio},
        severity=severity,
    ))
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

    registry.record(EntropyInjection(
        injection_id=registry.next_id("DRIFT"),
        target_file=f"{table_name}.csv",
        target_column=derived_col,
        target_rows=sorted(indices),
        layer="computational",
        dimension="derived_consistency",
        sub_dimension="formula_drift",
        detector_id="derived_value",
        injection_type="drift_formula",
        parameters={"source_cols": source_cols, "error_ratio": error_ratio},
        severity=severity,
    ))
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
        except (ValueError, TypeError):
            continue
        if d >= cutoff:
            val_values[i] = round(float(val_values[i]) * shift_factor, 2)
            affected.append(i)

    df = df.with_columns(_safe_series(value_col, val_values, df[value_col].dtype))

    registry.record(EntropyInjection(
        injection_id=registry.next_id("TDRIFT"),
        target_file=f"{table_name}.csv",
        target_column=value_col,
        target_rows=sorted(affected),
        layer="value",
        dimension="temporal_stability",
        sub_dimension="distribution_shift",
        detector_id="temporal_drift",
        injection_type="inject_temporal_drift",
        parameters={"shift_date": shift_date, "shift_factor": shift_factor},
        severity=severity,
    ))
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
            except (ValueError, TypeError):
                continue
            factor = rng.uniform(*factor_range)
            # Avoid factor ≈ 1.0 (undetectable)
            if abs(factor - 1.0) < 0.05:
                factor = factor_range[1]
            values[i] = round(val * factor, 2)
            actual_affected.append(i)

    df = df.with_columns(_safe_series(col, values, df[col].dtype))

    registry.record(EntropyInjection(
        injection_id=registry.next_id("XMATCH"),
        target_file=f"{table_name}.csv",
        target_column=col,
        target_rows=sorted(actual_affected),
        layer="structural",
        dimension="cross_table_consistency",
        sub_dimension="gl_invoice_mismatch",
        detector_id="cross_table_consistency",
        injection_type="break_gl_invoice_match",
        parameters={"ratio": ratio, "factor_range": list(factor_range)},
        severity=severity,
    ))
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
            except (ValueError, TypeError):
                continue
            factor = rng.uniform(*factor_range)
            if abs(factor - 1.0) < 0.03:
                factor = factor_range[1]
            values[i] = round(val * factor, 2)
            actual_affected.append(i)

    df = df.with_columns(_safe_series(col, values, df[col].dtype))

    registry.record(EntropyInjection(
        injection_id=registry.next_id("XMATCH"),
        target_file=f"{table_name}.csv",
        target_column=col,
        target_rows=sorted(actual_affected),
        layer="structural",
        dimension="cross_table_consistency",
        sub_dimension="payment_bank_mismatch",
        detector_id="cross_table_consistency",
        injection_type="break_payment_bank_match",
        parameters={"ratio": ratio, "factor_range": list(factor_range)},
        severity=severity,
    ))
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
            except (ValueError, TypeError):
                continue
            # Percentage error (±error_range)
            error_pct = rng.uniform(*error_range) * rng.choice([1, -1]) / 100.0
            values[i] = round(val * (1 + error_pct), 2)
            actual_affected.append(i)

    df = df.with_columns(_safe_series(col, values, df[col].dtype))

    registry.record(EntropyInjection(
        injection_id=registry.next_id("XBAL"),
        target_file=f"{table_name}.csv",
        target_column=col,
        target_rows=sorted(actual_affected),
        layer="computational",
        dimension="cross_table_consistency",
        sub_dimension="trial_balance_gl_mismatch",
        detector_id="derived_value_consistency",
        injection_type="break_trial_balance",
        parameters={"ratio": ratio, "error_range": list(error_range)},
        severity=severity,
    ))
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
        except (ValueError, TypeError):
            continue
        if a_nonzero and b_nonzero:
            # Already both populated — zero one out
            if rng.random() < 0.5:
                b_vals[i] = zero_b
            else:
                a_vals[i] = zero_a
            affected.append(i)

    df = df.with_columns([
        _safe_series(col_a, a_vals, df[col_a].dtype),
        _safe_series(col_b, b_vals, df[col_b].dtype),
    ])

    registry.record(EntropyInjection(
        injection_id=registry.next_id("MUTEX"),
        target_file=f"{table_name}.csv",
        target_column=f"{col_a}/{col_b}",
        target_rows=sorted(affected),
        layer="structural",
        dimension="dimensional_structure",
        sub_dimension="mutual_exclusivity",
        detector_id="dimensional_entropy",
        injection_type="create_mutual_exclusivity",
        parameters={"col_a": col_a, "col_b": col_b},
        severity=severity,
    ))
    return df
