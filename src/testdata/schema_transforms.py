"""Schema transforms — reshape DataFrames to different normalization levels.

Entropy injections run *before* these transforms, so injection specs use
the original (fully-normalized) table names.  The transform returns a
table-name mapping that ``run_scenario`` uses to update registry entries.

Normalization levels:
    full     — 8 tables, ERP schema export (identity transform)
    partial  — 6 tables, merge parent-child pairs
    flat     — 5 tables, inline lookups into journal/GL
    single   — 1 mega-table, everything joined + pivoted
"""

from __future__ import annotations

import uuid
from typing import Literal

import polars as pl

from testdata.families import (
    Fold,
    Merge,
    ambiguous_key_columns,
    key_columns,
    legacy_names,
    natural_key_prefixes,
)
from testdata.families import folds as family_folds
from testdata.families import merges as family_merges

NormalizationLevel = Literal["full", "partial", "flat", "single"]


# --- Column naming styles ---

ColumnStyle = Literal["snake_case", "camelCase", "PascalCase", "legacy"]

_LEGACY_NAMES: dict[str, str] = legacy_names()


def _to_camel(s: str) -> str:
    parts = s.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _to_pascal(s: str) -> str:
    return "".join(p.capitalize() for p in s.split("_"))


def restyle_column_name(col: str, style: ColumnStyle) -> str:
    """Rename a single snake_case column to *style* — the scalar of ``apply_column_style``.

    Exposed so ground-truth exporters (entropy_map, metadata_truth) can restyle the
    column names they reference through the exact same rule the data columns were
    restyled by, instead of re-deriving it.
    """
    if style == "camelCase":
        return _to_camel(col)
    if style == "PascalCase":
        return _to_pascal(col)
    if style == "legacy":
        return _LEGACY_NAMES.get(col, col.upper())
    return col  # snake_case — identity


def apply_column_style(
    dataframes: dict[str, pl.DataFrame],
    style: ColumnStyle,
) -> dict[str, pl.DataFrame]:
    """Rename columns across all DataFrames to the requested naming style.

    snake_case is the identity (no-op). Other styles apply a transformation.
    """
    if style == "snake_case":
        return dataframes

    out: dict[str, pl.DataFrame] = {}
    for name, df in dataframes.items():
        mapping = {col: restyle_column_name(col, style) for col in df.columns}
        out[name] = df.rename(mapping)

    return out


# --- Key strategies ---

KeyStrategy = Literal["surrogate", "natural", "uuid", "composite"]

# Columns that are primary or foreign keys (column_name → table where it's the PK)
_KEY_COLUMNS: dict[str, str] = key_columns()
_AMBIGUOUS_KEYS: frozenset[str] = ambiguous_key_columns()

# Natural key mappings: surrogate -> human-readable prefix
_NATURAL_KEYS: dict[str, str] = natural_key_prefixes()


def apply_key_strategy(
    dataframes: dict[str, pl.DataFrame],
    strategy: KeyStrategy,
    seed: int = 42,
) -> dict[str, pl.DataFrame]:
    """Transform key columns across all DataFrames to the requested strategy.

    Args:
        dataframes: Table name → DataFrame mapping.
        strategy: Key strategy to apply.
        seed: Seed for deterministic UUID generation.

    Returns:
        Transformed DataFrames with consistent key remapping across tables.
    """
    if strategy == "surrogate":
        return dataframes  # Already the default format

    # Build a global remapping: old_value → new_value per column
    remap: dict[str, dict[str, str]] = {}

    for table_name, df in dataframes.items():
        for col in df.columns:
            if col not in _KEY_COLUMNS:
                # Unknown, or claimed by two tables: remapping a column whose
                # values come from two unrelated id spaces would fuse them.
                continue
            if col in remap:
                continue  # Already built from another table

            unique_vals = df[col].drop_nulls().unique().sort().to_list()
            if strategy == "uuid":
                rng = _SeededUUID(seed, col)
                remap[col] = {v: rng.next() for v in unique_vals}
            elif strategy == "natural":
                prefix = _NATURAL_KEYS.get(col, col.upper())
                remap[col] = {v: _to_natural_key(v, prefix) for v in unique_vals}
            elif strategy == "composite":
                # Composite: prefix with table context
                remap[col] = {v: f"{table_name}::{v}" for v in unique_vals}

    # Apply remapping to all tables (both PK and FK columns)
    out: dict[str, pl.DataFrame] = {}
    for table_name, df in dataframes.items():
        for col in df.columns:
            if col in remap:
                mapping = remap[col]
                df = df.with_columns(
                    pl.col(col)
                    .replace_strict(
                        mapping,
                        default=pl.col(col),
                    )
                    .alias(col)
                )
        out[table_name] = df

    return out


class _SeededUUID:
    """Deterministic UUID generator seeded by column name."""

    def __init__(self, seed: int, col: str) -> None:
        import random as _random

        self._rng = _random.Random(hash((seed, col)))

    def next(self) -> str:
        return str(uuid.UUID(int=self._rng.getrandbits(128), version=4))


def _to_natural_key(surrogate: str, prefix: str) -> str:
    """Convert a surrogate key like 'V-0001' to a natural key like 'VENDOR-ACME-001'."""
    # Extract numeric part if present
    parts = surrogate.split("-")
    if len(parts) >= 2 and parts[-1].isdigit():
        num = int(parts[-1])
        return f"{prefix}-{num:05d}"
    return f"{prefix}-{surrogate}"


# --- Pivots ---


def pivot_trial_balance_wide(df: pl.DataFrame) -> pl.DataFrame:
    """Pivot trial balance from tall (account × period rows) to wide (accounts as rows, periods as columns).

    Returns a DataFrame with one row per account and columns for each period's
    debit/credit balances: ``2025_01_debit``, ``2025_01_credit``, etc.

    The spine is every account that appears in ANY period, not the accounts present
    in the first one. Seeding off period 1 drops an account whose first movement comes
    later — which is not hypothetical: inventory shrinkage first posts at the end of
    quarter one, so that account vanished from the wide export entirely.
    """
    periods = sorted(df["period"].unique().to_list())
    result = df.select("account_id").unique(maintain_order=True)

    for period in periods:
        period_df = df.filter(pl.col("period") == period).select(
            "account_id",
            pl.col("debit_balance").alias(f"{period.replace('-', '_')}_debit"),
            pl.col("credit_balance").alias(f"{period.replace('-', '_')}_credit"),
        )
        result = result.join(period_df, on="account_id", how="left")

    return result


def pivot_journal_lines_wide(df: pl.DataFrame) -> pl.DataFrame:
    """Pivot journal lines from separate debit/credit columns to a single amount + side column.

    Converts: line_id, entry_id, account_id, debit, credit
    To:        line_id, entry_id, account_id, amount, side
    Where side is 'debit' or 'credit' and amount is the non-zero value.
    """
    debit_rows = (
        df.filter(pl.col("debit") > 0)
        .with_columns(
            pl.col("debit").alias("amount"),
            pl.lit("debit").alias("side"),
        )
        .drop("debit", "credit")
    )

    credit_rows = (
        df.filter(pl.col("credit") > 0)
        .with_columns(
            pl.col("credit").alias("amount"),
            pl.lit("credit").alias("side"),
        )
        .drop("debit", "credit")
    )

    return pl.concat([debit_rows, credit_rows]).sort("line_id")


# --- Normalization ---


def apply_normalization(
    dataframes: dict[str, pl.DataFrame],
    level: NormalizationLevel = "full",
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """Reshape *dataframes* according to *level*.

    Returns:
        (transformed_dataframes, old_name → new_name mapping)
        The mapping only contains entries for tables whose name changed.
    """
    if level == "full":
        return dataframes, {}

    out = dict(dataframes)
    mapping: dict[str, str] = {}

    # partial: collapse every declared parent/child pair
    for merge in family_merges():
        out, mapping = _apply_merge(merge, out, mapping)

    if level == "partial":
        return out, mapping

    # flat: additionally inline the declared dimension folds
    for fold in family_folds():
        out, mapping = _apply_fold(fold, out, mapping)

    if level == "flat":
        return out, mapping

    # single: join everything into one mega-table
    out, mapping = _build_single_table(out, mapping)

    return out, mapping


# ---------------------------------------------------------------------------
# partial helpers
# ---------------------------------------------------------------------------


def _apply_merge(
    merge: Merge,
    dfs: dict[str, pl.DataFrame],
    mapping: dict[str, str],
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """``merge.name = merge.spine LEFT JOIN merge.joined ON merge.on``.

    Driven entirely by the family declaration, so a new family's header/item pair
    collapses at ``partial`` without this function learning its table names.

    A corpus missing either side — a probe-only fixture, or one generated before a
    family existed — is skipped rather than failed, so old fixtures stay loadable.
    """
    if merge.spine not in dfs or merge.joined not in dfs:
        return dfs, mapping

    spine = dfs.pop(merge.spine)
    joined = dfs.pop(merge.joined)
    if merge.rename:
        joined = joined.rename(dict(merge.rename))

    dfs[merge.name] = spine.join(joined, on=merge.on, how="left")
    mapping[merge.spine] = merge.name
    mapping[merge.joined] = merge.name
    return dfs, mapping


# ---------------------------------------------------------------------------
# flat helpers
# ---------------------------------------------------------------------------


def _apply_fold(
    fold: Fold,
    dfs: dict[str, pl.DataFrame],
    mapping: dict[str, str],
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """Inline ``fold.dimension`` into each fact in ``fold.into``, renaming as declared.

    A fact may take a new name once it carries the dimension (``journal_data`` becomes
    ``general_ledger``); one that keeps its own name needs no mapping entry. Facts the
    fold does not list keep their bare FK column — that key_only exposure is a graded
    acceptance class, not an omission (``bank_transactions.account_id`` is the case).
    """
    if fold.dimension not in dfs:
        return dfs, mapping

    dimension = dfs.pop(fold.dimension)
    if fold.rename:
        dimension = dimension.rename(dict(fold.rename))

    for fact, result in fold.into.items():
        if fact not in dfs:
            continue
        df = dfs.pop(fact)
        dfs[result] = df.join(dimension, on=fold.on, how="left")
        if result != fact:
            # Anything that already pointed at the pre-fold name follows it.
            for old, new in list(mapping.items()):
                if new == fact:
                    mapping[old] = result
    return dfs, mapping


# ---------------------------------------------------------------------------
# single helpers
# ---------------------------------------------------------------------------


def _build_single_table(
    dfs: dict[str, pl.DataFrame],
    mapping: dict[str, str],
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """Combine general_ledger with invoice_data and bank_transactions into one table.

    The mega-table uses the general_ledger as the spine and stacks invoice_data and
    bank_transactions onto it diagonally. Everything else the corpus still holds folds
    in conceptually and is dropped — **derived, not listed**. The drop set used to be a
    literal tuple that had to grow with every family (it gained six names when the
    operating chain and inventory landed), and a family that forgot to appear in it
    left a table dangling beside the mega-table, so ``single`` quietly produced two.

    A name that only ever existed as a merge RESULT (``sales_data``) is dropped without
    a mapping entry: its constituents are already mapped, and recording the derived name
    would put a table that never existed into the rename map.
    """
    gl = dfs.pop("general_ledger")

    # Add a source column to track provenance
    gl = gl.with_columns(pl.lit("gl").alias("source_table"))

    # Add invoice data as additional rows (different structure → vertical stack with nulls)
    parts = [gl]

    for name, label in (("invoice_data", "invoice"), ("bank_transactions", "bank")):
        if name in dfs:
            part = dfs.pop(name)
            parts.append(part.with_columns(pl.lit(label).alias("source_table")))

    # Vertical concat with diagonal join (fills missing columns with null)
    mega = pl.concat(parts, how="diagonal")

    # Map all original tables to mega_table
    for old_name in list(mapping.keys()):
        mapping[old_name] = "mega_table"
    mapping["bank_transactions"] = "mega_table"

    derived_names = {merge.name for merge in family_merges()}
    for name in list(dfs):
        dfs.pop(name)
        if name not in derived_names:
            mapping[name] = "mega_table"

    dfs["mega_table"] = mega
    return dfs, mapping
