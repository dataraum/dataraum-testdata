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

NormalizationLevel = Literal["full", "partial", "flat", "single"]


# --- Column naming styles ---

ColumnStyle = Literal["snake_case", "camelCase", "PascalCase", "legacy"]

_LEGACY_NAMES: dict[str, str] = {
    "entry_id": "JRNL_ID",
    "line_id": "LN_ID",
    "account_id": "ACCT_NO",
    "account_name": "ACCT_NM",
    "account_type": "ACCT_TYP",
    "parent_id": "PRNT_ACCT",
    "parent_account_id": "PRNT_ACCT",
    "debit": "DR_AMT",
    "credit": "CR_AMT",
    "debit_balance": "DR_BAL",
    "credit_balance": "CR_BAL",
    "cost_center": "CC",
    "invoice_id": "INV_NO",
    "vendor_id": "VNDR_ID",
    "due_date": "DUE_DT",
    "payment_terms": "PMT_TERMS",
    "payment_id": "PMT_ID",
    "payment_date": "PMT_DT",
    "payment_amount": "PMT_AMT",
    "payment_currency": "PMT_CCY",
    "payment_method": "PMT_MTHD",
    "txn_id": "TXN_ID",
    "counterparty": "CNTRPRTY",
    "reconciled": "RECON_FLG",
    "from_ccy": "FROM_CCY",
    "to_ccy": "TO_CCY",
    "created_by": "CRTD_BY",
    "date": "TXN_DT",
    "amount": "AMT",
    "currency": "CCY",
    "description": "DESC",
    "status": "STAT",
    "reference": "REF",
    "period": "PRD",
    "rate": "FX_RATE",
    "source": "SRC",
    "method": "MTHD",
    "account_currency": "ACCT_CCY",
}


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
_KEY_COLUMNS: dict[str, str] = {
    "entry_id": "journal_entries",
    "line_id": "journal_lines",
    "account_id": "chart_of_accounts",
    "invoice_id": "invoices",
    "payment_id": "payments",
    "txn_id": "bank_transactions",
    "vendor_id": "invoices",
}

# Natural key mappings: surrogate → human-readable pattern
_NATURAL_KEYS: dict[str, str] = {
    "entry_id": "JE",
    "line_id": "JL",
    "account_id": "ACCT",
    "invoice_id": "INV",
    "payment_id": "PAY",
    "txn_id": "BT",
    "vendor_id": "V",
}


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
    """
    periods = sorted(df["period"].unique().to_list())

    result = df.filter(pl.col("period") == periods[0]).select(
        "account_id",
        pl.col("debit_balance").alias(f"{periods[0].replace('-', '_')}_debit"),
        pl.col("credit_balance").alias(f"{periods[0].replace('-', '_')}_credit"),
    )

    for period in periods[1:]:
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

    # partial: merge parent→child pairs
    out, mapping = _merge_journal_data(out, mapping)
    out, mapping = _merge_invoice_data(out, mapping)
    out, mapping = _merge_sales_data(out, mapping)

    if level == "partial":
        return out, mapping

    # flat: additionally inline lookups
    out, mapping = _inline_chart_of_accounts(out, mapping)

    if level == "flat":
        return out, mapping

    # single: join everything into one mega-table
    out, mapping = _build_single_table(out, mapping)

    return out, mapping


# ---------------------------------------------------------------------------
# partial helpers
# ---------------------------------------------------------------------------


def _merge_journal_data(
    dfs: dict[str, pl.DataFrame],
    mapping: dict[str, str],
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """journal_data = journal_lines LEFT JOIN journal_entries ON entry_id."""
    lines = dfs.pop("journal_lines")
    entries = dfs.pop("journal_entries")

    # No column conflicts — join directly.
    journal_data = lines.join(entries, on="entry_id", how="left")

    dfs["journal_data"] = journal_data
    mapping["journal_lines"] = "journal_data"
    mapping["journal_entries"] = "journal_data"
    return dfs, mapping


def _merge_sales_data(
    dfs: dict[str, pl.DataFrame],
    mapping: dict[str, str],
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """sales_data = sales_order_lines LEFT JOIN sales_orders ON order_id (DAT-884).

    The operating chain's parent→child pair, folded exactly like journal_data — the
    header/item split is the shape a real ERP presents, and collapsing it is what the
    ``partial`` level exists to test the engine against. Customers and products stay
    as lookups at this level; they are dimension masters, not the order's parent.

    A corpus generated before the chain existed simply has neither table — skip
    rather than fail, so old fixtures stay loadable.
    """
    if "sales_order_lines" not in dfs or "sales_orders" not in dfs:
        return dfs, mapping

    lines = dfs.pop("sales_order_lines")
    orders = dfs.pop("sales_orders")
    sales_data = lines.join(orders, on="order_id", how="left")

    dfs["sales_data"] = sales_data
    mapping["sales_order_lines"] = "sales_data"
    mapping["sales_orders"] = "sales_data"
    return dfs, mapping


def _merge_invoice_data(
    dfs: dict[str, pl.DataFrame],
    mapping: dict[str, str],
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """invoice_data = invoices LEFT JOIN payments ON invoice_id.

    Conflicting payment columns are prefixed with ``payment_``.
    """
    invoices = dfs.pop("invoices")
    payments = dfs.pop("payments")

    # Rename conflicting + ambiguous payment columns before join.
    payments = payments.rename(
        {
            "date": "payment_date",
            "amount": "payment_amount",
            "currency": "payment_currency",
            "method": "payment_method",
        }
    )

    invoice_data = invoices.join(payments, on="invoice_id", how="left")

    dfs["invoice_data"] = invoice_data
    mapping["invoices"] = "invoice_data"
    mapping["payments"] = "invoice_data"
    return dfs, mapping


# ---------------------------------------------------------------------------
# flat helpers
# ---------------------------------------------------------------------------


def _inline_chart_of_accounts(
    dfs: dict[str, pl.DataFrame],
    mapping: dict[str, str],
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """Inline chart_of_accounts into journal_data → general_ledger, and enrich
    trial_balance + balance_sheet with account metadata.

    Every fact that carries ``account_id`` AND has a dimension table to lose gets the
    fold — that is what denormalizing a warehouse does, and it keeps the three facts'
    account axes conformed to one concept. ``balance_sheet`` is included so the
    conformed group has three members rather than two: a consumer that needs >= 2
    facts per shared dimension (the aggregation-lineage witness) then has slack, and
    the corpus stops sitting exactly on that boundary where one missing axis silently
    zeroes the whole group. It also restores level parity — at `full` the witness
    already reconciles balance_sheet.ending_balance (cumulative/STOCK) against the
    journal, and that pairing needs a shared account axis to survive `flat`.

    ``bank_transactions`` deliberately does NOT get the fold: its account_id stays a
    key_only exposure (the Layer-A blind-spot boundary — 2 distinct values, invisible
    to any overlap measure). Keep it that way; it is a graded acceptance class.
    """
    coa = dfs.pop("chart_of_accounts")

    # Prepare CoA columns: rename to avoid conflicts.
    coa_renamed = coa.rename(
        {
            "name": "account_name",
            "parent_id": "parent_account_id",
            "currency": "account_currency",
        }
    )

    # general_ledger = journal_data LEFT JOIN coa ON account_id
    journal_data = dfs.pop("journal_data")
    general_ledger = journal_data.join(coa_renamed, on="account_id", how="left")
    dfs["general_ledger"] = general_ledger

    # Update mapping: anything that pointed to journal_data now points to general_ledger
    for old, new in list(mapping.items()):
        if new == "journal_data":
            mapping[old] = "general_ledger"

    # trial_balance / balance_sheet = <fact> LEFT JOIN coa ON account_id.
    # Both keep their names, so no mapping entry is needed for either.
    for fact in ("trial_balance", "balance_sheet"):
        df = dfs.pop(fact)
        dfs[fact] = df.join(coa_renamed, on="account_id", how="left")

    return dfs, mapping


# ---------------------------------------------------------------------------
# single helpers
# ---------------------------------------------------------------------------


def _build_single_table(
    dfs: dict[str, pl.DataFrame],
    mapping: dict[str, str],
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """Combine general_ledger with invoice_data and bank_transactions into one table.

    The mega-table uses the general_ledger as the spine and left-joins
    invoice_data (on date + amount overlap) and bank_transactions.
    Non-joinable tables (fx_rates, trial_balance) are dropped.
    """
    gl = dfs.pop("general_ledger")

    # Add a source column to track provenance
    gl = gl.with_columns(pl.lit("gl").alias("source_table"))

    # Add invoice data as additional rows (different structure → vertical stack with nulls)
    parts = [gl]

    if "invoice_data" in dfs:
        inv = dfs.pop("invoice_data")
        inv = inv.with_columns(pl.lit("invoice").alias("source_table"))
        parts.append(inv)

    if "bank_transactions" in dfs:
        bank = dfs.pop("bank_transactions")
        bank = bank.with_columns(pl.lit("bank").alias("source_table"))
        parts.append(bank)

    # Vertical concat with diagonal join (fills missing columns with null)
    mega = pl.concat(parts, how="diagonal")

    dfs["mega_table"] = mega

    # Map all original tables to mega_table
    for old_name in list(mapping.keys()):
        mapping[old_name] = "mega_table"
    mapping["bank_transactions"] = "mega_table"

    # Drop the non-joinable period/lookup tables — they fold into the single table
    # conceptually (fx_rates, trial_balance, balance_sheet, and the probe tables —
    # stock/flow, formula, relationship — when a strategy generated them). Without
    # this, balance_sheet dangled and "single" produced two tables.
    for name in (
        "fx_rates",
        "trial_balance",
        "balance_sheet",
        "measure_probes",
        "probe_events",
        "formula_probes",
        "ref_entities",
        "ref_activity",
        "customers",
        "products",
        "ar_invoices",
        "receipts",
    ):
        if name in dfs:
            dfs.pop(name)
            mapping[name] = "mega_table"

    # The merged sales_data folds in too, but it is a DERIVED name: its constituents
    # (sales_orders / sales_order_lines) are already mapped above, so recording it
    # again would put a table that never existed into the mapping.
    dfs.pop("sales_data", None)

    return dfs, mapping
