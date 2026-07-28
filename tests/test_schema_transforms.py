"""Tests for schema normalization transforms."""

from __future__ import annotations

import polars as pl
import pytest

from testdata.canonical.finance.generators import generate_finance_dataset
from testdata.entropy.registry import EntropyInjection, InjectionRegistry
from testdata.export import dataset_to_dataframes
from testdata.schema_transforms import (
    apply_column_style,
    apply_key_strategy,
    apply_normalization,
    pivot_journal_lines_wide,
    pivot_trial_balance_wide,
)


@pytest.fixture(scope="module")
def base_dataframes() -> dict[str, pl.DataFrame]:
    """Generate a small dataset and convert to DataFrames (shared across tests)."""
    ds = generate_finance_dataset(seed=42, months=3)
    return dataset_to_dataframes(ds)


# ---------------------------------------------------------------------------
# full
# ---------------------------------------------------------------------------


def test_full_is_identity(base_dataframes):
    """'full' returns all 8 tables unchanged."""
    dfs, mapping = apply_normalization(dict(base_dataframes), "full")
    assert mapping == {}
    assert set(dfs.keys()) == set(base_dataframes.keys())
    for name, df in dfs.items():
        assert df.shape == base_dataframes[name].shape


# ---------------------------------------------------------------------------
# partial
# ---------------------------------------------------------------------------


def test_partial_merges_journals(base_dataframes):
    """partial produces journal_data with entry + line fields."""
    dfs, mapping = apply_normalization(dict(base_dataframes), "partial")

    assert "journal_data" in dfs
    assert "journal_lines" not in dfs
    assert "journal_entries" not in dfs

    jd = dfs["journal_data"]
    # line fields
    assert "line_id" in jd.columns
    assert "debit" in jd.columns
    # entry fields
    assert "date" in jd.columns
    assert "description" in jd.columns
    assert "created_by" in jd.columns

    # Row count matches journal_lines (LEFT JOIN preserves left side)
    assert len(jd) == len(base_dataframes["journal_lines"])


def test_partial_merges_invoices(base_dataframes):
    """partial produces invoice_data with prefixed payment columns."""
    dfs, mapping = apply_normalization(dict(base_dataframes), "partial")

    assert "invoice_data" in dfs
    assert "invoices" not in dfs
    assert "payments" not in dfs

    inv = dfs["invoice_data"]
    # Original invoice fields
    assert "invoice_id" in inv.columns
    assert "amount" in inv.columns
    assert "date" in inv.columns
    # Prefixed payment fields
    assert "payment_date" in inv.columns
    assert "payment_amount" in inv.columns
    assert "payment_currency" in inv.columns
    assert "payment_method" in inv.columns

    # LEFT JOIN: row count >= invoices (multiple payments per invoice possible)
    assert len(inv) >= len(base_dataframes["invoices"])


def test_partial_unchanged_tables(base_dataframes):
    """partial leaves other tables untouched."""
    dfs, _ = apply_normalization(dict(base_dataframes), "partial")
    for name in ("chart_of_accounts", "bank_transactions", "fx_rates", "trial_balance"):
        assert name in dfs
        assert dfs[name].shape == base_dataframes[name].shape


def test_partial_table_count(base_dataframes):
    """partial produces 7 tables (6 + the standalone balance_sheet period table)."""
    dfs, _ = apply_normalization(dict(base_dataframes), "partial")
    assert len(dfs) == 12  # + customers, products, sales_data, ar_invoices, receipts


# ---------------------------------------------------------------------------
# flat
# ---------------------------------------------------------------------------


def test_flat_inlines_accounts(base_dataframes):
    """flat inlines chart_of_accounts into general_ledger."""
    dfs, _ = apply_normalization(dict(base_dataframes), "flat")

    assert "general_ledger" in dfs
    assert "chart_of_accounts" not in dfs
    assert "journal_data" not in dfs

    gl = dfs["general_ledger"]
    assert "account_name" in gl.columns
    assert "account_type" in gl.columns
    assert "parent_account_id" in gl.columns
    assert "account_currency" in gl.columns
    # Original journal line fields still present
    assert "line_id" in gl.columns
    assert "debit" in gl.columns


def test_flat_enriches_trial_balance(base_dataframes):
    """flat adds account metadata to trial_balance."""
    dfs, _ = apply_normalization(dict(base_dataframes), "flat")

    tb = dfs["trial_balance"]
    assert "account_name" in tb.columns
    assert "account_type" in tb.columns
    assert "parent_account_id" in tb.columns
    assert "account_currency" in tb.columns
    # Row count unchanged
    assert len(tb) == len(base_dataframes["trial_balance"])


def test_flat_table_count(base_dataframes):
    """flat produces 6 tables (5 + the standalone balance_sheet period table)."""
    dfs, _ = apply_normalization(dict(base_dataframes), "flat")
    assert len(dfs) == 11  # + customers, products, sales_data, ar_invoices, receipts


# ---------------------------------------------------------------------------
# registry remap
# ---------------------------------------------------------------------------


def test_registry_remap():
    """remap_tables updates target_file entries correctly."""
    registry = InjectionRegistry()
    registry.record(
        EntropyInjection(
            injection_id="INJ-0001",
            target_file="journal_lines.csv",
            target_column="debit",
            target_rows=[0, 1],
            layer="value",
            dimension="accuracy",
            sub_dimension="type_corruption",
            detector_id="type_detector",
            injection_type="corrupt_types",
        )
    )
    registry.record(
        EntropyInjection(
            injection_id="INJ-0002",
            target_file="bank_transactions.csv",
            target_column="amount",
            target_rows=[5],
            layer="value",
            dimension="distribution",
            sub_dimension="benford",
            detector_id="benford_detector",
            injection_type="break_benford",
        )
    )

    mapping = {"journal_lines": "journal_data", "journal_entries": "journal_data"}
    registry.remap_tables(mapping)

    assert registry.injections[0].target_file == "journal_data.csv"
    # bank_transactions not in mapping — unchanged
    assert registry.injections[1].target_file == "bank_transactions.csv"


def test_registry_remap_flat():
    """Flat mapping chains journal_lines → general_ledger."""
    registry = InjectionRegistry()
    registry.record(
        EntropyInjection(
            injection_id="INJ-0001",
            target_file="journal_lines.csv",
            target_column="debit",
            target_rows=[0],
            layer="value",
            dimension="accuracy",
            sub_dimension="type_corruption",
            detector_id="type_detector",
            injection_type="corrupt_types",
        )
    )

    # flat mapping goes directly from original name to final name
    mapping = {
        "journal_lines": "general_ledger",
        "journal_entries": "general_ledger",
        "invoices": "invoice_data",
        "payments": "invoice_data",
    }
    registry.remap_tables(mapping)

    assert registry.injections[0].target_file == "general_ledger.csv"


# ---------------------------------------------------------------------------
# single
# ---------------------------------------------------------------------------


def test_single_produces_one_table(base_dataframes):
    """single normalization produces exactly 1 table."""
    dfs, mapping = apply_normalization(dict(base_dataframes), "single")
    assert len(dfs) == 1
    assert "mega_table" in dfs


def test_single_has_source_column(base_dataframes):
    """mega_table has source_table column for provenance tracking."""
    dfs, _ = apply_normalization(dict(base_dataframes), "single")
    mega = dfs["mega_table"]
    assert "source_table" in mega.columns
    sources = set(mega["source_table"].unique().to_list())
    assert "gl" in sources
    assert "invoice" in sources
    assert "bank" in sources


def test_single_preserves_gl_rows(base_dataframes):
    """mega_table GL rows >= original journal_lines count."""
    dfs, _ = apply_normalization(dict(base_dataframes), "single")
    mega = dfs["mega_table"]
    gl_rows = mega.filter(pl.col("source_table") == "gl")
    assert len(gl_rows) >= len(base_dataframes["journal_lines"])


# ---------------------------------------------------------------------------
# column naming styles
# ---------------------------------------------------------------------------


def test_snake_case_is_identity(base_dataframes):
    """snake_case returns columns unchanged."""
    result = apply_column_style(dict(base_dataframes), "snake_case")
    for name, df in result.items():
        assert df.columns == base_dataframes[name].columns


def test_camel_case(base_dataframes):
    """camelCase converts snake_case columns."""
    result = apply_column_style(dict(base_dataframes), "camelCase")
    je = result["journal_entries"]
    assert "entryId" in je.columns
    assert "createdBy" in je.columns


def test_pascal_case(base_dataframes):
    """PascalCase capitalizes each word."""
    result = apply_column_style(dict(base_dataframes), "PascalCase")
    je = result["journal_entries"]
    assert "EntryId" in je.columns
    assert "CreatedBy" in je.columns


def test_legacy_style(base_dataframes):
    """legacy uses abbreviated uppercase names."""
    result = apply_column_style(dict(base_dataframes), "legacy")
    jl = result["journal_lines"]
    assert "DR_AMT" in jl.columns
    assert "CR_AMT" in jl.columns
    assert "ACCT_NO" in jl.columns
    assert "CC" in jl.columns


# ---------------------------------------------------------------------------
# pivots
# ---------------------------------------------------------------------------


def test_trial_balance_wide_pivot(base_dataframes):
    """Wide pivot creates period columns for trial balance."""
    tb = base_dataframes["trial_balance"]
    wide = pivot_trial_balance_wide(tb)
    periods = sorted(tb["period"].unique().to_list())
    # One row per account
    accounts = tb["account_id"].unique()
    assert len(wide) == len(accounts)
    # Should have columns for each period
    for period in periods:
        col_prefix = period.replace("-", "_")
        assert f"{col_prefix}_debit" in wide.columns
        assert f"{col_prefix}_credit" in wide.columns


def test_journal_lines_wide_pivot(base_dataframes):
    """Wide pivot creates amount + side columns."""
    jl = base_dataframes["journal_lines"]
    wide = pivot_journal_lines_wide(jl)
    assert "amount" in wide.columns
    assert "side" in wide.columns
    assert "debit" not in wide.columns
    assert "credit" not in wide.columns
    sides = set(wide["side"].unique().to_list())
    assert sides == {"debit", "credit"}
    # Row count should match non-zero lines
    assert len(wide) == len(jl)  # Every line has exactly one non-zero side


# ---------------------------------------------------------------------------
# key strategies
# ---------------------------------------------------------------------------


def test_surrogate_is_identity(base_dataframes):
    """surrogate returns DataFrames unchanged."""
    result = apply_key_strategy(dict(base_dataframes), "surrogate")
    for name, df in result.items():
        assert df.columns == base_dataframes[name].columns
        assert df.shape == base_dataframes[name].shape


def test_natural_keys(base_dataframes):
    """natural converts surrogate IDs to prefix-based keys."""
    result = apply_key_strategy(dict(base_dataframes), "natural")
    je = result["journal_entries"]
    # entry_id should now start with "JE-"
    sample = je["entry_id"][0]
    assert sample.startswith("JE-")

    jl = result["journal_lines"]
    # line_id should start with "JL-"
    assert jl["line_id"][0].startswith("JL-")
    # Foreign key entry_id in journal_lines matches PK in journal_entries
    je_ids = set(je["entry_id"].unique().to_list())
    jl_entry_ids = set(jl["entry_id"].drop_nulls().unique().to_list())
    assert jl_entry_ids.issubset(je_ids)


def test_uuid_keys(base_dataframes):
    """uuid converts surrogate IDs to UUID format."""
    result = apply_key_strategy(dict(base_dataframes), "uuid")
    je = result["journal_entries"]
    sample = je["entry_id"][0]
    # Should look like a UUID (8-4-4-4-12 hex)
    parts = sample.split("-")
    assert len(parts) == 5
    assert len(parts[0]) == 8

    # FK consistency: journal_lines.entry_id values exist in journal_entries.entry_id
    jl = result["journal_lines"]
    je_ids = set(je["entry_id"].unique().to_list())
    jl_entry_ids = set(jl["entry_id"].drop_nulls().unique().to_list())
    assert jl_entry_ids.issubset(je_ids)


def test_uuid_deterministic(base_dataframes):
    """Same seed produces identical UUIDs."""
    r1 = apply_key_strategy(dict(base_dataframes), "uuid", seed=42)
    r2 = apply_key_strategy(dict(base_dataframes), "uuid", seed=42)
    assert r1["journal_entries"]["entry_id"].to_list() == r2["journal_entries"]["entry_id"].to_list()


def test_composite_keys(base_dataframes):
    """composite prefixes keys with table name."""
    result = apply_key_strategy(dict(base_dataframes), "composite")
    je = result["journal_entries"]
    sample = je["entry_id"][0]
    assert "::" in sample
