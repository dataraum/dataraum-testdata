"""Tests for schema normalization transforms."""

from __future__ import annotations

import polars as pl
import pytest

from testdata.canonical.finance.generators import generate_finance_dataset
from testdata.entropy.registry import EntropyInjection, InjectionRegistry
from testdata.export import dataset_to_dataframes
from testdata.schema_transforms import apply_normalization


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
    """partial produces 6 tables."""
    dfs, _ = apply_normalization(dict(base_dataframes), "partial")
    assert len(dfs) == 6


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
    """flat produces 5 tables."""
    dfs, _ = apply_normalization(dict(base_dataframes), "flat")
    assert len(dfs) == 5


# ---------------------------------------------------------------------------
# registry remap
# ---------------------------------------------------------------------------

def test_registry_remap():
    """remap_tables updates target_file entries correctly."""
    registry = InjectionRegistry()
    registry.record(EntropyInjection(
        injection_id="INJ-0001",
        target_file="journal_lines.csv",
        target_column="debit",
        target_rows=[0, 1],
        layer="value",
        dimension="accuracy",
        sub_dimension="type_corruption",
        detector_id="type_detector",
        injection_type="corrupt_types",
    ))
    registry.record(EntropyInjection(
        injection_id="INJ-0002",
        target_file="bank_transactions.csv",
        target_column="amount",
        target_rows=[5],
        layer="value",
        dimension="distribution",
        sub_dimension="benford",
        detector_id="benford_detector",
        injection_type="break_benford",
    ))

    mapping = {"journal_lines": "journal_data", "journal_entries": "journal_data"}
    registry.remap_tables(mapping)

    assert registry.injections[0].target_file == "journal_data.csv"
    # bank_transactions not in mapping — unchanged
    assert registry.injections[1].target_file == "bank_transactions.csv"


def test_registry_remap_flat():
    """Flat mapping chains journal_lines → general_ledger."""
    registry = InjectionRegistry()
    registry.record(EntropyInjection(
        injection_id="INJ-0001",
        target_file="journal_lines.csv",
        target_column="debit",
        target_rows=[0],
        layer="value",
        dimension="accuracy",
        sub_dimension="type_corruption",
        detector_id="type_detector",
        injection_type="corrupt_types",
    ))

    # flat mapping goes directly from original name to final name
    mapping = {
        "journal_lines": "general_ledger",
        "journal_entries": "general_ledger",
        "invoices": "invoice_data",
        "payments": "invoice_data",
    }
    registry.remap_tables(mapping)

    assert registry.injections[0].target_file == "general_ledger.csv"
