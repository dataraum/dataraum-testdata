"""The registry is the single declaration of what tables exist.

These tests are the guarantee the registry exists for: a family added to the
dataset but not declared — or declared without its keys — fails here rather than
silently losing its columns in a key or column-style transform, which is exactly
what happened to the operating chain before the registry existed.
"""

from __future__ import annotations

from testdata.canonical.finance.models import FinanceDataset
from testdata.export import TABLE_NAMES
from testdata.families import (
    FAMILIES,
    all_tables,
    ambiguous_key_columns,
    key_columns,
    legacy_names,
    natural_key_prefixes,
)
from testdata.schema_transforms import apply_key_strategy


def test_every_dataset_table_is_declared_by_a_family() -> None:
    declared = set(all_tables())
    on_the_dataset = set(FinanceDataset.model_fields)
    assert on_the_dataset - declared == set(), "dataset tables missing a family declaration"
    assert declared - on_the_dataset == set(), "families declare tables the dataset does not carry"


def test_exporter_follows_the_registry() -> None:
    assert set(TABLE_NAMES) == set(all_tables())


def test_every_declared_table_declares_a_primary_key() -> None:
    undeclared = {table for fam in FAMILIES for table in fam.tables if table not in fam.primary_keys}
    # Tables keyed by a compound of FKs rather than one column.
    compound_grain = {"fx_rates", "trial_balance", "balance_sheet", "inventory_positions"}
    assert undeclared == compound_grain


def test_no_family_declares_a_table_twice() -> None:
    tables = all_tables()
    assert len(tables) == len(set(tables))


def test_operating_chain_keys_reach_the_transforms() -> None:
    # The regression this registry exists for: these were absent from the key and
    # legacy maps while present in the exporter.
    keys = key_columns()
    for column, table in [
        ("customer_id", "customers"),
        ("product_id", "products"),
        ("order_line_id", "sales_order_lines"),
        ("ar_invoice_id", "ar_invoices"),
        ("receipt_id", "receipts"),
    ]:
        assert keys[column] == table
        assert column in natural_key_prefixes()
        assert column in legacy_names()


def test_order_id_is_ambiguous_and_therefore_not_remapped() -> None:
    # The sales order and the role-play probe fact both call their key `order_id`
    # over unrelated id spaces; fusing them would invent a join that is not there.
    assert "order_id" in ambiguous_key_columns()
    assert "order_id" not in key_columns()


def test_natural_keys_rewrite_the_operating_chain() -> None:
    import polars as pl

    frames = {
        "customers": pl.DataFrame({"customer_id": ["C-0001", "C-0002"]}),
        "sales_order_lines": pl.DataFrame({"order_line_id": ["SOL-0001"], "customer_id": ["C-0001"]}),
    }
    out = apply_key_strategy(frames, "natural", seed=42)

    assert out["customers"]["customer_id"].to_list() == ["CUST-00001", "CUST-00002"]
    # The FK follows the PK, or the join breaks.
    assert out["sales_order_lines"]["customer_id"].to_list() == ["CUST-00001"]
