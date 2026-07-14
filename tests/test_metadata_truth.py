"""Tests for the agent-layer metadata-truth export (DAT-682)."""

from __future__ import annotations

from pathlib import Path

import yaml

from testdata.metadata_truth import (
    canonical_metadata_truth,
    export_metadata_truth,
    remap_metadata_truth,
)
from testdata.scenarios.runner import run_scenario
from testdata.schema_transforms import apply_normalization

# Partial normalization's table rename, mirrored here so the remap assertions are
# pinned to a known mapping (journal_lines+journal_entries → journal_data;
# invoices+payments → invoice_data).
_PARTIAL_MAPPING = {
    "journal_lines": "journal_data",
    "journal_entries": "journal_data",
    "invoices": "invoice_data",
    "payments": "invoice_data",
}


# ---------------------------------------------------------------------------
# canonical shape
# ---------------------------------------------------------------------------


def test_canonical_has_every_graded_section() -> None:
    truth = canonical_metadata_truth()
    assert truth["vertical"] == "finance"
    for section in (
        "metric_additivity",
        "stock_flow",
        "reconciles_structurally",
        "relationships",
        "table_roles",
        "semantic_roles",
        "business_concepts",
        "cycles",
        "folded_dimensions",
        "degenerate_ids",
    ):
        assert section in truth, section
    assert len(truth["relationships"]) == 9
    assert set(truth["metric_additivity"]) == {"metrics", "measures"}
    # folded truth is level-specific — canonical (`full`) folds nothing.
    assert truth["folded_dimensions"] == []
    assert truth["degenerate_ids"] == []


def test_identity_remap_is_canonical() -> None:
    """full / snake_case (no mapping) leaves the truth byte-for-byte canonical."""
    canonical = canonical_metadata_truth()
    assert remap_metadata_truth(canonical) == canonical


def test_remap_does_not_mutate_input() -> None:
    canonical = canonical_metadata_truth()
    remap_metadata_truth(canonical, table_mapping=_PARTIAL_MAPPING)
    assert canonical == canonical_metadata_truth()


# ---------------------------------------------------------------------------
# table remap under a merge
# ---------------------------------------------------------------------------


def test_merge_collapsed_fk_is_dropped_self_fk_kept() -> None:
    out = remap_metadata_truth(canonical_metadata_truth(), table_mapping=_PARTIAL_MAPPING)
    rels = {(r["from"], r["to"]) for r in out["relationships"]}

    # journal_lines.entry_id → journal_entries.entry_id collapses into journal_data → dropped.
    assert ("journal_data.entry_id", "journal_data.entry_id") not in rels
    # payments.invoice_id → invoices.invoice_id collapses into invoice_data → dropped.
    assert not any(f == "invoice_data.invoice_id" and t == "invoice_data.invoice_id" for f, t in rels)
    # A cross-table FK whose endpoints land in DIFFERENT merged tables survives, remapped.
    assert ("invoice_data.entry_id", "journal_data.entry_id") in rels
    # The genuine self-FK is kept (endpoints were the same table before the merge, too).
    assert ("chart_of_accounts.parent_id", "chart_of_accounts.account_id") in rels
    # Two FKs dropped, seven kept.
    assert len(out["relationships"]) == 7


def test_table_references_follow_the_mapping() -> None:
    out = remap_metadata_truth(canonical_metadata_truth(), table_mapping=_PARTIAL_MAPPING)
    assert out["stock_flow"]["journal_data.debit"] == "additive"
    assert "journal_lines.debit" not in out["stock_flow"]
    assert "journal_data.debit" in out["semantic_roles"]["measure"]
    # table_roles / cycles carry merged names, de-duplicated.
    assert "journal_data" in out["table_roles"]["facts"]
    je_cycle = next(c for c in out["cycles"] if c["canonical_type"] == "journal_entry_cycle")
    assert je_cycle["key_tables"] == ["journal_data"]  # both source tables merged → one


def test_metric_additivity_is_shape_invariant() -> None:
    """metric_additivity keys are ontology names — untouched by any schema remap."""
    canonical = canonical_metadata_truth()
    out = remap_metadata_truth(canonical, table_mapping=_PARTIAL_MAPPING)
    assert out["metric_additivity"] == canonical["metric_additivity"]


# ---------------------------------------------------------------------------
# folded dimensions (DAT-757)
# ---------------------------------------------------------------------------


def test_full_and_partial_fold_nothing() -> None:
    """Normalized levels have no folded dimensions and no degenerate ids."""
    for level in (None, "full", "partial"):
        out = remap_metadata_truth(canonical_metadata_truth(), level=level)
        assert out["folded_dimensions"] == [], level
        assert out["degenerate_ids"] == [], level


def test_flat_folds_account_dim_into_two_facts() -> None:
    """`flat` folds chart_of_accounts (concept `account`) into general_ledger AND
    trial_balance — the cross-fact concept-identity case (DAT-757)."""
    out = remap_metadata_truth(canonical_metadata_truth(), level="flat")
    account = next(f for f in out["folded_dimensions"] if f["concept"] == "account")
    assert account["fold_key"] == "account_id"
    assert set(account["folded_into"]) == {"general_ledger", "trial_balance"}
    assert out["degenerate_ids"] == ["general_ledger.line_id"]


def test_single_folds_onto_the_mega_table() -> None:
    out = remap_metadata_truth(canonical_metadata_truth(), level="single")
    account = next(f for f in out["folded_dimensions"] if f["concept"] == "account")
    assert account["folded_into"] == ["mega_table"]


def test_folded_truth_matches_the_denormalized_schema() -> None:
    """truth ↔ data bind: every folded attribute / fold key / degenerate id the truth
    names actually EXISTS as a column in the `flat` normalized schema (DAT-757)."""
    from testdata.canonical.finance.generators import generate_finance_dataset
    from testdata.export import dataset_to_dataframes

    dfs = dataset_to_dataframes(generate_finance_dataset(seed=42, months=2))
    normalized, _ = apply_normalization(dict(dfs), "flat")
    truth = remap_metadata_truth(canonical_metadata_truth(), level="flat")

    for fold in truth["folded_dimensions"]:
        for fact in fold["folded_into"]:
            cols = set(normalized[fact].columns)
            assert fold["fold_key"] in cols, f"{fact} missing fold_key {fold['fold_key']}"
            missing = [a for a in fold["attributes"] if a not in cols]
            assert not missing, f"{fact} missing folded attributes {missing}"

    for qualified in truth["degenerate_ids"]:
        table, col = qualified.split(".", 1)
        assert col in set(normalized[table].columns), f"{table} missing degenerate id {col}"


# ---------------------------------------------------------------------------
# column-style remap
# ---------------------------------------------------------------------------


def test_column_style_restyles_qualified_names() -> None:
    out = remap_metadata_truth(canonical_metadata_truth(), column_style="legacy")
    # debit → DR_AMT, credit → CR_AMT (the legacy dictionary).
    assert out["stock_flow"]["journal_lines.DR_AMT"] == "additive"
    assert "journal_lines.CR_AMT" in out["semantic_roles"]["measure"]
    assert out["relationships"][0]["from"] == "journal_lines.JRNL_ID"


# ---------------------------------------------------------------------------
# export I/O + partial-normalization consistency
# ---------------------------------------------------------------------------


def test_export_writes_parsable_yaml(tmp_path: Path) -> None:
    export_metadata_truth(tmp_path)
    written = yaml.safe_load((tmp_path / "metadata_truth.yaml").read_text())
    assert written == canonical_metadata_truth()


def test_export_table_mapping_matches_live_normalization() -> None:
    """The mapping we pin the remap to is the one apply_normalization actually emits."""
    from testdata.canonical.finance.generators import generate_finance_dataset
    from testdata.export import dataset_to_dataframes

    dfs = dataset_to_dataframes(generate_finance_dataset(seed=42, months=2))
    _, mapping = apply_normalization(dict(dfs), "partial")
    assert mapping == _PARTIAL_MAPPING


# ---------------------------------------------------------------------------
# end-to-end wiring — emitted for every scenario
# ---------------------------------------------------------------------------


def test_run_scenario_emits_metadata_truth(tmp_path: Path) -> None:
    run_scenario("month-end-close", strategy_name="clean", months=2, output_dir=tmp_path)
    written = yaml.safe_load((tmp_path / "metadata_truth.yaml").read_text())
    # month-end-close is `full` normalization → canonical.
    assert written == canonical_metadata_truth()


def test_run_scenario_multi_source_emits_top_level_truth(tmp_path: Path) -> None:
    run_scenario("multi-system-recon", strategy_name="clean", months=2, output_dir=tmp_path)
    assert (tmp_path / "metadata_truth.yaml").exists()
    written = yaml.safe_load((tmp_path / "metadata_truth.yaml").read_text())
    assert len(written["relationships"]) >= 1
