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
    # the operating chain's header/item fold (DAT-884)
    "sales_order_lines": "sales_data",
    "sales_orders": "sales_data",
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
        "reconciles_with",
        "cycles",
        "folded_dimensions",
        "degenerate_ids",
        "bus_matrix",
    ):
        assert section in truth, section
    assert len(truth["relationships"]) == 9
    assert set(truth["metric_additivity"]) == {"metrics", "measures"}
    # reconciles_with (DAT-725 P2) is derived: the witness edges cover exactly the
    # structurally-reconciling measures, and the multi-grounding concepts are the
    # >= 2-relation fan-ins of business_concepts.required.
    reconciles = truth["reconciles_with"]
    assert {e["measure"] for e in reconciles["aggregation_lineage"]} == set(
        truth["reconciles_structurally"]
    )
    assert {e["event_table"] for e in reconciles["aggregation_lineage"]} == {"journal_lines"}
    assert {m["concept"]: m["relations"] for m in reconciles["multi_grounding"]} == {
        "account_balance": ["balance_sheet", "trial_balance"],
        "transaction_amount": ["bank_transactions", "invoices", "payments"],
    }
    # folded truth is level-specific — canonical (`full`) folds nothing.
    assert truth["folded_dimensions"] == []
    assert truth["degenerate_ids"] == []
    # canonical bus matrix: the four account-referencing facts, all `referenced`.
    assert truth["bus_matrix"] == {
        "balance_sheet": {"account": {"provenance": "referenced", "key": "account_id"}},
        "bank_transactions": {"account": {"provenance": "referenced", "key": "account_id"}},
        "journal_lines": {"account": {"provenance": "referenced", "key": "account_id"}},
        "trial_balance": {"account": {"provenance": "referenced", "key": "account_id"}},
    }


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


def test_flat_folds_account_dim_into_three_facts() -> None:
    """`flat` folds chart_of_accounts (concept `account`) into general_ledger,
    trial_balance AND balance_sheet — the cross-fact concept-identity case (DAT-757).

    Three, not two: a consumer requiring >= 2 facts per shared dimension has no slack
    at two, so one missed axis silently zeroes the group.
    """
    out = remap_metadata_truth(canonical_metadata_truth(), level="flat")
    account = next(f for f in out["folded_dimensions"] if f["concept"] == "account")
    assert account["fold_key"] == "account_id"
    assert set(account["folded_into"]) == {"general_ledger", "trial_balance", "balance_sheet"}
    assert out["degenerate_ids"] == ["general_ledger.line_id"]


def test_flat_fold_carries_the_coincidental_bijection_attribute() -> None:
    """`opened_date` folds in as an ATTRIBUTE of the account, never an alias of it.

    It is unique per account, so on the fact grain it is 1:1 with account_id and
    statistically identical to the true account_name alias — the negative half the
    dimension-identity judge must reject.
    """
    out = remap_metadata_truth(canonical_metadata_truth(), level="flat")
    account = next(f for f in out["folded_dimensions"] if f["concept"] == "account")
    assert "opened_date" in account["attributes"]
    assert "opened_date" != account["fold_key"]


def test_single_folds_onto_the_mega_table() -> None:
    out = remap_metadata_truth(canonical_metadata_truth(), level="single")
    account = next(f for f in out["folded_dimensions"] if f["concept"] == "account")
    assert account["folded_into"] == ["mega_table"]


def test_flat_bus_matrix_splits_folded_and_key_only() -> None:
    """At `flat` the account concept is folded into the three facts that take the
    inline, while bank_transactions keeps a bare key (CoA removed) — key_only.

    bank_transactions is the surviving key_only case BY DESIGN: its account_id has 2
    distinct values, statistically invisible to any overlap measure, so it is the
    Layer-A blind-spot boundary the harness reports rather than asserts (DAT-756/757).
    """
    out = remap_metadata_truth(canonical_metadata_truth(), level="flat")
    bus = out["bus_matrix"]
    assert bus["general_ledger"]["account"]["provenance"] == "folded"
    assert bus["trial_balance"]["account"]["provenance"] == "folded"
    assert bus["balance_sheet"]["account"] == {"provenance": "folded", "key": "account_id"}
    assert bus["bank_transactions"]["account"] == {"provenance": "key_only", "key": "account_id"}


def test_single_bus_matrix_is_mega_folded() -> None:
    out = remap_metadata_truth(canonical_metadata_truth(), level="single")
    assert out["bus_matrix"] == {
        "mega_table": {"account": {"provenance": "folded", "key": "account_id"}}
    }


def test_flat_drops_ghost_relationships() -> None:
    """A level that REMOVES a table (CoA at flat/single) must not export relationships
    pointing at it — the pre-2c export kept FKs to the nonexistent chart_of_accounts."""
    for level in ("flat", "single"):
        out = remap_metadata_truth(canonical_metadata_truth(), level=level)
        for rel in out["relationships"]:
            for side in ("from", "to"):
                assert not str(rel[side]).startswith("chart_of_accounts."), (level, rel)


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

    # bus matrix binds too: every cell's fact table exists and carries its key column.
    for fact, concepts in truth["bus_matrix"].items():
        assert fact in normalized, f"bus_matrix fact {fact} missing from flat schema"
        for concept, cell in concepts.items():
            assert cell["key"] in set(normalized[fact].columns), (
                f"{fact}.{cell['key']} ({concept}, {cell['provenance']}) missing"
            )


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
    """The mappings we pin the remap to are the ones apply_normalization actually emits
    — for partial AND for the level-default mappings the bus matrix relies on (2c)."""
    from testdata.canonical.finance.generators import generate_finance_dataset
    from testdata.export import dataset_to_dataframes
    from testdata.metadata_truth import _LEVEL_TABLE_MAPPINGS

    dfs = dataset_to_dataframes(generate_finance_dataset(seed=42, months=2))
    _, mapping = apply_normalization(dict(dfs), "partial")
    assert mapping == _PARTIAL_MAPPING
    for level, pinned in _LEVEL_TABLE_MAPPINGS.items():
        _, live = apply_normalization(dict(dfs), level)  # type: ignore[arg-type]
        assert live == pinned, f"{level}: pinned level mapping drifted from the transform"


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


# ---------------------------------------------------------------------------
# measured_in / units (DAT-731, CAP-measured-in-truth)
# ---------------------------------------------------------------------------


def test_measured_in_binds_the_models() -> None:
    """The pairing is INTROSPECTED from the models and compared to the authored truth,
    so the truth cannot drift from the structure it claims to author: every Decimal
    measure on a model with exactly ONE Currency-typed column pairs with that column;
    a multi-Currency model (FXRate — a ratio between currencies) and no-Currency
    models (TrialBalance / BalanceSheet) carry unit_column=None."""
    import typing
    from decimal import Decimal

    from testdata.canonical.finance.models import Currency, FinanceDataset

    expected: dict[str, str | None] = {}
    for table, field in FinanceDataset.model_fields.items():
        (model,) = typing.get_args(field.annotation)
        currency_cols = [
            name for name, f in model.model_fields.items() if f.annotation is Currency
        ]
        measures = [name for name, f in model.model_fields.items() if f.annotation is Decimal]
        measures += [
            name
            for name, c in model.model_computed_fields.items()
            if c.return_type is Decimal
        ]
        for measure in measures:
            expected[f"{table}.{measure}"] = (
                f"{table}.{currency_cols[0]}" if len(currency_cols) == 1 else None
            )

    truth = canonical_metadata_truth()
    authored = {e["measure"]: e["unit_column"] for e in truth["measured_in"]}
    assert authored == expected
    # every entry starts single-currency; fx_rates.rate is flagged dimensionless
    assert all(e["cross_unit"] is False for e in truth["measured_in"])
    flagged = {e["measure"] for e in truth["measured_in"] if e.get("dimensionless")}
    assert flagged == {"fx_rates.rate"}


def test_measured_in_flat_fold_creates_unit_columns() -> None:
    """The CoA fold lands account_currency on the balance facts at flat — the truth
    gains those same-table pairings (mirroring _inline_chart_of_accounts), while the
    line-grain measures keep their OWN currency and merged names carry the renames."""
    flat = remap_metadata_truth(canonical_metadata_truth(), level="flat")
    entries = {e["measure"]: e["unit_column"] for e in flat["measured_in"]}
    assert entries["trial_balance.debit_balance"] == "trial_balance.account_currency"
    assert entries["trial_balance.credit_balance"] == "trial_balance.account_currency"
    assert entries["balance_sheet.ending_balance"] == "balance_sheet.account_currency"
    assert entries["general_ledger.debit"] == "general_ledger.currency"
    assert entries["invoice_data.amount"] == "invoice_data.currency"
    assert entries["invoice_data.payment_amount"] == "invoice_data.payment_currency"
    # canonical (full): no fold, no same-table unit source for the balances
    fe = {e["measure"]: e["unit_column"] for e in canonical_metadata_truth()["measured_in"]}
    assert fe["trial_balance.debit_balance"] is None
    assert fe["balance_sheet.ending_balance"] is None


def test_export_cross_unit_is_data_derived(tmp_path: Path) -> None:
    """cross_unit comes from the FRAMES, never the injection config: a mixed unit
    column flips True, an all-USD one stays False, an absent table is left alone —
    an injector that silently no-ops can never produce a false truth."""
    import polars as pl

    from testdata.metadata_truth import export_metadata_truth as export

    frames = {
        "invoices": pl.DataFrame({"currency": ["USD", "EUR", "USD"]}),
        "payments": pl.DataFrame({"currency": ["USD", "USD", "USD"]}),
    }
    export(tmp_path, dataframes=frames)
    written = yaml.safe_load((tmp_path / "metadata_truth.yaml").read_text())
    flags = {e["measure"]: e["cross_unit"] for e in written["measured_in"]}
    assert flags["invoices.amount"] is True
    assert flags["payments.amount"] is False
    assert flags["journal_lines.debit"] is False  # table absent from frames → default


def test_run_scenario_unit_columns_exist_in_exported_frames(tmp_path: Path) -> None:
    """Truth-integrity bind: every non-null unit_column the truth names must exist in
    the run's exported frames (full AND flat) — the drift guard that keeps the
    pairing honest against schema-transform changes."""
    for level_scenario in ("month-end-close",):
        result = run_scenario(level_scenario, strategy_name="clean", months=2, output_dir=tmp_path)
        frames = result["dataframes"]
        written = yaml.safe_load((tmp_path / "metadata_truth.yaml").read_text())
        for entry in written["measured_in"]:
            if not entry["unit_column"]:
                continue
            table, _, col = entry["unit_column"].partition(".")
            assert table in frames and col in frames[table].columns, entry
            assert entry["cross_unit"] is False  # clean corpus is single-currency
    flat_frames, mapping = apply_normalization(dict(frames), "flat")
    flat_truth = remap_metadata_truth(canonical_metadata_truth(), table_mapping=mapping, level="flat")
    for entry in flat_truth["measured_in"]:
        if not entry["unit_column"]:
            continue
        table, _, col = entry["unit_column"].partition(".")
        assert table in flat_frames and col in flat_frames[table].columns, entry


# ---------------------------------------------------------------------------
# role-playing FKs (DAT-788/DAT-419, CAP-roleplay-fk-fixture)
# ---------------------------------------------------------------------------


def _roleplay_frames() -> dict:
    from testdata.canonical.finance.generators import generate_finance_dataset
    from testdata.entropy.injectors import inject_role_playing_fks
    from testdata.entropy.registry import InjectionRegistry
    import random as _random

    ds = generate_finance_dataset(
        seed=42, months=2, roleplay_addresses=10, roleplay_orders=40, roleplay_deliveries=60
    )
    from testdata.export import dataset_to_dataframes

    dfs = dataset_to_dataframes(ds)
    dfs["orders"] = inject_role_playing_fks(
        df=dfs["orders"], registry=InjectionRegistry(), table_name="orders",
        rng=_random.Random(1), dataframes=dfs, seed=7,
    )
    return dfs


def test_roleplay_truth_is_data_conditional(tmp_path: Path) -> None:
    """The role-play sections appear ONLY when the frames carry the shape; a corpus
    without the probe tables keeps canonical truth exactly (fk_roles empty)."""
    from testdata.metadata_truth import export_metadata_truth as export

    # without the shape: canonical equality (fk_roles stays {})
    export(tmp_path / "clean")
    clean = yaml.safe_load((tmp_path / "clean" / "metadata_truth.yaml").read_text())
    assert clean == canonical_metadata_truth()
    assert clean["fk_roles"] == {}
    assert len(clean["relationships"]) == 9

    # with the shape: fk_roles + roled relationships + table/timestamp roles
    export(tmp_path / "role", dataframes=_roleplay_frames())
    truth = yaml.safe_load((tmp_path / "role" / "metadata_truth.yaml").read_text())
    assert truth["fk_roles"] == {
        "orders.bill_to_addr": "bill_to",
        "orders.ship_to_addr": "ship_to",
        "deliveries.delivery_addr": "ship_to",
    }
    assert len(truth["relationships"]) == 13
    roled = {r["from"]: r.get("fk_role") for r in truth["relationships"] if "fk_role" in r}
    assert roled == truth["fk_roles"]
    assert "orders" in truth["table_roles"]["facts"]
    assert "deliveries" in truth["table_roles"]["facts"]
    assert "addresses" in truth["table_roles"]["dimensions"]
    assert "orders.order_date" in truth["semantic_roles"]["timestamp"]


def test_run_scenario_roleplay_end_to_end(tmp_path: Path) -> None:
    """The sizing gate + injector + conditional truth through the real runner: a
    strategy targeting the orders probe table materializes the shape; entropy_map
    records the genuine_clean legs; metadata_truth carries the role truth."""
    strategy = tmp_path / "roleplay-test.yaml"
    strategy.write_text(
        "name: roleplay-test\n"
        "level: low\n"
        "description: role-play probe shape e2e\n"
        "injections:\n"
        "  - injector: inject_role_playing_fks\n"
        "    table: orders\n"
        "    params:\n"
        "      seed: 20260722\n"
        "      severity: low\n"
    )
    out = tmp_path / "out"
    run_scenario("month-end-close", strategy_file=strategy, months=2, output_dir=out)

    for table in ("addresses", "orders", "deliveries"):
        assert (out / f"{table}.csv").exists(), table
    truth = yaml.safe_load((out / "metadata_truth.yaml").read_text())
    assert len(truth["fk_roles"]) == 3
    assert len(truth["relationships"]) == 13
    emap = yaml.safe_load((out / "entropy_map.yaml").read_text())
    rolefk = [i for i in emap["injections"] if i["injection_type"] == "inject_role_playing_fks"]
    assert len(rolefk) == 3
    assert all(i["parameters"]["stratum"] == "genuine_clean" for i in rolefk)
