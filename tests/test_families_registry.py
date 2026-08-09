"""The registry is the single declaration of what tables exist.

These tests are the guarantee the registry exists for: a family added to the
dataset but not declared — or declared without its keys — fails here rather than
silently losing its columns in a key or column-style transform, which is exactly
what happened to the operating chain before the registry existed.
"""

from __future__ import annotations

from testdata.canonical.finance.models import Corpus
from testdata.families import (
    FAMILIES,
    all_tables,
    ambiguous_key_columns,
    default_tables,
    event_fact,
    folds,
    key_columns,
    legacy_names,
    merges,
    natural_key_prefixes,
)
from testdata.schema_transforms import apply_key_strategy, apply_normalization


def test_every_dataset_table_is_declared_by_a_family() -> None:
    declared = set(all_tables())
    on_the_dataset = set(Corpus.model_fields)
    assert on_the_dataset - declared == set(), "dataset tables missing a family declaration"
    assert declared - on_the_dataset == set(), "families declare tables the dataset does not carry"


def test_each_container_fragment_carries_exactly_its_family() -> None:
    """The corpus composes one fragment per family, not one growing list of lists.

    A family declaring a table with no home on the container — or a fragment carrying a
    table no family declares — fails here, at the declaration, rather than at the first
    key transform that quietly skips it.
    """
    from testdata.canonical.finance.models import (
        CoreLedgerTables,
        InventoryTables,
        OperatingChainTables,
        ProbeTables,
    )

    fragments = {
        "core_ledger": CoreLedgerTables,
        "operating_chain": OperatingChainTables,
        "inventory": InventoryTables,
        "probes": ProbeTables,
    }
    by_name = {fam.name: fam for fam in FAMILIES}
    assert set(fragments) == set(by_name), "a family exists without a container fragment"
    for name, fragment in fragments.items():
        assert set(fragment.model_fields) == set(by_name[name].tables), name


def test_exporter_follows_the_registry() -> None:
    """The exporter reads ``Corpus.tables``, so it emits what the families declare."""
    from testdata.canonical.finance.generators import generate_finance_dataset

    corpus = generate_finance_dataset(seed=42, months=1)
    assert list(corpus.tables) == [t for t in all_tables() if t in Corpus.model_fields]
    # Probe tables are empty unless a strategy asks for them, and an empty table is
    # omitted from the export rather than written as a headerless file.
    assert {t for t, rows in corpus.tables.items() if rows} == set(default_tables())


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


# --- the shape transforms follow the registry too ---------------------------


def test_every_declared_merge_and_fold_names_declared_tables() -> None:
    """A transform declaration pointing at a table nobody declares is dead wiring."""
    declared = set(all_tables())
    merged_names = {merge.name for merge in merges()}
    for merge in merges():
        assert {merge.spine, merge.joined} <= declared, merge.name
    for fold in folds():
        assert fold.dimension in declared, fold.dimension
        # A fold's facts are post-merge names, so either a real table or a merge result.
        assert set(fold.into) <= declared | merged_names, fold.concept


def test_a_new_family_reshapes_without_editing_the_transform(monkeypatch) -> None:
    """The §3 exit criterion for schema transforms, exercised rather than asserted.

    A family declares a header/item pair and a dimension fold; `apply_normalization`
    collapses both without knowing either table exists. Before the declarations moved
    into the registry, the merge and inline functions named finance tables in their own
    bodies, so a new family's pair simply never collapsed — silently, and visible only
    as a table count.
    """
    import polars as pl

    from testdata.families import Family, Fold, Merge

    shipments = Family(
        name="shipments",
        description="A stand-in family, declared only for this test.",
        tables=("shipment_headers", "shipment_lines", "carriers"),
        merges=(
            Merge(name="shipment_data", spine="shipment_lines", joined="shipment_headers", on="shipment_id"),
        ),
        folds=(
            Fold(
                concept="carrier",
                dimension="carriers",
                on="carrier_id",
                into={"shipment_data": "shipment_ledger"},
                rename={"name": "carrier_name"},
                attributes=("carrier_name",),
            ),
        ),
    )
    monkeypatch.setattr("testdata.families.FAMILIES", (shipments,))

    frames = {
        "shipment_headers": pl.DataFrame({"shipment_id": ["S1"], "carrier_id": ["C1"]}),
        "shipment_lines": pl.DataFrame({"line_id": ["L1"], "shipment_id": ["S1"], "carrier_id": ["C1"]}),
        "carriers": pl.DataFrame({"carrier_id": ["C1"], "name": ["Fastly"]}),
    }

    partial, mapping = apply_normalization(dict(frames), "partial")
    assert set(partial) == {"shipment_data", "carriers"}
    assert mapping == {"shipment_lines": "shipment_data", "shipment_headers": "shipment_data"}

    flat, mapping = apply_normalization(dict(frames), "flat")
    assert set(flat) == {"shipment_ledger"}
    assert mapping == {"shipment_lines": "shipment_ledger", "shipment_headers": "shipment_ledger"}
    assert "carrier_name" in flat["shipment_ledger"].columns


def test_no_table_arrives_without_structural_truth() -> None:
    """§1's second rule with teeth: a table that lands without its truth is not done.

    Classification is the checkable half — every table a family declares must be a fact,
    a dimension, or explicitly ambiguous. The operating chain shipped a release absent
    from the published structural truth entirely; this is what would have caught it.

    Probe families are exempt: their truth is DATA-conditional (emitted only when a
    strategy materializes the shape), so asserting it unconditionally would publish
    roles for tables the corpus does not carry.
    """
    for fam in FAMILIES:
        if fam.optional:
            continue
        classified = set(fam.structure.facts) | set(fam.structure.dimensions) | set(fam.structure.ambiguous)
        assert classified == set(fam.tables), fam.name


def test_structural_truth_only_speaks_about_declared_tables() -> None:
    """A role or unit on a column of a table nobody declares is unreachable truth."""
    declared = set(all_tables())
    for fam in FAMILIES:
        st = fam.structure
        qualified = (
            set(st.measures)
            | set(st.timestamps)
            | set(st.stock_flow)
            | set(st.reconciles_structurally)
            | set(st.measured_in)
            | set(st.business_concepts)
        )
        for column in qualified:
            assert column.partition(".")[0] in declared, (fam.name, column)


def test_the_corpus_has_exactly_one_event_fact() -> None:
    """The finest grain a structural witness ties to. A subledger with its own finer
    fact says so per measure (`subledger_event_fact`), not by claiming the corpus's."""
    owners = [fam.name for fam in FAMILIES if fam.structure.event_fact]
    assert owners == ["core_ledger"]
    assert event_fact() == "journal_lines"
