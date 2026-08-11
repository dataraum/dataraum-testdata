"""Tests for entropy injection functions."""

import random

import polars as pl
import pytest

from testdata.entropy.injectors import (
    add_duplicate_fk_paths,
    break_benford,
    break_gl_invoice_match,
    break_payment_bank_match,
    break_referential_integrity,
    break_trial_balance,
    corrupt_dates,
    corrupt_types,
    create_mutual_exclusivity,
    drift_formula,
    inject_driver_effect,
    inject_outliers,
    inject_temporal_drift,
    introduce_nulls,
    mix_units,
    obscure_column_names,
)
from testdata.entropy.registry import InjectionRegistry


def _make_registry():
    return InjectionRegistry()


def _make_rng(seed=42):
    return random.Random(seed)


def test_corrupt_types():
    df = pl.DataFrame({"amount": [100.0, 200.0, 300.0, 400.0, 500.0] * 20})
    reg = _make_registry()
    result = corrupt_types(df, "amount", ratio=0.10, registry=reg, table_name="test", rng=_make_rng())
    # Some values should now be non-numeric strings
    str_values = result["amount"].to_list()
    non_numeric = [v for v in str_values if v not in [str(x) for x in [100.0, 200.0, 300.0, 400.0, 500.0]]]
    assert len(non_numeric) > 0
    assert len(reg) == 1
    assert reg.injections[0].defect == "type_fidelity"


def test_introduce_nulls():
    df = pl.DataFrame({"cost_center": ["CC100"] * 100})
    reg = _make_registry()
    result = introduce_nulls(df, "cost_center", ratio=0.15, registry=reg, table_name="test", rng=_make_rng())
    null_count = result["cost_center"].null_count()
    assert null_count >= 10  # ~15 expected
    assert null_count <= 25
    assert reg.injections[0].defect == "completeness"


def test_inject_outliers():
    df = pl.DataFrame({"credit": [100.0] * 100})
    reg = _make_registry()
    result = inject_outliers(df, "credit", ratio=0.05, factor=10.0, registry=reg, table_name="test", rng=_make_rng())
    values = result["credit"].to_list()
    outliers = [v for v in values if abs(v) > 500]
    assert len(outliers) >= 3
    assert reg.injections[0].defect == "distribution"


def test_break_benford():
    rng = _make_rng()
    df = pl.DataFrame({"amount": [10 ** rng.uniform(1, 4) for _ in range(1000)]})
    reg = _make_registry()
    result = break_benford(df, "amount", registry=reg, table_name="test", rng=_make_rng())
    # Check that round numbers were introduced
    values = result["amount"].to_list()
    round_vals = [v for v in values if v % 100 == 0]
    assert len(round_vals) > 100  # Many round numbers injected
    assert reg.injections[0].defect == "distribution"


def test_obscure_column_names():
    df = pl.DataFrame({"vendor_id": ["V-001"], "payment_terms": ["net_30"]})
    reg = _make_registry()
    result = obscure_column_names(df, {"vendor_id": "vid", "payment_terms": "pt"}, registry=reg, table_name="test")
    assert "vid" in result.columns
    assert "pt" in result.columns
    assert "vendor_id" not in result.columns
    assert len(reg) == 2  # One record per renamed column


def test_mix_units():
    df = pl.DataFrame({"amount": [1000.0] * 100})
    reg = _make_registry()
    result = mix_units(df, "amount", "EUR", ratio=0.10, registry=reg, table_name="test", rng=_make_rng(), fx_rate=1.1)
    values = result["amount"].to_list()
    # ~10 values should be 1100.0 (1000 * 1.1)
    converted = [v for v in values if abs(v - 1100.0) < 0.01]
    assert len(converted) >= 5
    assert reg.injections[0].defect == "unit_consistency"


def test_corrupt_dates():
    df = pl.DataFrame({"date": ["2025-03-15"] * 50})
    reg = _make_registry()
    result = corrupt_dates(df, "date", ["MM/DD/YYYY", "DD/MM/YYYY"], registry=reg, table_name="test", rng=_make_rng())
    values = result["date"].to_list()
    # Should contain reformatted dates
    non_iso = [v for v in values if "-" not in v or "/" in v]
    assert len(non_iso) > 0
    assert reg.injections[0].defect == "format_consistency"


def test_break_referential_integrity():
    df = pl.DataFrame({"invoice_id": [f"INV-{i:04d}" for i in range(100)]})
    reg = _make_registry()
    result = break_referential_integrity(df, "invoice_id", ratio=0.05, registry=reg, table_name="test", rng=_make_rng())
    values = result["invoice_id"].to_list()
    orphans = [v for v in values if v.startswith("ORPHAN")]
    assert len(orphans) >= 3
    assert reg.injections[0].defect == "referential_integrity"


def test_add_duplicate_fk_paths():
    df = pl.DataFrame({"entry_id": [f"JE-{i:04d}" for i in range(100)]})
    reg = _make_registry()
    result = add_duplicate_fk_paths(
        df,
        "entry_id",
        "je_ref",
        registry=reg,
        table_name="test",
        rng=_make_rng(),
        noise_ratio=0.10,
    )
    assert "je_ref" in result.columns
    # Most values should match
    matching = sum(1 for a, b in zip(result["entry_id"].to_list(), result["je_ref"].to_list()) if a == b)
    assert matching >= 85


def test_drift_formula():
    df = pl.DataFrame({"debit_balance": [1000.0] * 100})
    reg = _make_registry()
    result = drift_formula(
        df,
        "debit_balance",
        source_cols=["account_id"],
        error_ratio=0.05,
        registry=reg,
        table_name="test",
        rng=_make_rng(),
    )
    values = result["debit_balance"].to_list()
    drifted = [v for v in values if abs(v - 1000.0) > 0.001]
    assert len(drifted) >= 3
    assert reg.injections[0].defect == "derived_consistency"


def test_inject_temporal_drift():
    df = pl.DataFrame(
        {
            "amount": [100.0] * 10,
            "date": [f"2025-{m:02d}-15" for m in range(1, 11)],
        }
    )
    reg = _make_registry()
    result = inject_temporal_drift(
        df,
        "amount",
        "date",
        shift_date="2025-06-01",
        shift_factor=2.0,
        registry=reg,
        table_name="test",
    )
    values = result["amount"].to_list()
    # First 5 months (Jan-May) should be unchanged
    assert all(v == 100.0 for v in values[:5])
    # Months 6+ should be doubled
    assert all(v == 200.0 for v in values[5:])


def test_create_mutual_exclusivity():
    df = pl.DataFrame(
        {
            "debit": [100.0, 0.0, 50.0, 0.0, 200.0],
            "credit": [0.0, 100.0, 50.0, 200.0, 0.0],
        }
    )
    reg = _make_registry()
    result = create_mutual_exclusivity(df, "debit", "credit", registry=reg, table_name="test", rng=_make_rng())
    # Row 2 had both non-zero: one should now be zero
    debit_vals = result["debit"].to_list()
    credit_vals = result["credit"].to_list()
    for d, c in zip(debit_vals, credit_vals):
        assert d == 0.0 or c == 0.0, f"Both populated: debit={d}, credit={c}"


def test_break_gl_invoice_match():
    df = pl.DataFrame({"amount": [1000.0] * 100})
    reg = _make_registry()
    result = break_gl_invoice_match(
        df,
        "amount",
        ratio=0.10,
        registry=reg,
        table_name="invoices",
        rng=_make_rng(),
    )
    values = result["amount"].to_list()
    changed = [v for v in values if abs(v - 1000.0) > 0.01]
    assert len(changed) >= 5  # ~10% of 100
    assert len(reg) == 1
    assert reg.injections[0].defect == "cross_table_consistency"
    assert reg.injections[0].defect_detail == "gl_invoice_mismatch"


def test_break_payment_bank_match():
    df = pl.DataFrame({"amount": [500.0] * 100})
    reg = _make_registry()
    result = break_payment_bank_match(
        df,
        "amount",
        ratio=0.08,
        registry=reg,
        table_name="payments",
        rng=_make_rng(),
    )
    values = result["amount"].to_list()
    changed = [v for v in values if abs(v - 500.0) > 0.01]
    assert len(changed) >= 4  # ~8% of 100
    assert len(reg) == 1
    assert reg.injections[0].defect == "cross_table_consistency"
    assert reg.injections[0].defect_detail == "payment_bank_mismatch"


def test_break_trial_balance():
    df = pl.DataFrame({"debit_balance": [10000.0] * 100})
    reg = _make_registry()
    result = break_trial_balance(
        df,
        "debit_balance",
        ratio=0.10,
        registry=reg,
        table_name="trial_balance",
        rng=_make_rng(),
    )
    values = result["debit_balance"].to_list()
    changed = [v for v in values if abs(v - 10000.0) > 0.01]
    assert len(changed) >= 5
    assert len(reg) == 1
    assert reg.injections[0].defect == "cross_table_consistency"
    assert reg.injections[0].defect_detail == "trial_balance_gl_mismatch"


def _one_sided_journal_lines() -> pl.DataFrame:
    """A one-sided-measure fixture: debit nonzero on DR rows, zero on CR rows, across
    five cost_centers + a null slice (a null slice is no slice)."""
    centers = ["CC100", "CC200", "CC300", "CC400", "CC500"]
    cc: list[str | None] = []
    debit: list[float] = []
    for c in centers:
        for k in range(20):  # DR lines: nonzero
            cc.append(c)
            debit.append(100.0 + k)
        for _ in range(20):  # CR lines: zero (one-sided)
            cc.append(c)
            debit.append(0.0)
    for k in range(15):  # unlabelled rows — never scaled
        cc.append(None)
        debit.append(50.0 + k)
    return pl.DataFrame({"cost_center": cc, "debit": debit})


def test_inject_driver_effect():
    df = _one_sided_journal_lines()
    original = df["debit"].to_list()
    factors = [1.2, 1.5, 2.0, 3.0]
    reg = _make_registry()

    result = inject_driver_effect(
        df,
        col="debit",
        slice_col="cost_center",
        factors=factors,
        seed=20260714,
        registry=reg,
        table_name="journal_lines",
        rng=_make_rng(),
    )

    # One record per (value, factor): exactly the ladder, one factor per distinct value.
    recs = reg.injections
    assert len(recs) == len(factors)
    assert {r.defect for r in recs} == {"driver_effect"}
    assert sorted(r.parameters["factor"] for r in recs) == factors
    assigned = {r.parameters["value"]: r.parameters["factor"] for r in recs}
    assert len(assigned) == len(factors)  # distinct values
    assert set(assigned) < {"CC100", "CC200", "CC300", "CC400", "CC500"}  # one left as reference

    # Every nonzero debit in a scaled group is exactly f x original; zeros stay zero
    # (scale-invariant); the reference center and the null slice are untouched.
    ccs = result["cost_center"].to_list()
    new = result["debit"].to_list()
    for i, c in enumerate(ccs):
        factor = assigned.get(c)
        if factor is None:
            assert new[i] == original[i]  # reference / unlabelled unchanged
        else:
            assert new[i] == round(original[i] * factor, 2)

    # target_rows are exactly the materially-moved (nonzero) rows of each scaled group.
    for r in recs:
        value = r.parameters["value"]
        expected = sorted(i for i, c in enumerate(ccs) if c == value and original[i] != 0.0)
        assert r.target_rows == expected
        assert r.parameters["n_scaled"] == len(expected) > 0
        assert r.parameters["slice_column"] == "cost_center"
        assert r.parameters["measure"] == "debit"
        assert r.parameters["family"] == "driver_effect"


def test_inject_driver_effect_reproduces_by_seed():
    """The value→factor assignment is seed-derived and order-independent — same seed,
    same assignment; a different seed may pick a different center for a given factor."""
    factors = [1.2, 1.5, 2.0, 3.0]

    def _assign(seed: int) -> dict[str, float]:
        reg = _make_registry()
        inject_driver_effect(
            _one_sided_journal_lines(),
            col="debit",
            slice_col="cost_center",
            factors=factors,
            seed=seed,
            registry=reg,
            table_name="journal_lines",
            rng=_make_rng(),
        )
        return {r.parameters["value"]: r.parameters["factor"] for r in reg.injections}

    assert _assign(20260714) == _assign(20260714)


def test_inject_driver_effect_needs_enough_values():
    df = pl.DataFrame({"cost_center": ["CC100"] * 50 + ["CC200"] * 50, "debit": [100.0] * 100})
    reg = _make_registry()
    with pytest.raises(ValueError, match="labelled value"):
        inject_driver_effect(
            df,
            col="debit",
            slice_col="cost_center",
            factors=[1.2, 1.5, 2.0, 3.0],
            seed=1,
            registry=reg,
            table_name="journal_lines",
            rng=_make_rng(),
        )


def test_registry_summary():
    reg = _make_registry()
    rng = _make_rng()
    df = pl.DataFrame({"amount": [100.0] * 100})
    corrupt_types(df, "amount", 0.05, reg, "t1", rng)
    introduce_nulls(df, "amount", 0.05, reg, "t2", rng)
    assert len(reg) == 2
    summary = reg.summary()
    assert "by_layer" in summary
    assert "by_defect" in summary


def test_mix_units_declared_flips_unit_column():
    """The declared variant (unit_col given) writes alt_currency into the unit column
    on exactly the converted rows — a genuinely multi-currency, honestly-declared
    table (the shape the cross-unit gate must flag)."""
    df = pl.DataFrame({"amount": [1000.0] * 100, "currency": ["USD"] * 100})
    reg = _make_registry()
    result = mix_units(
        df,
        "amount",
        "EUR",
        ratio=0.10,
        registry=reg,
        table_name="test",
        rng=_make_rng(),
        fx_rate=1.1,
        unit_col="currency",
    )
    units = result["currency"].to_list()
    values = result["amount"].to_list()
    eur_rows = [i for i, u in enumerate(units) if u == "EUR"]
    assert len(eur_rows) >= 5
    # declaration and conversion move together: every EUR row carries a converted value
    assert all(abs(values[i] - 1100.0) < 0.01 for i in eur_rows)
    assert reg.injections[0].defect_detail == "declared_mixed_currency"
    assert reg.injections[0].parameters["unit_col"] == "currency"


def test_mix_units_undeclared_leaves_unit_column_untouched():
    """The original shape: values converted, unit column untouched — invisible to any
    reader of the declared unit surface, so data-derived cross_unit stays False."""
    df = pl.DataFrame({"amount": [1000.0] * 100, "currency": ["USD"] * 100})
    reg = _make_registry()
    result = mix_units(
        df,
        "amount",
        "EUR",
        ratio=0.10,
        registry=reg,
        table_name="test",
        rng=_make_rng(),
        fx_rate=1.1,
    )
    assert set(result["currency"].to_list()) == {"USD"}
    assert reg.injections[0].defect_detail == "mixed_currency"
    assert reg.injections[0].parameters["unit_col"] is None


def test_inject_role_playing_fks_shape_and_records():
    """The role-play family (DAT-788/DAT-419): FK integrity into the address
    dimension, role consistency BY CONSTRUCTION (delivery_addr == parent order's
    ship_to), a same-address trap fraction, and one genuine_clean record per role
    column. Deterministic under an explicit seed."""
    from testdata.canonical.finance.generators import generate_finance_dataset
    from testdata.entropy.injectors import inject_role_playing_fks
    from testdata.export import dataset_to_dataframes

    ds = generate_finance_dataset(seed=42, months=2, roleplay_addresses=10, roleplay_orders=50, roleplay_deliveries=80)
    dfs = dataset_to_dataframes(ds)
    reg = _make_registry()
    dfs["orders"] = inject_role_playing_fks(
        df=dfs["orders"],
        registry=reg,
        table_name="orders",
        rng=_make_rng(),
        dataframes=dfs,
        seed=7,
    )

    addr_ids = set(dfs["addresses"]["address_id"].to_list())
    bill = dfs["orders"]["bill_to_addr"].to_list()
    ship = dfs["orders"]["ship_to_addr"].to_list()
    assert set(bill) <= addr_ids and set(ship) <= addr_ids
    assert any(b == s for b, s in zip(bill, ship))  # the same-address trap leg
    assert any(b != s for b, s in zip(bill, ship))  # roles genuinely distinct

    ship_by_order = dict(zip(dfs["orders"]["order_id"].to_list(), ship))
    for oid, daddr in zip(dfs["deliveries"]["order_id"].to_list(), dfs["deliveries"]["delivery_addr"].to_list()):
        assert daddr == ship_by_order[oid]  # role consistency by construction

    assert len(reg.injections) == 3
    assert all(i.parameters["stratum"] == "genuine_clean" for i in reg.injections)
    assert {(i.target_file, i.target_column, i.parameters["fk_role"]) for i in reg.injections} == {
        ("orders.csv", "bill_to_addr", "bill_to"),
        ("orders.csv", "ship_to_addr", "ship_to"),
        ("deliveries.csv", "delivery_addr", "ship_to"),
    }

    # determinism: same explicit seed -> identical assignment
    dfs2 = dataset_to_dataframes(ds)
    dfs2["orders"] = inject_role_playing_fks(
        df=dfs2["orders"],
        registry=_make_registry(),
        table_name="orders",
        rng=random.Random(999),
        dataframes=dfs2,
        seed=7,
    )
    assert dfs2["orders"]["bill_to_addr"].to_list() == bill
    assert dfs2["deliveries"]["delivery_addr"].to_list() == dfs["deliveries"]["delivery_addr"].to_list()


# --- Scoped injections (A4) ---


def _mid_frames():
    """The reference corpus as frames — the grain scoped injections target."""
    import functools

    from testdata.canonical.finance.generators import generate_finance_dataset
    from testdata.export import dataset_to_dataframes

    @functools.lru_cache(maxsize=1)
    def _build():
        return generate_finance_dataset(seed=7, months=12, profile="mid")

    return dataset_to_dataframes(_build())


def test_slice_rows_reaches_dimensions_through_declared_joins():
    """A table is scoped by a path, never by a guessed column name."""
    from testdata.entropy.scoping import UnscopableTable, slice_rows

    dfs = _mid_frames()
    lines = dfs["sales_order_lines"]

    # customer-side, reached through sales_orders; product-side, local; and time.
    by_segment = slice_rows(lines, "sales_order_lines", dfs, scope={"segment": ["Enterprise"]})
    by_group = slice_rows(lines, "sales_order_lines", dfs, scope={"product_group": ["Instruments"]})
    by_period = slice_rows(lines, "sales_order_lines", dfs, periods=["2025-09"])
    assert 0 < len(by_segment) < lines.height
    assert 0 < len(by_group) < lines.height
    assert 0 < len(by_period) < lines.height

    # Dimensions INTERSECT, as a lever's scope does.
    both = slice_rows(lines, "sales_order_lines", dfs, scope={"segment": ["Enterprise"]}, periods=["2025-09"])
    assert set(both) == set(by_segment) & set(by_period)

    # Unscoped is every row, with no join performed.
    assert slice_rows(lines, "sales_order_lines", dfs) == list(range(lines.height))

    # A table with no declared path REFUSES rather than silently injecting table-wide.
    with pytest.raises(UnscopableTable):
        slice_rows(dfs["fx_rates"], "fx_rates", dfs, scope={"segment": ["Enterprise"]})


def test_scoped_mix_units_lands_only_on_its_slice():
    from testdata.entropy.injectors import mix_units
    from testdata.entropy.scoping import slice_rows

    dfs = _mid_frames()
    lines = dfs["sales_order_lines"]
    before = lines["line_amount"].to_list()
    reg = _make_registry()

    scope, periods = {"segment": ["Enterprise"]}, ["2025-09", "2025-10"]
    out = mix_units(
        df=lines,
        col="line_amount",
        alt_currency="EUR",
        ratio=1.0,
        fx_rate=1.15,
        registry=reg,
        table_name="sales_order_lines",
        rng=random.Random(1),
        dataframes=dfs,
        scope=scope,
        periods=periods,
    )
    expected = set(slice_rows(lines, "sales_order_lines", dfs, scope=scope, periods=periods))
    after = out["line_amount"].to_list()
    moved = {i for i in range(len(before)) if before[i] != after[i]}
    assert moved == expected

    params = reg.injections[0].parameters
    assert params["scope"] == scope and params["periods"] == periods
    assert params["slice_rows"] == len(expected)


def test_the_confusable_pair_is_matched_by_construction():
    """The sharpest test in the brief: same apparent magnitude, different cause.

    A `ratio: 1.0` unit-mix at fx F over a slice, and a price `rate` lever with
    factor F over the SAME slice, are the same arithmetic on the same rows — so the
    pair is matched analytically rather than tuned. What separates them is only
    whether the business changed: under the lever units, discount and customer
    behaviour are consistent with the higher amounts; under the artifact nothing
    else moved, because nothing else happened.
    """
    from decimal import Decimal

    from testdata.canonical.finance.generators import Lever, generate_finance_dataset
    from testdata.entropy.injectors import mix_units
    from testdata.export import dataset_to_dataframes

    kw = dict(seed=7, months=12, profile="mid")
    scope = {"segment": ["Enterprise"]}
    periods = [f"2025-{m:02d}" for m in range(7, 13)]  # period_k=6 onward

    base = generate_finance_dataset(**kw)
    levered = generate_finance_dataset(
        **kw, lever=Lever(period_k=6, type="rate", driver="price", factor=1.15, scope=scope)
    )
    lever_delta = float(
        sum(line.line_amount for line in levered.sales_order_lines)
        - sum(line.line_amount for line in base.sales_order_lines)
    )

    dfs = dataset_to_dataframes(base)
    reg = _make_registry()
    mix_units(
        df=dfs["sales_order_lines"],
        col="line_amount",
        alt_currency="EUR",
        ratio=1.0,
        fx_rate=1.15,
        registry=reg,
        table_name="sales_order_lines",
        rng=random.Random(1),
        dataframes=dfs,
        scope=scope,
        periods=periods,
    )
    artifact_delta = reg.injections[0].parameters["apparent_delta"]

    # Matched to rounding — the two must be indistinguishable on magnitude alone.
    assert abs(artifact_delta / lever_delta - 1.0) < 1e-4, f"artifact {artifact_delta} vs lever {lever_delta}"

    # And genuinely different underneath: the lever moved unit prices, the artifact
    # left every unit price exactly where it was.
    base_price = {line.order_line_id: line.unit_price for line in base.sales_order_lines}
    lev_price = {line.order_line_id: line.unit_price for line in levered.sales_order_lines}
    assert any(base_price[k] != lev_price[k] for k in base_price)
    assert dfs["sales_order_lines"]["unit_price"].to_list() == [
        float(base_price[k]) if isinstance(base_price[k], Decimal) else base_price[k]
        for k in dfs["sales_order_lines"]["order_line_id"].to_list()
    ]


def test_delivery_gap_records_keys_because_indices_do_not_survive_deletion():
    """The one injector that changes the row count, and why its truth differs.

    Every other injector mutates in place, so `target_rows` are positions a consumer
    can look up. Here the rows are gone and a position would name whichever row
    shifted up into it — an innocent survivor labelled as the defect.
    """
    from testdata.entropy.injectors import drop_slice_rows

    dfs = _mid_frames()
    lines = dfs["sales_order_lines"]
    reg = _make_registry()

    out = drop_slice_rows(
        df=lines,
        registry=reg,
        table_name="sales_order_lines",
        rng=random.Random(1),
        dataframes=dfs,
        scope={"segment": ["Enterprise"]},
        periods=["2025-09"],
    )
    record = reg.injections[0]
    params = record.parameters

    assert record.target_rows == [], "positional indices cannot name deleted rows"
    assert params["removed_count"] > 0
    assert params["rows_before"] - params["removed_count"] == params["rows_after"] == out.height

    # The keys name rows that are genuinely absent, and only those.
    survivors = set(out["order_line_id"].to_list())
    removed = set(params["removed_keys"])
    assert removed and not (removed & survivors)
    assert survivors | removed == set(lines["order_line_id"].to_list())


def test_scoped_duplicate_fk_path_diverges_only_inside_the_slice():
    """The second path exists everywhere; only the scoped rows disagree with it."""
    from testdata.entropy.injectors import add_duplicate_fk_paths
    from testdata.entropy.scoping import slice_rows

    dfs = _mid_frames()
    lines = dfs["sales_order_lines"]
    scope = {"product_group": ["Instruments"]}

    reg = _make_registry()
    out = add_duplicate_fk_paths(
        df=lines,
        existing_fk_col="order_id",
        new_col_name="order_ref",
        registry=reg,
        table_name="sales_order_lines",
        rng=random.Random(1),
        noise_ratio=0.2,
        dataframes=dfs,
        scope=scope,
    )
    assert out.height == lines.height, "the column is added, no rows are dropped"
    assert reg.injections[0].parameters["scope"] == scope

    diverged = {i for i, (a, b) in enumerate(zip(out["order_id"].to_list(), out["order_ref"].to_list())) if a != b}
    assert diverged and diverged <= set(slice_rows(lines, "sales_order_lines", dfs, scope=scope))


def test_repeated_orphan_is_the_most_common_value_not_a_rare_one():
    """One orphan reused everywhere — every rare-value heuristic goes silent."""
    df = pl.DataFrame({"invoice_id": [f"INV-{i:04d}" for i in range(200)]})
    reg = _make_registry()
    out = break_referential_integrity(
        df,
        "invoice_id",
        ratio=0.15,
        registry=reg,
        table_name="payments",
        rng=_make_rng(),
        variant="repeated_orphan",
    )
    values = out["invoice_id"].to_list()
    orphans = {v for v in values if v.startswith("ORPHAN")}
    assert len(orphans) == 1, "the whole point: ONE invented id, not one per row"

    orphan = orphans.pop()
    counts = {v: values.count(v) for v in set(values)}
    top = max(counts, key=lambda v: counts[v])
    assert top == orphan, "the orphan is the column's MOST frequent key, not a rare one"
    assert counts[orphan] == 30

    rec = reg.injections[0]
    assert rec.defect == "referential_integrity"
    assert rec.defect_detail == "repeated_orphan_foreign_key"
    assert rec.parameters["orphan_value"] == orphan
    assert rec.parameters["distinct_orphans"] == 1
    assert len(rec.target_rows) == 30


def test_shuffled_fks_all_resolve_and_leave_the_value_distribution_alone():
    """Every value exists in the parent; only the pairing is wrong."""
    parent = pl.DataFrame({"invoice_id": [f"INV-{i:04d}" for i in range(200)]})
    df = pl.DataFrame({"invoice_id": [f"INV-{i:04d}" for i in range(200)]})
    reg = _make_registry()
    out = break_referential_integrity(
        df,
        "invoice_id",
        ratio=0.2,
        registry=reg,
        table_name="payments",
        rng=_make_rng(),
        variant="shuffled",
    )
    before, after = df["invoice_id"].to_list(), out["invoice_id"].to_list()

    known = set(parent["invoice_id"].to_list())
    assert set(after) <= known, "no invented ids — every FK still resolves"
    assert sorted(after) == sorted(before), "the multiset is untouched; only the pairing moved"

    rec = reg.injections[0]
    assert rec.defect_detail == "shuffled_foreign_keys"
    assert rec.parameters["all_values_exist_in_parent"] is True

    # target_rows names exactly the rows that moved, and correct_values repairs them.
    moved = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert moved == rec.target_rows
    assert [before[i] for i in moved] == rec.parameters["correct_values"]


def test_banding_keeps_the_shuffle_a_relationship_defect_not_an_amount_defect():
    """Unbanded, a swap also breaks amount agreement — a detector wins without joining."""
    amounts = [float(i) for i in range(200)]
    frames = {
        "payments": pl.DataFrame({"invoice_id": [f"INV-{i:04d}" for i in range(200)], "amount": amounts}),
        # the parent, so "what SHOULD this row's amount be" is answerable
        "invoices": pl.DataFrame({"invoice_id": [f"INV-{i:04d}" for i in range(200)], "amount": amounts}),
    }
    owed = dict(zip(frames["invoices"]["invoice_id"].to_list(), frames["invoices"]["amount"].to_list()))

    def worst_disagreement(variant_kwargs):
        out = break_referential_integrity(
            df=frames["payments"],
            fk_col="invoice_id",
            ratio=1.0,
            registry=_make_registry(),
            table_name="payments",
            rng=_make_rng(3),
            variant="shuffled",
            **variant_kwargs,
        )
        return max(abs(owed[fk] - amt) for fk, amt in zip(out["invoice_id"].to_list(), out["amount"].to_list()))

    unbanded = worst_disagreement({})
    banded = worst_disagreement({"within_col": "amount", "bands": 8})
    assert banded < unbanded / 4, (
        "banded swaps stay inside a magnitude band, so the amounts still agree and only the relationship is wrong"
    )


def test_the_fk_variants_are_one_family_with_one_answer_key_shape():
    """Three shapes, one record shape — defect_detail is what separates them."""
    records = []
    for variant in ("distinct_orphan", "repeated_orphan", "shuffled"):
        df = pl.DataFrame({"invoice_id": [f"INV-{i:04d}" for i in range(100)]})
        reg = _make_registry()
        break_referential_integrity(
            df,
            "invoice_id",
            ratio=0.1,
            registry=reg,
            table_name="payments",
            rng=_make_rng(),
            variant=variant,
        )
        records.append(reg.injections[0])

    assert {r.defect for r in records} == {"referential_integrity"}
    assert {r.injection_type for r in records} == {"break_referential_integrity"}
    assert len({r.defect_detail for r in records}) == 3
    assert all(r.parameters["ratio"] == 0.1 for r in records)
    assert all(r.target_rows for r in records)


def test_the_default_fk_record_says_exactly_what_it_always_did():
    """Opt-in only: no variant, no scope → the record does not grow."""
    df = pl.DataFrame({"invoice_id": [f"INV-{i:04d}" for i in range(100)]})
    reg = _make_registry()
    break_referential_integrity(df, "invoice_id", ratio=0.05, registry=reg, table_name="test", rng=_make_rng())
    assert reg.injections[0].parameters == {"ratio": 0.05}
    assert reg.injections[0].defect_detail == "orphaned_foreign_keys"


def test_scoped_fk_breakage_lands_on_one_feed_and_no_other():
    """A bad feed breaks its own slice's keys, not the table's."""
    dfs = _mid_frames()
    orders = dfs["sales_orders"]
    scope = {"segment": ["Enterprise"]}

    reg = _make_registry()
    out = break_referential_integrity(
        df=orders,
        fk_col="customer_id",
        ratio=0.3,
        registry=reg,
        table_name="sales_orders",
        rng=random.Random(5),
        variant="repeated_orphan",
        dataframes=dfs,
        scope=scope,
    )
    broken = {i for i, v in enumerate(out["customer_id"].to_list()) if str(v).startswith("ORPHAN")}
    enterprise = set(
        i
        for i, v in enumerate(orders["customer_id"].to_list())
        if v in set(dfs["customers"].filter(pl.col("segment") == "Enterprise")["customer_id"].to_list())
    )
    assert broken and broken <= enterprise
    assert reg.injections[0].parameters["scope"] == scope
    assert 0 < reg.injections[0].parameters["slice_share"] < 1


def test_exact_value_swaps_leave_the_amount_surface_untouched():
    """The honest hard case: nothing numeric is wrong, only the pairing."""
    dfs = _mid_frames()
    payments, invoices = dfs["payments"], dfs["invoices"]
    owed = dict(zip(invoices["invoice_id"].to_list(), invoices["amount"].to_list()))
    invoice_date = dict(zip(invoices["invoice_id"].to_list(), invoices["date"].to_list()))

    def exact_match_rate(df):
        pairs = zip(df["invoice_id"].to_list(), df["amount"].to_list())
        agree = [abs(float(owed[fk]) - float(amt)) < 0.005 for fk, amt in pairs if fk in owed]
        return sum(agree) / len(agree)

    reg = _make_registry()
    out = break_referential_integrity(
        df=payments,
        fk_col="invoice_id",
        ratio=0.10,
        registry=reg,
        table_name="payments",
        rng=random.Random(1043),
        variant="shuffled",
        within_col="amount",
        bands=None,
    )
    rec = reg.injections[0]

    # Every swap is between rows owing the identical amount, so the agreement rate is
    # not merely close to the clean one — it is the SAME rate, to the cent.
    assert exact_match_rate(out) == exact_match_rate(payments)
    assert rec.parameters["grouping"] == "exact_value"
    assert rec.parameters["eligible_rows"] > len(rec.target_rows)

    # And yet the relationship is genuinely broken: a real share of the moved
    # payments now claim to settle an invoice that did not exist yet.
    fks, dates = out["invoice_id"].to_list(), out["date"].to_list()
    early = sum(1 for i in rec.target_rows if dates[i] < invoice_date[fks[i]])
    assert early / len(rec.target_rows) > 0.2, (
        "a defect nothing at all can see is not a test — the temporal incoherence "
        "is what a consumer reasoning about the join is supposed to find"
    )


def test_the_shuffled_strategy_ships_a_repairable_answer_key():
    """correct_values restores the corpus exactly, which is what makes repair gradeable."""
    from testdata.scenarios.runner import run_scenario

    out = run_scenario("month-end-close", strategy_name="fk-shuffled", seed=42, months=12)
    rec = next(i for i in out["registry"].injections if i.defect == "referential_integrity")
    broken = out["dataframes"]["payments"]["invoice_id"].to_list()

    repaired = list(broken)
    for row, correct in zip(rec.target_rows, rec.parameters["correct_values"], strict=True):
        repaired[row] = correct

    clean = run_scenario("month-end-close", strategy_name="clean", seed=42, months=12)
    assert repaired == clean["dataframes"]["payments"]["invoice_id"].to_list()
    assert repaired != broken
