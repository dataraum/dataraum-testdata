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
    assert reg.injections[0].detector_id == "type_fidelity"


def test_introduce_nulls():
    df = pl.DataFrame({"cost_center": ["CC100"] * 100})
    reg = _make_registry()
    result = introduce_nulls(df, "cost_center", ratio=0.15, registry=reg, table_name="test", rng=_make_rng())
    null_count = result["cost_center"].null_count()
    assert null_count >= 10  # ~15 expected
    assert null_count <= 25
    assert reg.injections[0].detector_id == "null_ratio"


def test_inject_outliers():
    df = pl.DataFrame({"credit": [100.0] * 100})
    reg = _make_registry()
    result = inject_outliers(df, "credit", ratio=0.05, factor=10.0, registry=reg, table_name="test", rng=_make_rng())
    values = result["credit"].to_list()
    outliers = [v for v in values if abs(v) > 500]
    assert len(outliers) >= 3
    assert reg.injections[0].detector_id == "outlier_rate"


def test_break_benford():
    rng = _make_rng()
    df = pl.DataFrame({"amount": [10 ** rng.uniform(1, 4) for _ in range(1000)]})
    reg = _make_registry()
    result = break_benford(df, "amount", registry=reg, table_name="test", rng=_make_rng())
    # Check that round numbers were introduced
    values = result["amount"].to_list()
    round_vals = [v for v in values if v % 100 == 0]
    assert len(round_vals) > 100  # Many round numbers injected
    assert reg.injections[0].detector_id == "benford"


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
    assert reg.injections[0].detector_id == "unit_entropy"


def test_corrupt_dates():
    df = pl.DataFrame({"date": ["2025-03-15"] * 50})
    reg = _make_registry()
    result = corrupt_dates(df, "date", ["MM/DD/YYYY", "DD/MM/YYYY"], registry=reg, table_name="test", rng=_make_rng())
    values = result["date"].to_list()
    # Should contain reformatted dates
    non_iso = [v for v in values if "-" not in v or "/" in v]
    assert len(non_iso) > 0
    assert reg.injections[0].detector_id == "temporal_entropy"


def test_break_referential_integrity():
    df = pl.DataFrame({"invoice_id": [f"INV-{i:04d}" for i in range(100)]})
    reg = _make_registry()
    result = break_referential_integrity(df, "invoice_id", ratio=0.05, registry=reg, table_name="test", rng=_make_rng())
    values = result["invoice_id"].to_list()
    orphans = [v for v in values if v.startswith("ORPHAN")]
    assert len(orphans) >= 3
    assert reg.injections[0].detector_id == "relationship_entropy"


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
    assert reg.injections[0].detector_id == "derived_value"


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
    assert reg.injections[0].detector_id == "cross_table_consistency"
    assert reg.injections[0].sub_dimension == "gl_invoice_mismatch"


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
    assert reg.injections[0].detector_id == "cross_table_consistency"
    assert reg.injections[0].sub_dimension == "payment_bank_mismatch"


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
    assert reg.injections[0].detector_id == "derived_value_consistency"
    assert reg.injections[0].sub_dimension == "trial_balance_gl_mismatch"


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
        df, col="debit", slice_col="cost_center", factors=factors, seed=20260714,
        registry=reg, table_name="journal_lines", rng=_make_rng(),
    )

    # One record per (value, factor): exactly the ladder, one factor per distinct value.
    recs = reg.injections
    assert len(recs) == len(factors)
    assert {r.detector_id for r in recs} == {"driver_rankings"}
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
        assert r.parameters["dimension"] == "cost_center"
        assert r.parameters["measure"] == "debit"
        assert r.parameters["family"] == "driver_effect"


def test_inject_driver_effect_reproduces_by_seed():
    """The value→factor assignment is seed-derived and order-independent — same seed,
    same assignment; a different seed may pick a different center for a given factor."""
    factors = [1.2, 1.5, 2.0, 3.0]

    def _assign(seed: int) -> dict[str, float]:
        reg = _make_registry()
        inject_driver_effect(
            _one_sided_journal_lines(), col="debit", slice_col="cost_center", factors=factors,
            seed=seed, registry=reg, table_name="journal_lines", rng=_make_rng(),
        )
        return {r.parameters["value"]: r.parameters["factor"] for r in reg.injections}

    assert _assign(20260714) == _assign(20260714)


def test_inject_driver_effect_needs_enough_values():
    df = pl.DataFrame({"cost_center": ["CC100"] * 50 + ["CC200"] * 50, "debit": [100.0] * 100})
    reg = _make_registry()
    with pytest.raises(ValueError, match="labelled value"):
        inject_driver_effect(
            df, col="debit", slice_col="cost_center", factors=[1.2, 1.5, 2.0, 3.0], seed=1,
            registry=reg, table_name="journal_lines", rng=_make_rng(),
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
    assert "by_detector" in summary
