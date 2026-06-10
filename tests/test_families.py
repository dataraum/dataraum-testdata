"""Generative injection families — the null_tokens family + injector (DAT-450).

A family is parameterized + seed-recorded: different seeds → different surface,
same semantics; the recorded seed reproduces exactly (AC1). Markers (is-null)
cluster, decoys (is-value) smear, and both are recorded as ground-truth labels
the calibration rig scores witnesses against.
"""

import random

import polars as pl
import pytest

from testdata.entropy.families import (
    CURATED_VOCAB,
    NullTokenFamilyParams,
    StockFlowFamilyParams,
    mint_decoy,
    sample_mixed_units_family,
    sample_null_token_family,
    sample_stock_flow_family,
)
from testdata.entropy.injectors import (
    inject_null_token_family,
    inject_scale_mix,
    inject_stock_flow_probes,
)
from testdata.entropy.registry import InjectionRegistry


def test_recorded_seed_reproduces_exactly() -> None:
    a = sample_null_token_family(20260609)
    b = sample_null_token_family(20260609)
    assert a == b


def test_different_seeds_vary_the_surface() -> None:
    # Across a spread of seeds the sampled marker sets are mostly distinct — a
    # detector cannot memorize one fixed token list.
    marker_sets = {sample_null_token_family(s).markers for s in range(40)}
    assert len(marker_sets) > 30


def test_sample_is_well_formed() -> None:
    for s in range(50):
        fam = sample_null_token_family(s)
        assert 2 <= len(fam.markers) <= 6
        assert len(set(fam.markers)) == len(fam.markers)  # distinct
        assert set(fam.in_vocab_markers) <= set(fam.markers)
        assert set(fam.in_vocab_markers) <= set(CURATED_VOCAB)
        assert 0.0 <= fam.vocab_coverage <= 1.0
        assert 0.05 <= fam.marker_ratio <= 0.075
        assert 0.015 <= fam.decoy_ratio <= 0.025
        # Combined cast-failure rate stays under the typing min_confidence margin
        # (0.85) so the corrupted column still infers numeric and quarantines.
        assert fam.marker_ratio + fam.decoy_ratio <= 0.10


def test_combined_ratio_guard_rejects_over_the_typing_threshold() -> None:
    # A strategy override that would push the corrupted column below typing
    # min_confidence (0.85) — and thus to VARCHAR, never quarantined — is rejected
    # at construction, not silently shipped (the DAT-450 live-run failure mode).
    with pytest.raises(ValueError, match="min_confidence"):
        NullTokenFamilyParams(marker_ratio=(0.05, 0.12), decoy_ratio=(0.02, 0.05))  # upper 0.17
    NullTokenFamilyParams()  # defaults (0.075 + 0.025 = 0.10) are safe → no raise


def test_decoy_cluster_size_zero_by_default_and_sampled_when_set() -> None:
    assert sample_null_token_family(5).decoy_cluster_size == 0  # distinct decoys (smear)
    stress = sample_null_token_family(5, NullTokenFamilyParams(decoy_cluster_size=(2, 4)))
    assert 2 <= stress.decoy_cluster_size <= 4  # clustered is-value stress mode


def test_params_override_the_space() -> None:
    fam = sample_null_token_family(7, NullTokenFamilyParams(n_markers=(3, 3), vocab_coverage=(1.0, 1.0)))
    assert len(fam.markers) == 3
    assert set(fam.markers) == set(fam.in_vocab_markers)  # full coverage → all in vocab
    assert fam.vocab_coverage == 1.0


def test_decoys_are_genuine_unparseable_distinct_values() -> None:
    rng = random.Random(1)
    minted = [mint_decoy(rng, "currency") for _ in range(20)]
    # Genuine amounts that fail a plain float() cast (the quarantine condition).
    for value in minted:
        try:
            float(value)
            raise AssertionError(f"{value!r} parsed as a float — not a quarantined decoy")
        except ValueError:
            pass
    assert len(set(minted)) > 15  # distinct → they smear, not cluster


def _frame(n: int = 200) -> pl.DataFrame:
    return pl.DataFrame({"debit": [float(i) for i in range(n)]})


def test_injector_labels_markers_and_decoys() -> None:
    reg = InjectionRegistry()
    df = inject_null_token_family(
        _frame(), col="debit", seed=20260609, registry=reg,
        table_name="journal_lines", rng=random.Random(99),
    )
    (inj,) = reg.injections
    p = inj.parameters
    col = df["debit"].to_list()

    # Marker rows carry one of the marker set; decoy rows carry a minted decoy.
    assert {col[i] for i in p["marker_rows"]} <= set(p["markers"])
    assert {col[i] for i in p["decoy_rows"]} == set(p["decoys"]) or set(p["decoys"]).issuperset(
        {col[i] for i in p["decoy_rows"]}
    )
    # Marker and decoy rows are disjoint, and both are recorded as ground truth.
    assert not (set(p["marker_rows"]) & set(p["decoy_rows"]))
    assert inj.detector_id == "null_semantics"
    assert p["seed"] == 20260609


def test_mixed_units_family_reproduces_and_varies() -> None:
    assert sample_mixed_units_family(7) == sample_mixed_units_family(7)  # recorded seed reproduces
    for s in range(30):
        fam = sample_mixed_units_family(s)
        assert fam.scale_factor in (100, 1000, 10000)  # a clean decade, not a ×1.1 currency
        assert 0.15 <= fam.mix_ratio <= 0.40
    surfaces = {(sample_mixed_units_family(s).scale_factor, sample_mixed_units_family(s).mix_ratio) for s in range(30)}
    assert len(surfaces) > 10  # different seeds → different surface


def test_inject_scale_mix_records_and_scales() -> None:
    base = [float(100 + i) for i in range(200)]  # one scale (~100–300)
    df = pl.DataFrame({"amount": base})
    reg = InjectionRegistry()
    out = inject_scale_mix(
        df, col="amount", seed=42, registry=reg, table_name="invoices", rng=random.Random(0)
    )
    (inj,) = reg.injections
    assert inj.detector_id == "unit_consistency"
    assert inj.injection_type == "inject_scale_mix"
    scale = inj.parameters["scale_factor"]
    col = out["amount"].to_list()
    # the recorded rows are the base value × the scale factor; the rest are untouched.
    assert inj.target_rows
    for i in inj.target_rows:
        assert abs(col[i] - base[i] * scale) < 0.01
    untouched = set(range(200)) - set(inj.target_rows)
    assert all(col[i] == base[i] for i in untouched)


def test_injection_is_reproducible_from_the_seed() -> None:
    def run() -> list[object]:
        reg = InjectionRegistry()
        df = inject_null_token_family(
            _frame(), col="debit", seed=42, registry=reg,
            table_name="journal_lines", rng=random.Random(random.randint(0, 1_000_000)),
        )
        return df["debit"].to_list()

    # The shared `rng` differs between runs; the family seed fixes the result.
    assert run() == run()


# --- stock/flow family (DAT-445) -------------------------------------------


def test_stock_flow_family_reproduces_and_varies() -> None:
    assert sample_stock_flow_family(7) == sample_stock_flow_family(7)  # recorded seed reproduces
    name_sets = {
        tuple(c.name for c in sample_stock_flow_family(s).columns) for s in range(40)
    }
    assert len(name_sets) > 30  # different seeds → a different name surface


def test_stock_flow_sample_is_well_formed() -> None:
    for s in range(40):
        fam = sample_stock_flow_family(s)
        names = [c.name for c in fam.columns]
        assert names == list(dict.fromkeys(names))  # names unique within a draw
        labels = {c.is_stock for c in fam.columns}
        assert labels == {True, False}  # both classes present (n>=2, mixed)
        # The disjoint vocabularies make a label readable from the name: every stock
        # name contains a stock noun, every flow name a flow noun, never the other.
        for c in fam.columns:
            stocky = any(w in c.name for w in ("balance", "inventory", "cash", "on_hand",
                                               "outstanding", "level", "position", "closing",
                                               "ending", "opening", "headcount", "reserve"))
            flowy = any(w in c.name for w in ("monthly", "weekly", "period", "paid", "sold",
                                              "movement", "volume", "amount", "revenue",
                                              "sales", "deposits", "withdrawals"))
            if c.is_stock:
                assert stocky and not flowy, f"stock name leaked a flow word: {c.name}"
            else:
                assert flowy and not stocky, f"flow name leaked a stock word: {c.name}"


def test_stock_flow_params_override_the_space() -> None:
    fam = sample_stock_flow_family(3, StockFlowFamilyParams(n_columns=(20, 20), stock_fraction=(0.5, 0.5)))
    assert len(fam.columns) <= 20  # may dedup below n; never above
    n_stock = sum(c.is_stock for c in fam.columns)
    assert 0 < n_stock < len(fam.columns)  # mixed


def test_inject_stock_flow_probes_adds_labelled_columns() -> None:
    df = pl.DataFrame(
        {
            "series_id": [f"S{(i // 6):03d}" for i in range(18)],
            "period": [f"2025-{(i % 6) + 1:02d}" for i in range(18)],
        }
    )
    reg = InjectionRegistry()
    out = inject_stock_flow_probes(
        df, seed=20260610, registry=reg, table_name="measure_probes", rng=random.Random(0)
    )
    assert reg.injections
    for inj in reg.injections:
        assert inj.detector_id == "temporal_behavior"
        assert inj.injection_type == "inject_stock_flow_probes"
        assert inj.parameters["true_behavior"] in ("stock", "flow")
        assert inj.target_column in out.columns  # the measure column was added
        assert inj.target_column not in ("series_id", "period")  # grain preserved
    assert {"series_id", "period"} <= set(out.columns)
    behaviours = {inj.parameters["true_behavior"] for inj in reg.injections}
    assert behaviours == {"stock", "flow"}  # both classes present


def test_inject_stock_flow_probes_is_reproducible_from_the_seed() -> None:
    df = pl.DataFrame({"series_id": ["S000"] * 12, "period": [f"2025-{m + 1:02d}" for m in range(12)]})

    def run(shared_seed: int) -> pl.DataFrame:
        return inject_stock_flow_probes(
            df, seed=42, registry=InjectionRegistry(), table_name="measure_probes",
            rng=random.Random(shared_seed),
        )

    # The shared rng differs; the family seed fixes the columns AND the values.
    assert run(1).equals(run(2))
