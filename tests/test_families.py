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
    FORMULA_OPS,
    FormulaDivergenceFamilyParams,
    NullTokenFamilyParams,
    StockFlowFamilyParams,
    apply_operation,
    mint_decoy,
    sample_formula_divergence_family,
    sample_mixed_units_family,
    sample_null_token_family,
    sample_stock_flow_family,
)
from testdata.entropy.injectors import (
    inject_formula_divergence,
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
        _frame(),
        col="debit",
        seed=20260609,
        registry=reg,
        table_name="journal_lines",
        rng=random.Random(99),
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
    out = inject_scale_mix(df, col="amount", seed=42, registry=reg, table_name="invoices", rng=random.Random(0))
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
            _frame(),
            col="debit",
            seed=42,
            registry=reg,
            table_name="journal_lines",
            rng=random.Random(random.randint(0, 1_000_000)),
        )
        return df["debit"].to_list()

    # The shared `rng` differs between runs; the family seed fixes the result.
    assert run() == run()


# --- stock/flow family (DAT-445) -------------------------------------------


def test_stock_flow_family_reproduces_and_varies() -> None:
    assert sample_stock_flow_family(7) == sample_stock_flow_family(7)  # recorded seed reproduces
    name_sets = {tuple(c.name for c in sample_stock_flow_family(s).columns) for s in range(40)}
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
            stocky = any(
                w in c.name
                for w in (
                    "balance",
                    "inventory",
                    "cash",
                    "on_hand",
                    "outstanding",
                    "level",
                    "position",
                    "closing",
                    "ending",
                    "opening",
                    "headcount",
                    "reserve",
                )
            )
            flowy = any(
                w in c.name
                for w in (
                    "monthly",
                    "weekly",
                    "period",
                    "paid",
                    "sold",
                    "movement",
                    "volume",
                    "amount",
                    "revenue",
                    "sales",
                    "deposits",
                    "withdrawals",
                )
            )
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
    out = inject_stock_flow_probes(df, seed=20260610, registry=reg, table_name="measure_probes", rng=random.Random(0))
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
            df,
            seed=42,
            registry=InjectionRegistry(),
            table_name="measure_probes",
            rng=random.Random(shared_seed),
        )

    # The shared rng differs; the family seed fixes the columns AND the values.
    assert run(1).equals(run(2))


def test_stock_flow_ambiguity_produces_conflicting_cue_names() -> None:
    fam = sample_stock_flow_family(5, StockFlowFamilyParams(n_columns=(20, 20), ambiguity=(0.5, 0.5)))
    ambiguous = [c for c in fam.columns if c.ambiguous]
    clear = [c for c in fam.columns if not c.ambiguous]
    assert ambiguous and clear  # a mix of hard + clear columns
    _STOCK_CUES = {
        "balance",
        "level",
        "position",
        "closing",
        "opening",
        "outstanding",
        "inventory",
        "cash",
        "receivables",
        "payables",
        "debt",
        "equity",
        "reserve",
        "headcount",
        "asset",
        "provision",
    }
    _FLOW_CUES = {
        "monthly",
        "weekly",
        "movement",
        "volume",
        "paid",
        "revenue",
        "sales",
        "units",
        "interest",
        "expense",
        "deposits",
        "withdrawals",
        "spend",
        "shipments",
        "payouts",
    }
    # An ambiguous name carries BOTH a stock cue and a flow cue → it signals neither.
    for c in ambiguous:
        parts = set(c.name.split("_"))
        assert parts & _STOCK_CUES and parts & _FLOW_CUES, f"name not conflicting: {c.name}"
    # Default params stay clear-only — the shipped corpus + its 100% clear-name result.
    assert all(not c.ambiguous for c in sample_stock_flow_family(5).columns)


# --- formula_divergence family (DAT-442, ADR-0009 derived-value) ------------


def _probe_frame(n: int = 240) -> pl.DataFrame:
    return pl.DataFrame({"probe_id": [f"FP{i:05d}" for i in range(n)]})


def test_formula_divergence_family_reproduces_and_varies() -> None:
    assert sample_formula_divergence_family(7) == sample_formula_divergence_family(7)  # recorded seed reproduces
    surfaces = {
        tuple((g.target, g.mode, g.actual_formula) for g in sample_formula_divergence_family(s).groups)
        for s in range(40)
    }
    assert len(surfaces) > 30  # different seeds → a different group surface


def test_formula_divergence_sample_is_well_formed() -> None:
    for s in range(40):
        fam = sample_formula_divergence_family(s)
        names = [c for g in fam.groups for c in (g.source_a, g.source_b, g.target)]
        assert names == list(dict.fromkeys(names))  # all column names unique within a draw
        assert {g.mode for g in fam.groups} == {"agree", "wholesale", "partial"}  # all strata present
        for g in fam.groups:
            assert g.named_op in FORMULA_OPS
            # group-coherent naming: the theme prefixes all three columns (source attribution)
            assert all(c.startswith(f"{g.theme}_") for c in (g.source_a, g.source_b, g.target))
            if g.mode == "agree":
                assert g.actual_formula == g.named_formula
                assert g.divergence_ratio == 0.0 and g.discoverable
            elif g.mode == "wholesale":
                assert g.divergence_ratio == 1.0
                assert g.actual_formula != g.named_formula
            else:  # partial
                assert 0.15 <= g.divergence_ratio <= 0.6
                assert g.actual_formula != g.named_formula
            if g.factor is not None:  # the scaled stress kind: labelled out-of-space
                assert not g.discoverable and g.mode != "agree"
                assert g.named_op != "ratio"  # a near-1 factor on a tiny quotient could hide under tolerance
                assert g.actual_op == g.named_op
                assert 0.08 <= abs(g.factor - 1.0) <= 0.30
            elif g.mode != "agree":  # op-swap: a DISCOVERABLE alternate binary formula
                assert g.discoverable and g.actual_op != g.named_op and g.actual_op in FORMULA_OPS


def test_formula_divergence_params_override_the_space() -> None:
    fam = sample_formula_divergence_family(
        3, FormulaDivergenceFamilyParams(n_groups=(12, 12), scaled_fraction=(1.0, 1.0))
    )
    assert len(fam.groups) == 12
    # at full scaled_fraction every ELIGIBLE divergent group is scaled; only
    # ratio-named groups (tolerance guard) fall back to the op-swap kind.
    swapped = [g for g in fam.groups if g.mode != "agree" and g.factor is None]
    assert all(g.named_op == "ratio" for g in swapped)


def test_formula_divergence_guards_reject_degenerate_spaces() -> None:
    with pytest.raises(ValueError, match="n_groups"):
        FormulaDivergenceFamilyParams(n_groups=(2, 4))  # cannot carry all three strata
    with pytest.raises(ValueError, match="divergence_ratio"):
        FormulaDivergenceFamilyParams(divergence_ratio=(0.0, 0.5))  # 0 is agree, not partial
    with pytest.raises(ValueError, match="scaled_rate"):
        FormulaDivergenceFamilyParams(scaled_rate=(0.001, 0.1))  # hides under the 0.01 tolerance


def test_inject_formula_divergence_values_follow_the_labels() -> None:
    reg = InjectionRegistry()
    out = inject_formula_divergence(
        _probe_frame(), seed=20260611, registry=reg, table_name="formula_probes", rng=random.Random(0)
    )
    assert reg.injections
    modes = set()
    for inj in reg.injections:
        p = inj.parameters
        modes.add(p["divergence_mode"])
        assert inj.detector_id == "derived_value"
        assert inj.injection_type == "inject_formula_divergence"
        src_a, src_b = p["source_columns"]
        a, b = out[src_a].to_list(), out[src_b].to_list()
        t = out[inj.target_column].to_list()
        divergent = set(inj.target_rows)
        for i in range(len(t)):
            named = apply_operation(p["named_op"], a[i], b[i])
            if i not in divergent:
                # clean rows obey the NAMED formula within the engine's 0.01 grading tolerance
                assert abs(t[i] - named) < 0.01
            else:
                # divergent rows measurably VIOLATE the named formula ...
                assert abs(t[i] - named) > 0.01
                # ... and exactly follow the labelled actual formula
                if p["factor"] is not None:
                    assert t[i] == round(named * p["factor"], 2)
                else:
                    assert abs(t[i] - apply_operation(p["actual_op"], a[i], b[i])) < 0.01
        if p["divergence_mode"] == "agree":
            assert not divergent
        elif p["divergence_mode"] == "wholesale":
            assert len(divergent) == len(t)
        else:  # partial: the divergent-row fraction is the labelled ratio
            assert abs(len(divergent) / len(t) - p["divergence_ratio"]) < 0.05
    assert modes == {"agree", "wholesale", "partial"}


def test_inject_formula_divergence_preserves_grain_and_labels_targets_only() -> None:
    reg = InjectionRegistry()
    out = inject_formula_divergence(
        _probe_frame(40), seed=9, registry=reg, table_name="formula_probes", rng=random.Random(0)
    )
    assert "probe_id" in out.columns  # grain preserved
    targets = {inj.target_column for inj in reg.injections}
    for inj in reg.injections:
        p = inj.parameters
        assert inj.target_column in out.columns
        # one labelled record per TARGET; sources are unlabelled scaffolding in the frame
        assert set(p["source_columns"]) <= set(out.columns)
        assert not set(p["source_columns"]) & targets
        # the label vocabulary the rig scores witnesses against
        assert {"named_formula", "actual_formula", "divergence_mode", "divergence_ratio", "discoverable"} <= set(p)
        # values stay numeric — divergence is a different formula, never a token
        assert out[inj.target_column].dtype == pl.Float64


def test_inject_formula_divergence_is_reproducible_from_the_seed() -> None:
    def run(shared_seed: int) -> pl.DataFrame:
        return inject_formula_divergence(
            _probe_frame(60),
            seed=42,
            registry=InjectionRegistry(),
            table_name="formula_probes",
            rng=random.Random(shared_seed),
        )

    # The shared rng differs; the family seed fixes columns, values, AND divergent rows.
    assert run(1).equals(run(2))
