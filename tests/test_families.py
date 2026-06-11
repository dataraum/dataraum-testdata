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


# --- events-backed stock/flow strata (DAT-491) ------------------------------


_BACKED = {
    "backed_fraction": (1.0, 1.0),
    "broken_fraction": (0.5, 0.5),
}


def _probe_frame(n_series: int = 5, months: int = 8) -> pl.DataFrame:
    """The probe skeleton's (series, period) grain, in (series, period) row order."""
    return pl.DataFrame(
        {
            "series_id": [f"S{s:03d}" for s in range(n_series) for _ in range(months)],
            "period": [f"2025-{m + 1:02d}" for _ in range(n_series) for m in range(months)],
        }
    )


def _cell_sums(events: pl.DataFrame, delta_col: str) -> dict[tuple[str, str], float]:
    """Per-(series, period) event sums for one movements column."""
    out: dict[tuple[str, str], float] = {}
    for sid, date, amount in events.select("series_id", "event_date", delta_col).iter_rows():
        key = (sid, date[:7])
        out[key] = out.get(key, 0.0) + amount
    return out


def _stock_deltas(df: pl.DataFrame, col: str) -> dict[tuple[str, str], float]:
    """Per-(series, period) level deltas for one stock column, periods >= 2 only.

    The first period's movement runs from the (internal) opening level, which the
    exported table cannot see — exactly the cell the engine's stock hypothesis ignores.
    """
    out: dict[tuple[str, str], float] = {}
    prev: dict[str, float] = {}
    for sid, period, value in df.select("series_id", "period", col).iter_rows():
        if sid in prev:
            out[(sid, period)] = value - prev[sid]
        prev[sid] = value
    return out


def test_stock_flow_default_sample_is_unbacked() -> None:
    # The existing corpus contract: no events table, no backed strata, by default.
    for s in range(20):
        for c in sample_stock_flow_family(s).columns:
            assert not c.backed and not c.broken
            assert c.break_ratio == 0.0 and c.break_magnitude == 0.0


def test_stock_flow_backed_sampling_reproduces_and_varies() -> None:
    p = StockFlowFamilyParams(**_BACKED)
    assert sample_stock_flow_family(7, p) == sample_stock_flow_family(7, p)
    backed_sets = {tuple(c.name for c in sample_stock_flow_family(s, p).columns if c.backed) for s in range(30)}
    assert len(backed_sets) > 20  # different seeds → a different backed surface


def test_stock_flow_backing_is_well_formed() -> None:
    p = StockFlowFamilyParams(
        n_columns=(14, 20),
        backed_fraction=(0.5, 1.0),
        broken_fraction=(0.3, 0.7),
        break_ratio=(0.4, 0.9),
        break_magnitude=(0.3, 1.2),
    )
    for s in range(30):
        fam = sample_stock_flow_family(s, p)
        backed = [c for c in fam.columns if c.backed]
        assert backed, "backed_fraction >= 0.5 over >= 5 stocks must back at least one column"
        assert len(backed) <= 8  # engine convention cap (MAX_CONVENTION_COLUMNS)
        assert all(c.is_stock for c in backed)  # flows stay as they are
        for c in fam.columns:
            if c.broken:
                assert c.backed  # broken ⊆ backed
                assert 0.4 <= c.break_ratio <= 0.9
                assert 0.3 <= c.break_magnitude <= 1.2
            else:
                assert c.break_ratio == 0.0 and c.break_magnitude == 0.0


def test_stock_flow_backing_leaves_the_name_surface_unchanged() -> None:
    # Orthogonality: turning backing on must not move the name/label draws — the same
    # seed yields the identical (name, is_stock, ambiguous) surface either way.
    plain = sample_stock_flow_family(11, StockFlowFamilyParams(ambiguity=(0.3, 0.3)))
    backed = sample_stock_flow_family(11, StockFlowFamilyParams(ambiguity=(0.3, 0.3), **_BACKED))
    assert [(c.name, c.is_stock, c.ambiguous) for c in plain.columns] == [
        (c.name, c.is_stock, c.ambiguous) for c in backed.columns
    ]


def test_events_per_cell_lower_bound_guard() -> None:
    # One event per cell would tie the events grain to the probe grain — the engine's
    # lineage direction gate (events strictly finer) could reject the pairing.
    with pytest.raises(ValueError, match="direction gate"):
        StockFlowFamilyParams(events_per_cell=(1, 3))
    StockFlowFamilyParams()  # defaults are safe → no raise


def _run_backed_injector(
    df: pl.DataFrame, seed: int = 20260611, **params: object
) -> tuple[pl.DataFrame, pl.DataFrame, InjectionRegistry]:
    reg = InjectionRegistry()
    frames: dict[str, pl.DataFrame] = {}
    out = inject_stock_flow_probes(
        df,
        seed=seed,
        registry=reg,
        table_name="measure_probes",
        rng=random.Random(0),
        backed_fraction=[1.0, 1.0],
        dataframes=frames,
        **params,  # type: ignore[arg-type]
    )
    return out, frames["probe_events"], reg


def test_inject_backed_probes_emits_reconciling_events() -> None:
    df = _probe_frame()
    out, events, reg = _run_backed_injector(df, n_columns=[10, 10])

    backed = [i.parameters for i in reg.injections if i.parameters["backed"]]
    assert backed and all(p["events_table"] == "probe_events" for p in backed)
    assert set(events.columns) == {"event_id", "series_id", "event_date", *(p["events_column"] for p in backed)}
    # The shared slice dimension + the events' own time axis, inside each cell's month.
    assert set(events["series_id"].to_list()) == set(df["series_id"].to_list())
    assert all(len(d) == 10 and d[:7].startswith("2025-") for d in events["event_date"].to_list())
    # Strictly finer-grained than the probe table (the engine's direction gate), with
    # 2..6 events per (series, period) cell.
    cells = events.group_by("series_id", pl.col("event_date").str.slice(0, 7)).len()["len"].to_list()
    assert len(cells) == len(df)
    assert all(2 <= c <= 6 for c in cells)

    # The identity itself: per cell (periods >= 2), Σ events == the stock's delta.
    for p in (p for p in backed if p["reconciles"]):
        sums = _cell_sums(events, p["events_column"])
        deltas = _stock_deltas(out, p["name"])
        assert deltas
        for cell, delta in deltas.items():
            assert sums[cell] == pytest.approx(delta, abs=1e-6), (p["name"], cell)


def test_inject_broken_probes_break_by_the_sampled_amount() -> None:
    df = _probe_frame(n_series=5, months=8)
    out, events, reg = _run_backed_injector(
        df,
        n_columns=[10, 10],
        broken_fraction=[1.0, 1.0],
        break_ratio=[0.4, 0.4],
        break_magnitude=[1.0, 1.0],
    )
    broken = [i.parameters for i in reg.injections if i.parameters["backed"]]
    assert broken and all(not p["reconciles"] for p in broken)
    for p in broken:
        assert p["break_ratio"] == 0.4 and p["break_magnitude"] == 1.0
        sums = _cell_sums(events, p["events_column"])
        deltas = _stock_deltas(out, p["name"])
        # Per series: every verifiable cell deviates by ONE constant magnitude (sign
        # varies per cell) on broken series, by ~0 on intact ones.
        dev_by_series: dict[str, set[float]] = {}
        for (sid, period), delta in deltas.items():
            dev = round(abs(sums[(sid, period)] - delta), 2)
            dev_by_series.setdefault(sid, set()).add(dev)
        broken_series = {s for s, devs in dev_by_series.items() if max(devs) > 0.01}
        assert len(broken_series) == 2  # round(0.4 * 5 series), at least 1
        for sid in broken_series:
            assert len(dev_by_series[sid]) == 1, "per-cell deviation must be one constant magnitude"
            (magnitude,) = dev_by_series[sid]
            assert magnitude > 10.0  # break_magnitude 1.0 × mean |movement| — a real break
        for sid in set(dev_by_series) - broken_series:
            assert max(dev_by_series[sid]) <= 0.01  # intact series still reconcile


def test_inject_backed_probes_registry_labels() -> None:
    _, _, reg = _run_backed_injector(_probe_frame(), n_columns=[12, 12], broken_fraction=[0.5, 0.5])
    params = [i.parameters for i in reg.injections]
    for p in params:
        if p["true_behavior"] == "flow":
            assert not p["backed"] and p["events_column"] is None  # flows stay as they are
        if p["backed"]:
            assert p["events_column"] == f"{p['name']}_delta"
            assert p["reconciles"] == (p["break_ratio"] == 0.0)
    assert any(p["backed"] and p["reconciles"] for p in params)
    assert any(p["backed"] and not p["reconciles"] for p in params)


def test_inject_backed_probes_reproducible_from_the_seed() -> None:
    def run(shared_seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
        reg = InjectionRegistry()
        frames: dict[str, pl.DataFrame] = {}
        out = inject_stock_flow_probes(
            _probe_frame(),
            seed=42,
            registry=reg,
            table_name="measure_probes",
            rng=random.Random(shared_seed),
            backed_fraction=[1.0, 1.0],
            broken_fraction=[0.5, 0.5],
            dataframes=frames,
        )
        return out, frames["probe_events"]

    p1, e1 = run(1)
    p2, e2 = run(2)
    assert p1.equals(p2) and e1.equals(e2)  # the family seed fixes probes AND events
    # And the events surface varies across family seeds.
    reg = InjectionRegistry()
    frames: dict[str, pl.DataFrame] = {}
    inject_stock_flow_probes(
        _probe_frame(),
        seed=43,
        registry=reg,
        table_name="measure_probes",
        rng=random.Random(1),
        backed_fraction=[1.0, 1.0],
        dataframes=frames,
    )
    assert not frames["probe_events"].equals(e1)


def test_backed_probes_require_the_dataframes_dict() -> None:
    with pytest.raises(ValueError, match="dataframes"):
        inject_stock_flow_probes(
            _probe_frame(),
            seed=1,
            registry=InjectionRegistry(),
            table_name="measure_probes",
            rng=random.Random(0),
            backed_fraction=[1.0, 1.0],
        )


def test_unbacked_run_emits_no_events_table() -> None:
    frames: dict[str, pl.DataFrame] = {}
    inject_stock_flow_probes(
        _probe_frame(),
        seed=1,
        registry=InjectionRegistry(),
        table_name="measure_probes",
        rng=random.Random(0),
        dataframes=frames,
    )
    assert "probe_events" not in frames  # defaults = today's corpus, untouched


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
