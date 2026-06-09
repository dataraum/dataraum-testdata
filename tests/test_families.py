"""Generative injection families — the null_tokens family + injector (DAT-450).

A family is parameterized + seed-recorded: different seeds → different surface,
same semantics; the recorded seed reproduces exactly (AC1). Markers (is-null)
cluster, decoys (is-value) smear, and both are recorded as ground-truth labels
the calibration rig scores witnesses against.
"""

import random

import polars as pl

from testdata.entropy.families import (
    CURATED_VOCAB,
    NullTokenFamilyParams,
    mint_decoy,
    sample_null_token_family,
)
from testdata.entropy.injectors import inject_null_token_family
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
