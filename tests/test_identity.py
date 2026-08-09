"""The corpus identity stamp.

A corpus is a function of its parameters. The stamp is the claim that a given directory
came from a given argument list, and the claim is only worth something if the digest
moves with every input and the four files agree on what they describe.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import yaml

from testdata.families import FAMILIES, default_families
from testdata.identity import CorpusIdentity, generator_version
from testdata.scenarios.runner import run_scenario


def _identity(**overrides: object) -> CorpusIdentity:
    base = CorpusIdentity(scenario="month-end-close", strategy="clean", seed=42, months=12)
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_the_digest_is_recomputable_from_the_published_fields() -> None:
    """Nothing feeds the digest that the stamp does not publish.

    This is the whole contract: a consumer re-derives the id from the block it can
    see and checks it, rather than trusting a number whose inputs are hidden. If a
    field ever enters ``_payload`` without entering ``as_dict``, this fails.
    """
    identity = _identity()
    stamp = identity.as_dict()
    published = {k: v for k, v in stamp.items() if k != "id"}
    assert published == identity._payload()
    assert stamp["id"] == identity.corpus_id
    assert len(stamp["id"]) == 12


def test_every_parameter_moves_the_id() -> None:
    """A field that cannot change the digest has no business being in the identity."""
    base = _identity()
    variants = {
        "scenario": _identity(scenario="erp-migration"),
        "strategy": _identity(strategy="medium"),
        "seed": _identity(seed=43),
        "months": _identity(months=6),
        "normalization": _identity(normalization="flat"),
        "fiscal_start": _identity(fiscal_start="2024-07-01"),
        "version": _identity(version="99.0.0"),
        "families": _identity(families=(*default_families(), "supply")),
        "lever": _identity(lever={"type": "price_level", "period_k": 6, "factor": 1.15}),
    }
    ids = {name: v.corpus_id for name, v in variants.items()}
    for name, corpus_id in ids.items():
        assert corpus_id != base.corpus_id, f"{name} does not move the id"
    assert len(set(ids.values())) == len(ids), "two different parameter sets collide"


def test_the_lever_and_its_baseline_are_different_corpora() -> None:
    """The counterfactual pair is *defined* by differing.

    An identity that could not separate a levered run from its own baseline would
    certify the wrong corpus in the one place where being wrong is worst.
    """
    levered = _identity(lever={"type": "volume", "period_k": 6, "factor": 1.2})
    assert levered.corpus_id != levered.baseline().corpus_id
    assert levered.baseline().corpus_id == _identity().corpus_id
    assert levered.baseline().lever is None


def test_the_id_is_stable_across_processes() -> None:
    """Hardcoded because a digest that drifts between runs stamps nothing.

    ``hash()`` is salted per process; this is sha256 over canonical JSON. If this
    value changes, either the payload shape or a default moved — both are real
    breaking changes for anyone who pinned an id, so update the constant knowingly.
    """
    identity = CorpusIdentity(
        scenario="month-end-close",
        strategy="clean",
        seed=42,
        months=12,
        families=("core_ledger", "operating_chain", "inventory"),
        version="0.2.0",
    )
    assert identity.corpus_id == "524369963d65"


def test_the_family_set_is_the_declared_one() -> None:
    """Probe families are strategy-activated, so they are not in the tuple.

    The identity has to be knowable before generation, or it cannot be pinned and
    then regenerated from — which is the entire point of not freezing the bytes.
    """
    assert default_families() == tuple(f.name for f in FAMILIES if not f.optional)
    assert "probes" not in default_families()
    assert "inventory" in default_families(), "the family whose landing broke every stale corpus"


def test_the_version_is_read_not_restated() -> None:
    """Read from package metadata, so it cannot disagree with pyproject.toml."""
    version = generator_version()
    assert version != "0+unknown", "run `uv sync` — the package must be installed"
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert f'version = "{version}"' in pyproject


def test_every_output_file_carries_the_same_stamp() -> None:
    """Four files, one corpus. A directory whose files disagree is not evidence.

    Before this, ``manifest.yaml`` had its own hand-written version, ``ground_truth``
    knew the seed but not the scenario, ``metadata_truth`` knew neither, and
    ``entropy_map`` opened straight into ``injections:``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "out"
        result = run_scenario("month-end-close", strategy_name="medium", seed=7, months=3, output_dir=output)

        stamps = {
            name: yaml.safe_load((output / name).read_text())["corpus"]
            for name in ("manifest.yaml", "ground_truth.yaml", "metadata_truth.yaml", "entropy_map.yaml")
        }
        assert len(stamps) == 4
        assert len({s["id"] for s in stamps.values()}) == 1
        assert all(s == result["identity"].as_dict() for s in stamps.values())

        stamp = stamps["manifest.yaml"]
        assert stamp["scenario"] == "month-end-close"
        assert stamp["strategy"] == "medium"
        assert stamp["seed"] == 7
        assert stamp["months"] == 3
        assert stamp["families"] == list(default_families())
        assert stamp["version"] == generator_version()


def test_the_manifest_states_parameters_once() -> None:
    """Run parameters live in ``corpus``; ``run`` holds only what follows from the run.

    The manifest used to repeat scenario/strategy/seed/months under ``parameters``
    beside a literal version string — the one-fact-in-two-places shape that let the
    operating chain ship without its key maps.
    """
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "out"
        run_scenario("month-end-close", strategy_name="low", seed=5, months=2, output_dir=output)
        manifest = yaml.safe_load((output / "manifest.yaml").read_text())

        assert set(manifest) == {"generated_at", "corpus", "run", "files"}
        assert set(manifest["run"]) == {"injection_count"}
        assert manifest["run"]["injection_count"] > 0
        assert not set(manifest["run"]) & set(manifest["corpus"])


def test_a_changed_parameter_changes_the_written_id() -> None:
    """End to end: two runs differing in one parameter are distinguishable on disk."""
    with tempfile.TemporaryDirectory() as tmp:
        ids = []
        for seed in (11, 12):
            output = Path(tmp) / f"s{seed}"
            run_scenario("month-end-close", strategy_name="clean", seed=seed, months=2, output_dir=output)
            ids.append(yaml.safe_load((output / "manifest.yaml").read_text())["corpus"]["id"])
        assert ids[0] != ids[1]


def test_multi_source_exports_share_one_corpus_id() -> None:
    """Three sources, three conventions, one origin — and the files say so.

    A reconciliation scenario whose sources could not be shown to come from the same
    events would be missing its own premise.
    """
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "multi"
        run_scenario("multi-system-recon", strategy_name="clean", seed=9, months=2, output_dir=output)

        index = yaml.safe_load((output / "sources.yaml").read_text())
        top_id = index["corpus"]["id"]
        for src in index["sources"]:
            stamp = yaml.safe_load((output / src["directory"] / "manifest.yaml").read_text())["corpus"]
            assert stamp["id"] == top_id
        # The per-source convention is a run fact, not part of the corpus identity:
        # the same events exported twice are still the same events.
        first = yaml.safe_load((output / index["sources"][0]["directory"] / "manifest.yaml").read_text())
        assert first["run"]["source"] == index["sources"][0]["name"]
        assert "column_style" not in first["corpus"]


def test_intervention_names_its_counterfactual() -> None:
    """``re-run without the lever`` is only actionable if the baseline can be named."""
    with tempfile.TemporaryDirectory() as tmp:
        levered = Path(tmp) / "levered"
        baseline = Path(tmp) / "baseline"
        lever = {"type": "price_level", "period_k": 1, "factor": 1.2}
        run_scenario("month-end-close", strategy_name="clean", seed=3, months=3, output_dir=levered, lever=lever)
        run_scenario("month-end-close", strategy_name="clean", seed=3, months=3, output_dir=baseline)

        record = yaml.safe_load((levered / "intervention.yaml").read_text())
        baseline_stamp = yaml.safe_load((baseline / "manifest.yaml").read_text())["corpus"]

        assert record["corpus"]["lever"] == lever
        assert record["counterfactual_corpus_id"] == baseline_stamp["id"]
        assert record["corpus"]["id"] != baseline_stamp["id"]
