"""Corpus identity — the argument list of the function that produces a corpus.

``output/`` is gitignored and stays that way. A corpus is not an artifact to preserve;
it is reproducible from its parameters, and the parameters are the contract::

    (version, scenario, strategy, seed, months, normalization, family set, lever) -> corpus

§6 of ``docs/operating-model.md`` names six of those. Normalization and the lever are
here for the same reason as the other six: both change the bytes. The lever is the sharp
case — a levered run and its baseline are *defined* by differing, so an identity that
could not tell them apart would certify the wrong corpus in the one place it matters
most. ``baseline()`` therefore names the counterfactual's id explicitly.

Every field that feeds the digest is published, and nothing else does. That is a
constraint rather than a courtesy: a consumer can recompute ``corpus_id`` from the stamp
and check it, instead of trusting a number whose inputs it cannot see.

The family set is the *declared* one, not the tables that happened to materialize:
optional probe families are activated by the strategy, which is already a field. So the
whole tuple is known before generation runs, which is what makes it pinnable.

What this deliberately does not capture: edits to ``config/scenarios/*.yaml`` and
``config/strategies/*.yaml``. Those are inputs the scenario and strategy *names* stand
in for, and the version is the proxy for "the generator's own definitions moved". Which
means the version has to be bumped whenever generation semantics change — a stamp that
never moves is worse than none, because it asserts sameness that was never checked.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version
from typing import Any

from testdata.families import default_families


GENERATOR = "dataraum-testdata"


def generator_version() -> str:
    """The installed package version — read, never restated.

    ``export.py`` used to carry a literal ``"0.1.0"`` beside ``pyproject.toml``'s, which
    is the same one-of-two-places defect the family registry was built to end.
    """
    try:
        return _installed_version(GENERATOR)
    except PackageNotFoundError:  # a source tree with no install — say so rather than guess
        return "0+unknown"


@dataclass(frozen=True)
class CorpusIdentity:
    """The parameters that produce a corpus, and the digest over them."""

    scenario: str
    strategy: str
    seed: int
    months: int
    normalization: str = "full"
    lever: Mapping[str, Any] | None = None
    families: tuple[str, ...] = field(default_factory=default_families)
    generator: str = GENERATOR
    version: str = field(default_factory=generator_version)

    @property
    def corpus_id(self) -> str:
        """A short, stable digest of every published field.

        ``sha256`` over a canonical JSON rendering — stable across processes and
        platforms, which ``hash()`` is not.
        """
        canonical = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    def baseline(self) -> CorpusIdentity:
        """The same run without the lever — the exact same-seed counterfactual."""
        return replace(self, lever=None)

    def as_dict(self) -> dict[str, Any]:
        """The stamp as written into every truth file."""
        return {"id": self.corpus_id, **self._payload()}

    def describe(self) -> str:
        """One line, for a terminal."""
        lever = f" lever={self.lever['type']}" if self.lever else ""
        return (
            f"{self.generator} {self.version} · {self.scenario}/{self.strategy} · "
            f"seed {self.seed} · {self.months}mo · {self.normalization}{lever} · #{self.corpus_id}"
        )

    def _payload(self) -> dict[str, Any]:
        """The digest's input. Field order here is display order in the YAML."""
        return {
            "generator": self.generator,
            "version": self.version,
            "scenario": self.scenario,
            "strategy": self.strategy,
            "seed": self.seed,
            "months": self.months,
            "normalization": self.normalization,
            "families": list(self.families),
            "lever": dict(self.lever) if self.lever is not None else None,
        }
