"""Generative injection families — parameterized corruption generators (DAT-450).

A *family* is the ADR-0009 answer to "fixed fixtures are dead": a strategy declares
a family + a recorded seed, and the generator SAMPLES the concrete corruption
(tokens, values, rates) from a parameter space. Two runs with different seeds
produce a different surface but the same family semantics; a recorded seed
reproduces exactly. A detector that memorized a fixed ``rrFlp_11_zp00`` proves
nothing — recall over a sampled family proves capability.

The families also *double as the calibration rig*: because each sampled instance
carries a ground-truth label, running a witness over many samples and scoring its
agreement is how the shipped reliabilities are measured (DAT-450, the eval rig).

First-wave family implemented here: **null_tokens** (feeds the null_semantics
adjudication, DAT-457). It samples two LABELLED classes into a numeric column:

* **markers** (``is-null``) — sentinel SHAPES a human types to mean "no value":
  status words, error codes, punctuation runs. A SMALL set, each injected into
  MANY rows, so they *cluster* in the quarantine (the quarantine-clustering
  witness's is-null signal). ``vocab_coverage`` controls how many are in the
  curated null vocabulary vs novel sentinels (the vocabulary witness's axis).
* **decoys** (``is-value``) — GENUINE amounts that merely fail a numeric cast:
  locale-formatted numbers, currency-prefixed, unit-suffixed, annotated. Each is
  DISTINCT (minted fresh per row), so they *smear* across the quarantine — the
  negative class a witness must not mistake for a null marker.

The marker/decoy split is the whole point of calibration: reliability is a
witness's accuracy at telling a sentinel from a genuine-but-unparseable value, not
its eagerness to fire on anything quarantined.

Other first-wave families (garbage names, mixed units, stock/flow, heterogeneity)
are framework-ready but deferred until their witnesses land (DAT-446/428/445/473).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# --- marker grammar (is-null sentinels) ------------------------------------

# Curated null markers — values the vertical's null vocabulary already knows.
# Drawing in-vocab markers from this pool makes the vocabulary witness HIT them.
_VOCAB_MARKERS: tuple[str, ...] = (
    "N/A", "NA", "n/a", "NULL", "NONE", "NIL", "TBD", "-", "--", "?",
)

# Novel-sentinel shapes the curated vocabulary has NOT seen — composed from
# templates so the surface varies by seed (no memorizable fixed list).
_STATUS_WORDS: tuple[str, ...] = (
    "PENDING", "WITHHELD", "DISPUTED", "REDACTED", "UNKNOWN", "MISSING", "VOID",
    "REVIEW", "UNCONFIRMED", "ONHOLD", "RESTRICTED", "DEFERRED", "QUERIED",
    "SUPPRESSED", "OUTSTANDING",
)
_STATUS_TEMPLATES: tuple[str, ...] = ("{w}", "{w}...", "[{w}]", "({w})", "{w}*", "<{w}>")
_ERR_CODES: tuple[str, ...] = ("ERR", "VALUE", "REF", "DIV/0", "CALC", "NAME")
_ERR_TEMPLATES: tuple[str, ...] = ("#{c}", "#{c}!", "{c}_ERROR", "<{c}>")
_PUNCT_CHARS: tuple[str, ...] = (".", "*", "—", "#", "~")


def _novel_marker(rng: random.Random) -> str:
    """One novel sentinel-shaped token (not in the curated vocabulary)."""
    kind = rng.choice(("status", "err", "punct"))
    if kind == "status":
        return _STATUS_TEMPLATES[rng.randrange(len(_STATUS_TEMPLATES))].format(
            w=_STATUS_WORDS[rng.randrange(len(_STATUS_WORDS))]
        )
    if kind == "err":
        return _ERR_TEMPLATES[rng.randrange(len(_ERR_TEMPLATES))].format(
            c=_ERR_CODES[rng.randrange(len(_ERR_CODES))]
        )
    return _PUNCT_CHARS[rng.randrange(len(_PUNCT_CHARS))] * rng.randint(2, 5)


# --- decoy grammar (is-value genuine-but-unparseable values) ---------------

_DECOY_STYLES: tuple[str, ...] = ("locale_eu", "spaced", "currency", "unit", "annotated")


def mint_decoy(rng: random.Random, style: str) -> str:
    """One GENUINE amount that fails a numeric cast but is is-value (not a marker).

    Distinct per call (the integer part is randomized), so decoys smear across the
    quarantine instead of clustering — the property that separates them from
    sentinels for the quarantine-clustering witness.
    """
    whole = rng.randint(1, 9999)
    cents = rng.randint(0, 99)
    if style == "locale_eu":  # European: dot thousands, comma decimal
        return f"{whole:,}".replace(",", ".") + f",{cents:02d}"
    if style == "spaced":  # space thousands
        return f"{whole:,}".replace(",", " ") + f",{cents:02d}"
    if style == "currency":
        sym = rng.choice(("EUR ", "$", "£", "USD "))
        return f"{sym}{whole:,}.{cents:02d}"
    if style == "unit":
        return f"{whole / 10:.1f}".replace(".", rng.choice((".", ","))) + rng.choice(("k", "K", "M", " Mio."))
    return f"{whole}" + rng.choice((" (est.)", "*", " ¹", " ~", " p.a."))  # annotated


# --- the null_tokens family ------------------------------------------------


# A corrupted column only infers a numeric type — and thus quarantines its tokens
# for null_semantics to adjudicate — when parse_success ≥ min_confidence (0.85,
# phases/typing.yaml). parse_success ≈ 1 − (marker + decoy ratio), so the COMBINED
# upper bound is capped here, leaving ≥0.88 parse (a margin over 0.85). Enforced,
# not just documented: at 16% corruption journal_lines.debit fell to VARCHAR and the
# adjudication was silently skipped (DAT-450 live-run finding).
_MAX_COMBINED_RATIO = 0.12


@dataclass(frozen=True)
class NullTokenFamilyParams:
    """The parameter space the null_tokens generator samples from.

    Ranges (not fixed values): the generator draws a concrete instance per seed.
    A strategy may override any field; unset fields use these defaults.
    """

    n_markers: tuple[int, int] = (2, 6)          # distinct sentinel tokens (small → cluster)
    marker_ratio: tuple[float, float] = (0.05, 0.075)  # fraction of rows replaced by a marker
    decoy_ratio: tuple[float, float] = (0.015, 0.025)  # fraction replaced by a genuine decoy
    vocab_coverage: tuple[float, float] = (0.2, 0.8)   # fraction of markers in the curated vocab
    decoy_style: str | None = None               # fixed style, or None → sampled per instance
    # None → decoys are minted DISTINCT (count 1, they smear). A (lo, hi) range →
    # decoys are a small CLUSTERED is-value set of that many distinct values,
    # repeated — the stress mode that lets the rig measure quarantine_clustering's
    # false-positive rate (does it mistake a recurring genuine value for a sentinel?).
    decoy_cluster_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        worst = self.marker_ratio[1] + self.decoy_ratio[1]
        if worst > _MAX_COMBINED_RATIO:
            raise ValueError(
                f"null_tokens family: combined marker+decoy upper bound {worst:.3f} exceeds "
                f"{_MAX_COMBINED_RATIO} — the corrupted column would parse below typing "
                "min_confidence (0.85) and fall back to VARCHAR, so null_semantics never runs."
            )


@dataclass(frozen=True)
class NullTokenFamilySample:
    """One concrete, fully-labelled draw from the null_tokens family.

    ``markers`` cluster (is-null); ``decoy_style`` mints distinct is-value values.
    ``in_vocab_markers`` are the markers the curated vocabulary would recognise —
    the rest are novel sentinels. ``seed`` makes the draw reproducible.
    """

    seed: int
    markers: tuple[str, ...]
    in_vocab_markers: tuple[str, ...]
    decoy_style: str
    marker_ratio: float
    decoy_ratio: float
    # 0 → decoys are distinct (smear); >0 → a clustered is-value set of this many
    # distinct decoys, repeated (the quarantine-specificity stress mode).
    decoy_cluster_size: int = 0

    @property
    def vocab_coverage(self) -> float:
        return len(self.in_vocab_markers) / len(self.markers) if self.markers else 0.0


def sample_null_token_family(
    seed: int, params: NullTokenFamilyParams | None = None
) -> NullTokenFamilySample:
    """Sample one labelled null_tokens instance — deterministic in ``seed``.

    Different seeds → different markers/decoys/rates (surface varies); the same
    seed always reproduces the same instance (AC1). Markers are a small set so
    they cluster; ``vocab_coverage`` of them are drawn from the curated vocabulary
    (the witness will recognise these), the rest are novel sentinel shapes.
    """
    p = params or NullTokenFamilyParams()
    rng = random.Random(seed)

    n_markers = rng.randint(*p.n_markers)
    coverage = rng.uniform(*p.vocab_coverage)
    n_in_vocab = max(0, min(n_markers, round(n_markers * coverage)))

    in_vocab = tuple(rng.sample(_VOCAB_MARKERS, min(n_in_vocab, len(_VOCAB_MARKERS))))
    novel: list[str] = []
    seen = set(in_vocab)
    while len(novel) < n_markers - len(in_vocab):
        tok = _novel_marker(rng)
        if tok not in seen:
            seen.add(tok)
            novel.append(tok)

    markers = tuple(in_vocab) + tuple(novel)
    decoy_style = p.decoy_style or _DECOY_STYLES[rng.randrange(len(_DECOY_STYLES))]
    cluster_size = rng.randint(*p.decoy_cluster_size) if p.decoy_cluster_size else 0
    return NullTokenFamilySample(
        seed=seed,
        markers=markers,
        in_vocab_markers=tuple(in_vocab),
        decoy_style=decoy_style,
        marker_ratio=round(rng.uniform(*p.marker_ratio), 4),
        decoy_ratio=round(rng.uniform(*p.decoy_ratio), 4),
        decoy_cluster_size=cluster_size,
    )


# The curated-vocabulary view the calibration rig hands the vocabulary witness:
# exactly the markers the vertical's null list would know. Exposed so the rig and
# the injector agree on what "in-vocab" means without importing engine config.
CURATED_VOCAB: tuple[str, ...] = _VOCAB_MARKERS
