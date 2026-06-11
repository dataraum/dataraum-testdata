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
    "N/A",
    "NA",
    "n/a",
    "NULL",
    "NONE",
    "NIL",
    "TBD",
    "-",
    "--",
    "?",
)

# Novel-sentinel shapes the curated vocabulary has NOT seen — composed from
# templates so the surface varies by seed (no memorizable fixed list).
_STATUS_WORDS: tuple[str, ...] = (
    "PENDING",
    "WITHHELD",
    "DISPUTED",
    "REDACTED",
    "UNKNOWN",
    "MISSING",
    "VOID",
    "REVIEW",
    "UNCONFIRMED",
    "ONHOLD",
    "RESTRICTED",
    "DEFERRED",
    "QUERIED",
    "SUPPRESSED",
    "OUTSTANDING",
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
        return _ERR_TEMPLATES[rng.randrange(len(_ERR_TEMPLATES))].format(c=_ERR_CODES[rng.randrange(len(_ERR_CODES))])
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

    n_markers: tuple[int, int] = (2, 6)  # distinct sentinel tokens (small → cluster)
    marker_ratio: tuple[float, float] = (0.05, 0.075)  # fraction of rows replaced by a marker
    decoy_ratio: tuple[float, float] = (0.015, 0.025)  # fraction replaced by a genuine decoy
    vocab_coverage: tuple[float, float] = (0.2, 0.8)  # fraction of markers in the curated vocab
    decoy_style: str | None = None  # fixed style, or None → sampled per instance
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


def sample_null_token_family(seed: int, params: NullTokenFamilyParams | None = None) -> NullTokenFamilySample:
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


# --- the mixed_units family ------------------------------------------------
#
# Feeds unit_consistency (DAT-428): a numeric column secretly mixing SCALES under one
# declared unit (some values in kEUR among EUR). A SCALE factor (a power of ten), NOT
# a ×1.1 currency factor — a 10% shift is undetectable from values; a 1000× shift is a
# clean second mode in log-magnitude that the bimodality witness reads.

_SCALE_FACTORS: tuple[int, ...] = (100, 1000, 10000)


@dataclass(frozen=True)
class MixedUnitsFamilyParams:
    """The parameter space the mixed_units (scale-mix) generator samples from."""

    scale_factors: tuple[int, ...] = _SCALE_FACTORS  # the alternate scale (a clean decade)
    mix_ratio: tuple[float, float] = (0.15, 0.40)  # fraction of rows pushed to that scale


@dataclass(frozen=True)
class MixedUnitsFamilySample:
    """One concrete draw: which scale, how much of the column lands on it."""

    seed: int
    scale_factor: int
    mix_ratio: float


def sample_mixed_units_family(seed: int, params: MixedUnitsFamilyParams | None = None) -> MixedUnitsFamilySample:
    """Sample one labelled mixed_units instance — deterministic in ``seed``.

    Different seeds → different scale factor + ratio (surface varies); the recorded
    seed reproduces exactly (AC1).
    """
    p = params or MixedUnitsFamilyParams()
    rng = random.Random(f"mixed_units:{seed}")
    return MixedUnitsFamilySample(
        seed=seed,
        scale_factor=rng.choice(p.scale_factors),
        mix_ratio=round(rng.uniform(*p.mix_ratio), 4),
    )


# --- the stock/flow family (DAT-445) ---------------------------------------
#
# Feeds temporal_behavior (DAT-445): the two-witness stock/flow adjudication
# (ontology prior vs LLM claim). Unlike the value-corruption families, the LABEL is
# the column's semantics, not an injected token — each sample is a set of measure
# columns, each (clear_name, is_stock). A STOCK is a carried-forward point-in-time
# level (a balance/position that must NOT be summed across periods); a FLOW is a
# per-period movement (a transaction amount that accumulates). The clear name carries
# the signal — the LLM's job is to read the behaviour from it (kill-gate v3: the LLM
# is name-anchored, so a clear name yields the true read), and the rig scores that
# read to measure the llm_claim witness's reliability. The two name vocabularies are
# DELIBERATELY DISJOINT (no shared word like "total"/"net") so a clear name is
# genuinely clear; the e2e measures the LLM's accuracy on exactly these.

# Stock name pieces — a carried-forward LEVEL.
_STOCK_NOUNS: tuple[str, ...] = (
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
)
_STOCK_TEMPLATES: tuple[str, ...] = (
    "{n}_balance",
    "closing_{n}",
    "ending_{n}",
    "opening_{n}",
    "{n}_on_hand",
    "outstanding_{n}",
    "{n}_level",
    "{n}_position",
)
# Flow name pieces — a per-period MOVEMENT.
_FLOW_NOUNS: tuple[str, ...] = (
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
)
_FLOW_TEMPLATES: tuple[str, ...] = (
    "monthly_{n}",
    "weekly_{n}",
    "period_{n}",
    "{n}_paid",
    "{n}_sold",
    "{n}_movement",
    "{n}_volume",
    "{n}_amount",
)


@dataclass(frozen=True)
class StockFlowFamilyParams:
    """The parameter space the stock/flow generator samples from."""

    n_columns: tuple[int, int] = (8, 14)  # measure columns per probe table
    stock_fraction: tuple[float, float] = (0.4, 0.6)  # fraction that are stocks
    # Fraction of columns given an AMBIGUOUS (conflicting-cue) name instead of a clear
    # one — the debit_balance archetype: one stock cue + one flow cue, so the name does
    # NOT reliably signal the behaviour. Default 0 = clear-only (the existing corpus); a
    # strategy opts in to measure the llm_claim witness's reliability in the HARD regime
    # (DAT-450) — where it genuinely fails, the boundary with the DAT-491 reality witness.
    ambiguity: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class ProbeColumn:
    """One labelled measure column: a name + its true temporal behaviour."""

    name: str
    is_stock: bool  # True → stock (point_in_time), False → flow (additive)
    ambiguous: bool = False  # True → a conflicting-cue (hard) name, not a clear one


@dataclass(frozen=True)
class StockFlowFamilySample:
    """One concrete, fully-labelled draw: the measure columns for a probe table."""

    seed: int
    columns: tuple[ProbeColumn, ...]


def _clear_name(rng: random.Random, *, is_stock: bool) -> str:
    """A clear name: a noun + a structural template, both from one concern's vocabulary."""
    nouns, templates = (_STOCK_NOUNS, _STOCK_TEMPLATES) if is_stock else (_FLOW_NOUNS, _FLOW_TEMPLATES)
    return templates[rng.randrange(len(templates))].format(n=nouns[rng.randrange(len(nouns))])


# Conflicting cues for AMBIGUOUS names: one stock-flavoured word + one flow-flavoured
# word, so the name carries BOTH and signals neither (the debit_balance archetype).
_STOCK_CUES: tuple[str, ...] = (
    *_STOCK_NOUNS,
    "balance",
    "level",
    "position",
    "closing",
    "opening",
    "outstanding",
)
_FLOW_CUES: tuple[str, ...] = (*_FLOW_NOUNS, "monthly", "weekly", "movement", "volume", "paid")


def _ambiguous_name(rng: random.Random) -> str:
    """A conflicting-cue name, INDEPENDENT of the true behaviour — the hard regime.

    One stock cue + one flow cue in random order (e.g. ``inventory_movement``,
    ``sales_balance``): the name does not reliably indicate stock vs flow, so a
    name-anchored LLM is right roughly by chance. The true behaviour is carried by the
    VALUES, not the name — exactly the case where the ``llm_claim`` witness should be
    UNreliable, which the rig then measures (rather than the best-case clear regime).
    """
    parts = [rng.choice(_STOCK_CUES), rng.choice(_FLOW_CUES)]
    rng.shuffle(parts)
    return f"{parts[0]}_{parts[1]}"


def sample_stock_flow_family(seed: int, params: StockFlowFamilyParams | None = None) -> StockFlowFamilySample:
    """Sample one labelled stock/flow instance — deterministic in ``seed``.

    Draws ``n_columns`` measure columns, ``stock_fraction`` of them stocks, each with a
    UNIQUE clear name from the disjoint stock/flow vocabularies. Different seeds → a
    different name set (surface varies, no memorizable fixture); the same seed
    reproduces exactly (AC1). The label (``is_stock``) is the ground truth the rig
    scores the LLM's stock/flow read against.
    """
    p = params or StockFlowFamilyParams()
    rng = random.Random(f"stock_flow:{seed}")

    n = rng.randint(*p.n_columns)
    n_stock = max(1, min(n - 1, round(n * rng.uniform(*p.stock_fraction))))
    flags = [True] * n_stock + [False] * (n - n_stock)
    rng.shuffle(flags)

    n_ambig = round(n * rng.uniform(*p.ambiguity))
    ambig = [True] * n_ambig + [False] * (n - n_ambig)
    rng.shuffle(ambig)

    columns: list[ProbeColumn] = []
    seen: set[str] = set()
    for is_stock, is_ambig in zip(flags, ambig, strict=True):
        name = ""
        for _ in range(50):
            name = _ambiguous_name(rng) if is_ambig else _clear_name(rng, is_stock=is_stock)
            if name not in seen:
                break
        if name in seen:  # vocabulary exhausted for this label — skip the dup
            continue
        seen.add(name)
        columns.append(ProbeColumn(name=name, is_stock=is_stock, ambiguous=is_ambig))

    return StockFlowFamilySample(seed=seed, columns=tuple(columns))


# --- the formula_divergence family (DAT-442, ADR-0009 derived-value) --------
#
# Feeds derived_value: each sampled GROUP is three numeric columns — two sources and
# a target whose NAME advertises a formula over them (`freight_total` next to
# `freight_subtotal` + `freight_tax` advertises the sum) while its VALUES follow a
# sampled truth. The two witnesses being calibrated read different evidence: the LLM
# `llm_hypothesis` reads the NAME, `formula_discovery` reads the ROWS — divergence is
# exactly where they must disagree, and the labels score each one. Three strata:
#
# * **agree** — values follow the named formula: the true-negative class.
# * **wholesale** — every row follows a different formula B: the name-reader is
#   measurably wrong; the row-reader finds B (or grades the named formula broken).
# * **partial** — a sampled fraction of rows follow B, the rest the named formula:
#   the graded match rate degrades by exactly that fraction (generalizes
#   ``drift_formula``'s error_ratio into a labelled family).
#
# The formula language mirrors the engine's discovery/hypothesis space (binary
# ``a op b`` over two same-table numeric columns, op in + - * /; analysis/correlation/
# within_table/derived_columns.py + the column_annotation prompt). One DELIBERATE
# out-of-space stress kind exists — ``scaled`` (values = named formula × a constant
# factor, a tax/discount shift no two-column formula expresses) — and is LABELLED
# ``discoverable: false`` so the rig can stratify it. The stock/flow naming discipline
# carries over: a target suffix vocabulary is op-specific (total/gross = sum, net/
# outstanding = difference, ...) so a name genuinely advertises its formula, and value
# ranges are chosen so every alternate formula separates from the named one by far
# more than the 0.01 grading tolerance (no accidental matches).

FORMULA_OPS: tuple[str, ...] = ("sum", "difference", "product", "ratio")
_COMMUTATIVE_OPS = frozenset({"sum", "product"})


def apply_operation(op: str, a: float, b: float) -> float:
    """Evaluate one binary formula op — the family's (and the engine's) formula language."""
    if op == "sum":
        return a + b
    if op == "difference":
        return a - b
    if op == "product":
        return a * b
    if op == "ratio":
        return a / b
    raise ValueError(f"unknown formula operation: {op}")


def formula_identity(op: str, source_a: str, source_b: str) -> str:
    """Canonical formula identity, e.g. ``sum(net,tax)``.

    Mirrors the engine's claim identity (entropy/measurements/derived_value.py):
    lowercased operands, sorted for commutative operations — so the recorded label
    and the engine's adjudication slot share one spelling.
    """
    a, b = source_a.strip().lower(), source_b.strip().lower()
    if op in _COMMUTATIVE_OPS and b < a:
        a, b = b, a
    return f"{op}({a},{b})"


# Name patterns per op: (target_suffix, (source_a part, lo, hi), (source_b part, lo, hi)).
# The suffix vocabulary is op-specific so the target name advertises ITS op; the value
# ranges keep targets away from zero (zero-target rows are excluded from the engine's
# grading), keep divisors well above 1, and keep difference minuends strictly above
# subtrahends — which also makes every alternate-op value separate from the named
# formula's value by orders of magnitude more than the 0.01 grading tolerance.
_NamePart = tuple[str, float, float]
_FormulaPattern = tuple[str, _NamePart, _NamePart]
_OP_PATTERNS: dict[str, tuple[_FormulaPattern, ...]] = {
    "sum": (
        ("total", ("subtotal", 100.0, 900.0), ("tax", 10.0, 200.0)),
        ("gross", ("net", 200.0, 1500.0), ("tax", 20.0, 300.0)),
        ("total", ("base", 50.0, 600.0), ("fees", 5.0, 90.0)),
        ("total", ("principal", 1000.0, 8000.0), ("interest", 10.0, 400.0)),
    ),
    "difference": (
        ("net", ("gross", 500.0, 2000.0), ("tax", 20.0, 300.0)),
        ("net", ("gross", 300.0, 1200.0), ("deductions", 10.0, 250.0)),
        ("outstanding", ("invoiced", 1000.0, 5000.0), ("paid", 100.0, 800.0)),
        ("remaining", ("budget", 2000.0, 9000.0), ("spent", 100.0, 1500.0)),
    ),
    "product": (
        ("amount", ("quantity", 2.0, 50.0), ("unit_price", 5.0, 200.0)),
        ("cost", ("hours", 2.0, 80.0), ("hourly_rate", 40.0, 150.0)),
        ("value", ("units", 3.0, 60.0), ("price_per_unit", 5.0, 120.0)),
    ),
    "ratio": (
        ("unit_cost", ("cost", 500.0, 5000.0), ("units", 5.0, 80.0)),
        ("avg_price", ("revenue", 1000.0, 9000.0), ("quantity", 10.0, 90.0)),
        ("daily_rate", ("charge", 600.0, 4000.0), ("days", 2.0, 30.0)),
    ),
}

# Group themes — each group's three columns share a distinctive prefix so the LLM can
# attribute sources to targets unambiguously (the fair-shot condition for the
# name-reading witness); sampled WITHOUT replacement, so names never collide.
_FORMULA_THEMES: tuple[str, ...] = (
    "freight",
    "storage",
    "handling",
    "consulting",
    "licensing",
    "maintenance",
    "marketing",
    "logistics",
    "subscription",
    "training",
    "catering",
    "leasing",
    "insurance",
    "packaging",
    "inspection",
    "advertising",
)


@dataclass(frozen=True)
class FormulaDivergenceFamilyParams:
    """The parameter space the formula_divergence generator samples from."""

    n_groups: tuple[int, int] = (6, 10)  # labelled (source_a, source_b, target) groups
    agree_fraction: tuple[float, float] = (0.3, 0.45)  # fraction of groups that are clean
    wholesale_fraction: tuple[float, float] = (0.25, 0.4)  # all-rows-divergent fraction; rest partial
    divergence_ratio: tuple[float, float] = (0.15, 0.6)  # B-row fraction within a partial group
    scaled_fraction: tuple[float, float] = (0.0, 0.25)  # of divergent groups: B = scaled (out-of-space)
    scaled_rate: tuple[float, float] = (0.08, 0.30)  # |factor − 1| for the scaled stress kind

    def __post_init__(self) -> None:
        if self.n_groups[0] < 3:
            raise ValueError(
                "formula_divergence family: n_groups lower bound must be >= 3 — every draw "
                "carries all three strata (agree / wholesale / partial)."
            )
        if not (0.0 < self.divergence_ratio[0] and self.divergence_ratio[1] < 1.0):
            raise ValueError(
                "formula_divergence family: divergence_ratio must stay strictly inside (0, 1) "
                "— ratio 0 is the agree stratum, ratio 1 is wholesale."
            )
        if self.scaled_rate[0] < 0.02:
            raise ValueError(
                "formula_divergence family: scaled_rate below 0.02 — a near-1 factor can hide "
                "under the engine's 0.01 absolute grading tolerance on small targets."
            )


@dataclass(frozen=True)
class FormulaProbeGroup:
    """One labelled formula group: two source columns + a target with a named formula.

    ``named_op`` is what the target's NAME advertises over (source_a, source_b);
    the values follow ``actual_op`` (or the named formula × ``factor`` for the scaled
    kind) on ``divergence_ratio`` of the rows. ``mode`` is the stratum label.
    """

    theme: str
    target: str
    source_a: str
    source_b: str
    a_range: tuple[float, float]
    b_range: tuple[float, float]
    named_op: str
    mode: str  # agree | wholesale | partial
    actual_op: str  # == named_op for agree groups and for the scaled kind
    factor: float | None  # scaled kind: values = named formula × factor; None otherwise
    divergence_ratio: float  # 0.0 agree, 1.0 wholesale, sampled in (0, 1) for partial

    @property
    def named_formula(self) -> str:
        """The canonical formula the NAME advertises — the llm_hypothesis ground truth."""
        return formula_identity(self.named_op, self.source_a, self.source_b)

    @property
    def actual_formula(self) -> str:
        """The canonical formula the divergent VALUES follow — the data-side ground truth."""
        if self.factor is not None:
            return f"scaled({formula_identity(self.named_op, self.source_a, self.source_b)},x{self.factor})"
        return formula_identity(self.actual_op, self.source_a, self.source_b)

    @property
    def discoverable(self) -> bool:
        """Whether the divergent values' formula is in the engine's binary-op language."""
        return self.factor is None


@dataclass(frozen=True)
class FormulaDivergenceFamilySample:
    """One concrete, fully-labelled draw: the formula groups for a probe table."""

    seed: int
    groups: tuple[FormulaProbeGroup, ...]


def sample_formula_divergence_family(
    seed: int, params: FormulaDivergenceFamilyParams | None = None
) -> FormulaDivergenceFamilySample:
    """Sample one labelled formula_divergence instance — deterministic in ``seed``.

    Draws ``n_groups`` themed groups, splits them into the agree / wholesale / partial
    strata by SAMPLED fractions (each stratum non-empty), and gives every divergent
    group an alternate truth: an op-swap (a different in-space binary formula over the
    same sources — discoverable) or, for a sampled subset, the scaled out-of-space
    kind. Different seeds → different themes/ops/strata (surface varies); the same
    seed reproduces exactly (AC1).
    """
    p = params or FormulaDivergenceFamilyParams()
    rng = random.Random(f"formula_divergence:{seed}")

    n = min(rng.randint(*p.n_groups), len(_FORMULA_THEMES))
    themes = rng.sample(_FORMULA_THEMES, n)
    ops = [FORMULA_OPS[rng.randrange(len(FORMULA_OPS))] for _ in range(n)]
    patterns = [_OP_PATTERNS[op][rng.randrange(len(_OP_PATTERNS[op]))] for op in ops]

    # Stratum sizes from sampled fractions, clamped so all three strata are non-empty.
    n_agree = max(1, min(round(n * rng.uniform(*p.agree_fraction)), n - 2))
    n_wholesale = max(1, min(round(n * rng.uniform(*p.wholesale_fraction)), n - n_agree - 1))
    modes = ["agree"] * n_agree + ["wholesale"] * n_wholesale + ["partial"] * (n - n_agree - n_wholesale)
    rng.shuffle(modes)

    # The scaled (out-of-space) kind: a sampled subset of divergent groups. Ratio-named
    # targets are excluded — a near-1 factor on a small quotient could hide under the
    # engine's 0.01 absolute grading tolerance, muddying the label.
    divergent = [i for i in range(n) if modes[i] != "agree"]
    eligible = [i for i in divergent if ops[i] != "ratio"]
    n_scaled = min(len(eligible), round(len(divergent) * rng.uniform(*p.scaled_fraction)))
    scaled_idx = set(rng.sample(eligible, n_scaled))

    groups: list[FormulaProbeGroup] = []
    for i, theme in enumerate(themes):
        op, mode = ops[i], modes[i]
        suffix, (part_a, a_lo, a_hi), (part_b, b_lo, b_hi) = patterns[i]
        actual_op, factor = op, None
        if mode != "agree":
            if i in scaled_idx:
                rate = rng.uniform(*p.scaled_rate)
                factor = round(1.0 + rate if rng.random() < 0.5 else 1.0 - rate, 4)
            else:
                actual_op = rng.choice([o for o in FORMULA_OPS if o != op])
        ratio = 0.0
        if mode == "wholesale":
            ratio = 1.0
        elif mode == "partial":
            ratio = round(rng.uniform(*p.divergence_ratio), 4)
        groups.append(
            FormulaProbeGroup(
                theme=theme,
                target=f"{theme}_{suffix}",
                source_a=f"{theme}_{part_a}",
                source_b=f"{theme}_{part_b}",
                a_range=(a_lo, a_hi),
                b_range=(b_lo, b_hi),
                named_op=op,
                mode=mode,
                actual_op=actual_op,
                factor=factor,
                divergence_ratio=ratio,
            )
        )
    return FormulaDivergenceFamilySample(seed=seed, groups=tuple(groups))
