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
from dataclasses import dataclass, replace

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


# Engine cap on event-side convention columns (dataraum.analysis.lineage.processor
# MAX_CONVENTION_COLUMNS): an events table's numeric columns beyond the first 8
# (sorted) are never enumerated as conventions, so a backed stock past the cap could
# never reconcile. The sampler caps the backed set here so every backed label is
# actually measurable.
_MAX_BACKED_COLUMNS = 8


def stock_flow_events_column(name: str) -> str:
    """The probe_events column backing one stock measure column.

    One numeric movements column per backed stock — the signed convention the engine's
    aggregation-lineage discovery should find (``Σ events ≈ Δ stock`` per series/period).
    Shared by the injector (which writes it) and recorded in the registry parameters
    (``events_column``) as the rig's ground truth.
    """
    return f"{name}_delta"


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
    # --- events backing (DAT-491) --------------------------------------------------
    # Fraction of STOCK columns backed by a probe_events movements table whose
    # per-(series, period) sums reconcile to the stock's period-over-period deltas
    # (opening + Σ events = closing) — the exact identity the temporal_behavior
    # ``structural_reconciliation`` witness reads. Default 0 = no events table (the
    # existing corpus); a calibration strategy opts in to measure that witness's
    # reliability (its 0.85 in reliabilities.yaml is an uncalibrated placeholder
    # precisely because the DAT-450 corpus has no events). Orthogonal to the name
    # axes: backing changes the EVENTS, never the measure column's values or name.
    backed_fraction: tuple[float, float] = (0.0, 0.0)
    # Fraction of BACKED columns whose reconciliation is BROKEN: a sampled fraction of
    # series gets per-period event sums perturbed off the stock's deltas — measures the
    # witness's behaviour when the identity fails (it must degrade or abstain, never
    # confidently confirm).
    broken_fraction: tuple[float, float] = (0.0, 0.0)
    # Per broken column: fraction of its series broken (≥1 series). Intact series still
    # reconcile, so the engine's per-entity vote fraction (match_rate) degrades with it.
    break_ratio: tuple[float, float] = (0.5, 1.0)
    # Per broken column: relative size of the per-period perturbation, in units of the
    # series' mean absolute movement — i.e. ≈ the per-entity stock residual R_stock the
    # engine measures. The default range straddles the engine's FIRE_RESIDUAL_MAX = 0.5
    # abstain gate (reconcile.py) from both sides, so the rig traces the full response:
    # sub-gate breaks (entity still votes, residual elevated) through clear abstentions.
    break_magnitude: tuple[float, float] = (0.3, 1.2)
    # Events per (series, period) cell. Lower bound 2 keeps the events side STRICTLY
    # finer-grained than the probe table (one row per cell), which the engine's
    # lineage direction gate requires (event rows > measure rows over paired cells).
    events_per_cell: tuple[int, int] = (2, 6)

    def __post_init__(self) -> None:
        if self.events_per_cell[0] < 2:
            raise ValueError(
                "stock_flow family: events_per_cell lower bound must be >= 2 — the engine's "
                "aggregation-lineage direction gate needs the events side strictly finer-grained "
                "than the probe table (one probe row per (series, period) cell)."
            )


@dataclass(frozen=True)
class ProbeColumn:
    """One labelled measure column: a name + its true temporal behaviour."""

    name: str
    is_stock: bool  # True → stock (point_in_time), False → flow (additive)
    ambiguous: bool = False  # True → a conflicting-cue (hard) name, not a clear one
    # --- events backing (DAT-491): the structural_reconciliation rig's ground truth ---
    backed: bool = False  # True → probe_events carries this column's movements
    broken: bool = False  # True → a sampled fraction of series does NOT reconcile
    break_ratio: float = 0.0  # fraction of series broken (0.0 when not broken)
    break_magnitude: float = 0.0  # perturbation in mean-|movement| units (≈ R_stock)


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

    # Events backing (DAT-491) — assigned on the FINAL column set (after name dedup),
    # AFTER all name/label draws, so a recorded seed's name surface is unchanged by
    # turning backing on. Only stocks can be backed (the witness's identity is
    # opening + Σ events = closing); flows stay as they are.
    stock_idx = [i for i, c in enumerate(columns) if c.is_stock]
    n_backed = min(round(len(stock_idx) * rng.uniform(*p.backed_fraction)), _MAX_BACKED_COLUMNS)
    backed_idx = sorted(rng.sample(stock_idx, n_backed)) if n_backed else []
    n_broken = round(len(backed_idx) * rng.uniform(*p.broken_fraction))
    broken_idx = set(rng.sample(backed_idx, n_broken)) if n_broken else set()
    for i in backed_idx:
        broken = i in broken_idx
        columns[i] = replace(
            columns[i],
            backed=True,
            broken=broken,
            break_ratio=round(rng.uniform(*p.break_ratio), 4) if broken else 0.0,
            break_magnitude=round(rng.uniform(*p.break_magnitude), 4) if broken else 0.0,
        )

    return StockFlowFamilySample(seed=seed, columns=tuple(columns))
