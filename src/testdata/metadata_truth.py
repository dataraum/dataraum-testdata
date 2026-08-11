"""Structural ground truth for the corpus — the generator's own answers about shape.

``ground_truth.yaml`` answers *what the numbers are*; this module answers *what the
schema means*: the FK topology, which tables are facts and which dimensions, which
columns are measures and which timestamps, whether a measure is a stock or a flow,
which axes a metric may be rolled up along, the business cycles the corpus supports,
and the conformed-dimension matrix. Anything recovering structure from the data —
relationship detection, grain and role inference, stock/flow adjudication — is
gradeable against it. Exported as ``metadata_truth.yaml`` beside ``entropy_map.yaml``
and ``ground_truth.yaml``.

The truth is authored from the models and the generator design
(``canonical/finance/models.py``, ``generators.py``, ``ground_truth.py``), never
measured from the output — measuring it would grade the data against itself.

**Authored by the FAMILIES, published here.** Roles, stock/flow, units, concepts,
cycles and reconciliation lineage are declared in ``families.Structure``, beside the
tables they describe; this module merges what it finds and derives the per-level views
(remap, folds, bus matrix, measured_in). It used to hold one canonical blob naming
finance tables, which made a family's truth a second thing to remember one file away
from its tables — and ``test_families_registry`` now fails if a declared table is not
classified at all, so a table cannot arrive truth-free. What is still authored here is
``metric_additivity`` (engine vocabulary, not corpus truth — §11 open decision 1) and
the data-conditional role-play block, which is emitted only when a strategy has
materialized the shape.

Determinism split, because not every verdict is equally hard:
  * ``function_symmetry`` — fixed by structure alone (AVG / COUNT(DISTINCT) / ratio
    never reconcile across a partition; the hard structural relationships and roles).
    No upstream labelling can change the right answer, so these are assertable
    outright.
  * ``label_dependent`` — correct only if an upstream classification is correct
    (SUM over a flow is additive, SUM over a stock is not, COUNT of events is).
    A consumer grading these is also grading its own stock/flow call, so they belong
    in a diagnostic band rather than a hard assertion.

Remap-safety: the truth is authored at canonical (``full`` / snake_case)
table + column names. ``remap_metadata_truth`` rewrites table names through the
normalization ``table_mapping`` and column names through a ``column_style``, mirroring
how ``InjectionRegistry.remap_tables`` keeps ``entropy_map.yaml`` valid. Cross-table
FKs that collapse to a single table after a merge are dropped (no longer discoverable);
a genuine self-referential FK (``chart_of_accounts.parent_id``) is kept. Column renames
introduced by normalization *merges* (e.g. ``payments.amount -> invoice_data.payment_amount``)
are NOT reflected — the merged columns keep canonical names, the same table-only
contract ``entropy_map.yaml`` follows.

Folded-dimension identity truth (``folded_dimensions`` / ``degenerate_ids``) IS carried,
level-specific, for the denormalized (``flat`` / ``single``) shapes, so a wide variant is
not truth-free. A folded dimension is a referenced dimension whose FK-target table a
normalization level INLINED into a fact; the generator knows the fold because it performed
the join (``schema_transforms._apply_fold``, driven by the family's ``Fold``), and the folded columns'
identity is that of the source dimension table — two facts that fold the SAME source share
ONE dimension. Degenerate operational IDs (a fact's own primary key) identify nothing and
must abstain; asserting a hierarchy over them is the characteristic wide-data error.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from testdata.families import (
    FAMILIES,
    business_concepts,
    cycles,
    degenerate_ids,
    dimensionless_measures,
    event_fact,
    folded_dimensions,
    foreign_keys,
    measured_in,
    merge_column_renames,
    reconciles_structurally,
    semantic_roles,
    stock_flow,
    subledger_event_facts,
    table_roles,
)
from testdata.families import folds as family_folds
from testdata.families import table_mapping as family_table_mapping
from testdata.identity import CorpusIdentity
from testdata.schema_transforms import ColumnStyle, restyle_column_name

if TYPE_CHECKING:
    import polars as pl

VERTICAL = "finance"

# --- metric_additivity ------------------------------------------------------
# Drill additivity per target (metric = a KPI rolled up through its DAG; measure =
# a standard_field's extract). `categorical_additive` / `time_additive` say whether a
# breakdown by that axis class reconciles to the unsliced total; the reason names why
# it does not (engine vocab: stock / average / distinct_count / ratio), null when it
# reconciles. Keys are metric/measure names, NOT table.column — this section
# is schema-shape invariant (nothing to remap).
_RATIO: dict[str, Any] = {
    "determinism": "function_symmetry",
    "categorical_additive": False,
    "time_additive": False,
    "categorical_reason": "ratio",
    "time_reason": "ratio",
    "note": "a ratio (X / Y) — non-additive on every axis.",
}
_FLOW_METRIC: dict[str, Any] = {
    "determinism": "label_dependent",
    "categorical_additive": True,
    "time_additive": True,
    "categorical_reason": None,
    "time_reason": None,
    "note": "a sum/difference of flow measures — additive on both axes (needs flow labels).",
}
_FLOW_MEASURE: dict[str, Any] = {
    "determinism": "label_dependent",
    "categorical_additive": True,
    "time_additive": True,
    "categorical_reason": None,
    "time_reason": None,
    "note": "SUM of an income-statement flow — additive on both axes (needs a flow label).",
}
_MARGIN_NOTE = "a margin (X / revenue) — a ratio; non-additive on every axis."
_STOCK_MEASURE: dict[str, Any] = {
    "determinism": "label_dependent",
    "categorical_additive": True,
    "time_additive": False,
    "categorical_reason": None,
    "time_reason": "stock",
    "note": "SUM of a balance-sheet stock — reconciles across categories, time-stripped.",
}


def _metric_additivity() -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {
        "active_accounts": {
            "determinism": "function_symmetry",
            "categorical_additive": False,
            "time_additive": False,
            "categorical_reason": "distinct_count",
            "time_reason": "distinct_count",
            "note": "COUNT(DISTINCT account) — the distinct set overlaps across slices.",
        },
        "average_transaction_value": {
            "determinism": "function_symmetry",
            "categorical_additive": False,
            "time_additive": False,
            "categorical_reason": "average",
            "time_reason": "average",
            "note": "AVG — an average of averages does not reconcile on any axis.",
        },
        "current_ratio": {**_RATIO, "note": "current_assets / current_liabilities — a ratio; non-additive everywhere."},
        "gross_margin": {**_RATIO, "note": _MARGIN_NOTE},
        "ebitda_margin": {**_RATIO, "note": _MARGIN_NOTE},
        "net_margin": {**_RATIO, "note": _MARGIN_NOTE},
        "operating_margin": {**_RATIO, "note": _MARGIN_NOTE},
        "dso": {**_RATIO, "note": "days-sales-outstanding — a ratio (AR / revenue × days); non-additive."},
        # The pinned denominator is PURCHASES (testdata.oracle), not COGS — this note
        # described the old formula for a release after the definition moved, which is
        # what `test_oracle` now pins the two files together to prevent.
        "dpo": {**_RATIO, "note": "days-payable-outstanding — a ratio (AP / purchases × days); non-additive."},
        "transaction_count": {
            "determinism": "label_dependent",
            "categorical_additive": True,
            "time_additive": True,
            "categorical_reason": None,
            "time_reason": None,
            "note": "COUNT over the journal_lines event fact — additive (needs event-grain).",
        },
        "gross_profit": dict(_FLOW_METRIC),
        "operating_income": dict(_FLOW_METRIC),
        "ebitda": dict(_FLOW_METRIC),
        "net_income": dict(_FLOW_METRIC),
    }
    measures: dict[str, dict[str, Any]] = {
        "account": {
            "determinism": "function_symmetry",
            "categorical_additive": False,
            "time_additive": False,
            "categorical_reason": "distinct_count",
            "time_reason": "distinct_count",
            "note": "The active_accounts extract — COUNT(DISTINCT), grounded by one metric.",
        },
        "revenue": dict(_FLOW_MEASURE),
        "cost_of_goods_sold": dict(_FLOW_MEASURE),
        "operating_expense": dict(_FLOW_MEASURE),
        "depreciation": dict(_FLOW_MEASURE),
        "interest": dict(_FLOW_MEASURE),
        "tax": dict(_FLOW_MEASURE),
        "current_assets": dict(_STOCK_MEASURE),
        "current_liabilities": dict(_STOCK_MEASURE),
        "accounts_receivable": dict(_STOCK_MEASURE),
        "accounts_payable": dict(_STOCK_MEASURE),
        "transaction_amount": {
            "determinism": "label_dependent",
            "categorical_additive": False,
            "time_additive": False,
            "categorical_reason": "average",
            "time_reason": "average",
            "note": (
                "Grounded by BOTH transaction_count (COUNT) and average_transaction_value (AVG); "
                "the measure verdict is the deduped most-restrictive fold across metrics, so AVG wins "
                "— proves the shared-field fold end-to-end."
            ),
        },
    }
    return {"metrics": metrics, "measures": measures}


# --- stock/flow, reconciliation lineage, roles, units, concepts, cycles -----
# All READ from the family registry's `Structure` declarations. This module used to
# author one canonical blob naming finance tables — a second place to remember a
# family's roles, units and cycles, one file away from where its tables are declared.
_STOCK_FLOW: dict[str, str] = stock_flow()
_RECONCILES_STRUCTURALLY: list[str] = reconciles_structurally()
_SUBLEDGER_EVENT_FACT: dict[str, str] = subledger_event_facts()

# --- FK topology --------------------------------------------------
# The generator's TRUE FK topology, read from the family registry rather than
# re-listed here. That is the point: the operating chain shipped for a whole release
# without its FKs in this file, because a second list is a second thing to forget. A
# family now declares its joins where it declares its tables, and this file publishes
# what it finds. The registry also records what is deliberately absent — `currency`
# (an enum, no dimension table) and the trial_balance/balance_sheet period fan trap.
_RELATIONSHIPS: list[dict[str, str]] = [
    {"from": source, "to": target} for source, target in foreign_keys()
]

# --- table + column roles -----------------------------------------
# is_fact_table HARD where structure decides: measure-bearing = fact, pure reference =
# dimension. The `ambiguous` bucket is genuinely modelable either way (an event header
# with no measure; a reference row that does carry a rate) -> reported, not asserted.
# semantic_role per column: measure graded recall+precision, timestamp graded recall.
_TABLE_ROLES: dict[str, list[str]] = table_roles()
_SEMANTIC_ROLES: dict[str, list[str]] = semantic_roles()

# --- measured_in / units --------------------
# The unit column each MONETARY measure is denominated in, declared by the family that
# owns the table. ``cross_unit`` (declared unit column carries >1 distinct value) is
# DATA-DERIVED at export from the generated frames — never authored. All-USD by model
# default, so it is False everywhere unless an injector writes a second currency INTO
# the unit column (mix_units' declared variant); the undeclared variant converts values
# only and correctly leaves cross_unit False (the gate grades the DECLARED surface).
_MEASURED_IN: dict[str, str | None] = measured_in()
_DIMENSIONLESS: frozenset[str] = dimensionless_measures()

# Pairings CREATED by the CoA fold: ``account_currency`` lands ON the balance facts at
# ``flat``/``single``, becoming their same-table unit source. general_ledger's
# debit/credit keep journal_lines' own in-table currency (the line's unit column —
# account_currency is the account's attribute, not the line's denomination).
_MEASURED_IN_FOLD: dict[str, dict[str, str]] = {
    "flat": {
        "trial_balance.debit_balance": "trial_balance.account_currency",
        "trial_balance.credit_balance": "trial_balance.account_currency",
        "balance_sheet.ending_balance": "balance_sheet.account_currency",
    },
    "single": {
        "trial_balance.debit_balance": "trial_balance.account_currency",
        "trial_balance.credit_balance": "trial_balance.account_currency",
        "balance_sheet.ending_balance": "balance_sheet.account_currency",
    },
}


# The measure→concept bindings a metric's grounding depends on (a missing one means
# the metric cannot ground). Graded HARD for recall. Dimension-concept bindings are
# LLM-selective discriminators → reported, not required.
_BUSINESS_CONCEPTS: dict[str, dict[str, str]] = {"required": business_concepts()}

# --- reconciles_with -------------------------------------------
# The event fact the structural witness reconciles measures against: trial_balance and
# balance_sheet are DERIVED by aggregating journal lines, so the generator knows the
# event side of every structural reconciliation.
_EVENT_FACT = event_fact()


def _build_reconciles_with(
    required: dict[str, str],
    reconciles_structurally: list[str],
    event_table: str | None,
    table_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The expected post-P2 ``reconciles_with`` edge set — derived, never authored.

    Two deterministic producers (the ruling, moved into P2), both keyed at
    Grounding grain — a Grounding is per (concept, relation):

    * ``aggregation_lineage`` — the witness reified as a Grounding→Grounding
      reconciliation: each structurally-reconciling measure ties out against its
      event fact's aggregation (trial_balance balance ↔ GL sum). Derived from
      ``reconciles_structurally`` × the generator's event fact. An entry whose
      measure landed IN the event relation after a merge is dropped — a measure
      cannot reconcile against an aggregation of its own relation.
    * ``multi_grounding`` — any concept with >= 2 groundings reconciles its
      groundings pairwise. The fan-in is the concept's distinct required-binding
      tables; a concept whose bindings collapse into ONE relation after a merge
      is dropped (nothing left to reconcile).
    """
    tm = table_mapping or {}
    lineage: list[dict[str, Any]] = []
    for measure in reconciles_structurally:
        subledger = _SUBLEDGER_EVENT_FACT.get(measure)
        fact = _remap_table(subledger, tm) if subledger is not None else event_table
        if fact is None or measure.partition(".")[0] == fact:
            continue
        lineage.append({"measure": measure, "event_table": fact})
    fan_in: dict[str, set[str]] = {}
    for col, concept in required.items():
        fan_in.setdefault(concept, set()).add(col.partition(".")[0])
    multi = [
        {"concept": concept, "relations": sorted(tables)}
        for concept, tables in sorted(fan_in.items())
        if len(tables) >= 2
    ]
    return {"aggregation_lineage": lineage, "multi_grounding": multi}


# --- business cycles ----------------------------------------------
# The cycles this corpus REALISTICALLY supports, declared by the families that own the
# tables backing them. Three have a strong structural backbone → required (a miss is a
# real recall gap); period_close is real but weakly signalled → soft.
_CYCLES: list[dict[str, Any]] = cycles()

# --- folded dimensions --------------------------------------------
# A folded dimension = a referenced dimension whose FK-target table a normalization
# level INLINED into a fact (the denormalized / wide / OBT shape). READ from the family
# registry's ``Fold`` declarations rather than mirrored here: this block used to be
# hand-authored "to mirror the inline transform", and a mirror is a second copy
# of a fact — the shape that let the operating chain ship without its FKs.
#
# ``folded_into`` lists the facts that carry the fold — two facts folding the SAME
# source dimension share ONE dimension by concept identity (the cross-fact case, no
# name/value heuristic). The fold is level-specific: ``full`` / ``partial`` are still
# normalized → no folds; at ``single`` every fact has collapsed onto one spine.
def _folded_dimensions(level: str) -> list[dict[str, Any]]:
    if level not in ("flat", "single"):
        return []
    return [
        {
            "concept": fold.concept,
            "source_dimension": fold.dimension,
            "fold_key": fold.on,
            "attributes": list(fold.attributes),
            "folded_into": ["mega_table"] if level == "single" else sorted(set(fold.into.values())),
        }
        for fold in family_folds()
    ]

# --- bus matrix --------------------------------
# The Kimball bus matrix: fact table x dimension concept -> HOW the fact exposes the
# dimension. Fully DERIVED per level from _RELATIONSHIPS + _FOLDED_DIMENSIONS +
# _REMOVED_TABLES — no per-level authoring. Provenance vocabulary:
#   referenced — a surviving FK to a dimension table (the identity)
#   folded     — the dimension's attributes inlined into the fact
#   key_only   — the FK column survives but the dimension table was removed by the
#                transform (e.g. bank_transactions.account_id at `flat` after CoA is
#                inlined elsewhere) — identity reachable only through the concept.
# `key` is the fact-side column carrying the exposure (FK column / fold key).
# The temporal/period conformed dimension is deliberately absent: temporal identity is
# the workspace calendar, not a categorical concept.
#
# A dimension table's concept comes from the fold that inlines it — the fold already has
# to name the concept for two facts folding the same source to be known to share ONE
# dimension, so naming it twice would be one fact in two places.
_DIM_TABLE_CONCEPTS: dict[str, str] = {fold.dimension: fold.concept for fold in family_folds()}

# Tables a level REMOVES without a single-valued table_mapping entry: their content fans
# out into MULTIPLE facts (CoA inlines into general_ledger, trial_balance and
# balance_sheet at `flat`), so no `old -> new` rename can express it. Their inbound FKs
# stop being discoverable relationships; the bus matrix records the key_only exposure
# instead (bank_transactions.account_id is the surviving key_only case).
_REMOVED_TABLES: dict[str, frozenset[str]] = {
    "flat": folded_dimensions(),
    "single": folded_dimensions(),
}


def _level_table_mapping(level: str | None) -> dict[str, str]:
    """The table_mapping *level* emits — the DEFAULT when a caller passes ``level``
    without a live mapping (tests, offline truth builds). The runner still passes the
    live mapping, which wins.

    Computed from the registry's merge/fold declarations by the same name algebra
    ``apply_normalization`` executes over frames. It was a hand-written table per level
    until the family registry could answer the question; a live-consistency test still
    pins the two together.
    """
    return family_table_mapping(level or "full")


def _build_bus_matrix(
    level: str | None, table_mapping: dict[str, str], column_style: ColumnStyle
) -> dict[str, dict[str, dict[str, str]]]:
    """``{fact_table: {concept: {provenance, key}}}`` for *level* — derived, not authored."""
    removed = _REMOVED_TABLES.get(level or "", frozenset())
    bus: dict[str, dict[str, dict[str, str]]] = {}
    # folded exposures first — they win over the key_only residue of the same FK.
    for fold in _folded_dimensions(level or ""):
        for fact in fold["folded_into"]:
            bus.setdefault(fact, {})[fold["concept"]] = {
                "provenance": "folded",
                "key": restyle_column_name(fold["fold_key"], column_style),
            }
    for rel in _RELATIONSHIPS:
        from_table, _, from_col = str(rel["from"]).partition(".")
        to_table = str(rel["to"]).partition(".")[0]
        concept = _DIM_TABLE_CONCEPTS.get(to_table)
        ft = _remap_table(from_table, table_mapping)
        if concept is None or from_table == to_table or ft in removed:
            continue  # not a dimension reference / the dim's own self-FK / fact gone
        # a fold on the same (fact, concept) already claimed the exposure -> setdefault
        kind = "key_only" if to_table in removed else "referenced"
        bus.setdefault(ft, {}).setdefault(
            concept,
            {"provenance": kind, "key": restyle_column_name(from_col, column_style)},
        )
    return {t: dict(sorted(c.items())) for t, c in sorted(bus.items())}


# Degenerate dimensions: a fact's OWN operational primary key —
# grounds to no dimension concept and carries NO cross-table identity, so the right
# answer is to ABSTAIN, not to assert it as a folded hierarchy. ``general_ledger.line_id`` is the
# journal-line PK that survives the fold as the fact's own key. This is the exact
# surface on which a distinct-count ratio wrongly asserts a hierarchy over wide data
# (a near-key guard misses a heavy-tailed id).
# Declared by the family that owns the surviving key.


def _build_measured_in(
    level: str | None = None,
    table_mapping: dict[str, str] | None = None,
    column_style: ColumnStyle = "snake_case",
    column_renames: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """The measured_in truth for *level*, remapped — derived, not textually rewritten.

    Rebuilt from the statics per level (like ``folded_dimensions`` / ``bus_matrix``)
    because the pairing itself is level-dependent (the CoA fold creates unit columns).
    Entries whose measure lands in a REMOVED table are dropped; entries that collapse
    to the same remapped measure after a merge (``single``'s diagonal concat unions
    same-named columns) are deduped, first in canonical sort order wins.
    ``cross_unit`` defaults False here; the export overrides it from the DATA.
    """
    tm = table_mapping or {}
    removed = _REMOVED_TABLES.get(level or "", frozenset())
    pairs = dict(_MEASURED_IN)
    pairs.update(_MEASURED_IN_FOLD.get(level or "", {}))

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for measure, unit in sorted(pairs.items()):
        if measure.partition(".")[0] in removed:
            continue
        remapped_measure = _remap_qualified(measure, tm, column_style, column_renames)
        if remapped_measure in seen:
            continue
        seen.add(remapped_measure)
        entry: dict[str, Any] = {
            "measure": remapped_measure,
            "unit_column": (
                _remap_qualified(unit, tm, column_style, column_renames) if unit else None
            ),
            "cross_unit": False,
        }
        if measure in _DIMENSIONLESS:
            entry["dimensionless"] = True
        out.append(entry)
    return out


def canonical_metadata_truth() -> dict[str, Any]:
    """The finance corpus's agent-layer ground truth at canonical (``full``/snake) names.

    A fresh deep-copyable dict every call — callers (remap, export) may mutate it.
    ``folded_dimensions`` / ``degenerate_ids`` are empty at canonical (``full``) — they
    are populated per normalization level by ``remap_metadata_truth``.
    """
    return {
        "vertical": VERTICAL,
        "metric_additivity": _metric_additivity(),
        "stock_flow": dict(_STOCK_FLOW),
        "reconciles_structurally": list(_RECONCILES_STRUCTURALLY),
        # direction_reliable is part of the canonical shape (True at full — every
        # parent keeps its grain), so the identity-remap invariant holds.
        "relationships": [{**rel, "direction_reliable": True} for rel in deepcopy(_RELATIONSHIPS)],
        "table_roles": {k: list(v) for k, v in _TABLE_ROLES.items()},
        "semantic_roles": {k: list(v) for k, v in _SEMANTIC_ROLES.items()},
        "business_concepts": {"required": dict(_BUSINESS_CONCEPTS["required"])},
        "reconciles_with": _build_reconciles_with(
            _BUSINESS_CONCEPTS["required"], _RECONCILES_STRUCTURALLY, _EVENT_FACT
        ),
        "cycles": deepcopy(_CYCLES),
        "measured_in": _build_measured_in(),
        # FK-role truth: empty unless the run carries the role-play probe
        # shape — filled data-conditionally at export (_apply_roleplay_truth).
        "fk_roles": {},
        "folded_dimensions": [],
        "degenerate_ids": [],
        "bus_matrix": _build_bus_matrix("full", {}, "snake_case"),
    }


# --- remap ------------------------------------------------------------------


def _remap_table(table: str, table_mapping: dict[str, str]) -> str:
    return table_mapping.get(table, table)


# Column renames the merge transforms apply, READ from the merge declarations that
# perform them — payments' conflicting columns are prefixed before the join. (The fold
# renames are irrelevant here: the dimension table is REMOVED and its qualified
# references are dropped.) Keyed by qualified CANONICAL name. Every non-``full`` level
# runs every declared merge, so these apply at partial / flat / single alike.
_MERGE_COLUMN_RENAMES: dict[str, str] = merge_column_renames()


def _remap_qualified(
    name: str,
    table_mapping: dict[str, str],
    column_style: ColumnStyle,
    column_renames: dict[str, str] | None = None,
) -> str:
    """Rewrite a ``table.column`` reference through renames + table_mapping + style."""
    table, _, column = name.partition(".")
    if column_renames and name in column_renames:
        column = column_renames[name]
    return f"{_remap_table(table, table_mapping)}.{restyle_column_name(column, column_style)}"


def _build_folded_dimensions(level: str | None, column_style: ColumnStyle) -> list[dict[str, Any]]:
    """The folded-dimension truth for *level* (empty for ``full`` / ``partial`` / None).

    ``fold_key`` + ``attributes`` are column names → restyled to the export style;
    ``folded_into`` / ``source_dimension`` are already the post-transform table names.
    """
    out: list[dict[str, Any]] = []
    for fold in _folded_dimensions(level or ""):
        out.append(
            {
                "concept": fold["concept"],
                "source_dimension": fold["source_dimension"],
                "fold_key": restyle_column_name(fold["fold_key"], column_style),
                "attributes": [restyle_column_name(a, column_style) for a in fold["attributes"]],
                "folded_into": list(fold["folded_into"]),
            }
        )
    return out


def _build_degenerate_ids(
    level: str | None, table_mapping: dict[str, str], column_style: ColumnStyle
) -> list[str]:
    """Degenerate operational-ID columns for *level* (``table.column``, column restyled)."""
    return [
        _remap_qualified(q, table_mapping, column_style) for q in degenerate_ids(level or "")
    ]


def remap_metadata_truth(
    truth: dict[str, Any],
    *,
    table_mapping: dict[str, str] | None = None,
    column_style: ColumnStyle = "snake_case",
    level: str | None = None,
) -> dict[str, Any]:
    """Rewrite *truth* so every table/column reference matches the exported data.

    ``table_mapping`` is the ``old -> new`` table rename returned by
    ``apply_normalization``; ``column_style`` is the exported column naming style.
    Both default to the identity (``full`` / snake_case). ``metric_additivity`` is
    keyed by metric names (schema-shape invariant) and passes through untouched.
    ``level`` is the normalization level — it drives the folded-dimension truth
    (``folded_dimensions`` / ``degenerate_ids``), which is empty unless the level folds
    (``flat`` / ``single``); None/``full``/``partial`` leave them empty. When
    ``level`` is given without a live ``table_mapping``, the level's known mapping
    (derived from the registry) is used, so every section stays at post-transform names.
    """
    tm = table_mapping if table_mapping is not None else _level_table_mapping(level)
    out = deepcopy(truth)
    removed = _REMOVED_TABLES.get(level or "", frozenset())
    # Every non-full level runs the invoice merge and carries its column renames.
    renames = _MERGE_COLUMN_RENAMES if (level or "full") != "full" else {}

    def _keep(qualified: str) -> bool:
        """A reference into a REMOVED table is not gradeable — the table is gone."""
        return qualified.partition(".")[0] not in removed

    # stock_flow / reconciles_structurally / semantic_roles / business_concepts:
    # table.column references — renamed to the merged schema, removed-table refs dropped.
    out["stock_flow"] = {
        _remap_qualified(k, tm, column_style, renames): v for k, v in truth["stock_flow"].items() if _keep(k)
    }
    out["reconciles_structurally"] = [
        _remap_qualified(k, tm, column_style, renames) for k in truth["reconciles_structurally"] if _keep(k)
    ]
    out["semantic_roles"] = {
        role: [_remap_qualified(c, tm, column_style, renames) for c in cols if _keep(c)]
        for role, cols in truth["semantic_roles"].items()
    }
    out["business_concepts"] = {
        "required": {
            _remap_qualified(k, tm, column_style, renames): v
            for k, v in truth["business_concepts"]["required"].items()
            if _keep(k)
        }
    }

    # reconciles_with: RECOMPUTED from the remapped sections (mirrors bus_matrix's
    # derived approach) — the per-level drop rules (measure merged into its event
    # relation; a concept's fan-in collapsing below 2) live in the builder, so a
    # textual remap of the canonical edge set would be wrong at `single`.
    out["reconciles_with"] = _build_reconciles_with(
        out["business_concepts"]["required"],
        out["reconciles_structurally"],
        None if _EVENT_FACT is None or _EVENT_FACT in removed else _remap_table(_EVENT_FACT, tm),
        tm,
    )

    # relationships: remap both endpoints; drop cross-table FKs that a merge collapsed
    # into a single table (no longer a discoverable relationship); keep genuine self-FKs.
    # Also drop FKs touching a table the level REMOVED outright (`_REMOVED_TABLES`) —
    # a relationship to a nonexistent table is not discoverable; the surviving key
    # column's exposure is recorded in `bus_matrix` as key_only instead.
    # str | bool: `direction_reliable` is a bool alongside the qualified-name strings.
    rels: list[dict[str, str | bool]] = []
    for rel in truth["relationships"]:
        ft, _, _ = str(rel["from"]).partition(".")
        tt, _, _ = str(rel["to"]).partition(".")
        new_ft, new_tt = _remap_table(ft, tm), _remap_table(tt, tm)
        collapsed_by_merge = ft != tt and new_ft == new_tt
        if collapsed_by_merge or ft in removed or tt in removed:
            continue
        rels.append(
            {
                "from": _remap_qualified(str(rel["from"]), tm, column_style),
                "to": _remap_qualified(str(rel["to"]), tm, column_style),
                # Direction is graded only while the parent (to) side kept its
                # grain. A merge that folds the parent into a line-grain fact
                # destroys the key (journal_entries.entry_id repeats per GL
                # line), and a uniqueness-canonical orientation
                # (#495 many→one) legitimately flips — grade those undirected
                # (Philipp's ruling, 2026-07-16).
                "direction_reliable": tt not in tm,
            }
        )
    out["relationships"] = rels

    # table_roles / cycles.key_tables: table-name lists (dedupe after a merge;
    # a REMOVED table has no row in the corpus and cannot carry a role).
    out["table_roles"] = {
        role: _dedupe([_remap_table(t, tm) for t in tables if t not in removed])
        for role, tables in truth["table_roles"].items()
    }
    out["cycles"] = [
        {**cyc, "key_tables": _dedupe([_remap_table(t, tm) for t in cyc["key_tables"] if t not in removed])}
        for cyc in truth["cycles"]
    ]

    # measured_in: rebuilt per level (the CoA fold CREATES unit columns at flat/single,
    # so a textual remap of the canonical pairing would miss them — same derived
    # approach as folded_dimensions / bus_matrix). cross_unit stays False here; the
    # export overrides it from the generated data.
    out["measured_in"] = _build_measured_in(level, tm, column_style, renames)

    # folded_dimensions / degenerate_ids: authored at post-transform names, selected by
    # level. Empty unless the level actually folds a dimension into a fact.
    out["folded_dimensions"] = _build_folded_dimensions(level, column_style)
    out["degenerate_ids"] = _build_degenerate_ids(level, tm, column_style)
    # bus_matrix: fully derived per level (referenced FKs + folds + key_only residue).
    out["bus_matrix"] = _build_bus_matrix(level or "full", tm, column_style)
    return out


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving de-duplication."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# --- role-playing FKs -------------
# Truth for the conditional role-play probe shape, authored from the probe DESIGN
# (models.Address/Order/Delivery + inject_role_playing_fks): one dimension, one fact
# with TWO FK roles to it, a second fact sharing the ship_to role under a different
# name. Emitted ONLY when the run's frames carry the shape (the probe tables are
# never renamed by normalization, so canonical names ARE export names). The
# bus_matrix is deliberately NOT extended: its {fact: {concept: {key}}} schema
# cannot express two roled exposures of one concept — that inexpressibility is
# 's point; role truth lives in fk_roles + the relationships' fk_role field.
_ROLEPLAY_TABLES: tuple[str, ...] = ("addresses", "orders", "deliveries")
_ROLEPLAY_RELATIONSHIPS: list[dict[str, str]] = [
    {"from": "orders.bill_to_addr", "to": "addresses.address_id", "fk_role": "bill_to"},
    {"from": "orders.ship_to_addr", "to": "addresses.address_id", "fk_role": "ship_to"},
    {"from": "deliveries.delivery_addr", "to": "addresses.address_id", "fk_role": "ship_to"},
    {"from": "deliveries.order_id", "to": "orders.order_id"},
]


def _apply_relationship_assertability(
    truth: dict[str, Any], dataframes: Mapping[str, Any]
) -> None:
    """Drop FK claims whose TARGET column is no longer a key in the exported frames.

    A foreign key references a KEY — a column whose values are unique. A merge can fold
    a parent into a finer grain (``journal_entries`` into ``general_ledger`` at ``flat``,
    where ``entry_id`` repeats once per GL line), and the surviving column is then an
    ordinary repeating attribute. The edge is no longer a foreign key at that shape, so
    asserting one states something false about the corpus.

    ``direction_reliable`` (2026-07-16) already covers the softer half of this — a merged
    parent can make a uniqueness-canonical orientation flip, so grade the edge
    undirected. It does NOT cover the case measured on clean-flat 2026-07-29, where the
    target's uniqueness collapses to 0.455 and the edge is unrecoverable in EITHER
    orientation (the reverse containment is ~0.18). Undirected grading cannot rescue an
    edge that is not an inclusion dependency in any direction.

    DATA-DERIVED, on purpose — the ``cross_unit`` precedent. The static remap knows a
    merge happened but not what it did to the values; only the frames know whether the
    key survived. Deriving it means a merge that PRESERVES uniqueness keeps its edge
    asserted, and no hand-maintained exception list can drift.

    The test is the DEFINITION of a key (no duplicate values), deliberately not the
    uniqueness tolerance of whatever is being graded. Filtering our truth through the
    gate under test would make the oracle unable to ever fail on that gate — the
    instrument tuned to the thing it measures.

    Dropped edges are recorded under ``relationships_not_assertable`` rather than
    deleted silently: a truth that quietly shrinks reads as a corpus that got easier.
    """
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for rel in truth.get("relationships") or []:
        table, _, column = str(rel["to"]).partition(".")
        df = dataframes.get(table)
        if df is None or column not in df.columns:
            kept.append(rel)
            continue
        values = df[column].drop_nulls()
        if len(values) and values.n_unique() < len(values):
            dropped.append(
                {
                    **rel,
                    "reason": (
                        f"target {rel['to']} is not a key in this shape "
                        f"({values.n_unique()}/{len(values)} distinct) — a merge folded "
                        "the parent into a finer grain, so no foreign key exists here"
                    ),
                }
            )
            continue
        kept.append(rel)
    truth["relationships"] = kept
    if dropped:
        truth["relationships_not_assertable"] = dropped


def _apply_roleplay_truth(truth: dict[str, Any], dataframes: Mapping[str, Any]) -> None:
    """Extend *truth* with the role-play sections when the shape is present (data-
    conditional, like cross_unit): relationships (+fk_role on the roled edges),
    table_roles, timestamp semantic_roles, and the fk_roles map O6 grades pairing
    against. A corpus without the probe tables is untouched — canonical equality
    holds for every other strategy."""
    present = all(
        t in dataframes and len(dataframes[t]) > 0 for t in _ROLEPLAY_TABLES
    )
    if not present:
        return
    truth["relationships"] = truth["relationships"] + [
        {**rel, "direction_reliable": True} for rel in deepcopy(_ROLEPLAY_RELATIONSHIPS)
    ]
    truth["table_roles"]["facts"] = truth["table_roles"]["facts"] + ["orders", "deliveries"]
    truth["table_roles"]["dimensions"] = truth["table_roles"]["dimensions"] + ["addresses"]
    truth["semantic_roles"]["timestamp"] = truth["semantic_roles"]["timestamp"] + [
        "orders.order_date",
        "deliveries.delivery_date",
    ]
    truth["fk_roles"] = {
        rel["from"]: rel["fk_role"] for rel in _ROLEPLAY_RELATIONSHIPS if "fk_role" in rel
    }


# --- export -----------------------------------------------------------------

_HEADER = (
    "# metadata_truth.yaml — agent-layer ground truth for the finance corpus.\n"
    "# GENERATED by dataraum-testdata (src/testdata/metadata_truth.py); do not hand-edit.\n"
    "# Table/column names are remapped to match this run's exported schema.\n"
)


def drop_absent_optional_tables(truth: dict[str, Any], present: Iterable[str]) -> None:
    """Strip claims about optional tables this corpus does not carry.

    An optional family declares what its tables MEAN, and that declaration is true
    whenever they exist. Publishing it unconditionally makes the truth file assert
    that a corpus has a dimension it does not have — a false answer key, which is
    worse than a missing one. So the declaration is filtered against the frames that
    actually shipped, the same way the role-play and cross-unit blocks already are.
    """
    tables = set(present)
    absent = {table for fam in FAMILIES if fam.optional for table in fam.tables if table not in tables}
    if not absent:
        return
    for role, tables in truth["table_roles"].items():
        truth["table_roles"][role] = [t for t in tables if t not in absent]
    truth["relationships"] = [
        rel
        for rel in truth["relationships"]
        if str(rel["from"]).partition(".")[0] not in absent and str(rel["to"]).partition(".")[0] not in absent
    ]


def export_metadata_truth(
    output_dir: Path,
    *,
    table_mapping: dict[str, str] | None = None,
    column_style: ColumnStyle = "snake_case",
    level: str = "full",
    dataframes: Mapping[str, pl.DataFrame] | None = None,
    identity: CorpusIdentity | None = None,
) -> None:
    """Write ``metadata_truth.yaml`` to *output_dir*, remapped to the run's schema.

    Emitted for every scenario alongside ``entropy_map.yaml`` / ``ground_truth.yaml``.
    ``table_mapping`` comes from ``apply_normalization``; ``column_style`` is the
    exported naming style (snake_case for single-source; the multi-source top-level
    file stays canonical, mirroring the top-level ``entropy_map.yaml``). ``level`` is the
    normalization level driving the folded-dimension truth; ``full`` folds none.

    ``dataframes`` (the runner's post-injection, post-normalization frames) drives the
    DATA-DERIVED ``measured_in.cross_unit`` flags: a measure's declared unit column
    carrying >1 distinct value → True. Deriving from the data rather than the
    injection config is deliberate (the corrupt_dates lesson): an injector that
    silently no-ops can never produce a false truth, and the undeclared mix_units
    variant (values converted, unit column untouched) correctly stays False.

    ``identity`` stamps which corpus this structure describes. Structural truth moves
    with the family set more than any other file does — a new family adds tables,
    joins and roles — so it is the file most worth being able to date.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = remap_metadata_truth(
        canonical_metadata_truth(), table_mapping=table_mapping, column_style=column_style, level=level
    )
    if dataframes is not None:
        for entry in truth["measured_in"]:
            if not entry["unit_column"]:
                continue
            table, _, unit_col = str(entry["unit_column"]).partition(".")
            df = dataframes.get(table)
            if df is not None and unit_col in df.columns:
                entry["cross_unit"] = df[unit_col].drop_nulls().n_unique() > 1
        _apply_relationship_assertability(truth, dataframes)
        # role-play shape truth: emitted only when the frames carry it
        _apply_roleplay_truth(truth, dataframes)
        drop_absent_optional_tables(truth, dataframes)
    if identity is not None:
        truth = {"corpus": identity.as_dict(), **truth}
    with open(output_dir / "metadata_truth.yaml", "w") as f:
        f.write(_HEADER)
        yaml.dump(truth, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
