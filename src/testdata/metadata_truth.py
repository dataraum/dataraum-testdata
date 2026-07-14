"""Agent-layer ground truth for the finance corpus — the generator's own answers.

The engine persists agent-authored / derived metadata in ``current_*`` read views
(relationships, table/semantic roles, column concepts, stock/flow, cycles, metric
additivity). The eval grades that agent layer the way detectors are graded against
``entropy_map.yaml`` — but the oracle needs a truth to grade against, and the
generator already *knows* every one of those answers. This module is that truth,
authored from the finance MODELS + generator design (``canonical/finance/models.py``,
``generators.py``, ``ground_truth.py``), and exported alongside ``entropy_map.yaml``
and ``ground_truth.yaml`` as ``metadata_truth.yaml`` (DAT-682).

It is the testdata-side home of what the eval hand-authored under DAT-684/685/686/718;
the eval now consumes this export, so the two never drift (an eval Tier-1 consistency
test binds them).

Determinism split (the anti-Goodhart grammar, mirrored by the eval oracle):
  * ``function_symmetry`` verdicts (AVG / COUNT(DISTINCT) / ratio never reconcile;
    the four HARD structural relationships/roles) are fixed by structure alone and
    HARD-asserted — no upstream label can game them.
  * ``label_dependent`` verdicts (SUM(flow) additive, SUM(stock) time-stripped,
    COUNT(event) additive) also need a correct upstream detector label and are graded
    as diagnostics (xfail, banded) — re-grading a detector deterministically through a
    downstream oracle would be Goodhart at the harness level.

Remap-safety (DAT-682 AC): the truth is authored at canonical (``full`` / snake_case)
table + column names. ``remap_metadata_truth`` rewrites table names through the
normalization ``table_mapping`` and column names through a ``column_style``, mirroring
how ``InjectionRegistry.remap_tables`` keeps ``entropy_map.yaml`` valid. Cross-table
FKs that collapse to a single table after a merge are dropped (no longer discoverable);
a genuine self-referential FK (``chart_of_accounts.parent_id``) is kept. Column renames
introduced by normalization *merges* (e.g. ``payments.amount -> invoice_data.payment_amount``)
are NOT reflected — the merged columns keep canonical names, the same table-only
contract ``entropy_map.yaml`` follows, and consistent with the wide-variant grading
being truth-free.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from testdata.schema_transforms import ColumnStyle, restyle_column_name

VERTICAL = "finance"

# --- metric_additivity ------------------------------------------------------
# Drill additivity per target (metric = a KPI rolled up through its DAG; measure =
# a standard_field's extract). `categorical_additive` / `time_additive` say whether a
# breakdown by that axis class reconciles to the unsliced total; the reason names why
# it does not (engine vocab: stock / average / distinct_count / ratio), null when it
# reconciles. Keys are ontology metric/measure names, NOT table.column — this section
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
        "dpo": {**_RATIO, "note": "days-payable-outstanding — a ratio (AP / COGS × days); non-additive."},
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


# --- stock/flow per measure column (DAT-685) --------------------------------
# The generator-known temporal_behavior: `additive` = per-period flow (sums across
# time); `point_in_time` = stock/level (does not sum across time). Design oracle:
# TrialBalance is per-period movement (FLOW); BalanceSheet.ending_balance is a
# carry-forward level (STOCK); fx_rates.rate is a price level (STOCK).
_STOCK_FLOW: dict[str, str] = {
    "journal_lines.debit": "additive",
    "journal_lines.credit": "additive",
    "journal_lines.net_amount": "additive",
    "invoices.amount": "additive",
    "payments.amount": "additive",
    "bank_transactions.amount": "additive",
    "trial_balance.debit_balance": "additive",
    "trial_balance.credit_balance": "additive",
    "balance_sheet.ending_balance": "point_in_time",
    "fx_rates.rate": "point_in_time",
}

# Measures that reconcile against a finer event fact via the DAT-491 structural
# stock/flow witness: the per-period movement tables reconcile `per_period` (FLOW),
# the carry-forward level reconciles `cumulative` (STOCK) vs journal_lines.
_RECONCILES_STRUCTURALLY: list[str] = [
    "trial_balance.debit_balance",
    "trial_balance.credit_balance",
    "balance_sheet.ending_balance",
]

# --- FK topology (DAT-684) --------------------------------------------------
# The generator's TRUE FK topology from the models' FK docstrings (not from what the
# LLM accepted — that would be circular). Enumerates every FK-bearing column across
# the 9 source tables. Deliberately EXCLUDES two near-misses: `currency` (Currency is
# an enum, no dimension table exists) and trial_balance.period ↔ balance_sheet.period
# (a shared conformed period dimension / fan trap, not an FK — the DAT-723 false
# positive the precision test catches).
_RELATIONSHIPS: list[dict[str, str]] = [
    {"from": "journal_lines.entry_id", "to": "journal_entries.entry_id"},
    {"from": "journal_lines.account_id", "to": "chart_of_accounts.account_id"},
    {"from": "invoices.entry_id", "to": "journal_entries.entry_id"},
    {"from": "payments.invoice_id", "to": "invoices.invoice_id"},
    {"from": "bank_transactions.account_id", "to": "chart_of_accounts.account_id"},
    {"from": "bank_transactions.payment_id", "to": "payments.payment_id"},
    {"from": "trial_balance.account_id", "to": "chart_of_accounts.account_id"},
    {"from": "balance_sheet.account_id", "to": "chart_of_accounts.account_id"},
    {"from": "chart_of_accounts.parent_id", "to": "chart_of_accounts.account_id"},
]

# --- table + column roles (DAT-685) -----------------------------------------
# is_fact_table HARD where structure decides: measure-bearing = fact, pure reference =
# dimension. journal_entries (event header, no measure) and fx_rates (rate lookup that
# also carries a measure) are genuinely modelable either way → reported, not asserted.
_TABLE_ROLES: dict[str, list[str]] = {
    "facts": ["journal_lines", "invoices", "payments", "bank_transactions", "trial_balance", "balance_sheet"],
    "dimensions": ["chart_of_accounts"],
    "ambiguous": ["journal_entries", "fx_rates"],
}

# semantic_role per column: measure graded recall+precision, timestamp graded recall
# (both load-bearing — drivers_phase filters measure, slicing reads timestamps). key /
# dimension / attribute are convention-dependent → reported by the oracle, not asserted.
_SEMANTIC_ROLES: dict[str, list[str]] = {
    "measure": [
        "journal_lines.debit",
        "journal_lines.credit",
        "journal_lines.net_amount",
        "invoices.amount",
        "payments.amount",
        "bank_transactions.amount",
        "trial_balance.debit_balance",
        "trial_balance.credit_balance",
        "balance_sheet.ending_balance",
        "fx_rates.rate",
    ],
    "timestamp": [
        "journal_entries.date",
        "invoices.date",
        "invoices.due_date",
        "payments.date",
        "bank_transactions.date",
        "fx_rates.date",
        "trial_balance.period",
        "balance_sheet.period",
    ],
}

# The measure→ontology-concept bindings metric grounding depends on (a missing one
# means the metric cannot ground). Graded HARD for recall. Dimension-concept bindings
# are LLM-selective discriminators → reported, not required.
_BUSINESS_CONCEPTS: dict[str, dict[str, str]] = {
    "required": {
        "journal_lines.debit": "debit",
        "journal_lines.credit": "credit",
        "trial_balance.debit_balance": "account_balance",
        "trial_balance.credit_balance": "account_balance",
        "balance_sheet.ending_balance": "account_balance",
        "invoices.amount": "transaction_amount",
        "payments.amount": "transaction_amount",
        "bank_transactions.amount": "transaction_amount",
    }
}

# --- business cycles (DAT-686) ----------------------------------------------
# The cycles this finance-9 corpus REALISTICALLY supports, derived from the finance
# cycle vocabulary × the corpus's actual tables + completion columns. Three have a
# strong structural backbone → required (a miss is a real recall gap); period_close is
# real but weakly signalled → soft.
_CYCLES: list[dict[str, Any]] = [
    {"canonical_type": "journal_entry_cycle", "key_tables": ["journal_entries", "journal_lines"], "required": True},
    {"canonical_type": "accounts_payable", "key_tables": ["invoices", "payments"], "required": True},
    {"canonical_type": "bank_reconciliation", "key_tables": ["bank_transactions", "payments"], "required": True},
    {"canonical_type": "period_close", "key_tables": ["trial_balance", "balance_sheet"], "required": False},
]


def canonical_metadata_truth() -> dict[str, Any]:
    """The finance corpus's agent-layer ground truth at canonical (``full``/snake) names.

    A fresh deep-copyable dict every call — callers (remap, export) may mutate it.
    """
    return {
        "vertical": VERTICAL,
        "metric_additivity": _metric_additivity(),
        "stock_flow": dict(_STOCK_FLOW),
        "reconciles_structurally": list(_RECONCILES_STRUCTURALLY),
        "relationships": deepcopy(_RELATIONSHIPS),
        "table_roles": {k: list(v) for k, v in _TABLE_ROLES.items()},
        "semantic_roles": {k: list(v) for k, v in _SEMANTIC_ROLES.items()},
        "business_concepts": {"required": dict(_BUSINESS_CONCEPTS["required"])},
        "cycles": deepcopy(_CYCLES),
    }


# --- remap ------------------------------------------------------------------


def _remap_table(table: str, table_mapping: dict[str, str]) -> str:
    return table_mapping.get(table, table)


def _remap_qualified(name: str, table_mapping: dict[str, str], column_style: ColumnStyle) -> str:
    """Rewrite a ``table.column`` reference through the table_mapping + column_style."""
    table, _, column = name.partition(".")
    return f"{_remap_table(table, table_mapping)}.{restyle_column_name(column, column_style)}"


def remap_metadata_truth(
    truth: dict[str, Any],
    *,
    table_mapping: dict[str, str] | None = None,
    column_style: ColumnStyle = "snake_case",
) -> dict[str, Any]:
    """Rewrite *truth* so every table/column reference matches the exported data.

    ``table_mapping`` is the ``old -> new`` table rename returned by
    ``apply_normalization``; ``column_style`` is the exported column naming style.
    Both default to the identity (``full`` / snake_case). ``metric_additivity`` is
    keyed by ontology names (schema-shape invariant) and passes through untouched.
    """
    tm = table_mapping or {}
    out = deepcopy(truth)

    # stock_flow / reconciles_structurally / semantic_roles / business_concepts:
    # table.column references.
    out["stock_flow"] = {_remap_qualified(k, tm, column_style): v for k, v in truth["stock_flow"].items()}
    out["reconciles_structurally"] = [_remap_qualified(k, tm, column_style) for k in truth["reconciles_structurally"]]
    out["semantic_roles"] = {
        role: [_remap_qualified(c, tm, column_style) for c in cols] for role, cols in truth["semantic_roles"].items()
    }
    out["business_concepts"] = {
        "required": {_remap_qualified(k, tm, column_style): v for k, v in truth["business_concepts"]["required"].items()}
    }

    # relationships: remap both endpoints; drop cross-table FKs that a merge collapsed
    # into a single table (no longer a discoverable relationship); keep genuine self-FKs.
    rels: list[dict[str, str]] = []
    for rel in truth["relationships"]:
        ft, _, _ = str(rel["from"]).partition(".")
        tt, _, _ = str(rel["to"]).partition(".")
        new_ft, new_tt = _remap_table(ft, tm), _remap_table(tt, tm)
        collapsed_by_merge = ft != tt and new_ft == new_tt
        if collapsed_by_merge:
            continue
        rels.append(
            {
                "from": _remap_qualified(str(rel["from"]), tm, column_style),
                "to": _remap_qualified(str(rel["to"]), tm, column_style),
            }
        )
    out["relationships"] = rels

    # table_roles / cycles.key_tables: table-name lists (dedupe after a merge).
    out["table_roles"] = {
        role: _dedupe([_remap_table(t, tm) for t in tables]) for role, tables in truth["table_roles"].items()
    }
    out["cycles"] = [
        {**cyc, "key_tables": _dedupe([_remap_table(t, tm) for t in cyc["key_tables"]])} for cyc in truth["cycles"]
    ]
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


# --- export -----------------------------------------------------------------

_HEADER = (
    "# metadata_truth.yaml — agent-layer ground truth for the finance corpus.\n"
    "# GENERATED by dataraum-testdata (src/testdata/metadata_truth.py); do not hand-edit.\n"
    "# Table/column names are remapped to match this run's exported schema.\n"
)


def export_metadata_truth(
    output_dir: Path,
    *,
    table_mapping: dict[str, str] | None = None,
    column_style: ColumnStyle = "snake_case",
) -> None:
    """Write ``metadata_truth.yaml`` to *output_dir*, remapped to the run's schema.

    Emitted for every scenario alongside ``entropy_map.yaml`` / ``ground_truth.yaml``.
    ``table_mapping`` comes from ``apply_normalization``; ``column_style`` is the
    exported naming style (snake_case for single-source; the multi-source top-level
    file stays canonical, mirroring the top-level ``entropy_map.yaml``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = remap_metadata_truth(
        canonical_metadata_truth(), table_mapping=table_mapping, column_style=column_style
    )
    with open(output_dir / "metadata_truth.yaml", "w") as f:
        f.write(_HEADER)
        yaml.dump(truth, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
