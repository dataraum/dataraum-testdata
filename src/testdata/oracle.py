"""The oracle contract — every published metric carries the definition that produced it.

``ground_truth.yaml`` used to publish numbers alone. Grading a consumer against it was
therefore manual: a defensible alternative definition read as a *delta to be argued*
rather than a *variant to be matched*. DPO is the canonical case — days-payable over
purchases and days-payable over total expenses are both correct answers to different
questions, and a consumer computing the second scored as wrong against the first.

This module holds the declaration. Every metric states its unit, its kind (which fixes
how a window aggregates it), the grains it is published at, the pinned formula, the
scope that formula ranges over, and its legitimate variants — each variant carrying its
own values, so a consumer's alternative grades as a *named* variant.

Two halves, and the seam matters for S0 (``docs/operating-model.md`` §3):

* the **mechanism** — :class:`Metric`, :class:`Variant`, the window rules and
  :func:`build_contract` — which is vertical-agnostic;
* the **finance registry** below it, which is not. When ``ground_truth`` splits into
  per-family fragments, each family declares its own metrics against this same
  mechanism and the registry becomes their union.

Values live nowhere in here. ``ground_truth._metric_values`` computes them and
:func:`build_contract` joins the two, refusing to publish a metric without values or
values without a definition — which is the whole point of having a contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# How a window aggregates a metric of each kind. Published per metric so a consumer
# reading a quarter out of monthly values does not have to guess: a stock takes the
# window's LAST period, a ratio is recomputed on the window's own aggregates (never
# averaged), a flow and a count sum.
_WINDOW_RULE: dict[str, str] = {
    "flow": "sum",
    "count": "sum",
    "stock": "last",
    "ratio": "recompute",
}

# Grains a metric may be published at. `month` / `year` are period grains keyed by
# period label; the rest are entity grains keyed by entity id.
_GRAINS: frozenset[str] = frozenset({"month", "year", "customer", "product_group"})

_UNITS: frozenset[str] = frozenset({"currency", "days", "percent", "count"})


@dataclass(frozen=True)
class Variant:
    """A legitimate alternative definition of a metric, published with its own values.

    Not a correction and not a footnote: both the pinned definition and the variant are
    right, and which one a consumer computes depends on what it can separate. The
    ``rationale`` says when a reader would land here rather than on the pinned formula.
    """

    id: str
    title: str
    definition: str
    rationale: str


@dataclass(frozen=True)
class Metric:
    """One published metric and the definition that produced it.

    ``kind`` fixes the window rule (see ``_WINDOW_RULE``) and, at the same time, the
    additivity verdict ``metadata_truth`` publishes for the same name — a ratio is
    non-additive on every axis, a stock reconciles across categories but not across
    time. ``tests/test_oracle.py`` pins the two files to each other.

    ``basis`` is ``derived`` for everything this corpus computes today. The slot exists
    because §5 requires a figure synthesized without a real basis to be *marked* as
    such rather than footnoted — Throughput cost variance is the known future case.

    ``breakdown`` names the axis when a metric's value is a mapping rather than a
    scalar (invoice counts by status, payment counts by method).

    ``families`` names the families whose tables the metric is computed from — usually
    one, sometimes two: ``revenue`` reads the GL at period grain and the order lines at
    entity grain, and saying so is what makes "a dimension without a graded metric is not
    lit" checkable rather than a slogan.
    """

    id: str
    title: str
    unit: str
    kind: str
    grains: tuple[str, ...]
    definition: str
    scope: str
    families: tuple[str, ...]
    variants: tuple[Variant, ...] = ()
    basis: str = "derived"
    breakdown: str | None = None

    def __post_init__(self) -> None:
        if self.unit not in _UNITS:
            raise ValueError(f"{self.id}: unknown unit {self.unit!r}")
        if self.kind not in _WINDOW_RULE:
            raise ValueError(f"{self.id}: unknown kind {self.kind!r}")
        unknown = set(self.grains) - _GRAINS
        if unknown or not self.grains:
            raise ValueError(f"{self.id}: bad grains {self.grains!r}")

    @property
    def window(self) -> str:
        return _WINDOW_RULE[self.kind]


# --- the finance registry ---------------------------------------------------
#
# Scope strings name accounts and columns rather than describing them in prose: a
# consumer has to be able to reproduce the filter, not agree with a summary of it.

_ENTITY_DERIVATION = (
    "Entity grain is derived from the order lines, not the GL — a GL posting cannot be "
    "attributed to a customer or a product group (the reason the operating chain exists)."
)

METRICS: tuple[Metric, ...] = (
    Metric(
        id="revenue",
        families=("core_ledger", "operating_chain"),
        title="Revenue",
        unit="currency",
        kind="flow",
        grains=("month", "year", "customer", "product_group"),
        definition="sum(journal_lines.credit) over accounts 4*, posted entries only",
        scope=(
            "All revenue accounts: 41xx product, 42xx service, 43xx other income. "
            f"{_ENTITY_DERIVATION} Entity values are sum(sales_order_lines.line_amount), "
            "so they sum to the `operating_revenue` variant and NOT to this metric — the "
            "gap is 43xx other income, which belongs to no customer."
        ),
        variants=(
            Variant(
                id="operating_revenue",
                title="Operating revenue (product + service)",
                definition="sum(journal_lines.credit) over accounts 41* and 42*",
                rationale=(
                    "Excludes 43xx other income (interest, FX). This is the figure the "
                    "order lines reconstruct to the cent, so it is the denominator a "
                    "consumer computing margins off the operating chain necessarily uses."
                ),
            ),
        ),
    ),
    Metric(
        id="cogs",
        families=("core_ledger", "operating_chain"),
        title="Cost of goods sold",
        unit="currency",
        kind="flow",
        grains=("month", "year", "customer", "product_group"),
        definition="sum(journal_lines.debit) over account 5100, posted entries only",
        scope=(
            "5100 carries cost of sale ONLY — vendor purchases post to their own expense "
            "accounts, which is what made purchases separable from COGS. "
            f"{_ENTITY_DERIVATION} Entity values are sum(sales_order_lines.line_cost), "
            "which ties to the GL account exactly: both sides are units x standard_cost."
        ),
    ),
    Metric(
        id="expenses",
        families=("core_ledger",),
        title="Total expenses",
        unit="currency",
        kind="flow",
        grains=("month", "year"),
        definition="sum(journal_lines.debit) over accounts 5*, posted entries only",
        scope="Every expense account, cost of sale included.",
        variants=(
            Variant(
                id="operating_expenses",
                title="Operating expenses (excluding cost of sale)",
                definition="expenses - cogs",
                rationale=(
                    "The opex-only base: what the firm spends to run, separated from what "
                    "it spends to deliver. Sized off the scale anchor (§7), so it moves "
                    "with the business rather than with a fixed invoice count."
                ),
            ),
        ),
    ),
    Metric(
        id="gross_profit",
        families=("core_ledger",),
        title="Gross profit",
        unit="currency",
        kind="flow",
        grains=("month", "year"),
        definition="revenue - cogs",
        scope=(
            "The textbook gross profit. Until the contract landed this id carried "
            "revenue - TOTAL expenses, which is operating income — the mislabelling the "
            "pinned definition exists to prevent."
        ),
        variants=(
            Variant(
                id="gross_profit_on_operating_revenue",
                title="Gross profit on operating revenue",
                definition="operating_revenue - cogs",
                rationale=(
                    "Drops 43xx other income from the top line. A consumer deriving gross "
                    "profit from the operating chain lands here, because the order lines "
                    "carry no interest income."
                ),
            ),
        ),
    ),
    Metric(
        id="operating_income",
        families=("core_ledger",),
        title="Operating income",
        unit="currency",
        kind="flow",
        grains=("month", "year"),
        definition="revenue - expenses",
        scope="All revenue accounts less all expense accounts. No tax or interest split.",
    ),
    Metric(
        id="gross_margin",
        families=("core_ledger",),
        title="Gross margin",
        unit="percent",
        kind="ratio",
        grains=("month", "year"),
        definition="gross_profit[w] / revenue[w] * 100",
        scope="Recomputed on the window's own aggregates — never an average of monthly margins.",
    ),
    Metric(
        id="operating_margin",
        families=("core_ledger",),
        title="Operating margin",
        unit="percent",
        kind="ratio",
        grains=("month", "year"),
        definition="operating_income[w] / revenue[w] * 100",
        scope="Recomputed on the window's own aggregates — never an average of monthly margins.",
    ),
    Metric(
        id="purchases",
        families=("core_ledger",),
        title="Purchases",
        unit="currency",
        kind="flow",
        grains=("month", "year"),
        definition="sum(invoices.amount) where status != cancelled, by invoice date",
        scope=(
            "Vendor-bill credits to AP, read off the invoice documents rather than the GL. "
            "A cancelled bill never posted, so it never created a payable and is not a "
            "purchase. Only computable since goods bills became separable from expense "
            "bills — which is why the pinned DPO could not use it before."
        ),
    ),
    Metric(
        id="ar_balance",
        families=("core_ledger",),
        title="Accounts receivable",
        unit="currency",
        kind="stock",
        grains=("month", "year"),
        definition="cumulative sum(debit - credit) over accounts 1210, 1220",
        scope="Carried forward from the first period of the corpus; there is no opening balance.",
    ),
    Metric(
        id="ap_balance",
        families=("core_ledger",),
        title="Accounts payable",
        unit="currency",
        kind="stock",
        grains=("month", "year"),
        definition="cumulative sum(credit - debit) over accounts 2110, 2120",
        scope="Carried forward from the first period of the corpus; there is no opening balance.",
    ),
    Metric(
        id="cash_balance",
        families=("core_ledger",),
        title="Cash",
        unit="currency",
        kind="stock",
        grains=("month", "year"),
        definition="cumulative sum(debit - credit) over accounts 1110, 1120",
        scope="Carried forward from the first period of the corpus; there is no opening balance.",
    ),
    Metric(
        id="inventory_balance",
        families=("core_ledger", "inventory"),
        title="Inventory",
        unit="currency",
        kind="stock",
        grains=("month", "year"),
        definition="cumulative sum(debit - credit) over account 1400",
        scope=(
            "Ties exactly to sum(inventory_positions.value) per period — both sides are "
            "valued at standard cost, so the tie is exact rather than approximate."
        ),
    ),
    Metric(
        id="dso",
        families=("core_ledger",),
        title="Days sales outstanding",
        unit="days",
        kind="ratio",
        grains=("month", "year"),
        definition="ar_balance[end of w] / revenue[w] * days[w]",
        scope="Closing receivable over the window's revenue; 0.0 when revenue is 0.",
    ),
    Metric(
        id="dpo",
        families=("core_ledger",),
        title="Days payable outstanding",
        unit="days",
        kind="ratio",
        grains=("month", "year"),
        definition="ap_balance[end of w] / purchases[w] * days[w]",
        scope="Closing payable over the window's purchases; 0.0 when purchases are 0.",
        variants=(
            Variant(
                id="dpo_on_total_expenses",
                title="DPO on total expenses",
                definition="ap_balance[end of w] / expenses[w] * days[w]",
                rationale=(
                    "The expense-denominator family — what a consumer without a separable "
                    "purchases figure necessarily computes. Both are correct answers to "
                    "different questions; this one runs high because payroll and rent sit "
                    "in the denominator without ever passing through AP."
                ),
            ),
        ),
    ),
    Metric(
        id="dio",
        families=("core_ledger", "inventory"),
        title="Days inventory outstanding",
        unit="days",
        kind="ratio",
        grains=("month", "year"),
        definition="inventory_balance[end of w] / cogs[w] * days[w]",
        scope="Closing stock over the window's cost of sale; 0.0 when COGS is 0.",
    ),
    Metric(
        id="cash_conversion_cycle",
        families=("core_ledger", "inventory"),
        title="Cash conversion cycle",
        unit="days",
        kind="ratio",
        grains=("month", "year"),
        definition="dio + dso - dpo",
        scope=(
            "Composed from the ROUNDED components, not the raw ones: a consumer that "
            "recombines the published DIO, DSO and DPO must land on the published CCC, or "
            "the oracle is grading its own rounding error."
        ),
        variants=(
            Variant(
                id="cash_conversion_cycle_on_expense_dpo",
                title="CCC on the expense-denominator DPO",
                definition="dio + dso - dpo_on_total_expenses",
                rationale=(
                    "The cycle a consumer lands on when it computes DPO the other way. "
                    "Published so the difference is one named substitution rather than an "
                    "unexplained gap in a headline number."
                ),
            ),
        ),
    ),
    Metric(
        id="revenue_growth_pct",
        families=("core_ledger",),
        title="Revenue growth, month over month",
        unit="percent",
        kind="ratio",
        grains=("month",),
        definition="(revenue[m] - revenue[m-1]) / revenue[m-1] * 100",
        scope=(
            "Null for the first period — there is no prior month inside the corpus, and a "
            "0.0 there would read as flat growth rather than as no basis."
        ),
    ),
    Metric(
        id="free_cash_flow",
        families=("core_ledger",),
        title="Free cash flow",
        unit="currency",
        kind="flow",
        grains=("month", "year"),
        definition="sum(bank_transactions.amount) in w",
        scope=(
            "Net bank movement: inflows positive, outflows negative. Negative across year "
            "one at every scale profile, and that is the corpus being honest — a firm "
            "starting from zero funds a full receivable and a full inventory position out "
            "of the same year's collections (the cold start, §7)."
        ),
    ),
    Metric(
        id="db1",
        families=("operating_chain",),
        title="Contribution margin (DB1)",
        unit="currency",
        kind="flow",
        grains=("customer", "product_group"),
        definition="sum(line_amount) - sum(line_cost) over the entity's order lines",
        scope=(
            "Exact, not estimated: both sides are constructed on the line "
            "(units x unit_price, units x standard_cost), so this is an answer key. "
            f"{_ENTITY_DERIVATION}"
        ),
    ),
    Metric(
        id="db1_pct",
        families=("operating_chain",),
        title="Contribution margin ratio",
        unit="percent",
        kind="ratio",
        grains=("customer", "product_group"),
        definition="db1[entity] / revenue[entity] * 100",
        scope=(
            "The comparable form — a grader matching an engine against the absolute DB1 "
            "is also grading entity size, which the scale profile sets."
        ),
    ),
    Metric(
        id="units_sold",
        families=("operating_chain",),
        title="Units sold",
        unit="count",
        kind="count",
        grains=("customer", "product_group"),
        definition="sum(sales_order_lines.units) over the entity's order lines",
        scope=_ENTITY_DERIVATION,
    ),
    Metric(
        id="order_count",
        families=("operating_chain",),
        title="Orders",
        unit="count",
        kind="count",
        grains=("customer", "product_group"),
        definition="count(distinct order_id) over the entity's order lines",
        scope=(
            "Distinct at order grain, so a product group appearing on three lines of one "
            "order counts once. " + _ENTITY_DERIVATION
        ),
    ),
    Metric(
        id="invoice_count",
        families=("core_ledger",),
        title="Invoices by status",
        unit="count",
        kind="count",
        grains=("month", "year"),
        definition="count(invoices) in w, grouped by status",
        scope="Every vendor invoice dated in the window, cancelled ones included.",
        breakdown="status",
    ),
    Metric(
        id="payment_count",
        families=("core_ledger",),
        title="Payments by method",
        unit="count",
        kind="count",
        grains=("month", "year"),
        definition="count(payments) in w, grouped by method",
        scope="Every payment dated in the window.",
        breakdown="method",
    ),
)

@dataclass(frozen=True)
class Invariant:
    """A structural property of the corpus, declared by the family that guarantees it.

    Not a metric: an invariant has no unit and no grain, it either holds or it does not
    — except for the reconciliation rate, which is a *measured* rate published as a
    figure. §5 is explicit that the rate is an authored expectation and never assumed to
    be 1.0: a consumer reporting a perfect rate has overcleaned, and that is a failure,
    so no band is asserted here and the judgement stays the consumer's.

    ``statement`` is what must hold, in reproducible terms — a consumer has to be able
    to re-check it, not agree with a summary of it.
    """

    id: str
    family: str
    statement: str
    kind: str = "holds"  # holds | rate


INVARIANTS: tuple[Invariant, ...] = (
    Invariant(
        id="journal_balanced",
        family="core_ledger",
        statement="For every journal entry, sum(debit) == sum(credit).",
    ),
    Invariant(
        id="trial_balance_balanced",
        family="core_ledger",
        statement="For every period, sum(debit_balance) == sum(credit_balance) across the trial balance.",
    ),
    Invariant(
        id="invoice_payment_matched",
        family="core_ledger",
        statement="Every invoice with status=paid has a payment whose amount equals the invoice amount.",
    ),
    Invariant(
        id="bank_reconciliation_rate",
        family="core_ledger",
        kind="rate",
        statement=(
            "Share of bank_transactions with reconciled=true. An AUTHORED expectation, "
            "never 1.0 — a consumer reporting a perfect rate has overcleaned."
        ),
    ),
    Invariant(
        id="inventory_rollforward_holds",
        family="inventory",
        statement=(
            "For every (product, location, period): closing[p-1] + sum(movement units in p) == closing[p]. "
            "Per key, not merely in aggregate — this is what separates a stock table from a table of "
            "numbers that look like stock."
        ),
    ),
    Invariant(
        id="inventory_ties_to_gl",
        family="inventory",
        statement=(
            "For every period, sum(inventory_positions.value) equals the cumulative balance of GL 1400. "
            "Both sides are valued at standard cost, so the tie is exact; a tolerance here would be "
            "hiding something."
        ),
    ),
)

INVARIANTS_BY_ID: dict[str, Invariant] = {i.id: i for i in INVARIANTS}
METRICS_BY_ID: dict[str, Metric] = {m.id: m for m in METRICS}
METRIC_IDS: frozenset[str] = frozenset(METRICS_BY_ID)
VARIANT_IDS: frozenset[str] = frozenset(v.id for m in METRICS for v in m.variants)

# Named surfaces that ``estimate_injection_impact`` reports against which are NOT
# metrics: integrity properties of the corpus rather than figures with a value. They are
# declared so the impact table cannot quietly grow a target that looks like a metric id
# and has no definition behind it (``tests/test_oracle.py`` holds the line).
INTEGRITY_SURFACES: frozenset[str] = frozenset(
    {
        "invoice_totals",
        "invoice_gl_consistency",
        "payment_bank_consistency",
        "referential_integrity",
        "trial_balance_accuracy",
        "benford_compliance",
        "temporal_stability",
    }
)


# --- binding values to definitions ------------------------------------------

# A value map is {metric or variant id: {grain: {key: value}}}.
ValueMap = Mapping[str, Mapping[str, Mapping[str, Any]]]


def build_contract(values: ValueMap, variant_values: ValueMap) -> list[dict[str, Any]]:
    """Join the registry to computed values, refusing anything unpaired.

    A metric without values would publish a definition nothing was measured against; a
    value without a metric would publish a number with no definition — the exact state
    the contract replaces. Both raise rather than being dropped silently, because a
    truth file that quietly shrinks reads as a corpus that got easier.

    Grains are checked too: a metric declares the grains it is published at, and a value
    map that covers a different set means one of the two moved without the other.
    """
    unknown = set(values) - METRIC_IDS
    if unknown:
        raise ValueError(f"values for unknown metrics: {sorted(unknown)}")
    unknown_variants = set(variant_values) - VARIANT_IDS
    if unknown_variants:
        raise ValueError(f"values for unknown variants: {sorted(unknown_variants)}")

    out: list[dict[str, Any]] = []
    for metric in METRICS:
        if metric.id not in values:
            raise ValueError(f"metric {metric.id!r} has a definition but no values")
        got = set(values[metric.id])
        if got != set(metric.grains):
            raise ValueError(
                f"metric {metric.id!r} declares grains {sorted(metric.grains)}, values carry {sorted(got)}"
            )

        entry: dict[str, Any] = {
            "id": metric.id,
            "title": metric.title,
            "unit": metric.unit,
            "kind": metric.kind,
            "window": metric.window,
            "basis": metric.basis,
            "grains": list(metric.grains),
            "definition": metric.definition,
            "scope": metric.scope,
        }
        if metric.breakdown:
            entry["breakdown"] = metric.breakdown
        entry["values"] = {grain: dict(values[metric.id][grain]) for grain in metric.grains}

        variants: list[dict[str, Any]] = []
        for variant in metric.variants:
            if variant.id not in variant_values:
                raise ValueError(f"variant {variant.id!r} of {metric.id!r} has a definition but no values")
            vgrains = set(variant_values[variant.id])
            if not vgrains or not vgrains <= set(metric.grains):
                raise ValueError(
                    f"variant {variant.id!r} carries grains {sorted(vgrains)}, "
                    f"which is not a non-empty subset of {sorted(metric.grains)}"
                )
            variants.append(
                {
                    "id": variant.id,
                    "title": variant.title,
                    "definition": variant.definition,
                    "rationale": variant.rationale,
                    "values": {g: dict(variant_values[variant.id][g]) for g in metric.grains if g in vgrains},
                }
            )
        if variants:
            entry["variants"] = variants
        out.append(entry)
    return out


def build_invariants(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Join the invariant declarations to their computed results, refusing anything unpaired.

    Same contract as :func:`build_contract`: a declaration with no result would publish a
    property nothing checked, and a result with no declaration would publish a verdict
    with no statement of what it means.
    """
    unknown = set(results) - set(INVARIANTS_BY_ID)
    if unknown:
        raise ValueError(f"results for undeclared invariants: {sorted(unknown)}")

    out: list[dict[str, Any]] = []
    for invariant in INVARIANTS:
        if invariant.id not in results:
            raise ValueError(f"invariant {invariant.id!r} is declared but never checked")
        entry: dict[str, Any] = {
            "id": invariant.id,
            "family": invariant.family,
            "kind": invariant.kind,
            "statement": invariant.statement,
        }
        if invariant.kind == "rate":
            entry["value"] = results[invariant.id]
        else:
            entry["holds"] = bool(results[invariant.id])
        out.append(entry)
    return out
