"""Ground truth calculator for deterministic finance datasets.

Computes known-correct financial metrics from clean Corpus Pydantic
models. Outputs a GroundTruth object that can be serialized to ground_truth.yaml
for use by downstream evaluators and test assertions.
"""

from __future__ import annotations

import calendar
from collections import Counter
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from testdata.canonical.finance.models import (
    Corpus,
    InvoiceStatus,
    JournalStatus,
)
from testdata.identity import CorpusIdentity
from testdata.oracle import INTEGRITY_SURFACES, METRIC_IDS, build_contract, build_invariants

# Account range prefixes for metric classification
_REVENUE_PREFIX = "4"
# Product (41xx) and service (42xx) revenue — the operating top line, excluding 43xx
# other income. This is the split the order lines can reconstruct: every sales posting
# lands in 41xx/42xx, and interest income belongs to no customer.
_OPERATING_REVENUE_PREFIXES = ("41", "42")
_EXPENSE_PREFIX = "5"
_AR_ACCOUNTS = {"1210", "1220"}
_AP_ACCOUNTS = {"2110", "2120"}
_CASH_ACCOUNTS = {"1110", "1120"}
_INVENTORY_ACCOUNT = "1400"
_COGS_ACCOUNT = "5100"


def _q(d: Decimal) -> Decimal:
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _ratio_pct(numerator: Decimal, denominator: Decimal) -> float:
    """A margin in percent, 0.0 on a zero base — never a division by nothing."""
    return round(float(numerator / denominator * 100), 2) if denominator else 0.0


# --- Models ---


class PeriodMetrics(BaseModel):
    """Financial metrics for a single month.

    The definitions these fields carry are published beside them — see
    :mod:`testdata.oracle`, which pins each one and names its legitimate variants.
    Every ``*_on_expenses`` / ``operating_*`` field here is the value behind one of
    those variants, computed once so the two files cannot disagree.

    ``dpo`` divides the payable by **purchases** — the vendor-bill credits to AP —
    which is the textbook definition and became computable only once goods bills
    existed. ``dpo_on_expenses`` carries the older total-expense denominator as a
    named alternative rather than an unexplained delta: it is what a consumer without
    a separable purchases figure necessarily computes, and both are correct answers
    to different questions. ``cash_conversion_cycle`` uses the purchases one.

    ``gross_profit`` is ``revenue - cogs``. It used to be ``revenue - total expenses``,
    which is operating income — a mislabelling that survived precisely because no
    definition was published next to the number.
    """

    period: str
    revenue: Decimal
    operating_revenue: Decimal
    expenses: Decimal
    operating_expenses: Decimal
    gross_profit: Decimal
    gross_profit_on_operating_revenue: Decimal
    operating_income: Decimal
    gross_margin: float
    operating_margin: float
    cogs: Decimal
    purchases: Decimal
    ar_balance: Decimal
    ap_balance: Decimal
    cash_balance: Decimal
    inventory_balance: Decimal
    free_cash_flow: Decimal
    dso: float
    dpo: float
    dpo_on_expenses: float
    dio: float
    cash_conversion_cycle: float
    cash_conversion_cycle_on_expenses: float
    revenue_growth_pct: float | None = None
    invoice_count: dict[str, int]
    payment_count: dict[str, int]


class AnnualMetrics(BaseModel):
    """Aggregate annual financial metrics. See :class:`PeriodMetrics` on the two DPOs."""

    total_revenue: Decimal
    total_operating_revenue: Decimal
    total_expenses: Decimal
    total_operating_expenses: Decimal
    gross_profit: Decimal
    gross_profit_on_operating_revenue: Decimal
    operating_income: Decimal
    gross_margin: float
    operating_margin: float
    total_cogs: Decimal
    total_purchases: Decimal
    ending_ar_balance: Decimal
    ending_ap_balance: Decimal
    ending_cash_balance: Decimal
    ending_inventory_balance: Decimal
    annual_dso: float
    annual_dpo: float
    annual_dpo_on_expenses: float
    annual_dio: float
    annual_cash_conversion_cycle: float
    annual_cash_conversion_cycle_on_expenses: float
    free_cash_flow: Decimal


class ContributionMargin(BaseModel):
    """True DB1 for one entity — revenue minus cost of sale.

    The first **per-entity unit metric with truth** this corpus can carry, and the
    reason the operating chain exists. Both sides are derived from the order lines
    by construction (``units × unit_price`` and ``units × standard_cost``), so this
    is exact, not estimated — an answer key, not a plausible number.

    ``db1_pct`` is the contribution margin ratio; a grader comparing an engine
    metric should use it rather than the absolute figure when entity sizes differ.
    """

    entity: str  # customer_id or product_group
    revenue: Decimal
    cost_of_sale: Decimal
    db1: Decimal
    db1_pct: float
    units: int
    orders: int


class Invariants(BaseModel):
    """Structural integrity checks on the dataset.

    The two inventory ones are the stock subledger's contract: the roll-forward
    ``opening + Σ movements = closing`` per product, location and period, and the tie
    from Σ position value to the GL inventory account. A generator that produces a
    stock table without both is producing a plausible table, not a gradeable one.
    """

    journal_balanced: bool
    trial_balance_balanced: bool
    invoice_payment_matched: bool
    bank_reconciliation_rate: float
    inventory_rollforward_holds: bool = True
    inventory_ties_to_gl: bool = True


class InjectionImpact(BaseModel):
    """Estimated impact of an injection on a metric or an integrity surface.

    ``metric`` is either a metric id from :mod:`testdata.oracle` — in which case the
    reader can look up the definition the estimate is against — or one of that module's
    declared ``INTEGRITY_SURFACES``, which are properties of the corpus rather than
    figures with a value. Nothing else may appear: a target that looks like a metric id
    and has no definition behind it is the shape this contract exists to remove.
    """

    metric: str
    expected_error_pct: float
    affected_by: list[str]


class DimensionProfile(BaseModel):
    """What is true of a high-cardinality dimension's frequency distribution.

    A corpus whose dimensions are all small and near-uniform lets a consumer answer
    "the top 10 merchants" by reading the table. Once the axis has thousands of
    members drawn from a power law, that question has a real answer that is only
    obtainable by aggregating — and this is that answer, so the claim can be graded
    rather than believed.

    ``head`` is the exact top-20 with their counts, because a top-N claim is what
    consumers actually make. ``coverage`` is the cumulative share at several N, which
    is what distinguishes a genuine power law from a merely large dimension:
    ``distinct_observed`` well below ``pool_size`` with a large ``seen_once`` is the
    signature — a long tail the sample never fully reaches.
    """

    table: str
    column: str
    law: str
    exponent: float
    pool_size: int
    rows: int
    distinct_observed: int
    seen_once: int
    coverage: dict[str, float]
    head: list[dict[str, Any]]


def dimension_profile(
    rows: list[str | None], *, table: str, column: str, pool_size: int, exponent: float
) -> DimensionProfile:
    """Measure a dimension's realised distribution — from the data, never from the spec.

    The exponent is what was ASKED for; every other number here is what the corpus
    actually holds. A sample of n rows from a pool of N never reaches every member, so
    publishing the pool size as the cardinality would put a figure on disk that no
    query can reproduce.
    """
    counts = Counter(value for value in rows if value is not None)
    total = sum(counts.values())
    ranked = counts.most_common()
    coverage = {
        str(n): round(sum(count for _, count in ranked[:n]) / total, 6)
        for n in (1, 5, 10, 50, 100, 500, 1000)
        if n <= len(ranked)
    }
    return DimensionProfile(
        table=table,
        column=column,
        law="zipf",
        exponent=exponent,
        pool_size=pool_size,
        rows=total,
        distinct_observed=len(counts),
        seen_once=sum(1 for count in counts.values() if count == 1),
        coverage=coverage,
        head=[
            {"value": value, "rows": count, "share": round(count / total, 6)}
            for value, count in ranked[:20]
        ],
    )


class GroundTruth(BaseModel):
    """Complete ground truth for a generated finance dataset.

    Metrics only. *Which* corpus these are true of is ``CorpusIdentity``'s job, and
    stating it twice is how the two answers start to disagree — the seed, strategy,
    months and fiscal start used to be restated here beside a ``generator: finance``
    that named the vertical rather than the generator.
    """

    annual: AnnualMetrics
    monthly: list[PeriodMetrics]
    invariants: Invariants
    injection_impact: list[InjectionImpact] = []
    # — the operating-model answer key. Empty on corpora generated before
    # the chain existed; a grader must treat absence as "not gradeable", never as 0.
    db1_by_customer: list[ContributionMargin] = []
    db1_by_product_group: list[ContributionMargin] = []
    # — the realised shape of any high-cardinality dimension the corpus carries.
    # Empty unless one was asked for; absence means the corpus has no such axis.
    dimensions: list[DimensionProfile] = []


# --- Calculation ---


def calculate_ground_truth(
    dataset: Corpus,
    *,
    fiscal_start: date | None = None,
    months: int = 12,
    merchant_exponent: float = 1.05,
) -> GroundTruth:
    """Compute all ground truth metrics from a clean Corpus.

    Takes only what it computes with. ``seed`` and ``strategy`` were arguments that
    were recorded and never read — provenance wearing the shape of a parameter.

    Args:
        dataset: The finance dataset (ideally clean, pre-injection).
        fiscal_start: First day of fiscal year.
        months: Number of months in the dataset.
        merchant_exponent: The Zipf exponent the payer dimension was asked for —
            recorded beside the realised distribution, which is measured from the
            data. Ignored when the corpus carries no such dimension.

    Returns:
        GroundTruth with annual metrics, monthly metrics, and invariants.
    """
    if fiscal_start is None:
        fiscal_start = date(2025, 1, 1)

    # Build entry lookup: entry_id -> (date, status)
    entry_info = {e.entry_id: (e.date, e.status) for e in dataset.journal_entries}

    # Build period list
    periods: list[str] = []
    for offset in range(months):
        year = fiscal_start.year + (fiscal_start.month + offset - 1) // 12
        month = (fiscal_start.month + offset - 1) % 12 + 1
        periods.append(f"{year:04d}-{month:02d}")

    # --- GL aggregation by period ---
    # Accumulate per-period movements by account
    period_account_debits: dict[tuple[str, str], Decimal] = {}
    period_account_credits: dict[tuple[str, str], Decimal] = {}

    for line in dataset.journal_lines:
        info = entry_info.get(line.entry_id)
        if info is None:
            continue
        entry_date, status = info
        if status != JournalStatus.POSTED:
            continue

        period_str = entry_date.strftime("%Y-%m")
        if period_str not in periods:
            continue  # Outside fiscal year

        key = (period_str, line.account_id)
        period_account_debits[key] = period_account_debits.get(key, Decimal("0")) + line.debit
        period_account_credits[key] = period_account_credits.get(key, Decimal("0")) + line.credit

    # --- Monthly metrics ---
    cumulative_ar = Decimal("0")
    cumulative_ap = Decimal("0")
    cumulative_cash = Decimal("0")
    cumulative_inventory = Decimal("0")
    prev_revenue: Decimal | None = None

    # Purchases per period: the vendor-bill credits to AP, read off the invoice
    # documents rather than the GL. A cancelled bill never posted, so it never
    # created a payable and is not a purchase.
    purchases_by_period: dict[str, Decimal] = {}
    for inv in dataset.invoices:
        p = inv.date.strftime("%Y-%m")
        if p not in periods or inv.status == InvoiceStatus.CANCELLED:
            continue
        purchases_by_period[p] = purchases_by_period.get(p, Decimal("0")) + inv.amount

    # Pre-compute invoice counts by period and status
    invoice_by_period: dict[str, dict[str, int]] = {}
    for inv in dataset.invoices:
        p = inv.date.strftime("%Y-%m")
        if p not in periods:
            continue
        counts = invoice_by_period.setdefault(p, {})
        counts[inv.status.value] = counts.get(inv.status.value, 0) + 1

    # Pre-compute payment counts by period and method
    payment_by_period: dict[str, dict[str, int]] = {}
    for pay in dataset.payments:
        p = pay.date.strftime("%Y-%m")
        if p not in periods:
            continue
        counts = payment_by_period.setdefault(p, {})
        counts[pay.method.value] = counts.get(pay.method.value, 0) + 1

    # Free cash flow per period: net bank movement, inflows positive.
    fcf_by_period: dict[str, Decimal] = {}
    for bt in dataset.bank_transactions:
        p = bt.date.strftime("%Y-%m")
        if p in periods:
            fcf_by_period[p] = fcf_by_period.get(p, Decimal("0")) + bt.amount

    monthly_metrics: list[PeriodMetrics] = []

    for period_str in periods:
        year_int = int(period_str[:4])
        month_int = int(period_str[5:7])
        days_in_period = calendar.monthrange(year_int, month_int)[1]

        # Revenue: credits to revenue accounts
        revenue = Decimal("0")
        for acct_id in _all_accounts_with_prefix(period_account_credits, period_str, _REVENUE_PREFIX):
            revenue += period_account_credits.get((period_str, acct_id), Decimal("0"))

        # Operating revenue: product + service only. The order lines reconstruct this
        # figure to the cent; they cannot reconstruct `revenue`, which carries 43xx
        # other income on top.
        operating_revenue = Decimal("0")
        for prefix in _OPERATING_REVENUE_PREFIXES:
            for acct_id in _all_accounts_with_prefix(period_account_credits, period_str, prefix):
                operating_revenue += period_account_credits.get((period_str, acct_id), Decimal("0"))

        # Expenses: debits to expense accounts
        expenses = Decimal("0")
        for acct_id in _all_accounts_with_prefix(period_account_debits, period_str, _EXPENSE_PREFIX):
            expenses += period_account_debits.get((period_str, acct_id), Decimal("0"))

        # Balance sheet: cumulative
        for acct in _AR_ACCOUNTS:
            cumulative_ar += period_account_debits.get((period_str, acct), Decimal("0"))
            cumulative_ar -= period_account_credits.get((period_str, acct), Decimal("0"))

        for acct in _AP_ACCOUNTS:
            cumulative_ap += period_account_credits.get((period_str, acct), Decimal("0"))
            cumulative_ap -= period_account_debits.get((period_str, acct), Decimal("0"))

        for acct in _CASH_ACCOUNTS:
            cumulative_cash += period_account_debits.get((period_str, acct), Decimal("0"))
            cumulative_cash -= period_account_credits.get((period_str, acct), Decimal("0"))

        cumulative_inventory += period_account_debits.get((period_str, _INVENTORY_ACCOUNT), Decimal("0"))
        cumulative_inventory -= period_account_credits.get((period_str, _INVENTORY_ACCOUNT), Decimal("0"))

        cogs = period_account_debits.get((period_str, _COGS_ACCOUNT), Decimal("0"))
        purchases = purchases_by_period.get(period_str, Decimal("0"))

        # DSO: (AR / Revenue) × days_in_period (avoid div by zero)
        dso = float(cumulative_ar / revenue * days_in_period) if revenue > 0 else 0.0

        # DPO: (AP / Purchases) × days_in_period — the pinned definition, plus the
        # total-expense denominator as the named alternative.
        dpo = float(cumulative_ap / purchases * days_in_period) if purchases > 0 else 0.0
        dpo_expenses = float(cumulative_ap / expenses * days_in_period) if expenses > 0 else 0.0

        # DIO: (Inventory / COGS) × days_in_period
        dio = float(cumulative_inventory / cogs * days_in_period) if cogs > 0 else 0.0

        # Revenue growth MoM
        growth: float | None = None
        if prev_revenue is not None and prev_revenue > 0:
            growth = round(float((revenue - prev_revenue) / prev_revenue * 100), 2)

        monthly_metrics.append(
            PeriodMetrics(
                period=period_str,
                revenue=_q(revenue),
                operating_revenue=_q(operating_revenue),
                expenses=_q(expenses),
                operating_expenses=_q(expenses - cogs),
                gross_profit=_q(revenue - cogs),
                gross_profit_on_operating_revenue=_q(operating_revenue - cogs),
                operating_income=_q(revenue - expenses),
                gross_margin=_ratio_pct(revenue - cogs, revenue),
                operating_margin=_ratio_pct(revenue - expenses, revenue),
                cogs=_q(cogs),
                purchases=_q(purchases),
                ar_balance=_q(cumulative_ar),
                ap_balance=_q(cumulative_ap),
                cash_balance=_q(cumulative_cash),
                inventory_balance=_q(cumulative_inventory),
                free_cash_flow=_q(fcf_by_period.get(period_str, Decimal("0"))),
                dso=round(dso, 1),
                dpo=round(dpo, 1),
                dpo_on_expenses=round(dpo_expenses, 1),
                dio=round(dio, 1),
                # Composed from the ROUNDED components, not the raw ones. An answer
                # key has to be self-consistent: a consumer that recombines the
                # published DIO, DSO and DPO must land on the published CCC, or the
                # oracle is grading its own rounding error.
                cash_conversion_cycle=round(round(dio, 1) + round(dso, 1) - round(dpo, 1), 1),
                cash_conversion_cycle_on_expenses=round(
                    round(dio, 1) + round(dso, 1) - round(dpo_expenses, 1), 1
                ),
                revenue_growth_pct=growth,
                invoice_count=invoice_by_period.get(period_str, {}),
                payment_count=payment_by_period.get(period_str, {}),
            )
        )

        prev_revenue = revenue

    # --- Annual metrics ---
    total_revenue = sum((m.revenue for m in monthly_metrics), Decimal("0"))
    total_operating_revenue = sum((m.operating_revenue for m in monthly_metrics), Decimal("0"))
    total_expenses = sum((m.expenses for m in monthly_metrics), Decimal("0"))
    last = monthly_metrics[-1] if monthly_metrics else None

    # FCF: sum of all bank transaction amounts (positive = inflow, negative = outflow)
    fcf = sum(fcf_by_period.values(), Decimal("0"))

    total_cogs = sum((m.cogs for m in monthly_metrics), Decimal("0"))
    total_purchases = sum((m.purchases for m in monthly_metrics), Decimal("0"))

    total_days = sum(calendar.monthrange(int(p[:4]), int(p[5:7]))[1] for p in periods)
    annual_dso = float(last.ar_balance / total_revenue * total_days) if last and total_revenue > 0 else 0.0
    annual_dpo = float(last.ap_balance / total_purchases * total_days) if last and total_purchases > 0 else 0.0
    annual_dpo_exp = float(last.ap_balance / total_expenses * total_days) if last and total_expenses > 0 else 0.0
    annual_dio = float(last.inventory_balance / total_cogs * total_days) if last and total_cogs > 0 else 0.0

    annual = AnnualMetrics(
        total_revenue=_q(total_revenue),
        total_operating_revenue=_q(total_operating_revenue),
        total_expenses=_q(total_expenses),
        total_operating_expenses=_q(total_expenses - total_cogs),
        gross_profit=_q(total_revenue - total_cogs),
        gross_profit_on_operating_revenue=_q(total_operating_revenue - total_cogs),
        operating_income=_q(total_revenue - total_expenses),
        gross_margin=_ratio_pct(total_revenue - total_cogs, total_revenue),
        operating_margin=_ratio_pct(total_revenue - total_expenses, total_revenue),
        total_cogs=_q(total_cogs),
        total_purchases=_q(total_purchases),
        ending_ar_balance=_q(last.ar_balance) if last else Decimal("0"),
        ending_ap_balance=_q(last.ap_balance) if last else Decimal("0"),
        ending_cash_balance=_q(last.cash_balance) if last else Decimal("0"),
        ending_inventory_balance=_q(last.inventory_balance) if last else Decimal("0"),
        annual_dso=round(annual_dso, 1),
        annual_dpo=round(annual_dpo, 1),
        annual_dpo_on_expenses=round(annual_dpo_exp, 1),
        annual_dio=round(annual_dio, 1),
        annual_cash_conversion_cycle=round(
            round(annual_dio, 1) + round(annual_dso, 1) - round(annual_dpo, 1), 1
        ),
        annual_cash_conversion_cycle_on_expenses=round(
            round(annual_dio, 1) + round(annual_dso, 1) - round(annual_dpo_exp, 1), 1
        ),
        free_cash_flow=_q(fcf),
    )

    # --- Invariants ---
    invariants = _check_invariants(dataset, entry_info)

    return GroundTruth(
        annual=annual,
        monthly=monthly_metrics,
        invariants=invariants,
        db1_by_customer=_contribution_margin(dataset, by="customer"),
        db1_by_product_group=_contribution_margin(dataset, by="product_group"),
        dimensions=_dimension_profiles(dataset, merchant_exponent),
    )


def _dimension_profiles(dataset: Corpus, merchant_exponent: float) -> list[DimensionProfile]:
    """The high-cardinality axes this corpus carries — none, on most corpora."""
    if not dataset.merchants:
        return []
    return [
        dimension_profile(
            [txn.merchant_id for txn in dataset.bank_transactions],
            table="bank_transactions",
            column="merchant_id",
            pool_size=len(dataset.merchants),
            exponent=merchant_exponent,
        )
    ]


def _contribution_margin(
    dataset: Corpus, *, by: str
) -> list[ContributionMargin]:
    """True DB1 per customer or per product group, from the order lines.

    Computed off the operating chain rather than the GL: the line carries both sides
    (``line_amount``, ``line_cost``) by construction, so this is an exact answer key.
    Deriving it from GL postings instead would re-introduce the very ambiguity the
    chain removed — account 5100 no longer mixes cost-of-sale with vendor purchases,
    but the GL still cannot attribute a posting to a customer or a product group.

    Returns [] when the corpus has no chain, so a grader sees "not gradeable" rather
    than a zero it might mistake for a measurement.
    """
    if not dataset.sales_order_lines:
        return []

    customer_of = {o.order_id: o.customer_id for o in dataset.sales_orders}
    group_of = {p.product_id: p.product_group for p in dataset.products}

    agg: dict[str, dict[str, Any]] = {}
    for line in dataset.sales_order_lines:
        if by == "customer":
            key = customer_of.get(line.order_id, "")
        else:
            key = group_of.get(line.product_id, "")
        if not key:
            continue
        bucket = agg.setdefault(
            key,
            {"revenue": Decimal("0"), "cost": Decimal("0"), "units": 0, "orders": set()},
        )
        bucket["revenue"] += line.line_amount
        bucket["cost"] += line.line_cost
        bucket["units"] += line.units
        bucket["orders"].add(line.order_id)

    out: list[ContributionMargin] = []
    for key in sorted(agg):
        b = agg[key]
        revenue, cost = _q(b["revenue"]), _q(b["cost"])
        db1 = _q(revenue - cost)
        out.append(
            ContributionMargin(
                entity=key,
                revenue=revenue,
                cost_of_sale=cost,
                db1=db1,
                db1_pct=round(float(db1 / revenue * 100), 2) if revenue else 0.0,
                units=int(b["units"]),
                orders=len(b["orders"]),
            )
        )
    return out


def _all_accounts_with_prefix(
    movements: dict[tuple[str, str], Decimal],
    period: str,
    prefix: str,
) -> set[str]:
    """Find all account IDs with a given prefix that have movements in a period."""
    return {acct for (p, acct) in movements if p == period and acct.startswith(prefix)}


def _check_invariants(
    dataset: Corpus,
    entry_info: dict[str, tuple[date, JournalStatus]],
) -> Invariants:
    """Check structural invariants on the dataset."""
    # 1. Every journal entry is balanced
    lines_by_entry: dict[str, tuple[Decimal, Decimal]] = {}
    for line in dataset.journal_lines:
        d, c = lines_by_entry.get(line.entry_id, (Decimal("0"), Decimal("0")))
        lines_by_entry[line.entry_id] = (d + line.debit, c + line.credit)

    journal_balanced = all(d == c for d, c in lines_by_entry.values())

    # 2. Trial balance balanced per period
    tb_by_period: dict[str, tuple[Decimal, Decimal]] = {}
    for tb in dataset.trial_balance:
        d, c = tb_by_period.get(tb.period, (Decimal("0"), Decimal("0")))
        tb_by_period[tb.period] = (d + tb.debit_balance, c + tb.credit_balance)

    tb_balanced = all(d == c for d, c in tb_by_period.values())

    # 3. Every PAID invoice has a payment with matching amount
    paid_invoices = {inv.invoice_id: inv.amount for inv in dataset.invoices if inv.status == InvoiceStatus.PAID}
    payment_by_invoice = {pay.invoice_id: pay.amount for pay in dataset.payments}
    invoice_matched = all(
        inv_id in payment_by_invoice and payment_by_invoice[inv_id] == amount
        for inv_id, amount in paid_invoices.items()
    )

    # 4. Bank reconciliation rate
    total_bank = len(dataset.bank_transactions)
    reconciled = sum(1 for bt in dataset.bank_transactions if bt.reconciled)
    recon_rate = reconciled / total_bank if total_bank > 0 else 1.0

    rollforward, ties_to_gl = _check_inventory(dataset, entry_info)

    return Invariants(
        journal_balanced=journal_balanced,
        trial_balance_balanced=tb_balanced,
        invoice_payment_matched=invoice_matched,
        bank_reconciliation_rate=round(recon_rate, 4),
        inventory_rollforward_holds=rollforward,
        inventory_ties_to_gl=ties_to_gl,
    )


def _check_inventory(
    dataset: Corpus,
    entry_info: dict[str, tuple[date, JournalStatus]],
) -> tuple[bool, bool]:
    """The stock subledger's two contracts. ``(True, True)`` when there is no stock.

    * **Roll-forward** — for every (product, location, period),
      ``closing[p-1] + Σ movement units in p == closing[p]``. This is the invariant
      that distinguishes a stock table from a table of numbers that happen to look
      like stock, and it holds per key, not just in aggregate.
    * **Ties to GL** — Σ position value in a period equals the cumulative balance of
      the inventory account. Both sides are valued at standard cost, so the tie is
      exact rather than approximate; a tolerance here would be hiding something.
    """
    if not dataset.inventory_positions:
        return True, True

    periods = sorted({p.period for p in dataset.inventory_positions})

    moved: dict[tuple[str, str, str], int] = {}
    for movement in dataset.stock_movements:
        key = (movement.product_id, movement.location_id, movement.date.strftime("%Y-%m"))
        moved[key] = moved.get(key, 0) + movement.units

    closing: dict[tuple[str, str, str], int] = {
        (p.product_id, p.location_id, p.period): p.units_on_hand for p in dataset.inventory_positions
    }
    rollforward = True
    for (product, location, period), units in closing.items():
        index = periods.index(period)
        opening = 0 if index == 0 else closing.get((product, location, periods[index - 1]), 0)
        if opening + moved.get((product, location, period), 0) != units:
            rollforward = False
            break

    value_by_period: dict[str, Decimal] = {}
    for position in dataset.inventory_positions:
        value_by_period[position.period] = value_by_period.get(position.period, Decimal("0")) + position.value

    gl_balance = Decimal("0")
    gl_by_period: dict[str, Decimal] = {}
    movement_by_period: dict[str, Decimal] = {}
    for line in dataset.journal_lines:
        info = entry_info.get(line.entry_id)
        if info is None or info[1] != JournalStatus.POSTED or line.account_id != _INVENTORY_ACCOUNT:
            continue
        period = info[0].strftime("%Y-%m")
        movement_by_period[period] = movement_by_period.get(period, Decimal("0")) + line.debit - line.credit
    for period in sorted(set(movement_by_period) | set(value_by_period)):
        gl_balance += movement_by_period.get(period, Decimal("0"))
        gl_by_period[period] = gl_balance

    ties_to_gl = all(value_by_period[p] == gl_by_period.get(p) for p in value_by_period)
    return rollforward, ties_to_gl


# --- The oracle contract ---

# (metric id, PeriodMetrics field, AnnualMetrics field). One row per period-grain
# metric, so a metric published without a definition is a missing row in ``oracle``
# rather than a silent divergence between the two files.
_PERIOD_METRICS: tuple[tuple[str, str, str], ...] = (
    ("revenue", "revenue", "total_revenue"),
    ("cogs", "cogs", "total_cogs"),
    ("expenses", "expenses", "total_expenses"),
    ("gross_profit", "gross_profit", "gross_profit"),
    ("operating_income", "operating_income", "operating_income"),
    ("gross_margin", "gross_margin", "gross_margin"),
    ("operating_margin", "operating_margin", "operating_margin"),
    ("purchases", "purchases", "total_purchases"),
    ("ar_balance", "ar_balance", "ending_ar_balance"),
    ("ap_balance", "ap_balance", "ending_ap_balance"),
    ("cash_balance", "cash_balance", "ending_cash_balance"),
    ("inventory_balance", "inventory_balance", "ending_inventory_balance"),
    ("free_cash_flow", "free_cash_flow", "free_cash_flow"),
    ("dso", "dso", "annual_dso"),
    ("dpo", "dpo", "annual_dpo"),
    ("dio", "dio", "annual_dio"),
    ("cash_conversion_cycle", "cash_conversion_cycle", "annual_cash_conversion_cycle"),
)

# The same, for the variant values. Each variant is computed once, here, and published
# under the metric it is an alternative to.
_PERIOD_VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("operating_revenue", "operating_revenue", "total_operating_revenue"),
    ("operating_expenses", "operating_expenses", "total_operating_expenses"),
    (
        "gross_profit_on_operating_revenue",
        "gross_profit_on_operating_revenue",
        "gross_profit_on_operating_revenue",
    ),
    ("dpo_on_total_expenses", "dpo_on_expenses", "annual_dpo_on_expenses"),
    (
        "cash_conversion_cycle_on_expense_dpo",
        "cash_conversion_cycle_on_expenses",
        "annual_cash_conversion_cycle_on_expenses",
    ),
)

# (metric id, ContributionMargin field) for the per-entity unit metrics. §5: a
# dimension without at least one graded per-entity metric is not lit.
_ENTITY_METRICS: tuple[tuple[str, str], ...] = (
    ("revenue", "revenue"),
    ("cogs", "cost_of_sale"),
    ("db1", "db1"),
    ("db1_pct", "db1_pct"),
    ("units_sold", "units"),
    ("order_count", "orders"),
)


def _year_label(periods: list[str]) -> str:
    """The key the ``year`` grain is published under.

    A 12-month corpus starting in January is just its year. Anything else says so
    explicitly rather than picking one of the years it spans and hoping.
    """
    if not periods:
        return "year"
    years = sorted({p[:4] for p in periods})
    return years[0] if len(years) == 1 else f"{years[0]}..{years[-1]}"


def _sum_counts(rows: list[dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for row in rows:
        for key, count in row.items():
            total[key] = total.get(key, 0) + count
    return dict(sorted(total.items()))


def invariant_contract(truth: GroundTruth) -> list[dict[str, Any]]:
    """The declared invariants with their computed verdicts, each naming its family."""
    checks = truth.invariants
    return build_invariants(
        {
            "journal_balanced": checks.journal_balanced,
            "trial_balance_balanced": checks.trial_balance_balanced,
            "invoice_payment_matched": checks.invoice_payment_matched,
            "bank_reconciliation_rate": checks.bank_reconciliation_rate,
            "inventory_rollforward_holds": checks.inventory_rollforward_holds,
            "inventory_ties_to_gl": checks.inventory_ties_to_gl,
        }
    )


def metric_contract(truth: GroundTruth) -> list[dict[str, Any]]:
    """The §5 contract: every metric with its pinned definition, variants and values.

    Derived from *truth* rather than stored on it. The computed metrics are the single
    source; this is the view a consumer grades against, and building it on demand is
    what keeps the definition and the number from drifting into two facts.
    """
    year = _year_label([m.period for m in truth.monthly])
    values: dict[str, dict[str, dict[str, Any]]] = {}
    variant_values: dict[str, dict[str, dict[str, Any]]] = {}

    for metric_id, month_field, annual_field in _PERIOD_METRICS:
        values[metric_id] = {
            "month": {m.period: getattr(m, month_field) for m in truth.monthly},
            "year": {year: getattr(truth.annual, annual_field)},
        }
    for variant_id, month_field, annual_field in _PERIOD_VARIANTS:
        variant_values[variant_id] = {
            "month": {m.period: getattr(m, month_field) for m in truth.monthly},
            "year": {year: getattr(truth.annual, annual_field)},
        }

    # Month grain only — the first period has no prior month inside the corpus.
    values["revenue_growth_pct"] = {"month": {m.period: m.revenue_growth_pct for m in truth.monthly}}

    for metric_id, field_name in (("invoice_count", "invoice_count"), ("payment_count", "payment_count")):
        per_month = {m.period: dict(getattr(m, field_name)) for m in truth.monthly}
        values[metric_id] = {
            "month": per_month,
            "year": {year: _sum_counts(list(per_month.values()))},
        }

    entities = {"customer": truth.db1_by_customer, "product_group": truth.db1_by_product_group}
    for metric_id, cm_field in _ENTITY_METRICS:
        entry = values.setdefault(metric_id, {})
        for grain, rows in entities.items():
            entry[grain] = {row.entity: getattr(row, cm_field) for row in rows}

    return build_contract(values, variant_values)


# --- Export ---


def export_ground_truth(truth: GroundTruth, output_dir: Path, identity: CorpusIdentity | None = None) -> None:
    """Write ground_truth.yaml to the output directory.

    What ships is the **contract**: every metric with the definition that produced it,
    its variants, and its values at every grain it is published at. The raw ``annual`` /
    ``monthly`` / ``db1_by_*`` blocks are gone from the file — each of their numbers now
    appears exactly once, under a definition. They are still on :class:`GroundTruth` for
    callers computing in-process; publishing them beside the contract would restate every
    figure without its definition, which is how ``gross_profit`` spent a release meaning
    operating income.

    Invariants ship the same way: each one names the family that guarantees it and
    states, reproducibly, what must hold — a bare ``journal_balanced: true`` said nothing
    a consumer could re-check.

    ``identity`` stamps the corpus these numbers were computed from. Without it a
    consumer holding a stale directory grades against the wrong answer key and has no
    way to notice — which is exactly what happened when the inventory family landed
    and every previously-generated corpus silently stopped matching its own seed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if identity is not None:
        data["corpus"] = identity.as_dict()
    data["metrics"] = metric_contract(truth)
    data["invariants"] = invariant_contract(truth)
    data["injection_impact"] = [impact.model_dump() for impact in truth.injection_impact]
    # Present only when the corpus HAS such an axis — an always-emitted empty list
    # would rewrite ground_truth.yaml for every corpus that never asked for one.
    if truth.dimensions:
        data["dimensions"] = [profile.model_dump() for profile in truth.dimensions]
    with open(output_dir / "ground_truth.yaml", "w") as f:
        yaml.dump(_to_yaml_dict(data), f, default_flow_style=False, sort_keys=False)


def _to_yaml_dict(obj: Any) -> Any:
    """Recursively convert Decimal to float for YAML serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_yaml_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_yaml_dict(v) for v in obj]
    return obj


# --- Injection impact estimation ---

# Maps injection_type → list of (metric, estimation_fn)
# estimation_fn(params) → approximate error percentage on that metric
_IMPACT_RULES: dict[str, list[tuple[str, str]]] = {
    # Value-layer injections
    "corrupt_type": [
        ("revenue", "ratio"),
        ("expenses", "ratio"),
    ],
    "introduce_nulls": [
        ("revenue", "ratio"),
        ("expenses", "ratio"),
    ],
    "inject_outliers": [
        ("revenue", "ratio_x_factor"),
        ("expenses", "ratio_x_factor"),
    ],
    # Semantic-layer injections
    "mix_units": [
        ("invoice_totals", "ratio_x_fx_delta"),
    ],
    # Structural-layer injections
    "break_gl_invoice_match": [
        ("invoice_gl_consistency", "ratio"),
    ],
    "break_payment_bank_match": [
        ("payment_bank_consistency", "ratio"),
    ],
    "break_referential_integrity": [
        ("referential_integrity", "ratio"),
    ],
    # Computational-layer injections
    "drift_formula": [
        ("trial_balance_accuracy", "error_ratio"),
    ],
    "break_trial_balance": [
        ("trial_balance_accuracy", "ratio"),
    ],
    # Distribution injections
    "break_benford": [
        ("benford_compliance", "fixed_60"),
    ],
    "inject_temporal_drift": [
        ("temporal_stability", "shift_factor"),
    ],
}


_UNDEFINED_IMPACT_TARGETS = {
    metric for rules in _IMPACT_RULES.values() for metric, _ in rules
} - METRIC_IDS - INTEGRITY_SURFACES
if _UNDEFINED_IMPACT_TARGETS:  # pragma: no cover — a wiring error, and loud at import
    raise ValueError(
        f"injection impact reports against undefined targets: {sorted(_UNDEFINED_IMPACT_TARGETS)}. "
        "Add a Metric to testdata.oracle, or declare it in INTEGRITY_SURFACES."
    )


def estimate_injection_impact(
    injections: list[dict[str, Any]],
) -> list[InjectionImpact]:
    """Estimate metric deviation from known injection parameters.

    Args:
        injections: List of injection dicts from registry.export_dicts().

    Returns:
        List of InjectionImpact estimates per affected metric.
    """
    # Accumulate impacts by metric
    metric_impacts: dict[str, tuple[float, list[str]]] = {}

    for inj in injections:
        inj_type = inj.get("injection_type", "")
        params = inj.get("parameters", {})
        target = f"{inj_type}:{inj.get('target_file', '')}"

        rules = _IMPACT_RULES.get(inj_type, [])
        for metric, method in rules:
            error_pct = _estimate_error(method, params)
            if error_pct <= 0:
                continue

            prev_error, prev_sources = metric_impacts.get(metric, (0.0, []))
            # Compound: independent errors combine
            combined = prev_error + error_pct - (prev_error * error_pct / 100.0)
            metric_impacts[metric] = (combined, prev_sources + [target])

    return [
        InjectionImpact(
            metric=metric,
            expected_error_pct=round(error, 2),
            affected_by=sources,
        )
        for metric, (error, sources) in sorted(metric_impacts.items())
    ]


def _estimate_error(method: str, params: dict[str, Any]) -> float:
    """Estimate error percentage from injection parameters."""
    if method == "ratio":
        return params.get("ratio", 0) * 100.0
    if method == "error_ratio":
        return params.get("error_ratio", 0) * 100.0
    if method == "ratio_x_factor":
        ratio = params.get("ratio", 0)
        factor = params.get("factor", 1)
        return ratio * (factor - 1) * 100.0
    if method == "ratio_x_fx_delta":
        ratio = params.get("ratio", 0)
        fx_rate = params.get("fx_rate", 1.0)
        return ratio * abs(fx_rate - 1.0) * 100.0
    if method == "fixed_60":
        return 60.0  # ~60% of values affected by break_benford
    if method == "shift_factor":
        factor = params.get("shift_factor", 1.0)
        return abs(factor - 1.0) * 50.0  # ~50% of data affected (post-cutoff)
    return 0.0
