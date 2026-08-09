"""Ground truth calculator for deterministic finance datasets.

Computes known-correct financial metrics from clean FinanceDataset Pydantic
models. Outputs a GroundTruth object that can be serialized to ground_truth.yaml
for use by downstream evaluators and test assertions.
"""

from __future__ import annotations

import calendar
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from testdata.canonical.finance.models import (
    FinanceDataset,
    InvoiceStatus,
    JournalStatus,
)
from testdata.identity import CorpusIdentity

# Account range prefixes for metric classification
_REVENUE_PREFIX = "4"
_EXPENSE_PREFIX = "5"
_AR_ACCOUNTS = {"1210", "1220"}
_AP_ACCOUNTS = {"2110", "2120"}
_CASH_ACCOUNTS = {"1110", "1120"}
_INVENTORY_ACCOUNT = "1400"
_COGS_ACCOUNT = "5100"


def _q(d: Decimal) -> Decimal:
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# --- Models ---


class PeriodMetrics(BaseModel):
    """Financial metrics for a single month.

    ``dpo`` divides the payable by **purchases** — the vendor-bill credits to AP —
    which is the textbook definition and became computable only once goods bills
    existed. ``dpo_on_expenses`` carries the older total-expense denominator as a
    named alternative rather than an unexplained delta: it is what a consumer without
    a separable purchases figure necessarily computes, and both are correct answers
    to different questions. ``cash_conversion_cycle`` uses the purchases one.
    """

    period: str
    revenue: Decimal
    expenses: Decimal
    gross_profit: Decimal
    cogs: Decimal
    purchases: Decimal
    ar_balance: Decimal
    ap_balance: Decimal
    cash_balance: Decimal
    inventory_balance: Decimal
    dso: float
    dpo: float
    dpo_on_expenses: float
    dio: float
    cash_conversion_cycle: float
    revenue_growth_pct: float | None = None
    invoice_count: dict[str, int]
    payment_count: dict[str, int]


class AnnualMetrics(BaseModel):
    """Aggregate annual financial metrics. See :class:`PeriodMetrics` on the two DPOs."""

    total_revenue: Decimal
    total_expenses: Decimal
    gross_profit: Decimal
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
    """Estimated impact of an injection on a financial metric."""

    metric: str
    expected_error_pct: float
    affected_by: list[str]


class GroundTruth(BaseModel):
    """Complete ground truth for a generated finance dataset."""

    generator: str = "finance"
    seed: int
    strategy: str
    fiscal_year_start: str
    months: int
    annual: AnnualMetrics
    monthly: list[PeriodMetrics]
    invariants: Invariants
    injection_impact: list[InjectionImpact] = []
    # — the operating-model answer key. Empty on corpora generated before
    # the chain existed; a grader must treat absence as "not gradeable", never as 0.
    db1_by_customer: list[ContributionMargin] = []
    db1_by_product_group: list[ContributionMargin] = []


# --- Calculation ---


def calculate_ground_truth(
    dataset: FinanceDataset,
    *,
    seed: int,
    strategy: str = "clean",
    fiscal_start: date | None = None,
    months: int = 12,
) -> GroundTruth:
    """Compute all ground truth metrics from a clean FinanceDataset.

    Args:
        dataset: The finance dataset (ideally clean, pre-injection).
        seed: The random seed used for generation.
        strategy: Strategy name (for labeling).
        fiscal_start: First day of fiscal year.
        months: Number of months in the dataset.

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

    monthly_metrics: list[PeriodMetrics] = []

    for period_str in periods:
        year_int = int(period_str[:4])
        month_int = int(period_str[5:7])
        days_in_period = calendar.monthrange(year_int, month_int)[1]

        # Revenue: credits to revenue accounts
        revenue = Decimal("0")
        for acct_id in _all_accounts_with_prefix(period_account_credits, period_str, _REVENUE_PREFIX):
            revenue += period_account_credits.get((period_str, acct_id), Decimal("0"))

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
                expenses=_q(expenses),
                gross_profit=_q(revenue - expenses),
                cogs=_q(cogs),
                purchases=_q(purchases),
                ar_balance=_q(cumulative_ar),
                ap_balance=_q(cumulative_ap),
                cash_balance=_q(cumulative_cash),
                inventory_balance=_q(cumulative_inventory),
                dso=round(dso, 1),
                dpo=round(dpo, 1),
                dpo_on_expenses=round(dpo_expenses, 1),
                dio=round(dio, 1),
                # Composed from the ROUNDED components, not the raw ones. An answer
                # key has to be self-consistent: a consumer that recombines the
                # published DIO, DSO and DPO must land on the published CCC, or the
                # oracle is grading its own rounding error.
                cash_conversion_cycle=round(round(dio, 1) + round(dso, 1) - round(dpo, 1), 1),
                revenue_growth_pct=growth,
                invoice_count=invoice_by_period.get(period_str, {}),
                payment_count=payment_by_period.get(period_str, {}),
            )
        )

        prev_revenue = revenue

    # --- Annual metrics ---
    total_revenue = sum((m.revenue for m in monthly_metrics), Decimal("0"))
    total_expenses = sum((m.expenses for m in monthly_metrics), Decimal("0"))
    last = monthly_metrics[-1] if monthly_metrics else None

    # FCF: sum of all bank transaction amounts (positive = inflow, negative = outflow)
    fcf = Decimal("0")
    for bt in dataset.bank_transactions:
        bt_period = bt.date.strftime("%Y-%m")
        if bt_period in periods:
            fcf += bt.amount

    total_cogs = sum((m.cogs for m in monthly_metrics), Decimal("0"))
    total_purchases = sum((m.purchases for m in monthly_metrics), Decimal("0"))

    total_days = sum(calendar.monthrange(int(p[:4]), int(p[5:7]))[1] for p in periods)
    annual_dso = float(last.ar_balance / total_revenue * total_days) if last and total_revenue > 0 else 0.0
    annual_dpo = float(last.ap_balance / total_purchases * total_days) if last and total_purchases > 0 else 0.0
    annual_dpo_exp = float(last.ap_balance / total_expenses * total_days) if last and total_expenses > 0 else 0.0
    annual_dio = float(last.inventory_balance / total_cogs * total_days) if last and total_cogs > 0 else 0.0

    annual = AnnualMetrics(
        total_revenue=_q(total_revenue),
        total_expenses=_q(total_expenses),
        gross_profit=_q(total_revenue - total_expenses),
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
        free_cash_flow=_q(fcf),
    )

    # --- Invariants ---
    invariants = _check_invariants(dataset, entry_info)

    return GroundTruth(
        seed=seed,
        strategy=strategy,
        fiscal_year_start=fiscal_start.isoformat(),
        months=months,
        annual=annual,
        monthly=monthly_metrics,
        invariants=invariants,
        db1_by_customer=_contribution_margin(dataset, by="customer"),
        db1_by_product_group=_contribution_margin(dataset, by="product_group"),
    )


def _contribution_margin(
    dataset: FinanceDataset, *, by: str
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
    dataset: FinanceDataset,
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
    dataset: FinanceDataset,
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


# --- Export ---


def export_ground_truth(truth: GroundTruth, output_dir: Path, identity: CorpusIdentity | None = None) -> None:
    """Write ground_truth.yaml to the output directory.

    ``identity`` stamps the corpus these numbers were computed from. Without it a
    consumer holding a stale directory grades against the wrong answer key and has no
    way to notice — which is exactly what happened when the inventory family landed
    and every previously-generated corpus silently stopped matching its own seed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _to_yaml_dict(truth.model_dump())
    if identity is not None:
        data = {"corpus": identity.as_dict(), **data}
    with open(output_dir / "ground_truth.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


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
