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

# Account range prefixes for metric classification
_REVENUE_PREFIX = "4"
_EXPENSE_PREFIX = "5"
_AR_ACCOUNTS = {"1210", "1220"}
_AP_ACCOUNTS = {"2110", "2120"}
_CASH_ACCOUNTS = {"1110", "1120"}


def _q(d: Decimal) -> Decimal:
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# --- Models ---


class PeriodMetrics(BaseModel):
    """Financial metrics for a single month."""

    period: str
    revenue: Decimal
    expenses: Decimal
    gross_profit: Decimal
    ar_balance: Decimal
    ap_balance: Decimal
    cash_balance: Decimal
    dso: float
    dpo: float
    revenue_growth_pct: float | None = None
    invoice_count: dict[str, int]
    payment_count: dict[str, int]


class AnnualMetrics(BaseModel):
    """Aggregate annual financial metrics."""

    total_revenue: Decimal
    total_expenses: Decimal
    gross_profit: Decimal
    ending_ar_balance: Decimal
    ending_ap_balance: Decimal
    ending_cash_balance: Decimal
    annual_dso: float
    annual_dpo: float
    free_cash_flow: Decimal


class Invariants(BaseModel):
    """Structural integrity checks on the dataset."""

    journal_balanced: bool
    trial_balance_balanced: bool
    invoice_payment_matched: bool
    bank_reconciliation_rate: float


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
    entry_info = {
        e.entry_id: (e.date, e.status) for e in dataset.journal_entries
    }

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
        period_account_debits[key] = (
            period_account_debits.get(key, Decimal("0")) + line.debit
        )
        period_account_credits[key] = (
            period_account_credits.get(key, Decimal("0")) + line.credit
        )

    # --- Monthly metrics ---
    cumulative_ar = Decimal("0")
    cumulative_ap = Decimal("0")
    cumulative_cash = Decimal("0")
    prev_revenue: Decimal | None = None

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
        for acct_id in _all_accounts_with_prefix(
            period_account_credits, period_str, _REVENUE_PREFIX
        ):
            revenue += period_account_credits.get(
                (period_str, acct_id), Decimal("0")
            )

        # Expenses: debits to expense accounts
        expenses = Decimal("0")
        for acct_id in _all_accounts_with_prefix(
            period_account_debits, period_str, _EXPENSE_PREFIX
        ):
            expenses += period_account_debits.get(
                (period_str, acct_id), Decimal("0")
            )

        # Balance sheet: cumulative
        for acct in _AR_ACCOUNTS:
            cumulative_ar += period_account_debits.get(
                (period_str, acct), Decimal("0")
            )
            cumulative_ar -= period_account_credits.get(
                (period_str, acct), Decimal("0")
            )

        for acct in _AP_ACCOUNTS:
            cumulative_ap += period_account_credits.get(
                (period_str, acct), Decimal("0")
            )
            cumulative_ap -= period_account_debits.get(
                (period_str, acct), Decimal("0")
            )

        for acct in _CASH_ACCOUNTS:
            cumulative_cash += period_account_debits.get(
                (period_str, acct), Decimal("0")
            )
            cumulative_cash -= period_account_credits.get(
                (period_str, acct), Decimal("0")
            )

        # DSO: (AR / Revenue) × days_in_period (avoid div by zero)
        dso = (
            float(cumulative_ar / revenue * days_in_period)
            if revenue > 0
            else 0.0
        )

        # DPO: (AP / Expenses) × days_in_period
        dpo = (
            float(cumulative_ap / expenses * days_in_period)
            if expenses > 0
            else 0.0
        )

        # Revenue growth MoM
        growth: float | None = None
        if prev_revenue is not None and prev_revenue > 0:
            growth = round(
                float((revenue - prev_revenue) / prev_revenue * 100), 2
            )

        monthly_metrics.append(
            PeriodMetrics(
                period=period_str,
                revenue=_q(revenue),
                expenses=_q(expenses),
                gross_profit=_q(revenue - expenses),
                ar_balance=_q(cumulative_ar),
                ap_balance=_q(cumulative_ap),
                cash_balance=_q(cumulative_cash),
                dso=round(dso, 1),
                dpo=round(dpo, 1),
                revenue_growth_pct=growth,
                invoice_count=invoice_by_period.get(period_str, {}),
                payment_count=payment_by_period.get(period_str, {}),
            )
        )

        prev_revenue = revenue

    # --- Annual metrics ---
    total_revenue = sum(m.revenue for m in monthly_metrics)
    total_expenses = sum(m.expenses for m in monthly_metrics)
    last = monthly_metrics[-1] if monthly_metrics else None

    # FCF: sum of all bank transaction amounts (positive = inflow, negative = outflow)
    fcf = Decimal("0")
    for bt in dataset.bank_transactions:
        bt_period = bt.date.strftime("%Y-%m")
        if bt_period in periods:
            fcf += bt.amount

    total_days = sum(
        calendar.monthrange(int(p[:4]), int(p[5:7]))[1] for p in periods
    )
    annual_dso = (
        float(last.ar_balance / total_revenue * total_days)
        if last and total_revenue > 0
        else 0.0
    )
    annual_dpo = (
        float(last.ap_balance / total_expenses * total_days)
        if last and total_expenses > 0
        else 0.0
    )

    annual = AnnualMetrics(
        total_revenue=_q(total_revenue),
        total_expenses=_q(total_expenses),
        gross_profit=_q(total_revenue - total_expenses),
        ending_ar_balance=_q(last.ar_balance) if last else Decimal("0"),
        ending_ap_balance=_q(last.ap_balance) if last else Decimal("0"),
        ending_cash_balance=_q(last.cash_balance) if last else Decimal("0"),
        annual_dso=round(annual_dso, 1),
        annual_dpo=round(annual_dpo, 1),
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
    )


def _all_accounts_with_prefix(
    movements: dict[tuple[str, str], Decimal],
    period: str,
    prefix: str,
) -> set[str]:
    """Find all account IDs with a given prefix that have movements in a period."""
    return {
        acct for (p, acct) in movements if p == period and acct.startswith(prefix)
    }


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
    paid_invoices = {
        inv.invoice_id: inv.amount
        for inv in dataset.invoices
        if inv.status == InvoiceStatus.PAID
    }
    payment_by_invoice = {
        pay.invoice_id: pay.amount for pay in dataset.payments
    }
    invoice_matched = all(
        inv_id in payment_by_invoice
        and payment_by_invoice[inv_id] == amount
        for inv_id, amount in paid_invoices.items()
    )

    # 4. Bank reconciliation rate
    total_bank = len(dataset.bank_transactions)
    reconciled = sum(1 for bt in dataset.bank_transactions if bt.reconciled)
    recon_rate = reconciled / total_bank if total_bank > 0 else 1.0

    return Invariants(
        journal_balanced=journal_balanced,
        trial_balance_balanced=tb_balanced,
        invoice_payment_matched=invoice_matched,
        bank_reconciliation_rate=round(recon_rate, 4),
    )


# --- Export ---


def export_ground_truth(truth: GroundTruth, output_dir: Path) -> None:
    """Write ground_truth.yaml to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _to_yaml_dict(truth.model_dump())
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
