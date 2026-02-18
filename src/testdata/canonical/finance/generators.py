"""Seed-based deterministic generators for canonical finance data."""

from __future__ import annotations

import random
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from .models import (
    AccountType,
    BankTransaction,
    ChartOfAccounts,
    Currency,
    FinanceDataset,
    FXRate,
    Invoice,
    InvoiceStatus,
    JournalEntry,
    JournalLine,
    JournalStatus,
    Payment,
    PaymentMethod,
    PaymentTerms,
    TrialBalance,
)

# --- Constants ---

COST_CENTERS = ["CC100", "CC200", "CC300", "CC400", "CC500"]
USERS = ["jsmith", "mwilson", "agarcia", "ljohnson", "klee", "rbrown"]
VENDOR_NAMES = [
    "Acme Corp", "Global Supply Co", "TechParts Inc", "Office Depot",
    "AWS", "CloudFlare", "Salesforce", "ADP Payroll",
    "Delta Airlines", "Marriott Hotels", "FedEx", "UPS",
    "Deloitte", "KPMG", "Ernst & Young", "PwC",
    "Google Workspace", "Microsoft", "Zoom", "Slack",
]
COUNTERPARTIES = VENDOR_NAMES + [
    "First National Bank", "Wells Fargo", "JPMorgan Chase",
    "Customer A", "Customer B", "Customer C", "Customer D", "Customer E",
]


def _benford_amount(rng: random.Random, min_val: float = 10.0, max_val: float = 100000.0) -> Decimal:
    """Generate a Benford-compliant amount using log-uniform distribution."""
    import math
    log_min = math.log10(min_val)
    log_max = math.log10(max_val)
    value = 10 ** (rng.uniform(log_min, log_max))
    return Decimal(str(round(value, 2)))


def _quantize(d: Decimal) -> Decimal:
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _random_date(rng: random.Random, start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


# --- Chart of Accounts ---

_ACCOUNT_TREE: list[tuple[str, str, AccountType, str | None]] = [
    # Assets
    ("1000", "Assets", AccountType.ASSET, None),
    ("1100", "Cash and Bank", AccountType.ASSET, "1000"),
    ("1110", "Operating Account", AccountType.ASSET, "1100"),
    ("1120", "Savings Account", AccountType.ASSET, "1100"),
    ("1130", "Petty Cash", AccountType.ASSET, "1100"),
    ("1200", "Accounts Receivable", AccountType.ASSET, "1000"),
    ("1210", "Trade Receivables", AccountType.ASSET, "1200"),
    ("1220", "Other Receivables", AccountType.ASSET, "1200"),
    ("1300", "Prepaid Expenses", AccountType.ASSET, "1000"),
    ("1400", "Inventory", AccountType.ASSET, "1000"),
    ("1500", "Fixed Assets", AccountType.ASSET, "1000"),
    ("1510", "Equipment", AccountType.ASSET, "1500"),
    ("1520", "Furniture", AccountType.ASSET, "1500"),
    ("1530", "Vehicles", AccountType.ASSET, "1500"),
    ("1590", "Accumulated Depreciation", AccountType.ASSET, "1500"),
    # Liabilities
    ("2000", "Liabilities", AccountType.LIABILITY, None),
    ("2100", "Accounts Payable", AccountType.LIABILITY, "2000"),
    ("2110", "Trade Payables", AccountType.LIABILITY, "2100"),
    ("2120", "Accrued Expenses", AccountType.LIABILITY, "2100"),
    ("2200", "Short-term Debt", AccountType.LIABILITY, "2000"),
    ("2300", "Long-term Debt", AccountType.LIABILITY, "2000"),
    ("2400", "Tax Payable", AccountType.LIABILITY, "2000"),
    ("2410", "Income Tax Payable", AccountType.LIABILITY, "2400"),
    ("2420", "Sales Tax Payable", AccountType.LIABILITY, "2400"),
    ("2500", "Deferred Revenue", AccountType.LIABILITY, "2000"),
    # Equity
    ("3000", "Equity", AccountType.EQUITY, None),
    ("3100", "Common Stock", AccountType.EQUITY, "3000"),
    ("3200", "Retained Earnings", AccountType.EQUITY, "3000"),
    ("3300", "Dividends", AccountType.EQUITY, "3000"),
    # Revenue
    ("4000", "Revenue", AccountType.REVENUE, None),
    ("4100", "Product Revenue", AccountType.REVENUE, "4000"),
    ("4110", "Domestic Sales", AccountType.REVENUE, "4100"),
    ("4120", "International Sales", AccountType.REVENUE, "4100"),
    ("4200", "Service Revenue", AccountType.REVENUE, "4000"),
    ("4210", "Consulting Fees", AccountType.REVENUE, "4200"),
    ("4220", "Support Contracts", AccountType.REVENUE, "4200"),
    ("4300", "Other Income", AccountType.REVENUE, "4000"),
    ("4310", "Interest Income", AccountType.REVENUE, "4300"),
    ("4320", "FX Gains", AccountType.REVENUE, "4300"),
    # Expenses
    ("5000", "Expenses", AccountType.EXPENSE, None),
    ("5100", "Cost of Goods Sold", AccountType.EXPENSE, "5000"),
    ("5200", "Salaries and Wages", AccountType.EXPENSE, "5000"),
    ("5210", "Salaries", AccountType.EXPENSE, "5200"),
    ("5220", "Benefits", AccountType.EXPENSE, "5200"),
    ("5230", "Payroll Taxes", AccountType.EXPENSE, "5200"),
    ("5300", "Rent and Utilities", AccountType.EXPENSE, "5000"),
    ("5310", "Office Rent", AccountType.EXPENSE, "5300"),
    ("5320", "Utilities", AccountType.EXPENSE, "5300"),
    ("5400", "Professional Services", AccountType.EXPENSE, "5000"),
    ("5410", "Legal Fees", AccountType.EXPENSE, "5400"),
    ("5420", "Audit Fees", AccountType.EXPENSE, "5400"),
    ("5500", "Travel and Entertainment", AccountType.EXPENSE, "5000"),
    ("5510", "Travel", AccountType.EXPENSE, "5500"),
    ("5520", "Meals", AccountType.EXPENSE, "5500"),
    ("5600", "Office Supplies", AccountType.EXPENSE, "5000"),
    ("5700", "Depreciation", AccountType.EXPENSE, "5000"),
    ("5800", "Insurance", AccountType.EXPENSE, "5000"),
    ("5900", "Other Expenses", AccountType.EXPENSE, "5000"),
    ("5910", "Bank Fees", AccountType.EXPENSE, "5900"),
    ("5920", "FX Losses", AccountType.EXPENSE, "5900"),
]

# Leaf accounts (the ones we actually book against)
_LEAF_ACCOUNTS: dict[AccountType, list[str]] = {}
for _aid, _name, _atype, _parent in _ACCOUNT_TREE:
    # An account is a leaf if no other account has it as parent
    pass

def _get_leaf_accounts() -> dict[AccountType, list[str]]:
    parent_ids = {row[3] for row in _ACCOUNT_TREE if row[3] is not None}
    result: dict[AccountType, list[str]] = {}
    for aid, _name, atype, _parent in _ACCOUNT_TREE:
        if aid not in parent_ids:
            result.setdefault(atype, []).append(aid)
    return result


def generate_chart_of_accounts() -> list[ChartOfAccounts]:
    return [
        ChartOfAccounts(account_id=aid, name=name, account_type=atype, parent_id=parent)
        for aid, name, atype, parent in _ACCOUNT_TREE
    ]


# --- Journal Entries & Lines ---

_JOURNAL_TEMPLATES = [
    # (description_template, debit_type, credit_type, amount_range)
    ("Vendor payment - {vendor}", AccountType.LIABILITY, AccountType.ASSET, (500, 50000)),
    ("Revenue recognition - {desc}", AccountType.ASSET, AccountType.REVENUE, (1000, 80000)),
    ("Payroll - {month}", AccountType.EXPENSE, AccountType.ASSET, (5000, 75000)),
    ("Office supplies", AccountType.EXPENSE, AccountType.LIABILITY, (50, 2000)),
    ("Depreciation - {month}", AccountType.EXPENSE, AccountType.ASSET, (500, 5000)),
    ("Insurance premium", AccountType.EXPENSE, AccountType.ASSET, (1000, 10000)),
    ("Rent payment - {month}", AccountType.EXPENSE, AccountType.ASSET, (3000, 15000)),
    ("Professional fees - {vendor}", AccountType.EXPENSE, AccountType.LIABILITY, (2000, 30000)),
    ("Tax accrual - {month}", AccountType.EXPENSE, AccountType.LIABILITY, (1000, 20000)),
    ("Bank fees", AccountType.EXPENSE, AccountType.ASSET, (10, 500)),
]


def _generate_journal_entries(
    rng: random.Random,
    fiscal_start: date,
    months: int,
) -> tuple[list[JournalEntry], list[JournalLine]]:
    leaf = _get_leaf_accounts()
    entries: list[JournalEntry] = []
    lines: list[JournalLine] = []
    entry_counter = 0
    line_counter = 0

    for month_offset in range(months):
        month_start = date(fiscal_start.year, fiscal_start.month + month_offset, 1)
        if month_start.month == 12:
            month_end = date(month_start.year, 12, 31)
        else:
            month_end = date(month_start.year, month_start.month + 1, 1) - timedelta(days=1)
        month_label = month_start.strftime("%B %Y")

        # Seasonal factor: Q4 has more activity
        seasonal = 1.0 + 0.3 * (month_start.month >= 10)
        entries_this_month = int(rng.gauss(420, 40) * seasonal)

        for _ in range(entries_this_month):
            entry_counter += 1
            entry_id = f"JE-{entry_counter:06d}"
            entry_date = _random_date(rng, month_start, month_end)

            template = rng.choice(_JOURNAL_TEMPLATES)
            desc_tmpl, debit_type, credit_type, (amt_min, amt_max) = template
            vendor = rng.choice(VENDOR_NAMES)
            desc = desc_tmpl.format(vendor=vendor, month=month_label, desc=f"Q{(month_start.month - 1) // 3 + 1}")

            status = JournalStatus.POSTED
            if rng.random() < 0.02:
                status = JournalStatus.REVERSED
            elif rng.random() < 0.05:
                status = JournalStatus.DRAFT

            entries.append(JournalEntry(
                entry_id=entry_id,
                date=entry_date,
                description=desc,
                status=status,
                created_by=rng.choice(USERS),
            ))

            # Generate balanced lines (1-3 debit lines, 1-3 credit lines)
            n_debit = rng.choices([1, 2, 3], weights=[70, 25, 5])[0]
            n_credit = rng.choices([1, 2, 3], weights=[70, 25, 5])[0]

            total_amount = _benford_amount(rng, amt_min, amt_max)

            # Split into debit amounts
            debit_amounts = _split_amount(rng, total_amount, n_debit)
            credit_amounts = _split_amount(rng, total_amount, n_credit)

            cost_center = rng.choice(COST_CENTERS) if rng.random() < 0.85 else None

            for i, amt in enumerate(debit_amounts):
                line_counter += 1
                lines.append(JournalLine(
                    line_id=f"JL-{line_counter:07d}",
                    entry_id=entry_id,
                    account_id=rng.choice(leaf[debit_type]),
                    debit=amt,
                    credit=Decimal("0.00"),
                    cost_center=cost_center,
                ))

            for i, amt in enumerate(credit_amounts):
                line_counter += 1
                lines.append(JournalLine(
                    line_id=f"JL-{line_counter:07d}",
                    entry_id=entry_id,
                    account_id=rng.choice(leaf[credit_type]),
                    debit=Decimal("0.00"),
                    credit=amt,
                    cost_center=cost_center,
                ))

    return entries, lines


def _split_amount(rng: random.Random, total: Decimal, n: int) -> list[Decimal]:
    """Split a total into n parts that sum exactly to total."""
    if n == 1:
        return [_quantize(total)]

    # Generate random proportions
    raw = [rng.random() for _ in range(n)]
    raw_sum = sum(raw)
    parts = [_quantize(total * Decimal(str(r / raw_sum))) for r in raw]

    # Fix rounding: adjust last element
    parts[-1] = _quantize(total - sum(parts[:-1]))
    return parts


# --- Invoices ---

def _generate_invoices(
    rng: random.Random,
    fiscal_start: date,
    months: int,
    count: int = 3000,
) -> list[Invoice]:
    invoices: list[Invoice] = []
    fiscal_end = date(fiscal_start.year, fiscal_start.month + months - 1, 28)

    # 80/20 vendor concentration: 20% of vendors get 80% of invoices
    top_vendors = VENDOR_NAMES[:4]
    other_vendors = VENDOR_NAMES[4:]

    terms_options = list(PaymentTerms)
    terms_days = {
        PaymentTerms.NET_30: 30,
        PaymentTerms.NET_60: 60,
        PaymentTerms.NET_90: 90,
        PaymentTerms.DUE_ON_RECEIPT: 0,
    }

    for i in range(count):
        inv_date = _random_date(rng, fiscal_start, fiscal_end)
        terms = rng.choice(terms_options)
        due = inv_date + timedelta(days=terms_days[terms])

        if rng.random() < 0.80:
            vendor = rng.choice(top_vendors)
        else:
            vendor = rng.choice(other_vendors)

        vendor_id = f"V-{VENDOR_NAMES.index(vendor) + 1:04d}"
        amount = _benford_amount(rng, 100, 50000)

        # Status depends on date relative to fiscal end
        days_since = (fiscal_end - inv_date).days
        if days_since > terms_days[terms] + 30:
            status = rng.choices(
                [InvoiceStatus.PAID, InvoiceStatus.CANCELLED],
                weights=[95, 5],
            )[0]
        elif days_since > terms_days[terms]:
            status = rng.choices(
                [InvoiceStatus.PAID, InvoiceStatus.OVERDUE, InvoiceStatus.PARTIAL],
                weights=[70, 20, 10],
            )[0]
        else:
            status = rng.choices(
                [InvoiceStatus.OPEN, InvoiceStatus.PAID],
                weights=[60, 40],
            )[0]

        invoices.append(Invoice(
            invoice_id=f"INV-{i + 1:06d}",
            vendor_id=vendor_id,
            date=inv_date,
            due_date=due,
            amount=amount,
            status=status,
            payment_terms=terms,
        ))

    return invoices


# --- Payments ---

def _generate_payments(
    rng: random.Random,
    invoices: list[Invoice],
) -> list[Payment]:
    payments: list[Payment] = []
    counter = 0
    methods = list(PaymentMethod)

    for inv in invoices:
        if inv.status in (InvoiceStatus.PAID, InvoiceStatus.PARTIAL):
            counter += 1
            pay_date = inv.date + timedelta(days=rng.randint(1, 45))

            if inv.status == InvoiceStatus.PARTIAL:
                pay_amount = _quantize(inv.amount * Decimal(str(rng.uniform(0.3, 0.9))))
            else:
                pay_amount = inv.amount

            payments.append(Payment(
                payment_id=f"PAY-{counter:06d}",
                invoice_id=inv.invoice_id,
                date=pay_date,
                amount=pay_amount,
                currency=inv.currency,
                method=rng.choice(methods),
            ))

    return payments


# --- Bank Transactions ---

def _generate_bank_transactions(
    rng: random.Random,
    fiscal_start: date,
    months: int,
    count: int = 8000,
) -> list[BankTransaction]:
    transactions: list[BankTransaction] = []
    fiscal_end = date(fiscal_start.year, fiscal_start.month + months - 1, 28)

    for i in range(count):
        txn_date = _random_date(rng, fiscal_start, fiscal_end)
        # Mix of debits (60%) and credits (40%)
        if rng.random() < 0.60:
            amount = -_benford_amount(rng, 50, 30000)
        else:
            amount = _benford_amount(rng, 100, 80000)

        counterparty = rng.choice(COUNTERPARTIES)
        ref_prefix = "TXN" if amount < 0 else "RCV"

        transactions.append(BankTransaction(
            txn_id=f"BT-{i + 1:07d}",
            date=txn_date,
            amount=amount,
            reference=f"{ref_prefix}-{rng.randint(100000, 999999)}",
            counterparty=counterparty,
            reconciled=rng.random() < 0.85,
        ))

    return transactions


# --- FX Rates ---

_FX_BASE_RATES: dict[tuple[Currency, Currency], float] = {
    (Currency.EUR, Currency.USD): 1.10,
    (Currency.GBP, Currency.USD): 1.27,
    (Currency.CHF, Currency.USD): 1.13,
    (Currency.JPY, Currency.USD): 0.0067,
    (Currency.USD, Currency.EUR): 0.91,
    (Currency.USD, Currency.GBP): 0.79,
    (Currency.USD, Currency.CHF): 0.88,
    (Currency.USD, Currency.JPY): 149.5,
}


def _generate_fx_rates(
    rng: random.Random,
    fiscal_start: date,
    months: int,
) -> list[FXRate]:
    rates: list[FXRate] = []

    for month_offset in range(months):
        month_start = date(fiscal_start.year, fiscal_start.month + month_offset, 1)
        if month_start.month == 12:
            month_end = date(month_start.year, 12, 31)
        else:
            month_end = date(month_start.year, month_start.month + 1, 1) - timedelta(days=1)

        # Weekly rates
        current = month_start
        while current <= month_end:
            for (from_ccy, to_ccy), base_rate in _FX_BASE_RATES.items():
                # Random walk around base rate (±3%)
                drift = rng.gauss(0, 0.01)
                rate_val = base_rate * (1 + drift)
                rates.append(FXRate(
                    from_ccy=from_ccy,
                    to_ccy=to_ccy,
                    date=current,
                    rate=Decimal(str(round(rate_val, 6))),
                ))
            current += timedelta(days=7)

    return rates


# --- Trial Balance ---

def _generate_trial_balance(
    journal_lines: list[JournalLine],
    fiscal_start: date,
    months: int,
) -> list[TrialBalance]:
    """Derive trial balance from journal lines (ensures consistency)."""
    return _derive_trial_balance(journal_lines, fiscal_start, months)


def _derive_trial_balance(
    journal_lines: list[JournalLine],
    fiscal_start: date,
    months: int,
) -> list[TrialBalance]:
    """Build monthly trial balance from journal lines.

    Since journal lines don't carry dates directly, we use the entry_id
    ordering (which is chronological) to assign periods.
    """
    # Group lines by account
    account_debits: dict[str, Decimal] = {}
    account_credits: dict[str, Decimal] = {}

    for line in journal_lines:
        account_debits[line.account_id] = account_debits.get(line.account_id, Decimal("0")) + line.debit
        account_credits[line.account_id] = account_credits.get(line.account_id, Decimal("0")) + line.credit

    # Create one TB row per account per month (cumulative balances)
    all_accounts = sorted(set(account_debits.keys()) | set(account_credits.keys()))
    result: list[TrialBalance] = []

    for month_offset in range(months):
        period = date(fiscal_start.year, fiscal_start.month + month_offset, 1)
        period_str = period.strftime("%Y-%m")

        # Scale balances proportionally by month (simplified: linear accumulation)
        fraction = Decimal(str((month_offset + 1) / months))

        for acct in all_accounts:
            total_debit = _quantize(account_debits.get(acct, Decimal("0")) * fraction)
            total_credit = _quantize(account_credits.get(acct, Decimal("0")) * fraction)
            result.append(TrialBalance(
                account_id=acct,
                period=period_str,
                debit_balance=total_debit,
                credit_balance=total_credit,
            ))

    return result


# --- Main Generator ---

def generate_finance_dataset(seed: int = 42, months: int = 12) -> FinanceDataset:
    """Generate a complete clean finance dataset.

    Args:
        seed: Random seed for reproducibility.
        months: Number of months to generate (fiscal year).

    Returns:
        FinanceDataset with all 8 tables populated.
    """
    rng = random.Random(seed)
    fiscal_start = date(2025, 1, 1)

    chart = generate_chart_of_accounts()
    entries, lines = _generate_journal_entries(rng, fiscal_start, months)
    invoices = _generate_invoices(rng, fiscal_start, months)
    payments = _generate_payments(rng, invoices)
    bank_txns = _generate_bank_transactions(rng, fiscal_start, months)
    fx_rates = _generate_fx_rates(rng, fiscal_start, months)
    trial_bal = _generate_trial_balance(lines, fiscal_start, months)

    return FinanceDataset(
        chart_of_accounts=chart,
        journal_entries=entries,
        journal_lines=lines,
        invoices=invoices,
        payments=payments,
        bank_transactions=bank_txns,
        fx_rates=fx_rates,
        trial_balance=trial_bal,
    )
