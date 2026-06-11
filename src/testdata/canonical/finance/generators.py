"""Seed-based deterministic generators for canonical finance data.

Event-driven architecture: business events cascade across tables to produce
a closed-loop accounting system where GL entries, invoices, payments, bank
transactions, and trial balance are numerically consistent.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from .models import (
    AccountType,
    BalanceSheet,
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
    MeasureProbe,
    Payment,
    PaymentMethod,
    PaymentTerms,
    RefActivity,
    RefEntity,
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
CUSTOMER_NAMES = [
    "Northwind Corp", "Contoso Ltd", "Adventure Works", "Fabrikam Inc",
    "Tailspin Toys", "Woodgrove Bank", "Litware Inc", "Proseware",
    "Alpine Ski House", "Trey Research", "Humongous Insurance", "Datum Corp",
    "A. Datum", "Coho Vineyard", "Lucerne Publishing", "Margie's Travel",
]

# --- Helpers ---

def _benford_amount(rng: random.Random, min_val: float = 10.0, max_val: float = 100000.0) -> Decimal:
    """Generate a Benford-compliant amount using log-uniform distribution."""
    log_min = math.log10(min_val)
    log_max = math.log10(max_val)
    value = 10 ** (rng.uniform(log_min, log_max))
    return Decimal(str(round(value, 2)))


def _quantize(d: Decimal) -> Decimal:
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _random_date(rng: random.Random, start: date, end: date) -> date:
    delta = (end - start).days
    if delta <= 0:
        return start
    return start + timedelta(days=rng.randint(0, delta))


def _split_amount(rng: random.Random, total: Decimal, n: int) -> list[Decimal]:
    """Split a total into n parts that sum exactly to total."""
    if n == 1:
        return [_quantize(total)]

    raw = [rng.random() for _ in range(n)]
    raw_sum = sum(raw)
    parts = [_quantize(total * Decimal(str(r / raw_sum))) for r in raw]

    # Fix rounding: adjust last element
    parts[-1] = _quantize(total - sum(parts[:-1]))
    return parts


def _month_end(year: int, month: int) -> date:
    """Return the last day of the given month."""
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def _month_start_end(fiscal_start: date, month_offset: int) -> tuple[date, date]:
    """Return (first_day, last_day) for fiscal_start + month_offset months."""
    year = fiscal_start.year + (fiscal_start.month + month_offset - 1) // 12
    month = (fiscal_start.month + month_offset - 1) % 12 + 1
    start = date(year, month, 1)
    end = _month_end(year, month)
    return start, end


# --- Counters ---

@dataclass
class _Counters:
    """Mutable counters for sequential ID generation across event types."""

    entry: int = 0
    line: int = 0
    invoice: int = 0
    payment: int = 0
    bank_txn: int = 0

    def next_entry(self) -> str:
        self.entry += 1
        return f"JE-{self.entry:06d}"

    def next_line(self) -> str:
        self.line += 1
        return f"JL-{self.line:07d}"

    def next_invoice(self) -> str:
        self.invoice += 1
        return f"INV-{self.invoice:06d}"

    def next_payment(self) -> str:
        self.payment += 1
        return f"PAY-{self.payment:06d}"

    def next_bank_txn(self) -> str:
        self.bank_txn += 1
        return f"BT-{self.bank_txn:07d}"


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


def _get_leaf_accounts() -> dict[AccountType, list[str]]:
    parent_ids = {row[3] for row in _ACCOUNT_TREE if row[3] is not None}
    result: dict[AccountType, list[str]] = {}
    for aid, _name, atype, _parent in _ACCOUNT_TREE:
        if aid not in parent_ids:
            result.setdefault(atype, []).append(aid)
    return result


# Specific account groups for event-driven generation
_AR_ACCOUNTS = ["1210", "1220"]
_CASH_ACCOUNTS = ["1110", "1120"]
_AP_ACCOUNTS = ["2110", "2120"]


def generate_chart_of_accounts() -> list[ChartOfAccounts]:
    return [
        ChartOfAccounts(account_id=aid, name=name, account_type=atype, parent_id=parent)
        for aid, name, atype, parent in _ACCOUNT_TREE
    ]


# --- Revenue Cycle: Sales → Cash Receipts ---

@dataclass
class _SaleRecord:
    """Internal tracking for a sale event (for cash receipt generation)."""

    entry_id: str
    sale_date: date
    amount: Decimal
    customer: str
    collected: bool = False


def _generate_revenue_entries(
    rng: random.Random,
    fiscal_start: date,
    months: int,
    counters: _Counters,
    *,
    sales_per_month: int = 250,
    q4_seasonal_boost: float = 0.3,
) -> tuple[list[JournalEntry], list[JournalLine], list[_SaleRecord]]:
    """Generate customer sale events: DR AR, CR Revenue.

    Returns GL entries/lines plus internal sale records for cash receipt generation.
    """
    leaf = _get_leaf_accounts()
    revenue_accounts = leaf.get(AccountType.REVENUE, [])
    # Exclude "Other Income" (interest, FX gains) from sales
    sales_revenue = [a for a in revenue_accounts if a.startswith(("41", "42"))]

    entries: list[JournalEntry] = []
    lines: list[JournalLine] = []
    sales: list[_SaleRecord] = []

    for month_offset in range(months):
        m_start, m_end = _month_start_end(fiscal_start, month_offset)

        # Q4 seasonal boost
        seasonal = 1.0 + q4_seasonal_boost * (m_start.month >= 10)
        count = int(rng.gauss(sales_per_month, sales_per_month * 0.1) * seasonal)

        for _ in range(count):
            sale_date = _random_date(rng, m_start, m_end)
            amount = _benford_amount(rng, 500, 80000)
            customer = rng.choice(CUSTOMER_NAMES)
            cost_center = rng.choice(COST_CENTERS) if rng.random() < 0.85 else None

            entry_id = counters.next_entry()
            entries.append(JournalEntry(
                entry_id=entry_id,
                date=sale_date,
                description=f"Revenue recognition - {customer}",
                status=JournalStatus.POSTED,
                created_by=rng.choice(USERS),
            ))

            # DR: AR
            ar_account = rng.choice(_AR_ACCOUNTS)
            lines.append(JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id=ar_account,
                debit=amount,
                credit=Decimal("0.00"),
                cost_center=cost_center,
            ))

            # CR: Revenue (may split across 1-2 revenue accounts)
            n_rev = rng.choices([1, 2], weights=[80, 20])[0]
            rev_amounts = _split_amount(rng, amount, n_rev)
            for rev_amt in rev_amounts:
                lines.append(JournalLine(
                    line_id=counters.next_line(),
                    entry_id=entry_id,
                    account_id=rng.choice(sales_revenue),
                    debit=Decimal("0.00"),
                    credit=rev_amt,
                    cost_center=cost_center,
                ))

            sales.append(_SaleRecord(
                entry_id=entry_id,
                sale_date=sale_date,
                amount=amount,
                customer=customer,
            ))

    return entries, lines, sales


def _generate_cash_receipts(
    rng: random.Random,
    sales: list[_SaleRecord],
    fiscal_start: date,
    months: int,
    counters: _Counters,
    *,
    collection_rate: float = 0.85,
) -> tuple[list[JournalEntry], list[JournalLine], list[BankTransaction]]:
    """Generate cash receipt events for collected sales: DR Cash, CR AR + Bank Txn."""
    fiscal_end = _month_start_end(fiscal_start, months - 1)[1]

    entries: list[JournalEntry] = []
    lines: list[JournalLine] = []
    bank_txns: list[BankTransaction] = []

    for sale in sales:
        # Older sales more likely to be collected
        days_outstanding = (fiscal_end - sale.sale_date).days
        if days_outstanding > 60:
            collect_prob = collection_rate
        elif days_outstanding > 30:
            collect_prob = collection_rate * 0.7
        else:
            collect_prob = collection_rate * 0.3

        if rng.random() >= collect_prob:
            continue

        sale.collected = True

        # Payment arrives 5-45 days after sale
        receipt_date = sale.sale_date + timedelta(days=rng.randint(5, 45))
        if receipt_date > fiscal_end:
            receipt_date = fiscal_end

        # Partial collection (5% chance)
        if rng.random() < 0.05:
            amount = _quantize(sale.amount * Decimal(str(rng.uniform(0.5, 0.95))))
        else:
            amount = sale.amount

        entry_id = counters.next_entry()
        entries.append(JournalEntry(
            entry_id=entry_id,
            date=receipt_date,
            description=f"Cash receipt - {sale.customer}",
            status=JournalStatus.POSTED,
            created_by=rng.choice(USERS),
        ))

        # DR: Cash
        cash_account = rng.choice(_CASH_ACCOUNTS)
        lines.append(JournalLine(
            line_id=counters.next_line(),
            entry_id=entry_id,
            account_id=cash_account,
            debit=amount,
            credit=Decimal("0.00"),
        ))

        # CR: AR
        lines.append(JournalLine(
            line_id=counters.next_line(),
            entry_id=entry_id,
            account_id=rng.choice(_AR_ACCOUNTS),
            debit=Decimal("0.00"),
            credit=amount,
        ))

        # Bank transaction (positive = credit/inflow)
        bank_txns.append(BankTransaction(
            txn_id=counters.next_bank_txn(),
            account_id=cash_account,
            date=receipt_date,
            amount=amount,
            reference=f"RCV-{rng.randint(100000, 999999)}",
            counterparty=sale.customer,
            reconciled=rng.random() < 0.90,
        ))

    return entries, lines, bank_txns


# --- Expenditure Cycle: Purchase Invoices → Vendor Payments ---

def _generate_purchase_invoices(
    rng: random.Random,
    fiscal_start: date,
    months: int,
    counters: _Counters,
    *,
    count: int = 3000,
) -> tuple[list[Invoice], list[JournalEntry], list[JournalLine]]:
    """Generate vendor/purchase invoices with GL entries: DR Expense, CR AP."""
    leaf = _get_leaf_accounts()
    expense_accounts = leaf.get(AccountType.EXPENSE, [])

    invoices: list[Invoice] = []
    entries: list[JournalEntry] = []
    lines_out: list[JournalLine] = []

    fiscal_end = _month_start_end(fiscal_start, months - 1)[1]

    # 80/20 vendor concentration
    top_vendors = VENDOR_NAMES[:4]
    other_vendors = VENDOR_NAMES[4:]

    terms_days = {
        PaymentTerms.NET_30: 30,
        PaymentTerms.NET_60: 60,
        PaymentTerms.NET_90: 90,
        PaymentTerms.DUE_ON_RECEIPT: 0,
    }

    for i in range(count):
        inv_date = _random_date(rng, fiscal_start, fiscal_end)
        terms = rng.choice(list(PaymentTerms))
        due = inv_date + timedelta(days=terms_days[terms])

        if rng.random() < 0.80:
            vendor = rng.choice(top_vendors)
        else:
            vendor = rng.choice(other_vendors)

        vendor_id = f"V-{VENDOR_NAMES.index(vendor) + 1:04d}"
        amount = _benford_amount(rng, 100, 50000)

        # Status depends on age relative to fiscal end
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

        invoice_id = counters.next_invoice()
        inv = Invoice(
            invoice_id=invoice_id,
            vendor_id=vendor_id,
            date=inv_date,
            due_date=due,
            amount=amount,
            status=status,
            payment_terms=terms,
        )
        invoices.append(inv)

        # GL entry: DR Expense, CR AP (only for non-cancelled invoices)
        if status != InvoiceStatus.CANCELLED:
            cost_center = rng.choice(COST_CENTERS) if rng.random() < 0.85 else None
            entry_id = counters.next_entry()
            inv.entry_id = entry_id

            entries.append(JournalEntry(
                entry_id=entry_id,
                date=inv_date,
                description=f"Vendor invoice - {vendor} - {invoice_id}",
                status=JournalStatus.POSTED,
                created_by=rng.choice(USERS),
            ))

            # DR: Expense (may split across 1-3 expense accounts)
            n_exp = rng.choices([1, 2, 3], weights=[70, 25, 5])[0]
            exp_amounts = _split_amount(rng, amount, n_exp)
            for exp_amt in exp_amounts:
                lines_out.append(JournalLine(
                    line_id=counters.next_line(),
                    entry_id=entry_id,
                    account_id=rng.choice(expense_accounts),
                    debit=exp_amt,
                    credit=Decimal("0.00"),
                    cost_center=cost_center,
                ))

            # CR: AP
            lines_out.append(JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id=rng.choice(_AP_ACCOUNTS),
                debit=Decimal("0.00"),
                credit=amount,
                cost_center=cost_center,
            ))

    return invoices, entries, lines_out


def _generate_vendor_payments(
    rng: random.Random,
    invoices: list[Invoice],
    counters: _Counters,
) -> tuple[list[Payment], list[JournalEntry], list[JournalLine], list[BankTransaction]]:
    """Generate vendor payment events: DR AP, CR Cash + Bank Txn (debit)."""
    payments: list[Payment] = []
    entries: list[JournalEntry] = []
    lines_out: list[JournalLine] = []
    bank_txns: list[BankTransaction] = []

    methods = list(PaymentMethod)
    vendor_name_by_id = {f"V-{i + 1:04d}": name for i, name in enumerate(VENDOR_NAMES)}

    for inv in invoices:
        if inv.status not in (InvoiceStatus.PAID, InvoiceStatus.PARTIAL):
            continue

        pay_date = inv.date + timedelta(days=rng.randint(1, 45))

        if inv.status == InvoiceStatus.PARTIAL:
            pay_amount = _quantize(inv.amount * Decimal(str(rng.uniform(0.3, 0.9))))
        else:
            pay_amount = inv.amount

        payment_id = counters.next_payment()
        payments.append(Payment(
            payment_id=payment_id,
            invoice_id=inv.invoice_id,
            date=pay_date,
            amount=pay_amount,
            currency=inv.currency,
            method=rng.choice(methods),
        ))

        # GL entry: DR AP, CR Cash
        entry_id = counters.next_entry()
        vendor_name = vendor_name_by_id.get(inv.vendor_id, inv.vendor_id)
        entries.append(JournalEntry(
            entry_id=entry_id,
            date=pay_date,
            description=f"Vendor payment - {vendor_name} - {inv.invoice_id}",
            status=JournalStatus.POSTED,
            created_by=rng.choice(USERS),
        ))

        # DR: AP
        lines_out.append(JournalLine(
            line_id=counters.next_line(),
            entry_id=entry_id,
            account_id=rng.choice(_AP_ACCOUNTS),
            debit=pay_amount,
            credit=Decimal("0.00"),
        ))

        # CR: Cash
        cash_account = rng.choice(_CASH_ACCOUNTS)
        lines_out.append(JournalLine(
            line_id=counters.next_line(),
            entry_id=entry_id,
            account_id=cash_account,
            debit=Decimal("0.00"),
            credit=pay_amount,
        ))

        # Bank transaction (negative = debit/outflow)
        bank_txns.append(BankTransaction(
            txn_id=counters.next_bank_txn(),
            account_id=cash_account,
            date=pay_date,
            amount=-pay_amount,
            reference=f"TXN-{rng.randint(100000, 999999)}",
            counterparty=vendor_name,
            reconciled=rng.random() < 0.90,
            payment_id=payment_id,
        ))

    return payments, entries, lines_out, bank_txns


# --- Operating Events: Payroll, Depreciation, Rent, Misc ---

def _generate_operating_events(
    rng: random.Random,
    fiscal_start: date,
    months: int,
    counters: _Counters,
) -> tuple[list[JournalEntry], list[JournalLine], list[BankTransaction]]:
    """Generate recurring operating events: payroll, depreciation, rent, etc."""
    entries: list[JournalEntry] = []
    lines_out: list[JournalLine] = []
    bank_txns: list[BankTransaction] = []

    for month_offset in range(months):
        m_start, m_end = _month_start_end(fiscal_start, month_offset)
        month_label = m_start.strftime("%B %Y")

        # --- Payroll (last day of month, cash outflow) ---
        payroll_amount = _benford_amount(rng, 30000, 75000)
        entry_id = counters.next_entry()
        entries.append(JournalEntry(
            entry_id=entry_id,
            date=m_end,
            description=f"Payroll - {month_label}",
            status=JournalStatus.POSTED,
            created_by="system",
        ))

        # Split across salary/benefits/taxes
        salary = _quantize(payroll_amount * Decimal("0.70"))
        benefits = _quantize(payroll_amount * Decimal("0.20"))
        taxes = _quantize(payroll_amount - salary - benefits)

        for acct, amt in [("5210", salary), ("5220", benefits), ("5230", taxes)]:
            lines_out.append(JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id=acct,
                debit=amt,
                credit=Decimal("0.00"),
            ))

        lines_out.append(JournalLine(
            line_id=counters.next_line(),
            entry_id=entry_id,
            account_id="1110",  # Operating account
            debit=Decimal("0.00"),
            credit=payroll_amount,
        ))

        bank_txns.append(BankTransaction(
            txn_id=counters.next_bank_txn(),
            account_id="1110",
            date=m_end,
            amount=-payroll_amount,
            reference=f"PAY-{m_start.strftime('%Y%m')}",
            counterparty="ADP Payroll",
            reconciled=True,
        ))

        # --- Rent (1st of month, cash outflow) ---
        rent_amount = _benford_amount(rng, 5000, 15000)
        entry_id = counters.next_entry()
        entries.append(JournalEntry(
            entry_id=entry_id,
            date=m_start,
            description=f"Rent payment - {month_label}",
            status=JournalStatus.POSTED,
            created_by="system",
        ))

        lines_out.append(JournalLine(
            line_id=counters.next_line(),
            entry_id=entry_id,
            account_id="5310",  # Office Rent
            debit=rent_amount,
            credit=Decimal("0.00"),
        ))
        lines_out.append(JournalLine(
            line_id=counters.next_line(),
            entry_id=entry_id,
            account_id="1110",  # Operating account
            debit=Decimal("0.00"),
            credit=rent_amount,
        ))

        bank_txns.append(BankTransaction(
            txn_id=counters.next_bank_txn(),
            account_id="1110",
            date=m_start,
            amount=-rent_amount,
            reference=f"RENT-{m_start.strftime('%Y%m')}",
            counterparty="Building Management",
            reconciled=True,
        ))

        # --- Depreciation (end of month, non-cash) ---
        depr_amount = _benford_amount(rng, 1000, 5000)
        entry_id = counters.next_entry()
        entries.append(JournalEntry(
            entry_id=entry_id,
            date=m_end,
            description=f"Depreciation - {month_label}",
            status=JournalStatus.POSTED,
            created_by="system",
        ))

        lines_out.append(JournalLine(
            line_id=counters.next_line(),
            entry_id=entry_id,
            account_id="5700",  # Depreciation expense
            debit=depr_amount,
            credit=Decimal("0.00"),
        ))
        lines_out.append(JournalLine(
            line_id=counters.next_line(),
            entry_id=entry_id,
            account_id="1590",  # Accumulated Depreciation
            debit=Decimal("0.00"),
            credit=depr_amount,
        ))

        # --- Insurance (1st of month, cash outflow) ---
        insurance = _benford_amount(rng, 1000, 5000)
        entry_id = counters.next_entry()
        entries.append(JournalEntry(
            entry_id=entry_id,
            date=m_start,
            description=f"Insurance premium - {month_label}",
            status=JournalStatus.POSTED,
            created_by="system",
        ))

        lines_out.append(JournalLine(
            line_id=counters.next_line(),
            entry_id=entry_id,
            account_id="5800",  # Insurance
            debit=insurance,
            credit=Decimal("0.00"),
        ))
        lines_out.append(JournalLine(
            line_id=counters.next_line(),
            entry_id=entry_id,
            account_id="1110",
            debit=Decimal("0.00"),
            credit=insurance,
        ))

        bank_txns.append(BankTransaction(
            txn_id=counters.next_bank_txn(),
            account_id="1110",
            date=m_start,
            amount=-insurance,
            reference=f"INS-{m_start.strftime('%Y%m')}",
            counterparty="Insurance Corp",
            reconciled=True,
        ))

        # --- Misc operating expenses (5-15 per month) ---
        misc_templates = [
            ("Office supplies", "5600"),
            ("Travel expense", "5510"),
            ("Meals and entertainment", "5520"),
            ("Utilities", "5320"),
            ("Bank fees", "5910"),
            ("Legal fees", "5410"),
            ("Audit fees", "5420"),
        ]

        n_misc = rng.randint(5, 15)
        for _ in range(n_misc):
            desc, account = rng.choice(misc_templates)
            misc_date = _random_date(rng, m_start, m_end)
            misc_amount = _benford_amount(rng, 50, 5000)
            cost_center = rng.choice(COST_CENTERS) if rng.random() < 0.85 else None

            entry_id = counters.next_entry()
            entries.append(JournalEntry(
                entry_id=entry_id,
                date=misc_date,
                description=f"{desc} - {month_label}",
                status=JournalStatus.POSTED,
                created_by=rng.choice(USERS),
            ))

            lines_out.append(JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id=account,
                debit=misc_amount,
                credit=Decimal("0.00"),
                cost_center=cost_center,
            ))
            lines_out.append(JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id="1110",
                debit=Decimal("0.00"),
                credit=misc_amount,
                cost_center=cost_center,
            ))

            # 70% are cash payments, 30% are accruals (no bank txn)
            if rng.random() < 0.70:
                bank_txns.append(BankTransaction(
                    txn_id=counters.next_bank_txn(),
                    account_id="1110",
                    date=misc_date,
                    amount=-misc_amount,
                    reference=f"TXN-{rng.randint(100000, 999999)}",
                    counterparty=rng.choice(VENDOR_NAMES),
                    reconciled=rng.random() < 0.85,
                ))

    return entries, lines_out, bank_txns


# --- Misc Bank Transactions (interest, transfers, unmatched) ---

def _generate_misc_bank_transactions(
    rng: random.Random,
    fiscal_start: date,
    months: int,
    counters: _Counters,
    *,
    count_per_month: int = 20,
) -> tuple[list[JournalEntry], list[JournalLine], list[BankTransaction]]:
    """Generate miscellaneous bank transactions with GL entries.

    Covers interest income, bank fees, and unreconciled items.
    """
    entries: list[JournalEntry] = []
    lines_out: list[JournalLine] = []
    bank_txns: list[BankTransaction] = []

    for month_offset in range(months):
        m_start, m_end = _month_start_end(fiscal_start, month_offset)

        for _ in range(count_per_month):
            txn_date = _random_date(rng, m_start, m_end)

            # 30% credits (interest, refunds), 70% debits (fees, misc charges)
            cash_account = rng.choice(_CASH_ACCOUNTS)
            if rng.random() < 0.30:
                amount = _benford_amount(rng, 10, 2000)
                ref = f"RCV-{rng.randint(100000, 999999)}"
                counterparty = rng.choice(["First National Bank", "Wells Fargo", "JPMorgan Chase"])
                gl_debit_account = cash_account
                gl_credit_account = "4310"  # Interest Income
            else:
                amount = -_benford_amount(rng, 5, 500)
                ref = f"FEE-{rng.randint(100000, 999999)}"
                counterparty = rng.choice(["First National Bank", "Wells Fargo", "JPMorgan Chase"])
                gl_debit_account = "5910"  # Bank Fees
                gl_credit_account = cash_account

            bank_txns.append(BankTransaction(
                txn_id=counters.next_bank_txn(),
                account_id=cash_account,
                date=txn_date,
                amount=amount,
                reference=ref,
                counterparty=counterparty,
                reconciled=rng.random() < 0.80,
            ))

            # GL entry for the bank transaction
            entry_id = counters.next_entry()
            entries.append(JournalEntry(
                entry_id=entry_id,
                date=txn_date,
                description=f"Bank {'credit' if amount > 0 else 'fee'} - {ref}",
                status=JournalStatus.POSTED,
                created_by="system",
            ))

            abs_amount = abs(amount)
            lines_out.append(JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id=gl_debit_account,
                debit=abs_amount,
                credit=Decimal("0.00"),
            ))
            lines_out.append(JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id=gl_credit_account,
                debit=Decimal("0.00"),
                credit=abs_amount,
            ))

    return entries, lines_out, bank_txns


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
        m_start, m_end = _month_start_end(fiscal_start, month_offset)

        # Weekly rates
        current = m_start
        while current <= m_end:
            for (from_ccy, to_ccy), base_rate in _FX_BASE_RATES.items():
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


# --- Trial Balance (derived from actual GL) ---

def _derive_trial_balance(
    all_entries: list[JournalEntry],
    all_lines: list[JournalLine],
    fiscal_start: date,
    months: int,
) -> list[TrialBalance]:
    """Build monthly trial balance from actual GL entries.

    Only includes POSTED entries. Groups by account and period,
    stores per-period activity (not cumulative) so that summing
    all periods yields correct totals for the accounting equation.
    """
    # Build entry_id -> (date, status) mapping
    entry_info = {e.entry_id: (e.date, e.status) for e in all_entries}

    # Accumulate per-period debits/credits by account
    period_movements: dict[tuple[str, str], tuple[Decimal, Decimal]] = {}

    for line in all_lines:
        info = entry_info.get(line.entry_id)
        if info is None:
            continue
        entry_date, status = info
        if status != JournalStatus.POSTED:
            continue

        period_str = entry_date.strftime("%Y-%m")
        key = (line.account_id, period_str)
        prev_d, prev_c = period_movements.get(key, (Decimal("0"), Decimal("0")))
        period_movements[key] = (prev_d + line.debit, prev_c + line.credit)

    # Build per-period activity rows for every (account, period)
    # that has any movement — including entries past fiscal year end.
    result: list[TrialBalance] = []

    for (acct, period_str), (month_d, month_c) in sorted(period_movements.items()):
        if month_d == 0 and month_c == 0:
            continue
        result.append(TrialBalance(
            account_id=acct,
            period=period_str,
            debit_balance=_quantize(month_d),
            credit_balance=_quantize(month_c),
        ))

    return result


def _derive_balance_sheet(
    all_entries: list[JournalEntry],
    all_lines: list[JournalLine],
    chart: list[ChartOfAccounts],
) -> list[BalanceSheet]:
    """Build point-in-time balances for balance-sheet accounts — a STOCK.

    For each balance-sheet account (asset/liability/equity) the ``ending_balance``
    is the running cumulative net movement (debit − credit), carried forward across
    EVERY period from the account's first activity onward — including periods with
    no movement (the balance persists). This is the carry-forward level that makes
    the column a stock, the structural opposite of ``trial_balance`` (per-period flow).
    """
    balance_sheet_accounts = {
        c.account_id
        for c in chart
        if c.account_type in (AccountType.ASSET, AccountType.LIABILITY, AccountType.EQUITY)
    }

    entry_info = {e.entry_id: (e.date, e.status) for e in all_entries}

    # Per-period NET movement (debit − credit) by account, posted entries only.
    period_net: dict[tuple[str, str], Decimal] = {}
    all_periods: set[str] = set()
    for line in all_lines:
        info = entry_info.get(line.entry_id)
        if info is None:
            continue
        entry_date, status = info
        if status != JournalStatus.POSTED:
            continue
        if line.account_id not in balance_sheet_accounts:
            continue
        period_str = entry_date.strftime("%Y-%m")
        all_periods.add(period_str)
        key = (line.account_id, period_str)
        period_net[key] = period_net.get(key, Decimal("0")) + line.debit - line.credit

    periods_sorted = sorted(all_periods)

    result: list[BalanceSheet] = []
    for acct in sorted(balance_sheet_accounts):
        # Skip accounts with no activity at all.
        first_idx = next(
            (i for i, p in enumerate(periods_sorted) if (acct, p) in period_net),
            None,
        )
        if first_idx is None:
            continue
        running = Decimal("0")
        # Carry forward across every period from first activity onward (dense).
        for period_str in periods_sorted[first_idx:]:
            running += period_net.get((acct, period_str), Decimal("0"))
            result.append(BalanceSheet(
                account_id=acct,
                period=period_str,
                ending_balance=_quantize(running),
            ))

    return result


def _generate_measure_probes(
    months: int, fiscal_start: date, n_series: int
) -> list[MeasureProbe]:
    """Skeleton ``(series_id, period)`` grain for the stock/flow probe table (DAT-445).

    ``n_series`` synthetic series × ``months`` periods — the grain that
    ``inject_stock_flow_probes`` fills with stock/flow measure columns. Rows are in
    ``(series, period)`` order so a stock column's per-series running total accumulates
    in row order. Empty when ``n_series == 0`` (no stock/flow strategy active).
    """
    if n_series <= 0:
        return []
    periods = [_month_start_end(fiscal_start, m)[0].strftime("%Y-%m") for m in range(months)]
    return [
        MeasureProbe(series_id=f"S{s:03d}", period=period)
        for s in range(n_series)
        for period in periods
    ]


def _generate_relationship_probes(n_parents: int, n_children: int) -> tuple[list[RefEntity], list[RefActivity]]:
    """Skeleton grains for the relationship-pairs probe tables (DAT-408/450).

    ``n_parents`` parent rows + ``n_children`` child rows — the grains that
    ``inject_relationship_pairs`` fills with labelled FK/code column pairs
    (genuine clean / orphan-broken / spurious-overlap strata). Empty when either
    count is 0 (no relationship strategy active).
    """
    if n_parents <= 0 or n_children <= 0:
        return [], []
    parents = [RefEntity(entity_seq=f"RE-{i:04d}") for i in range(n_parents)]
    children = [RefActivity(activity_seq=f"RA-{i:05d}") for i in range(n_children)]
    return parents, children


# --- Main Generator ---

def generate_finance_dataset(
    seed: int = 42,
    months: int = 12,
    fiscal_start: date | None = None,
    invoices_count: int | None = None,
    q4_seasonal_boost: float | None = None,
    probe_series: int = 0,
    relation_parents: int = 0,
    relation_children: int = 0,
    **_kwargs: object,
) -> FinanceDataset:
    """Generate a complete finance dataset with closed-loop accounting.

    Business events cascade across tables: sales create GL entries and AR,
    payments settle AP and create bank transactions, trial balance is derived
    from actual cumulative GL entries.

    Args:
        seed: Random seed for reproducibility.
        months: Number of months to generate (fiscal year).
        fiscal_start: First day of the fiscal year.
        invoices_count: Number of vendor/purchase invoices.
        q4_seasonal_boost: Fractional boost applied to Q4 revenue months.

    Returns:
        FinanceDataset with all 8 tables populated and numerically consistent.
    """
    rng = random.Random(seed)
    if fiscal_start is None:
        fiscal_start = date(2025, 1, 1)

    q4_boost = q4_seasonal_boost if q4_seasonal_boost is not None else 0.3
    inv_count = invoices_count if invoices_count is not None else 3000

    counters = _Counters()

    # Static tables
    chart = generate_chart_of_accounts()
    fx_rates = _generate_fx_rates(rng, fiscal_start, months)

    # ── Revenue cycle ──
    # Sales → GL entries (DR: AR, CR: Revenue)
    sale_entries, sale_lines, sale_records = _generate_revenue_entries(
        rng, fiscal_start, months, counters,
        sales_per_month=250,
        q4_seasonal_boost=q4_boost,
    )

    # Cash receipts → GL + Bank (DR: Cash, CR: AR)
    receipt_entries, receipt_lines, receipt_bank = _generate_cash_receipts(
        rng, sale_records, fiscal_start, months, counters,
    )

    # ── Expenditure cycle ──
    # Purchase invoices → Invoice + GL (DR: Expense, CR: AP)
    invoices, inv_entries, inv_lines = _generate_purchase_invoices(
        rng, fiscal_start, months, counters,
        count=inv_count,
    )

    # Vendor payments → Payment + GL + Bank (DR: AP, CR: Cash)
    payments, pay_entries, pay_lines, pay_bank = _generate_vendor_payments(
        rng, invoices, counters,
    )

    # ── Operating events ──
    # Payroll, depreciation, rent, misc expenses
    op_entries, op_lines, op_bank = _generate_operating_events(
        rng, fiscal_start, months, counters,
    )

    # ── Misc bank transactions ──
    # Interest, fees, unmatched items (with corresponding GL)
    misc_entries, misc_lines, misc_bank = _generate_misc_bank_transactions(
        rng, fiscal_start, months, counters,
        count_per_month=20,
    )

    # ── Assembly ──
    all_entries = sale_entries + receipt_entries + inv_entries + pay_entries + op_entries + misc_entries
    all_lines = sale_lines + receipt_lines + inv_lines + pay_lines + op_lines + misc_lines
    all_bank = receipt_bank + pay_bank + op_bank + misc_bank

    # Sort by date for chronological ordering
    all_entries.sort(key=lambda e: (e.date, e.entry_id))
    all_bank.sort(key=lambda t: (t.date, t.txn_id))

    # Derive trial balance (per-period movement, a flow) from actual GL entries
    trial_bal = _derive_trial_balance(all_entries, all_lines, fiscal_start, months)
    # Derive balance sheet (carry-forward ending balance, a stock) for BS accounts
    balance_sheet = _derive_balance_sheet(all_entries, all_lines, chart)
    # Skeleton grain for the stock/flow probe table — empty unless a strategy needs it.
    measure_probes = _generate_measure_probes(months, fiscal_start, probe_series)
    # Skeleton grains for the relationship probe tables — empty unless a strategy needs them.
    ref_entities, ref_activity = _generate_relationship_probes(relation_parents, relation_children)

    return FinanceDataset(
        chart_of_accounts=chart,
        journal_entries=all_entries,
        journal_lines=all_lines,
        invoices=invoices,
        payments=payments,
        bank_transactions=all_bank,
        fx_rates=fx_rates,
        trial_balance=trial_bal,
        balance_sheet=balance_sheet,
        measure_probes=measure_probes,
        ref_entities=ref_entities,
        ref_activity=ref_activity,
    )
