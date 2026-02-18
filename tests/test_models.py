"""Tests for canonical finance Pydantic models."""

from datetime import date
from decimal import Decimal

from testdata.canonical.finance.models import (
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


def test_chart_of_accounts_defaults():
    acct = ChartOfAccounts(account_id="1000", name="Cash", account_type=AccountType.ASSET)
    assert acct.parent_id is None
    assert acct.currency == Currency.USD


def test_journal_entry_defaults():
    entry = JournalEntry(
        entry_id="JE-001", date=date(2025, 1, 15),
        description="Test", created_by="user",
    )
    assert entry.status == JournalStatus.POSTED


def test_journal_line_debit_credit():
    line = JournalLine(
        line_id="JL-001", entry_id="JE-001", account_id="1000",
        debit=Decimal("100.00"), credit=Decimal("0.00"),
    )
    assert line.debit == Decimal("100.00")
    assert line.credit == Decimal("0.00")
    assert line.cost_center is None


def test_invoice_with_terms():
    inv = Invoice(
        invoice_id="INV-001", vendor_id="V-001",
        date=date(2025, 3, 1), due_date=date(2025, 3, 31),
        amount=Decimal("5000.00"), payment_terms=PaymentTerms.NET_30,
    )
    assert inv.status == InvoiceStatus.OPEN
    assert inv.currency == Currency.USD


def test_payment():
    pay = Payment(
        payment_id="PAY-001", invoice_id="INV-001",
        date=date(2025, 4, 1), amount=Decimal("5000.00"),
        method=PaymentMethod.WIRE,
    )
    assert pay.currency == Currency.USD


def test_bank_transaction():
    txn = BankTransaction(
        txn_id="BT-001", date=date(2025, 1, 5),
        amount=Decimal("-1500.00"), reference="TXN-123456",
        counterparty="Acme Corp",
    )
    assert txn.reconciled is False
    assert txn.amount < 0


def test_fx_rate():
    rate = FXRate(
        from_ccy=Currency.EUR, to_ccy=Currency.USD,
        date=date(2025, 1, 1), rate=Decimal("1.10"),
    )
    assert rate.source == "ECB"


def test_trial_balance():
    tb = TrialBalance(
        account_id="1000", period="2025-01",
        debit_balance=Decimal("10000.00"), credit_balance=Decimal("0.00"),
    )
    assert tb.debit_balance > tb.credit_balance


def test_finance_dataset_container():
    ds = FinanceDataset(
        chart_of_accounts=[], journal_entries=[], journal_lines=[],
        invoices=[], payments=[], bank_transactions=[],
        fx_rates=[], trial_balance=[],
    )
    assert len(ds.chart_of_accounts) == 0
