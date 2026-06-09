"""Pydantic models for canonical finance data (zero entropy baseline)."""

from __future__ import annotations

import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field


class AccountType(StrEnum):
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    REVENUE = "revenue"
    EXPENSE = "expense"


class JournalStatus(StrEnum):
    DRAFT = "draft"
    POSTED = "posted"
    REVERSED = "reversed"


class InvoiceStatus(StrEnum):
    OPEN = "open"
    PAID = "paid"
    PARTIAL = "partial"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class PaymentMethod(StrEnum):
    WIRE = "wire"
    ACH = "ach"
    CHECK = "check"
    CREDIT_CARD = "credit_card"


class Currency(StrEnum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CHF = "CHF"
    JPY = "JPY"


class PaymentTerms(StrEnum):
    NET_30 = "net_30"
    NET_60 = "net_60"
    NET_90 = "net_90"
    DUE_ON_RECEIPT = "due_on_receipt"


class ChartOfAccounts(BaseModel):
    account_id: str = Field(description="Unique account identifier, e.g. '1000'")
    name: str = Field(description="Human-readable account name")
    account_type: AccountType
    parent_id: str | None = Field(default=None, description="Parent account ID for hierarchy")
    currency: Currency = Currency.USD


class JournalEntry(BaseModel):
    entry_id: str = Field(description="Unique journal entry identifier")
    date: datetime.date
    description: str
    status: JournalStatus = JournalStatus.POSTED
    created_by: str


class JournalLine(BaseModel):
    line_id: str = Field(description="Unique line identifier")
    entry_id: str = Field(description="FK to JournalEntry")
    account_id: str = Field(description="FK to ChartOfAccounts")
    debit: Decimal = Field(default=Decimal("0.00"))
    credit: Decimal = Field(default=Decimal("0.00"))
    currency: Currency = Currency.USD
    cost_center: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def net_amount(self) -> Decimal:
        """Derived column: debit - credit. Enables correlation detection."""
        return self.debit - self.credit


class Invoice(BaseModel):
    invoice_id: str
    vendor_id: str
    date: datetime.date
    due_date: datetime.date
    amount: Decimal
    currency: Currency = Currency.USD
    status: InvoiceStatus = InvoiceStatus.OPEN
    payment_terms: PaymentTerms = PaymentTerms.NET_30
    entry_id: str | None = Field(default=None, description="FK to JournalEntry (None for cancelled)")


class Payment(BaseModel):
    payment_id: str
    invoice_id: str = Field(description="FK to Invoice")
    date: datetime.date
    amount: Decimal
    currency: Currency = Currency.USD
    method: PaymentMethod = PaymentMethod.WIRE


class BankTransaction(BaseModel):
    txn_id: str
    account_id: str = Field(description="FK to ChartOfAccounts (bank/cash account)")
    date: datetime.date
    amount: Decimal = Field(description="Positive=credit, negative=debit")
    currency: Currency = Currency.USD
    reference: str
    counterparty: str
    reconciled: bool = False
    payment_id: str | None = Field(default=None, description="FK to Payment (vendor payments only)")


class FXRate(BaseModel):
    from_ccy: Currency
    to_ccy: Currency
    date: datetime.date
    rate: Decimal
    source: str = "ECB"


class TrialBalance(BaseModel):
    account_id: str = Field(description="FK to ChartOfAccounts")
    period: str = Field(description="Period identifier, e.g. '2025-01'")
    debit_balance: Decimal = Field(default=Decimal("0.00"))
    credit_balance: Decimal = Field(default=Decimal("0.00"))


class BalanceSheet(BaseModel):
    """Point-in-time net balances for balance-sheet accounts — a STOCK.

    Unlike ``TrialBalance`` (per-period movement, a flow), ``ending_balance`` is a
    carry-forward level: ``ending_balance[period] = ending_balance[period-1] +
    net movement[period]`` (cumulative debit − credit). It persists across
    no-activity periods. Balance-sheet accounts only (asset/liability/equity).
    """

    account_id: str = Field(description="FK to ChartOfAccounts (balance-sheet accounts only)")
    period: str = Field(description="Period identifier, e.g. '2025-01'")
    ending_balance: Decimal = Field(
        default=Decimal("0.00"),
        description="Cumulative net balance (debit − credit) carried forward across periods",
    )


class FinanceDataset(BaseModel):
    """Container for a complete finance dataset."""

    chart_of_accounts: list[ChartOfAccounts]
    journal_entries: list[JournalEntry]
    journal_lines: list[JournalLine]
    invoices: list[Invoice]
    payments: list[Payment]
    bank_transactions: list[BankTransaction]
    fx_rates: list[FXRate]
    trial_balance: list[TrialBalance]
    balance_sheet: list[BalanceSheet]
