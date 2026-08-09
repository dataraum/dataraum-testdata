"""Pydantic models for canonical finance data (zero entropy baseline)."""

from __future__ import annotations

import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, computed_field

from testdata.families import all_tables


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


class InvoiceCategory(StrEnum):
    """What a vendor bill bought — the split DPO's denominator depends on.

    ``goods`` bills debit Inventory and are the *purchases* a textbook DPO divides by;
    ``expense`` bills debit an expense account directly. Before the inventory family
    the corpus had only the second kind, so "purchases" was not a computable quantity
    and every DPO had to fall back on total expenses.
    """

    EXPENSE = "expense"
    GOODS = "goods"


class StockMovementType(StrEnum):
    RECEIPT = "receipt"
    ISSUE = "issue"
    ADJUSTMENT = "adjustment"


class ChartOfAccounts(BaseModel):
    account_id: str = Field(description="Unique account identifier, e.g. '1000'")
    name: str = Field(description="Human-readable account name")
    account_type: AccountType
    parent_id: str | None = Field(default=None, description="Parent account ID for hierarchy")
    currency: Currency = Currency.USD
    # Unique per account BY CONSTRUCTION, which makes account_id <-> opened_date a
    # bijection. That is the point: once CoA is inlined (`flat`), both columns repeat
    # together on the fact grain, so the pair is a NON-KEY bijection — statistically
    # indistinguishable from the true account_id <-> account_name alias. It is not an
    # alias: it is an attribute OF the account. Collapsing the two would fuse "which
    # account" with "when was it opened" into one drill axis. Only meaning separates
    # them, so this column is the corpus's coincidental-bijection case — the negative
    # half the dimension-identity judge has to reject.
    opened_date: datetime.date = Field(description="Date the account was opened in the ledger")


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
    category: InvoiceCategory = Field(
        default=InvoiceCategory.EXPENSE,
        description="What the bill bought — goods bills are the purchases DPO divides by",
    )
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


class MeasureProbe(BaseModel):
    """Skeleton grain for the stock/flow probe table.

    A synthetic measures-over-time table: ``series_id`` × ``period`` rows that
    ``inject_stock_flow_probes`` fills with clearly-named stock/flow measure columns
    (the corpus for the temporal_behavior ``llm_claim`` witness + recall). Generated
    only when a strategy injects into it; empty otherwise.
    """

    series_id: str = Field(description="Probe series identifier")
    period: str = Field(description="Period identifier, e.g. '2025-01'")


class FormulaProbe(BaseModel):
    """Skeleton grain for the formula-divergence probe table.

    A synthetic per-record table: ``probe_id`` rows that ``inject_formula_divergence``
    fills with labelled (source_a, source_b, target) formula groups — the calibration
    corpus for the derived_value witnesses (formula_discovery + llm_hypothesis).
    Generated only when a strategy injects into it; empty otherwise.
    """

    probe_id: str = Field(description="Probe record identifier")


class RefEntity(BaseModel):
    """Skeleton row for the relationship-pairs PARENT probe table.

    A reference/master grain: one row per entity that ``inject_relationship_pairs``
    fills with labelled key/code columns (the genuine-FK pools and the spurious
    shared-code pools). Generated only when a strategy injects into the child
    probe table; empty otherwise.
    """

    entity_seq: str = Field(description="Probe parent row identifier")


class RefActivity(BaseModel):
    """Skeleton row for the relationship-pairs CHILD probe table.

    An activity grain referencing :class:`RefEntity` pools: one row per event that
    ``inject_relationship_pairs`` fills with labelled FK/code columns across the
    clean / orphan-broken / spurious-overlap strata. Generated only when a
    strategy injects into it; empty otherwise.
    """

    activity_seq: str = Field(description="Probe child row identifier")


class Address(BaseModel):
    """Dimension for the role-playing-FK probe shape.

    One address master that TWO differently-roled FK columns on the same fact
    reference (``orders.bill_to_addr`` + ``orders.ship_to_addr`` — the two-FKs-one-table-pair shape), plus a second fact sharing the ship_to role
    under a DIFFERENT column name (``deliveries.delivery_addr`` — the judge's
    conform path). Role columns are filled by ``inject_role_playing_fks``.
    Generated only when a strategy injects into the role-play facts; empty otherwise.
    """

    address_id: str = Field(description="Unique address identifier")
    street: str = Field(description="Street line")
    city: str = Field(description="City name")
    country: str = Field(description="Country code")


class Order(BaseModel):
    """Skeleton grain for the role-play fact with two FK roles.

    ``order_id`` × ``order_date`` rows; ``inject_role_playing_fks`` adds the
    ``bill_to_addr`` / ``ship_to_addr`` FK-role columns (both → addresses).
    Generated only when a strategy injects into it; empty otherwise.
    """

    order_id: str = Field(description="Unique order identifier")
    order_date: datetime.date


class Delivery(BaseModel):
    """Skeleton grain for the second role-play fact sharing the dimension.

    ``inject_role_playing_fks`` adds ``delivery_addr`` (→ addresses, ship_to ROLE
    under a different name — role-consistent with the parent order's ship_to by
    construction). Generated only when a strategy injects into the role-play facts.
    """

    delivery_id: str = Field(description="Unique delivery identifier")
    order_id: str = Field(description="FK to Order")
    delivery_date: datetime.date


class Customer(BaseModel):
    """A selling-side counterparty — the Demand dimension's canonical entity.

    Distinct from the role-play probe dimension (:class:`Address`): this is real
    operating master data every sales order references, present on every corpus.
    """

    customer_id: str = Field(description="Unique customer identifier")
    name: str = Field(description="Customer legal name")
    segment: str = Field(description="Commercial segment — the Demand ladder's rung")
    region: str = Field(description="Sales region")
    payment_terms: PaymentTerms = Field(description="Agreed settlement terms")
    # The validity window. A master table with no lifecycle cannot express churn, and
    # a corpus without churn never poses the case prior-period and peer comparisons
    # actually fail on: a customer whose "decline" is that they did not exist yet.
    created_date: datetime.date = Field(description="When the account was opened")
    churned_date: datetime.date | None = Field(default=None, description="When the account lapsed, if it did")


class Product(BaseModel):
    """A sellable item with a STANDARD COST — the Offer dimension's canonical entity.

    ``standard_cost`` is what makes gross profit and DB1 real: before this the corpus
    had no cost-of-sale relationship at all (COGS was a random slice of vendor
    purchases), so every margin metric was ungradeable in principle. Cost of sale is
    now ``units × standard_cost``, derivable per line and therefore per customer and
    per product group.
    """

    product_id: str = Field(description="Unique product identifier")
    name: str = Field(description="Product name")
    product_group: str = Field(description="Product group — the Offer ladder's rung")
    standard_cost: Decimal = Field(description="Standard cost per unit (the DB1 cost side)")
    list_price: Decimal = Field(description="List price per unit before discount")
    # Same reason as the customer window: a portfolio with no launches and no
    # discontinuations makes every year-over-year product comparison trivially valid.
    launched_date: datetime.date = Field(description="When the item became sellable")
    discontinued_date: datetime.date | None = Field(default=None, description="When it left the catalogue, if it did")


class SalesOrder(BaseModel):
    """A customer order header (the documented ERP sales-order shape)."""

    order_id: str = Field(description="Unique sales order identifier")
    customer_id: str = Field(description="FK to Customer")
    order_date: datetime.date
    status: str = Field(description="Order status")


class SalesOrderLine(BaseModel):
    """One ordered product at a quantity and a price — the grain revenue derives from.

    ``line_amount == units × unit_price`` and ``line_cost == units × standard_cost``
    BY CONSTRUCTION, so the GL revenue and COGS legs are derived rather than drawn,
    and DB1 per customer / per product group is exact.
    """

    order_line_id: str = Field(description="Unique order line identifier")
    order_id: str = Field(description="FK to SalesOrder")
    product_id: str = Field(description="FK to Product")
    units: int = Field(description="Quantity ordered")
    unit_price: Decimal = Field(description="Agreed price per unit")
    line_amount: Decimal = Field(description="units × unit_price")
    line_cost: Decimal = Field(description="units × product standard_cost")


class ARInvoice(BaseModel):
    """A CUSTOMER-side invoice — the AR half the corpus never had.

    ``invoices`` is vendor-side only, so days-sales-outstanding had no receivable
    document to measure against and the AR half of Capital was invisible. One AR
    invoice per sales order, amount tied to the order's revenue posting.
    """

    ar_invoice_id: str = Field(description="Unique AR invoice identifier")
    order_id: str = Field(description="FK to SalesOrder")
    customer_id: str = Field(description="FK to Customer")
    invoice_date: datetime.date
    due_date: datetime.date
    amount: Decimal = Field(description="Invoiced amount — the order's revenue")
    currency: Currency = Currency.USD
    status: InvoiceStatus = Field(description="Settlement status at fiscal close")


class Receipt(BaseModel):
    """Customer cash in against an AR invoice — the collection event DSO measures."""

    receipt_id: str = Field(description="Unique receipt identifier")
    ar_invoice_id: str = Field(description="FK to ARInvoice")
    customer_id: str = Field(description="FK to Customer")
    receipt_date: datetime.date
    amount: Decimal = Field(description="Amount collected (may be partial)")
    currency: Currency = Currency.USD
    method: PaymentMethod = Field(description="How the money arrived")


class StockMovement(BaseModel):
    """One movement of stock — the subledger under GL account 1400.

    ``units`` and ``value`` are **signed**: a receipt is positive, an issue negative,
    an adjustment either. The sign convention is the movement type's arithmetic, and
    it is what makes the table a genuine additive flow — the roll-forward
    ``opening + Σ units = closing`` is a plain sum, not a case expression over
    ``movement_type``. ``unit_cost`` stays positive; it is a rate, not a movement.

    Every movement carries the document it came from (``source_document``: the order
    line that issued it, the vendor bill that received it, the count that adjusted it)
    and the GL entry that posted it, so the subledger ties to the ledger both ways.
    """

    movement_id: str = Field(description="Unique movement identifier")
    product_id: str = Field(description="FK to Product")
    location_id: str = Field(description="Stocking location")
    date: datetime.date
    movement_type: StockMovementType
    units: int = Field(description="Signed quantity — receipt positive, issue negative")
    unit_cost: Decimal = Field(description="Standard cost per unit (positive)")
    value: Decimal = Field(description="units × unit_cost — signed, same sign as units")
    source_document: str = Field(description="The order line / vendor bill / count behind it")
    entry_id: str = Field(description="FK to JournalEntry")


class InventoryPosition(BaseModel):
    """Closing stock at ``(product_id, location_id, period)`` — a STOCK, never summed.

    The counterpart of :class:`TrialBalance` for goods: ``units_on_hand`` is a level
    carried forward, so summing it across periods is the error the temporal-behaviour
    truth exists to catch. Valued at standard cost, so Σ ``value`` over a period equals
    the GL balance of account 1400 exactly — the subledger-to-ledger tie.
    """

    product_id: str = Field(description="FK to Product")
    location_id: str = Field(description="Stocking location")
    period: str = Field(description="Period identifier, e.g. '2025-01'")
    units_on_hand: int = Field(description="Closing quantity — a level, not a movement")
    unit_cost: Decimal = Field(description="Standard cost per unit")
    value: Decimal = Field(description="units_on_hand × unit_cost")


# --- the corpus container, one fragment per family --------------------------
#
# One flat container of eight-plus lists was the shape before: a family's tables were
# appended to a list that grew with every release and belonged to nobody, so nothing
# connected `stock_movements` to the family that declares it in `families.py`. Each
# fragment below carries exactly one family's tables and ``test_families_registry``
# asserts the two agree, which is how a family that declares a table without giving it a
# home — or the reverse — fails immediately rather than at the first key transform.


class CoreLedgerTables(BaseModel):
    """`core_ledger` — the double-entry ledger and its subledgers."""

    chart_of_accounts: list[ChartOfAccounts]
    journal_entries: list[JournalEntry]
    journal_lines: list[JournalLine]
    invoices: list[Invoice]
    payments: list[Payment]
    bank_transactions: list[BankTransaction]
    fx_rates: list[FXRate]
    trial_balance: list[TrialBalance]
    balance_sheet: list[BalanceSheet]


class OperatingChainTables(BaseModel):
    """`operating_chain` — present on EVERY corpus, unlike the probe shapes."""

    customers: list[Customer] = []
    products: list[Product] = []
    sales_orders: list[SalesOrder] = []
    sales_order_lines: list[SalesOrderLine] = []
    ar_invoices: list[ARInvoice] = []
    receipts: list[Receipt] = []


class InventoryTables(BaseModel):
    """`inventory` — the stock subledger under GL 1400."""

    stock_movements: list[StockMovement] = []
    inventory_positions: list[InventoryPosition] = []


class ProbeTables(BaseModel):
    """`probes` — labelled shapes, materialized only when a strategy asks for them."""

    measure_probes: list[MeasureProbe] = []
    formula_probes: list[FormulaProbe] = []
    ref_entities: list[RefEntity] = []
    ref_activity: list[RefActivity] = []
    addresses: list[Address] = []
    orders: list[Order] = []
    deliveries: list[Delivery] = []


class Corpus(CoreLedgerTables, OperatingChainTables, InventoryTables, ProbeTables):
    """One corpus: every family's tables, over one key space and one calendar.

    Composed from the family fragments rather than accumulating their fields, so adding
    a family is writing its fragment and listing it here — not appending to a list that
    belongs to nobody. Attribute access stays typed (``corpus.journal_lines`` is
    ``list[JournalLine]``), which is what a table-keyed ``dict[str, list[BaseModel]]``
    would have cost: every consumer would then need a cast to touch a column.

    ``tables`` is the untyped view the generic consumers want — the exporter and the
    schema transforms work table-by-table and should never learn a table's name.
    """

    @property
    def tables(self) -> dict[str, list[Any]]:
        """``{table name: rows}`` for every declared table, in family order."""
        return {name: getattr(self, name) for name in all_tables() if hasattr(self, name)}
