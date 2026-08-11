"""Seed-based deterministic generators for canonical finance data.

Event-driven architecture: business events cascade across tables to produce
a closed-loop accounting system where GL entries, invoices, payments, bank
transactions, and trial balance are numerically consistent.
"""

from __future__ import annotations

import bisect
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from testdata.scale import (
    MERCHANT_CATEGORIES,
    MERCHANT_COUNTRIES,
    ScaleProfile,
    customer_names,
    get_profile,
    merchant_names,
    product_catalog,
    supplier_names,
)

from .models import (
    AccountType,
    Address,
    ARInvoice,
    BalanceSheet,
    BankTransaction,
    ChartOfAccounts,
    Currency,
    Customer,
    Delivery,
    Corpus,
    FormulaProbe,
    FXRate,
    InventoryPosition,
    Invoice,
    InvoiceCategory,
    InvoiceStatus,
    JournalEntry,
    JournalLine,
    JournalStatus,
    MeasureProbe,
    Merchant,
    Order,
    Payment,
    PaymentMethod,
    PaymentTerms,
    Product,
    Receipt,
    RefActivity,
    RefEntity,
    SalesOrder,
    SalesOrderLine,
    StockMovement,
    StockMovementType,
    TrialBalance,
)

# --- Constants ---

COST_CENTERS = ["CC100", "CC200", "CC300", "CC400", "CC500"]
USERS = ["jsmith", "mwilson", "agarcia", "ljohnson", "klee", "rbrown"]

# Where the operating-expense budget goes. Shares sum to 1.0 and are a *declared* cost
# structure, not one fitted to anything — which is the point: the total is now a
# function of what the firm contributes (§9's scale anchor), and this says how it is
# spent. Before, each of these was drawn from a fixed band with no reference to the
# size of the business, so the P&L sign was an artifact of a row count (§7).
_OPEX_ALLOCATION: dict[str, float] = {
    "vendor_bills": 0.42,
    "payroll": 0.40,
    "rent": 0.06,
    "depreciation": 0.05,
    "insurance": 0.03,
    "misc": 0.04,
}

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


def _month_offset(fiscal_start: date, when: date) -> int:
    """How many whole months *when* sits after the fiscal start."""
    return (when.year - fiscal_start.year) * 12 + when.month - fiscal_start.month


def _month_start_end(fiscal_start: date, month_offset: int) -> tuple[date, date]:
    """Return (first_day, last_day) for fiscal_start + month_offset months."""
    year = fiscal_start.year + (fiscal_start.month + month_offset - 1) // 12
    month = (fiscal_start.month + month_offset - 1) % 12 + 1
    start = date(year, month, 1)
    end = _month_end(year, month)
    return start, end


# --- Entity-keyed RNG streams ---


def _stream(seed: int, *key: object) -> random.Random:
    """A random stream keyed by stable entity IDENTITY, not by draw order.

    The operating chain draws only from these, never from the sequential
    ``rng`` the ledger cycles share. That is what makes a **volume** lever an exact
    same-seed counterfactual.

    Why it is load-bearing. The existing ``price_level`` lever is exact for a narrow
    reason: it rescales an amount *after* every draw and nothing downstream branches
    on the value, so the sequential stream is untouched (see :class:`Lever`). A volume
    lever changes how many events exist, so on one sequential stream run B would draw
    a different number of randoms than run A and every subsequent value would shift —
    the two corpora would differ everywhere, and the difference would no longer be
    attributable to the intervention. A counterfactual that is only approximately the
    same everywhere else is not ground truth; it is noise with a story.

    Keyed by identity, order *i* of a (customer, month) draws from
    ``_stream(seed, "order", customer_id, month, i)`` regardless of how many orders
    that month holds. So for a volume factor > 1 the baseline's orders are a strict
    **subset** of the levered run's, byte-identical, plus the added ones — and the
    added rows' revenue, COGS and DB1 contribution are computable to the cent.

    Same-key determinism is the whole contract, so the key must contain every
    coordinate that distinguishes the entity — a missing coordinate silently makes
    two different entities share a stream. Precedent: the string-keyed
    ``random.Random(f"null_token_family:{col}:{seed}")`` pattern in
    ``entropy/injectors.py``, which exists for the same order-independence reason.
    """
    return random.Random(":".join(str(k) for k in (*key, seed)))


# --- Counters ---


@dataclass
class _Counters:
    """Mutable counters for sequential ID generation across event types."""

    entry: int = 0
    line: int = 0
    invoice: int = 0
    payment: int = 0
    bank_txn: int = 0
    movement: int = 0

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

    def next_movement(self) -> str:
        self.movement += 1
        return f"STK-{self.movement:07d}"


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
    # A sibling of COGS, not a child of it: posting to a parent account while its
    # children also carry postings is a real ERP shape but a muddy one, and the
    # hierarchy truth would stop being a clean tree. Shrinkage is where the physical
    # count disagrees with the book — the only inventory expense that is not a sale.
    ("5150", "Inventory Shrinkage", AccountType.EXPENSE, "5000"),
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


_COA_EPOCH = date(2015, 1, 5)
_COA_OPEN_STRIDE_DAYS = 31


def generate_chart_of_accounts() -> list[ChartOfAccounts]:
    """The chart of accounts — static tree, no RNG.

    ``opened_date`` walks a fixed stride from ``_COA_EPOCH``, so it is strictly
    increasing and therefore UNIQUE across the tree by construction. Uniqueness is
    load-bearing, not decorative: a random date would collide (60 accounts over a few
    years is squarely in birthday-paradox territory) and a single collision breaks the
    account_id <-> opened_date bijection the identity judge is meant to be tested on.
    See ``ChartOfAccounts.opened_date`` for why that bijection matters.
    """
    return [
        ChartOfAccounts(
            account_id=aid,
            name=name,
            account_type=atype,
            parent_id=parent,
            opened_date=_COA_EPOCH + timedelta(days=_COA_OPEN_STRIDE_DAYS * i),
        )
        for i, (aid, name, atype, parent) in enumerate(_ACCOUNT_TREE)
    ]


# --- Revenue Cycle: Sales → Cash Receipts ---


# The dimensions a lever's scope may name, split by which entity carries them. The
# split is not cosmetic: an order COUNT is a property of (customer, month) and no
# product exists at the point it is drawn, so a product-side scope on a volume lever
# is not a narrower intervention — it is an unanswerable one, and it raises.
_CUSTOMER_SCOPE_DIMS = ("segment", "region", "customer_id")
_PRODUCT_SCOPE_DIMS = ("product_group", "product_id")
_SCOPE_DIMS = _CUSTOMER_SCOPE_DIMS + _PRODUCT_SCOPE_DIMS

_LEVER_TYPES = ("price_level", "volume", "rate", "mix")
_DRIVERS = ("price", "frequency", "collection_lag")
# `price_level` and `volume` predate the typed set and stay valid: they are these two
# drivers under their original names, so existing runs keep working and the generator
# still tests one vocabulary.
_LEGACY_DRIVER = {"price_level": "price", "volume": "frequency"}


@dataclass(frozen=True)
class _Scope:
    """A lever's scope resolved against the master data, as id sets.

    Resolved ONCE per run rather than per line: at ``large`` the alternative is a
    dict walk per order line for an answer that depends only on the entity. ``None``
    on a side means that side is unscoped — every entity matches — which is what
    keeps the unscoped path free of a membership test it would always pass.
    """

    customer_ids: frozenset[str] | None
    product_ids: frozenset[str] | None

    def covers_customer(self, customer_id: str) -> bool:
        return self.customer_ids is None or customer_id in self.customer_ids

    def covers_product(self, product_id: str) -> bool:
        return self.product_ids is None or product_id in self.product_ids


_UNSCOPED = _Scope(customer_ids=None, product_ids=None)

_ONE = Decimal("1")


@dataclass(frozen=True)
class _Factors:
    """What a lever multiplies by, per entity — one number, or one per member.

    A homogeneous lever moves its whole slice as a block, which answers "did this
    slice move". A HETEROGENEOUS one moves each member by its own declared amount,
    which is what makes "the aggregate moved, but not uniformly, and here is who
    moved how much" a question with an answer key rather than an inference. A single
    aggregate delta is consistent with infinitely many per-member stories; declaring
    the story is the only way to grade an attribution claim.

    Resolved to entity ids once per run, and looked up through one method whichever
    spelling was given, so the generation loop never branches on which it got and the
    homogeneous path stays byte-identical to the one that predates this.
    """

    uniform: Decimal
    by_member: Mapping[str, Decimal] | None = None
    side: str = ""  # "customer" or "product" — which id keys `by_member`

    def of(self, customer_id: str, product_id: str = "") -> Decimal:
        if self.by_member is None:
            return self.uniform
        return self.by_member.get(customer_id if self.side == "customer" else product_id, _ONE)

    def num(self, customer_id: str, product_id: str = "") -> float:
        """The same factor where the caller works in floats (counts, day lags)."""
        return float(self.of(customer_id, product_id))


_UNIFORM_ONE = _Factors(uniform=_ONE)


def _resolve_factors(
    lever: "Lever | None", customers: list[Customer], products: list[Product], scope: _Scope
) -> _Factors:
    """Turn a lever's declared factor — scalar or per-member map — into an id lookup.

    Expanded against the RESOLVED scope, so a member's factor reaches exactly the
    entities the scope already admits: with ``scope={"segment": [...], "region":
    ["EMEA"]}`` and a factor keyed by segment, a non-EMEA enterprise account is out of
    scope and never consulted, rather than picking up its segment's factor.
    """
    if lever is None or lever.effective_driver == "share":
        return _UNIFORM_ONE
    if not isinstance(lever.factor, Mapping):
        return _Factors(uniform=Decimal(str(lever.factor)))

    dim = lever.factor_dimension
    table = {str(member): Decimal(str(value)) for member, value in lever.factor.items()}
    if dim in _CUSTOMER_SCOPE_DIMS:
        return _Factors(
            uniform=_ONE,
            by_member={
                c.customer_id: table[str(getattr(c, dim))]
                for c in customers
                if scope.covers_customer(c.customer_id) and str(getattr(c, dim)) in table
            },
            side="customer",
        )
    return _Factors(
        uniform=_ONE,
        by_member={
            p.product_id: table[str(getattr(p, dim))]
            for p in products
            if scope.covers_product(p.product_id) and str(getattr(p, dim)) in table
        },
        side="product",
    )


_TREND_DIMENSIONS = ("price", "volume")


@dataclass(frozen=True)
class Trend:
    """A secular drift — the firm growing, not the firm changing.

    A lever answers "did something happen at period k". A TREND is the control for
    that question: prices and volumes creep a few percent a year for no reason at all,
    every metric rises monotonically, and the honest answer to "what changed in
    September" is *nothing*. Without a corpus where drift is the whole story, a
    consumer that flags every upward line is indistinguishable from one that is right.

    Rates are ANNUAL and compound continuously across the year — month *m* carries
    ``(1 + rate) ** (m / 12)`` — so period 0 is undrifted and the year-end factor is
    exactly ``1 + rate``. Deterministic in the month, so it adds no draw of its own on
    the price side and leaves a lever's same-seed counterfactual exact: both runs of a
    pair carry the identical trend, and it cancels.

    A trend is NOT a lever: it has no activation period, no scope, and no
    counterfactual, because there is nothing to attribute. It is a property of the
    corpus, and the corpus stamp is where it is recorded.
    """

    price: float = 0.0
    volume: float = 0.0

    def __post_init__(self) -> None:
        for name in _TREND_DIMENSIONS:
            rate = getattr(self, name)
            if rate <= -1.0:
                raise ValueError(f"trend {name}={rate} would drive the corpus to zero or below")

    @property
    def active(self) -> bool:
        return self.price != 0.0 or self.volume != 0.0

    def price_at(self, month_offset: int) -> Decimal:
        return Decimal(str((1.0 + self.price) ** (month_offset / 12.0)))

    def volume_at(self, month_offset: int) -> float:
        return (1.0 + self.volume) ** (month_offset / 12.0)

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in _TREND_DIMENSIONS if getattr(self, name)}


_NO_TREND = Trend()


@dataclass(frozen=True)
class _Applied:
    """One lever resolved against the master data — scope and factors, ready to use.

    Levers compose by MULTIPLYING where they meet. Two price levers over overlapping
    slices scale the same line twice; a price lever and a collection-lag lever act at
    different sites and simply both happen. That is what makes an interaction pair
    possible: run A, run B, run A+B, and the combined corpus is not the sum of the
    two single ones, because the second lever acts on amounts the first already moved.
    """

    lever: "Lever"
    scope: _Scope
    factors: _Factors


def _resolve_levers(
    levers: Sequence["Lever"], customers: list[Customer], products: list[Product]
) -> list[_Applied]:
    """Resolve every lever once per run, in declaration order."""
    resolved = []
    for lever in levers:
        scope = _resolve_scope(lever, customers, products)
        resolved.append(_Applied(lever, scope, _resolve_factors(lever, customers, products, scope)))
    return resolved


def _normalise_levers(lever: "Lever | None", levers: "Sequence[Lever] | None") -> tuple["Lever", ...]:
    """One lever or many, onto one list — and refuse the combinations that lie.

    ``mix`` measures its baseline share by re-running the order draw with no lever at
    all. Another lever in the same run moves that draw, so the measured share would
    describe a corpus that never existed and ``intervention.yaml`` would publish it as
    truth. Refused rather than silently approximated.
    """
    if lever is not None and levers:
        raise ValueError("pass `lever` or `levers`, not both")
    out = tuple(levers) if levers else ((lever,) if lever is not None else ())
    if len(out) > 1 and any(item.effective_driver == "share" for item in out):
        raise ValueError(
            "a `mix` lever cannot share a run with another lever: its baseline share is "
            "measured against a no-lever draw that the other lever would move"
        )
    return out


def _resolve_scope(
    lever: "Lever | None", customers: list[Customer], products: list[Product]
) -> _Scope:
    """Turn a lever's declared scope into the id sets the generation loop tests.

    Membership is decided on entity IDENTITY, never on a drawn value, so scoping a
    lever does not introduce control flow that branches on anything the lever moved
    — the property that keeps a same-seed pair exact (see :class:`Lever`).
    """
    if lever is None or not lever.scope:
        return _UNSCOPED
    scope = {dim: frozenset(members) for dim, members in lever.scope.items()}

    def ids(entities: list[Customer] | list[Product], dims: tuple[str, ...], key: str) -> frozenset[str] | None:
        active = {dim: members for dim, members in scope.items() if dim in dims}
        if not active:
            return None
        # Several dimensions INTERSECT — `segment: [Enterprise], region: [EMEA]` is
        # enterprise accounts in EMEA, not their union.
        return frozenset(
            str(getattr(e, key))
            for e in entities
            if all(str(getattr(e, dim)) in members for dim, members in active.items())
        )

    return _Scope(
        customer_ids=ids(customers, _CUSTOMER_SCOPE_DIMS, "customer_id"),
        product_ids=ids(products, _PRODUCT_SCOPE_DIMS, "product_id"),
    )


@dataclass(frozen=True)
class Lever:
    """A constructed intervention: a DGP parameter change at a known period.

    Unlike entropy injectors (post-hoc corruption of generated frames), a lever
    changes the generating process itself, so the effect propagates naturally
    through the event cascade (sales → AR/revenue → receipts → cash/bank → TB/BS).

    ``price_level``: the realised ``unit_price`` of every order line in
    ``month_offset >= period_k`` is scaled by ``factor``, and ``line_amount`` follows
    from it. Scaling happens after all random draws for the event — after the discount
    draw, so the discount off list is what it was — and no downstream control flow
    branches on amount values, so the RNG stream is identical with and without the
    lever. A same-seed pair is an exact counterfactual: revenue in months >= period_k
    scales by exactly ``factor``; receipts/cash follow with the collection lag; cost
    of sale and the expenditure cycle are untouched (a price change does not move
    volumes or costs).

    It applies at the **draw site**, not at the GL posting. It used to scale the
    revenue credit inside ``_generate_revenue_entries`` and leave
    ``sales_order_lines`` at baseline, which desynchronised the order lines from the
    ledger they are supposed to derive: ``operating_revenue`` no longer reconstructed
    from the lines, and every entity-grain metric (``db1_by_customer``,
    ``db1_by_product_group``) read identical in the levered run and its baseline —
    the lever moved the aggregate and moved nothing at the grain "which slice drove
    it" is asked at.

    ``volume``: the number of orders a customer places in ``month_offset >=
    period_k`` is scaled by ``factor``. Exact for a different and stronger reason: order *i* of a (customer, month) draws from its own identity-keyed
    stream, so the baseline's orders are a strict SUBSET of the levered run's —
    byte-identical, plus the added ones. The ledger cycles never draw from those
    streams, so purchases, payroll and depreciation are untouched, and the added
    rows' revenue, cost of sale and DB1 contribution are computable to the cent.

    ``rate``: shift one named ``driver`` WITHIN a slice, holding member shares fixed —
    ``price`` (realised unit price), ``frequency`` (orders per customer-month) or
    ``collection_lag`` (days from sale to receipt). The aggregate then moves through
    behaviour rather than composition, which is the half of "artifact or real change"
    that ``mix`` is the other half of.

    ``mix``: shift the SHARE of order activity toward the scoped members, from 20% to
    ``target_share``, while holding every within-member rate fixed. The complement's
    factor is *derived* rather than declared, so total activity stays put and the
    aggregate moves purely through composition. Declaring only the target's factor
    would move total volume too, and the lever would be a frequency change wearing a
    mix label. Shares are of order COUNT, so the scope is customer-side only.

    ``price_level`` and ``volume`` remain valid and mean exactly ``rate``/``price``
    and ``rate``/``frequency``; ``driver`` normalises all four onto one code path.

    ``scope`` narrows a lever to a subset of entities — ``{"segment":
    ["Enterprise"], "region": ["EMEA"]}`` is enterprise accounts in EMEA, since
    several dimensions intersect rather than union. Omitted, the lever applies to
    everything, which is the behaviour every unscoped run keeps. Scoping is what
    makes "a metric moved — which slice drove it" answerable by construction: the
    aggregate moves, and exactly one named slice moved under it.

    Membership is decided on entity identity, so a scoped lever is exact for the same
    reason an unscoped one is. What it costs is the global subset property of the
    frequency driver: with a scope, the baseline's orders are a strict subset of the
    levered run's *within the scope* and byte-identical outside it. That is a
    sharper statement than the unscoped one, not a weaker one, but it is a different
    statement and ``intervention.yaml`` says so.

    ``factor`` may be a MAP instead of a number — ``{"Enterprise": 1.15, "Mid-Market":
    1.05}`` — and then each member moves by its own amount. Its keys must be exactly
    one scope dimension's members, so the dimension is derived rather than declared
    twice. This is what separates "the aggregate moved" from "and here is who moved
    how much": a single delta is consistent with infinitely many per-member stories,
    so a consumer that recovers the total has shown nothing about attribution. With a
    per-member factor the story is on disk and an attribution claim is gradeable —
    naming the wrong member as the driver is wrong even when the total is right.
    ``mix`` takes no factor at all; its knob is ``target_share``.
    """

    period_k: int  # month offset (0-based) at which the lever activates
    # Multiplicative change — a scalar (1.15, the whole slice moves together) or a map
    # keyed by the members of ONE scope dimension ({"Enterprise": 1.15, "Mid-Market":
    # 1.05}), which moves each member by its own amount.
    factor: float | Mapping[str, float] = 1.0
    type: str = "price_level"
    # Plain JSON all the way down — dimension name -> member ids. The spec crosses a
    # process boundary and feeds the corpus digest, so a custom object here would
    # either fail to serialize or hash as its repr.
    scope: Mapping[str, Sequence[str]] | None = None
    driver: str | None = None  # required for `rate`; implied by the legacy type names
    target_share: float | None = None  # `mix` only — the scoped members' post-shift share

    @property
    def effective_driver(self) -> str:
        """One vocabulary for four spellings, so the generator tests a driver only."""
        if self.type in _LEGACY_DRIVER:
            return _LEGACY_DRIVER[self.type]
        if self.driver is not None:  # `rate`, and __post_init__ has checked it is set
            return self.driver
        return "share"

    @property
    def factor_dimension(self) -> str:
        """The scope dimension a per-member factor map is keyed by.

        Derived rather than declared: the map's keys must be exactly one scope
        dimension's members, so naming it again would be a second place to get it
        wrong. ``__post_init__`` has already established there is exactly one.
        """
        if not isinstance(self.factor, Mapping) or not self.scope:
            return ""
        keys = {str(k) for k in self.factor}
        return next(dim for dim, members in self.scope.items() if {str(m) for m in members} == keys)

    def __post_init__(self) -> None:
        if self.type not in _LEVER_TYPES:
            raise ValueError(f"unknown lever type: {self.type!r} (supported: {list(_LEVER_TYPES)})")
        if self.period_k < 0:
            raise ValueError(f"invalid lever: period_k={self.period_k}")
        if self.type == "rate" and self.driver not in _DRIVERS:
            raise ValueError(f"a rate lever needs driver in {list(_DRIVERS)}, got {self.driver!r}")
        if self.type != "rate" and self.driver is not None:
            raise ValueError(f"driver is a `rate` field; {self.type!r} implies its own")
        if self.type == "mix":
            if self.target_share is None or not 0.0 < self.target_share < 1.0:
                raise ValueError(f"a mix lever needs 0 < target_share < 1, got {self.target_share!r}")
            if not self.scope:
                raise ValueError("a mix lever needs a scope — the members whose share is shifted")
            if isinstance(self.factor, Mapping):
                raise ValueError("a mix lever takes no factor at all — its knob is target_share")
        elif isinstance(self.factor, Mapping):
            if self.target_share is not None:
                raise ValueError(f"target_share is a `mix` field; {self.type!r} takes a factor")
            self._check_factor_map()
        else:
            if self.target_share is not None:
                raise ValueError(f"target_share is a `mix` field; {self.type!r} takes a factor")
            if self.factor <= 0:
                raise ValueError(f"invalid lever: factor={self.factor}")
        if self.scope is None:
            return
        unknown = sorted(set(self.scope) - set(_SCOPE_DIMS))
        if unknown:
            raise ValueError(f"unknown scope dimension(s): {unknown} (supported: {list(_SCOPE_DIMS)})")
        empty = sorted(dim for dim, members in self.scope.items() if not members)
        if empty:
            raise ValueError(f"empty scope dimension(s): {empty} — omit the key to leave it unscoped")
        # Silently ignoring a scope would produce a corpus whose intervention.yaml
        # claims a narrower intervention than the one that ran. Order counts, shares
        # and collection are all decided before or above the product grain, so only
        # the price driver can honour a product-side scope.
        product_side = sorted(set(self.scope) & set(_PRODUCT_SCOPE_DIMS))
        if product_side and self.effective_driver != "price":
            raise ValueError(
                f"a {self.effective_driver!r} lever cannot scope on {product_side}: it acts at "
                "(customer, month) grain, before any product is chosen"
            )

    def _check_factor_map(self) -> None:
        """A per-member factor map must name members the scope already admits.

        The map's keys ARE one dimension's scope, so the two must agree exactly. A map
        that named members outside the scope would declare a change to entities the
        lever never touches; one that omitted a scoped member would leave that member
        moving by an amount nothing on disk states. Either way ``intervention.yaml``
        would describe a run that did not happen.
        """
        assert isinstance(self.factor, Mapping)  # narrowed by the caller
        if not self.factor:
            raise ValueError("an empty factor map declares nothing — give a scalar factor instead")
        bad = sorted(str(m) for m, v in self.factor.items() if not isinstance(v, int | float) or v <= 0)
        if bad:
            raise ValueError(f"factor map holds non-positive or non-numeric value(s) for: {bad}")
        if not self.scope:
            raise ValueError("a per-member factor needs a scope — the dimension its keys belong to")
        keys = {str(k) for k in self.factor}
        matching = [dim for dim, members in self.scope.items() if {str(m) for m in members} == keys]
        if len(matching) != 1:
            raise ValueError(
                f"a factor map must match exactly one scope dimension's members; keys {sorted(keys)} "
                f"match {matching or 'no dimension'} in scope "
                f"{ {dim: sorted(str(m) for m in members) for dim, members in self.scope.items()} }"
            )


@dataclass
class _SaleRecord:
    """Internal tracking for a sale event (for cash receipt generation)."""

    entry_id: str
    sale_date: date
    amount: Decimal
    customer: str
    order_id: str = ""
    customer_id: str = ""
    collected: bool = False


# --- The operating chain: customer → product → order → order line ---

_SEGMENTS = ["Enterprise", "Mid-Market", "SMB"]
_REGIONS = ["DACH", "Nordics", "Benelux", "UK&I"]

# Revenue accounts the chain posts to, by product group. Services revenue is a
# genuinely different account from product revenue — the split the Offer ladder
# groups by, and it keeps the existing 41xx/42xx revenue structure meaningful.
# Groups a larger profile adds fall back to product revenue; ``Analytics`` is the
# second genuinely service-like group, so it books there.
_GROUP_REVENUE_ACCOUNT: dict[str, str] = {
    "Instruments": "4110",
    "Controllers": "4120",
    "Consumables": "4110",
    "Services": "4210",
    "Analytics": "4210",
}
_DEFAULT_REVENUE_ACCOUNT = "4110"

_INVENTORY_ACCOUNT = "1400"
_COGS_ACCOUNT = "5100"


def _lifecycle(
    seed: int, kind: str, entity_id: str, fiscal_start: date, months: int, churn_fraction: float
) -> tuple[date, date | None]:
    """When an entity became live, and when it stopped — the birth/death window.

    Three populations, deliberately: most entities predate the fiscal year and survive
    it; a churn slice is born inside it; another dies inside it. Both cases are the
    ones prior-period and peer comparisons fail on — a customer whose "collapse" is
    that they did not exist in the comparison period, and a product whose "recovery"
    is that it was discontinued. A corpus without them lets a naive year-over-year
    method look correct.

    Keyed by entity identity, so churn is a property of the firm and not of the draw
    order — and therefore untouched by a volume lever.
    """
    draw = _stream(seed, "lifecycle", kind, entity_id)
    roll = draw.random()
    first, _ = _month_start_end(fiscal_start, 0)

    if roll < churn_fraction / 2 and months >= 4:
        # Born inside the window: no history before this month.
        born_at = draw.randint(1, max(1, months - 2))
        return _month_start_end(fiscal_start, born_at)[0], None
    if roll < churn_fraction and months >= 4:
        # Dies inside the window, after a real history behind it.
        died_at = draw.randint(2, max(2, months - 1))
        return first - timedelta(days=draw.randint(400, 2600)), _month_start_end(fiscal_start, died_at)[1]
    return first - timedelta(days=draw.randint(200, 3000)), None


def _is_active(created: date, ended: date | None, m_start: date, m_end: date) -> bool:
    """Whether an entity is live at any point in a month."""
    return created <= m_end and (ended is None or ended >= m_start)


def generate_customers(
    seed: int, profile: ScaleProfile, fiscal_start: date, months: int
) -> list[Customer]:
    """Customer master — identity is deterministic; only the lifecycle window is drawn.

    Names, segments and regions come from the index so that retuning any distribution
    never moves a customer id. The validity window is keyed by that id for the same
    reason.
    """
    out: list[Customer] = []
    for i, name in enumerate(customer_names(profile.customers)):
        customer_id = f"C-{i + 1:04d}"
        created, churned = _lifecycle(
            seed, "customer", customer_id, fiscal_start, months, profile.churn_fraction
        )
        out.append(
            Customer(
                customer_id=customer_id,
                name=name,
                segment=_SEGMENTS[i % len(_SEGMENTS)],
                region=_REGIONS[i % len(_REGIONS)],
                payment_terms=list(PaymentTerms)[i % len(PaymentTerms)],
                created_date=created,
                churned_date=churned,
            )
        )
    return out


def generate_merchants(seed: int, count: int) -> list[Merchant]:
    """The payer dimension's members — ``count`` of them, which is the point.

    A thin reference table on purpose. What this dimension contributes is not richer
    attributes but CARDINALITY and a frequency law: the corpus's other dimensions are
    small and near-uniform, so every top-N question over them can be answered by
    reading the table. This one cannot.
    """
    names = merchant_names(count)
    rows = []
    for i, name in enumerate(names):
        pick = _stream(seed, "merchant", str(i))
        rows.append(
            Merchant(
                merchant_id=f"MRC-{i + 1:06d}",
                merchant_name=name,
                category=pick.choice(MERCHANT_CATEGORIES),
                country=pick.choice(MERCHANT_COUNTRIES),
            )
        )
    return rows


def _zipf_cumulative(count: int, exponent: float) -> list[float]:
    """Cumulative Zipf weights over ranks 1..count — the inverse-CDF sampling table.

    Built once and binary-searched per row, because the alternative at ``large`` is a
    linear scan over thousands of merchants for every one of hundreds of thousands of
    bank lines.
    """
    weights = [1.0 / (rank**exponent) for rank in range(1, count + 1)]
    total = sum(weights)
    running = 0.0
    cumulative = []
    for weight in weights:
        running += weight / total
        cumulative.append(running)
    cumulative[-1] = 1.0  # close the interval against float drift
    return cumulative


def assign_merchants(
    seed: int, transactions: list[BankTransaction], merchants: list[Merchant], exponent: float
) -> None:
    """Attach a Zipf-drawn merchant to every bank transaction, in place.

    Drawn from a stream keyed by the transaction's own id, so the assignment does not
    depend on how many transactions exist or in what order they were minted — the same
    property every other entity-keyed draw in this generator has, and the reason
    turning the dimension on does not disturb a lever's counterfactual.

    Rank order is NOT id order: rank *r* maps to a merchant chosen by a fixed
    seed-derived permutation, so the most frequent merchant is not ``MRC-000001``. A
    corpus where the head is readable off the key is not testing an aggregation.
    """
    if not merchants:
        return
    cumulative = _zipf_cumulative(len(merchants), exponent)
    order = list(range(len(merchants)))
    random.Random(f"merchant_rank:{seed}").shuffle(order)
    for txn in transactions:
        rank = bisect.bisect_left(cumulative, _stream(seed, "merchant_draw", txn.txn_id).random())
        txn.merchant_id = merchants[order[min(rank, len(merchants) - 1)]].merchant_id


def generate_products(
    seed: int, profile: ScaleProfile, fiscal_start: date, months: int
) -> list[Product]:
    """Product master with standard cost — costs and margins designed, not drawn.

    ``list_price`` derives from the catalogue's margin target rather than a separate
    draw, so the true unit contribution is a designed quantity we can assert against.
    A declared fraction of the catalogue is priced *below the contribution threshold*:
    thin enough that the ordinary discount range drives realised contribution to zero
    or through it. A portfolio where every item earns cannot be pruned, and "which
    products should we drop" is a canonical Offer question with no answer here before.
    """
    out: list[Product] = []
    catalog = product_catalog(profile.products, profile.product_groups, profile.tail_product_fraction)
    for i, item in enumerate(catalog):
        product_id = f"P-{i + 1:04d}"
        standard_cost = _quantize(Decimal(str(item.standard_cost)))
        list_price = _quantize(standard_cost / Decimal(str(1.0 - item.margin_target)))
        launched, discontinued = _lifecycle(
            seed, "product", product_id, fiscal_start, months, profile.churn_fraction
        )
        out.append(
            Product(
                product_id=product_id,
                name=item.name,
                product_group=item.group,
                standard_cost=standard_cost,
                list_price=list_price,
                launched_date=launched,
                discontinued_date=discontinued,
            )
        )
    return out


def _scale_anchor(order_lines: list[SalesOrderLine]) -> Decimal:
    """What the firm contributes — revenue less cost of sale, over the whole window.

    The one number the expense base is sized against (§9). Contribution rather than
    revenue, because a firm's affordable overhead follows what its sales actually
    leave behind, and because contribution is already an exact quantity here: both
    sides derive from the same order line.
    """
    return sum((line.line_amount - line.line_cost for line in order_lines), Decimal("0"))


def _customer_weight(seed: int, customer_id: str, profile: ScaleProfile) -> float:
    """This customer's share of order intensity — Pareto, mean-normalised, capped.

    Revenue per customer has to be Pareto or concentration risk is unmeasurable: a
    book of uniform customers has no top 5% worth naming. Normalising by the
    distribution's mean keeps total volume on the profile's declared level, so the
    shape does not smuggle in a size change. The cap is a modelling choice — an
    uncapped Pareto occasionally draws a customer who *is* the firm — and it is
    declared on the profile rather than buried here.
    """
    draw = _stream(seed, "customer_weight", customer_id)
    alpha = profile.customer_pareto_alpha
    cap = profile.customer_weight_cap
    mean = alpha / (alpha - 1.0)
    return min(draw.paretovariate(alpha) / mean, cap) / _pareto_normalizer(alpha, cap)


def _pareto_normalizer(alpha: float, cap: float) -> float:
    """``E[min(X/E[X], cap)]`` for ``X ~ Pareto(alpha)``.

    Capping a heavy tail removes real mass, so dividing by the *uncapped* mean leaves
    average intensity below 1 and the profile's declared order volume quietly wrong.
    Normalising by the truncated mean keeps the shape a shape: concentration changes,
    total volume does not.
    """
    mean_x = alpha / (alpha - 1.0)
    threshold = mean_x * cap
    return (alpha / (mean_x * (1.0 - alpha))) * (threshold ** (1.0 - alpha) - 1.0) + cap * threshold ** (-alpha)


@dataclass(frozen=True)
class MixSolution:
    """What a ``mix`` lever has to solve before it can be applied.

    A share target is not a factor. Moving the scoped members from ``baseline_share``
    to ``target_share`` while holding TOTAL activity fixed needs two factors that
    satisfy ``s0·ft + (1-s0)·fc = 1``, and only one of them is declared. Deriving the
    complement is what separates a mix from a frequency change: scale the target
    alone and total volume moves with it, so the aggregate would shift through
    volume as well as composition and the lever would not mean what it says.

    ``baseline_share`` is measured, not assumed — it is the scoped members' share of
    the order counts the same seed draws with no lever at all, over the months from
    ``period_k`` on, which is the window the shift applies to.
    """

    baseline_share: float
    target_share: float
    target_factor: float
    complement_factor: float


def _mix_solution(
    seed: int,
    customers: list[Customer],
    products: list[Product],
    fiscal_start: date,
    months: int,
    profile: ScaleProfile,
    q4_seasonal_boost: float,
    lever: "Lever",
) -> MixSolution:
    """Measure the baseline share, then solve for the two factors.

    A pre-pass over ``_order_count`` at ``lever=None`` — one gauss draw per live
    customer-month, negligible beside generation — rather than an assumed starting
    share. It mirrors the generation loop's own gates (a customer places no orders
    before it exists or after it lapses; no orders in a month with no live product),
    because a share measured over customer-months that never produced an order is
    not the share the corpus actually has.

    Pure and deterministic given the same arguments, so the runner can call it again
    to record the solution in ``intervention.yaml`` without threading state out of
    generation.
    """
    scope = _resolve_scope(lever, customers, products)
    live_months = {
        m for m in range(months)
        if any(_is_active(p.launched_date, p.discontinued_date, *_month_start_end(fiscal_start, m)) for p in products)
    }

    target = complement = 0
    for customer in customers:
        weight = _customer_weight(seed, customer.customer_id, profile)
        in_scope = scope.covers_customer(customer.customer_id)
        for month_offset in range(lever.period_k, months):
            if month_offset not in live_months:
                continue
            m_start, m_end = _month_start_end(fiscal_start, month_offset)
            if not _is_active(customer.created_date, customer.churned_date, m_start, m_end):
                continue
            seasonal = 1.0 + q4_seasonal_boost * (m_start.month >= 10)
            n = _order_count(
                seed, customer.customer_id, month_offset,
                profile.orders_per_customer_month * weight, seasonal,
            )
            if in_scope:
                target += n
            else:
                complement += n

    total = target + complement
    s0 = target / total if total else 0.0
    s1 = lever.target_share if lever.target_share is not None else s0
    if not 0.0 < s0 < 1.0:
        # No baseline activity in the scope (or nothing outside it) — there is no
        # share to shift, and a factor of s1/0 is not a number. Say so rather than
        # emit an intervention whose truth is a division by zero.
        raise ValueError(
            f"mix lever cannot shift a share of {s0:.4f}: the scope holds "
            f"{target} of {total} orders from period_k={lever.period_k} on"
        )
    return MixSolution(
        baseline_share=s0,
        target_share=s1,
        target_factor=s1 / s0,
        complement_factor=(1.0 - s1) / (1.0 - s0),
    )


def mix_outcome(
    lever: "Lever",
    corpus: Corpus,
    *,
    seed: int,
    months: int,
    fiscal_start: date,
    q4_seasonal_boost: float = 0.3,
    profile: str | ScaleProfile | None = None,
) -> tuple[MixSolution, float]:
    """The mix a lever solved for, and the share the corpus actually landed on.

    The two differ: an order count is a rounded draw, so scaling by the solved
    factors lands NEAR the target rather than on it. Publishing only the request
    would put a number on disk that the data does not hold, which is the shape the
    oracle contract exists to remove.

    Reads the master data off the corpus rather than regenerating it, so the shares
    are measured against the rows that actually shipped.
    """
    solution = _mix_solution(
        seed, corpus.customers, corpus.products, fiscal_start, months,
        get_profile(profile), q4_seasonal_boost, lever,
    )
    scope = _resolve_scope(lever, corpus.customers, corpus.products)
    in_window = [
        order for order in corpus.sales_orders
        if _month_offset(fiscal_start, order.order_date) >= lever.period_k
    ]
    scoped = sum(1 for order in in_window if scope.covers_customer(order.customer_id))
    return solution, (scoped / len(in_window) if in_window else 0.0)


def _order_count(
    seed: int, customer_id: str, month_offset: int, base: float, seasonal: float,
    scale: float | None = None,
) -> int:
    """How many orders this customer places this month — the volume lever's target.

    Drawn from the (customer, month) stream, so changing the count perturbs nothing
    else. A ``volume`` lever scales the count from ``period_k`` on; because order *i*
    keys its own stream, the baseline's orders remain a strict SUBSET of the levered
    run's and the added rows are exactly attributable.

    ``scale`` is the combined multiplier the caller resolved for this customer-month —
    ``None`` when no lever reaches it. The draw happens either way and only the
    scaling is conditional, so an out-of-scope customer is byte-identical to its
    baseline rather than merely similar. Resolving it OUTSIDE keeps the scope tests,
    the period test and the composition of several levers in one place, and leaves
    this function a draw.
    """
    draw = _stream(seed, "order_count", customer_id, month_offset)
    n = max(0, round(draw.gauss(base * seasonal, base * 0.25)))
    if scale is None:
        return n
    return _scale_count(n, scale, seed, customer_id, month_offset)


def _scale_count(n: int, factor: float, seed: int, customer_id: str, month_offset: int) -> int:
    """Scale an order count without letting rounding eat the factor.

    ``round(n * factor)`` is badly biased at this grain. A customer-month holds 0-3
    orders, so a 2.5% shift rounds to nothing on nearly every one of them: a mix
    lever asked to move a segment's share from 34.1% to 35.0% landed on 34.3%,
    roughly a seventh of the way, and the shortfall grew as the requested shift got
    smaller. A lever that quietly under-delivers is worse than one that refuses,
    because ``intervention.yaml`` would still name the target.

    Stochastic rounding carries the fraction as a probability instead of discarding
    it, so ``E[result] == n * factor`` exactly and the aggregate lands on the target.
    Drawn from its OWN identity-keyed stream, so no other draw moves; and monotone in
    ``factor`` — at ``factor >= 1`` the result is never below ``n``, which is what
    keeps the baseline's orders a strict subset of the levered run's.
    """
    scaled = n * factor
    whole = int(scaled)
    return whole + (1 if _stream(seed, "count_scale", customer_id, month_offset).random() < scaled - whole else 0)


def generate_sales_orders(
    seed: int,
    customers: list[Customer],
    products: list[Product],
    fiscal_start: date,
    months: int,
    *,
    profile: ScaleProfile,
    q4_seasonal_boost: float = 0.3,
    lever: Lever | None = None,
    levers: Sequence[Lever] | None = None,
    trend: Trend = _NO_TREND,
) -> tuple[list[SalesOrder], list[SalesOrderLine]]:
    """The operating chain — customer orders with priced, costed lines.

    Draws ONLY from entity-keyed streams (never the sequential ``rng``), so the
    ledger cycles are independent of order volume and a volume lever stays an exact
    counterfactual.

    Two shapes carry the population (§9). Order **intensity** is scaled by the
    customer's Pareto weight, so revenue per customer is concentrated and "the top
    5%" means something. Order **size** is log-normal in units, so mean and median
    part company and a consumer reporting one for the other is visibly wrong. Both
    are drawn per entity, so neither disturbs the counterfactual.
    """
    orders: list[SalesOrder] = []
    lines: list[SalesOrderLine] = []
    log_median = math.log(profile.order_units_median)

    # A price lever acts HERE, on the realised unit price, not on the GL credit
    # derived from it — see :class:`Lever`. Scopes and per-member factors resolve to
    # entity ids once, so the loops below ask one question per entity either way.
    applied = _resolve_levers(_normalise_levers(lever, levers), customers, products)
    price_levers = [a for a in applied if a.lever.effective_driver == "price"]
    frequency_levers = [a for a in applied if a.lever.effective_driver == "frequency"]
    mix_lever = next((a for a in applied if a.lever.effective_driver == "share"), None)
    mix = (
        _mix_solution(
            seed, customers, products, fiscal_start, months, profile, q4_seasonal_boost, mix_lever.lever
        )
        if mix_lever is not None
        else None
    )

    # The sellable catalogue per month, computed once rather than per customer-month:
    # at `large` that inner filter would be 4,000 x 12 x 1,200 comparisons for an
    # answer that only depends on the month.
    live_by_month = [
        [p for p in products if _is_active(p.launched_date, p.discontinued_date, *_month_start_end(fiscal_start, m))]
        for m in range(months)
    ]
    # Deterministic in the month, so it costs one lookup rather than a draw, and both
    # runs of a lever pair carry the identical drift — it cancels in the difference.
    price_drift = [trend.price_at(m) for m in range(months)] if trend.price else None
    volume_drift = [trend.volume_at(m) for m in range(months)] if trend.volume else None

    for customer in customers:
        weight = _customer_weight(seed, customer.customer_id, profile)
        # Loop-invariant per customer; the product side is tested per line below.
        price_here = [a for a in price_levers if a.scope.covers_customer(customer.customer_id)]
        frequency_here = [a for a in frequency_levers if a.scope.covers_customer(customer.customer_id)]
        mix_here = mix_lever is not None and mix_lever.scope.covers_customer(customer.customer_id)
        for month_offset in range(months):
            m_start, m_end = _month_start_end(fiscal_start, month_offset)
            # A customer places no orders before it exists or after it lapses. This is
            # the birth/death case, and it must gate the ORDERS, not just the master
            # row — a validity window nothing respects is decoration.
            if not _is_active(customer.created_date, customer.churned_date, m_start, m_end):
                continue
            live = live_by_month[month_offset]
            if not live:
                continue
            seasonal = 1.0 + q4_seasonal_boost * (m_start.month >= 10)
            # Several frequency levers over one customer MULTIPLY, and the product is
            # applied in a single scaling call so the count draws from one stream —
            # scaling twice would consume the stream twice and desynchronise the
            # single-lever runs this is meant to be comparable with.
            count_scale = volume_drift[month_offset] if volume_drift is not None else 1.0
            for a in frequency_here:
                if month_offset >= a.lever.period_k:
                    count_scale *= a.factors.num(customer.customer_id)
            if mix is not None and mix_lever is not None and month_offset >= mix_lever.lever.period_k:
                count_scale *= mix.target_factor if mix_here else mix.complement_factor
            n_orders = _order_count(
                seed, customer.customer_id, month_offset,
                profile.orders_per_customer_month * weight, seasonal,
                count_scale if count_scale != 1.0 else None,
            )
            for i in range(n_orders):
                o = _stream(seed, "order", customer.customer_id, month_offset, i)
                order_id = f"SO-{customer.customer_id[2:]}-{month_offset:02d}-{i:03d}"
                order_date = _random_date(o, max(m_start, customer.created_date), m_end)
                orders.append(
                    SalesOrder(
                        order_id=order_id,
                        customer_id=customer.customer_id,
                        order_date=order_date,
                        status="open" if o.random() < 0.08 else "confirmed",
                    )
                )
                for j in range(o.choices([1, 2, 3], weights=[55, 30, 15])[0]):
                    line = _stream(seed, "order_line", order_id, j)
                    product = live[line.randrange(len(live))]
                    units = max(1, round(line.lognormvariate(log_median, profile.order_units_sigma)))
                    # Discount off list — the price realisation that makes per-customer
                    # DB1 differ from per-product DB1, and the reason the catalogue's
                    # thin tail actually goes negative rather than merely looking thin.
                    discount = Decimal(str(round(line.uniform(0.0, 0.18), 4)))
                    unit_price = _quantize(product.list_price * (Decimal("1") - discount))
                    if price_drift is not None:
                        unit_price = _quantize(unit_price * price_drift[month_offset])
                    for a in price_here:
                        if month_offset >= a.lever.period_k and a.scope.covers_product(product.product_id):
                            unit_price = _quantize(
                                unit_price * a.factors.of(customer.customer_id, product.product_id)
                            )
                    lines.append(
                        SalesOrderLine(
                            order_line_id=f"{order_id}-L{j + 1}",
                            order_id=order_id,
                            product_id=product.product_id,
                            units=units,
                            unit_price=unit_price,
                            line_amount=_quantize(unit_price * units),
                            line_cost=_quantize(product.standard_cost * units),
                        )
                    )
    return orders, lines


def _generate_revenue_entries(
    seed: int,
    orders: list[SalesOrder],
    order_lines: list[SalesOrderLine],
    customers: list[Customer],
    products: list[Product],
    counters: _Counters,
    fiscal_start: date,
) -> tuple[list[JournalEntry], list[JournalLine], list[_SaleRecord], dict[str, str]]:
    """Revenue and cost of sale, DERIVED from the order lines.

    Two GL entries per order, both derived rather than drawn:

    * **Revenue** — DR Accounts Receivable (order total), CR the product group's
      revenue account per line (``units × unit_price``).
    * **Cost of sale** — DR Cost of Goods Sold, CR Inventory (``units ×
      standard_cost``). This is the leg the corpus never had: before it, COGS was a
      random slice of vendor purchases with no link to any sale, so gross profit and
      every margin below it were ungradeable in principle.

    Returns entries, lines, the sale records the receipt cycle consumes, and the
    cost-of-sale entry per order — the link the stock subledger hangs its issue
    movements off, so every issue names the GL posting it detailed.

    Lever-free by construction. Both legs are a pure function of the order lines, so
    a ``price_level`` lever reaches the ledger by having already moved
    ``unit_price``, and a ``volume`` lever by having already added orders. Applying a
    factor here as well is what desynchronised the lines from the ledger; see
    :class:`Lever`.
    """
    lines_by_order: dict[str, list[SalesOrderLine]] = {}
    for line in order_lines:
        lines_by_order.setdefault(line.order_id, []).append(line)
    product_group = {p.product_id: p.product_group for p in products}
    customer_name = {c.customer_id: c.name for c in customers}

    entries: list[JournalEntry] = []
    gl_lines: list[JournalLine] = []
    sales: list[_SaleRecord] = []
    cogs_entry_by_order: dict[str, str] = {}

    for order in orders:
        rows = lines_by_order.get(order.order_id, [])
        if not rows:
            continue
        post = _stream(seed, "revenue_post", order.order_id)

        cost_center = post.choice(COST_CENTERS) if post.random() < 0.85 else None
        customer = customer_name.get(order.customer_id, order.customer_id)

        total = _quantize(sum((r.line_amount for r in rows), Decimal("0.00")))
        cost = _quantize(sum((r.line_cost for r in rows), Decimal("0.00")))

        # ── Revenue: DR AR / CR revenue per line ──
        entry_id = counters.next_entry()
        entries.append(
            JournalEntry(
                entry_id=entry_id,
                date=order.order_date,
                description=f"Revenue recognition - {customer}",
                status=JournalStatus.POSTED,
                created_by=post.choice(USERS),
            )
        )
        gl_lines.append(
            JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id=post.choice(_AR_ACCOUNTS),
                debit=total,
                credit=Decimal("0.00"),
                cost_center=cost_center,
            )
        )
        for row in rows:
            gl_lines.append(
                JournalLine(
                    line_id=counters.next_line(),
                    entry_id=entry_id,
                    account_id=_GROUP_REVENUE_ACCOUNT.get(product_group[row.product_id], _DEFAULT_REVENUE_ACCOUNT),
                    debit=Decimal("0.00"),
                    credit=row.line_amount,
                    cost_center=cost_center,
                )
            )

        # ── Cost of sale: DR COGS / CR Inventory ──
        if cost > 0:
            cogs_entry = counters.next_entry()
            entries.append(
                JournalEntry(
                    entry_id=cogs_entry,
                    date=order.order_date,
                    description=f"Cost of sale - {customer}",
                    status=JournalStatus.POSTED,
                    created_by=post.choice(USERS),
                )
            )
            gl_lines.append(
                JournalLine(
                    line_id=counters.next_line(), entry_id=cogs_entry,
                    account_id=_COGS_ACCOUNT, debit=cost, credit=Decimal("0.00"),
                    cost_center=cost_center,
                )
            )
            gl_lines.append(
                JournalLine(
                    line_id=counters.next_line(), entry_id=cogs_entry,
                    account_id=_INVENTORY_ACCOUNT, debit=Decimal("0.00"), credit=cost,
                    cost_center=cost_center,
                )
            )
            cogs_entry_by_order[order.order_id] = cogs_entry

        sales.append(
            _SaleRecord(
                entry_id=entry_id,
                sale_date=order.order_date,
                amount=total,
                customer=customer,
                order_id=order.order_id,
                customer_id=order.customer_id,
            )
        )

    return entries, gl_lines, sales, cogs_entry_by_order


_TERMS_DAYS: dict[PaymentTerms, int] = {
    PaymentTerms.NET_30: 30,
    PaymentTerms.NET_60: 60,
    PaymentTerms.NET_90: 90,
    PaymentTerms.DUE_ON_RECEIPT: 0,
}


def _generate_ar_invoices(
    sales: list[_SaleRecord],
    customers: list[Customer],
    fiscal_end: date,
) -> list[ARInvoice]:
    """One customer invoice per order — the AR document the corpus never had.

    ``invoices`` is vendor-side only, so days-sales-outstanding had no receivable to
    measure and the AR half of Capital was invisible. Amount ties to the
    order's revenue posting by construction, and the due date follows the customer's
    own agreed terms, so DSO is a property of the data rather than a constant.

    Status is derived from the receipt cycle afterwards (``_settle_ar_invoices``) —
    minting it here would guess at a collection the receipt cycle actually decides.
    """
    terms_of = {c.customer_id: c.payment_terms for c in customers}
    out: list[ARInvoice] = []
    for sale in sales:
        terms = terms_of.get(sale.customer_id, PaymentTerms.NET_30)
        due = sale.sale_date + timedelta(days=_TERMS_DAYS[terms])
        out.append(
            ARInvoice(
                ar_invoice_id=f"ARI-{sale.order_id[3:]}",
                order_id=sale.order_id,
                customer_id=sale.customer_id,
                invoice_date=sale.sale_date,
                due_date=due,
                amount=sale.amount,
                status=InvoiceStatus.OVERDUE if due < fiscal_end else InvoiceStatus.OPEN,
            )
        )
    return out


def _settle_ar_invoices(
    ar_invoices: list[ARInvoice], receipts: list[Receipt]
) -> None:
    """Set each AR invoice's status from what was actually collected against it.

    Derived, never drawn: an invoice reading ``paid`` with no receipt behind it — or
    ``open`` with one — is precisely the cross-table inconsistency a consumer's
    validation induction is supposed to catch, and it would be ours, not theirs.
    """
    collected: dict[str, Decimal] = {}
    for receipt in receipts:
        collected[receipt.ar_invoice_id] = (
            collected.get(receipt.ar_invoice_id, Decimal("0.00")) + receipt.amount
        )
    for invoice in ar_invoices:
        got = collected.get(invoice.ar_invoice_id, Decimal("0.00"))
        if got <= 0:
            continue
        invoice.status = (
            InvoiceStatus.PAID if got >= invoice.amount else InvoiceStatus.PARTIAL
        )


# --- Inventory: the stock subledger under GL 1400 ---

_LOCATIONS = ["WH-MAIN", "WH-SOUTH"]
_LOCATION_WEIGHTS = [70, 30]
_SHRINKAGE_ACCOUNT = "5150"

# Movements on the same day post in this order. Not cosmetic: a receipt landing after
# the day's issues would drive on-hand negative on the fiscal year's first day, a
# defect we would have invented ourselves.
_MOVEMENT_RANK = {
    StockMovementType.RECEIPT: 0,
    StockMovementType.ISSUE: 1,
    StockMovementType.ADJUSTMENT: 2,
}


@dataclass
class _InventoryCycle:
    """Everything the stock subledger produces, including its own payables."""

    movements: list[StockMovement] = dataclass_field(default_factory=list)
    positions: list[InventoryPosition] = dataclass_field(default_factory=list)
    invoices: list[Invoice] = dataclass_field(default_factory=list)
    payments: list[Payment] = dataclass_field(default_factory=list)
    entries: list[JournalEntry] = dataclass_field(default_factory=list)
    lines: list[JournalLine] = dataclass_field(default_factory=list)
    bank_transactions: list[BankTransaction] = dataclass_field(default_factory=list)


def _generate_inventory_cycle(
    seed: int,
    orders: list[SalesOrder],
    order_lines: list[SalesOrderLine],
    products: list[Product],
    cogs_entry_by_order: dict[str, str],
    fiscal_start: date,
    months: int,
    counters: _Counters,
    suppliers: list[str],
) -> _InventoryCycle:
    """Stock movements, positions, and the supplier bills that actually get paid.

    What this replaces and why. The previous replenishment posted one
    ``DR Inventory / CR AP`` per month at 1.02–1.18 × the month's cost of sale and
    nothing ever settled it: 95% of closing payables were permanently open and annual
    DPO read 271 days. The credit leg dangled because there was no purchasing event
    behind it — only a plug sized to make the asset side work.

    Now every receipt IS a vendor bill (``category=goods``), so it flows through the
    same payment cycle as every other payable, and *purchases* becomes a computable
    quantity for the first time — the denominator a textbook DPO wants and the corpus
    could not previously offer.

    The policy is a designed coverage target, not a forecast: each (product, location)
    holds roughly ``coverage`` months of its own average demand, replenished in whole
    case-size batches. Stating it that way is deliberate — a synthetic generator may
    use hindsight, but it must say so, because a consumer measuring "how good is this
    firm's planning" would otherwise be measuring our omniscience.

    Everything draws from entity-keyed streams, so a volume lever propagates into
    stock (more sales → more issues → more receipts → more payables) without
    perturbing any other cycle's draws.
    """
    cycle = _InventoryCycle()
    order_date = {o.order_id: o.order_date for o in orders}
    product_by_id = {p.product_id: p for p in products}

    # ── Issues: one movement per order line ──
    # Valued at the same standard cost the revenue cycle already posted to COGS, so
    # Σ issue value == Σ COGS to the cent. No second GL entry: the issue movement is
    # the subledger detail BEHIND the cost-of-sale entry, not another posting.
    demand: dict[tuple[str, str, int], int] = {}
    for line in order_lines:
        entry_id = cogs_entry_by_order.get(line.order_id)
        if entry_id is None:
            continue
        moved_on = order_date[line.order_id]
        offset = _month_offset(fiscal_start, moved_on)
        if not 0 <= offset < months:
            continue
        location = _stream(seed, "issue_location", line.order_line_id).choices(
            _LOCATIONS, weights=_LOCATION_WEIGHTS
        )[0]
        product = product_by_id[line.product_id]
        cycle.movements.append(
            StockMovement(
                movement_id="",
                product_id=line.product_id,
                location_id=location,
                date=moved_on,
                movement_type=StockMovementType.ISSUE,
                units=-line.units,
                unit_cost=product.standard_cost,
                value=-line.line_cost,
                source_document=line.order_line_id,
                entry_id=entry_id,
            )
        )
        key = (line.product_id, location, offset)
        demand[key] = demand.get(key, 0) + line.units

    # ── Replenishment and cycle counts, per (product, location) ──
    for product in products:
        for location in _LOCATIONS:
            monthly = [demand.get((product.product_id, location, m), 0) for m in range(months)]
            if sum(monthly) == 0:
                continue
            _replenish_one_stock(
                seed, cycle, product, location, monthly, fiscal_start, months, counters, suppliers
            )

    # ── Age and settle the goods payables ──
    # The whole point of the family. A goods bill is never CANCELLED: the pallet is on
    # the dock and the movement is on the books, so "this line was never really
    # bought" is a story the stock ledger already contradicts.
    fiscal_end = _month_start_end(fiscal_start, months - 1)[1]
    for bill in cycle.invoices:
        aging = _stream(seed, "goods_aging", bill.invoice_id)
        status = _aged_invoice_status(
            aging, (fiscal_end - bill.date).days, _TERMS_DAYS[bill.payment_terms]
        )
        bill.status = InvoiceStatus.PAID if status == InvoiceStatus.CANCELLED else status

    payments, pay_entries, pay_lines, pay_bank = _generate_vendor_payments(
        lambda inv: _stream(seed, "goods_payment", inv.invoice_id),
        cycle.invoices,
        counters,
        suppliers,
    )
    cycle.payments.extend(payments)
    cycle.entries.extend(pay_entries)
    cycle.lines.extend(pay_lines)
    cycle.bank_transactions.extend(pay_bank)

    # ── Movement ids, minted after the whole ledger is ordered ──
    cycle.movements.sort(
        key=lambda m: (m.date, _MOVEMENT_RANK[m.movement_type], m.product_id, m.location_id, m.source_document)
    )
    for movement in cycle.movements:
        movement.movement_id = counters.next_movement()

    return cycle


def _replenish_one_stock(
    seed: int,
    cycle: _InventoryCycle,
    product: Product,
    location: str,
    monthly_demand: list[int],
    fiscal_start: date,
    months: int,
    counters: _Counters,
    suppliers: list[str],
) -> None:
    """Plan, receive, count and value one (product, location) across the year.

    Appends receipts, adjustments, their GL and their vendor bills to *cycle*, and
    emits one :class:`InventoryPosition` per period — the closing level, which is why
    it is written even for a month with no movement at all.
    """
    policy = _stream(seed, "stock_policy", product.product_id, location)
    average = sum(monthly_demand) / months
    coverage = policy.uniform(0.8, 1.4)
    target_close = int(round(coverage * average))
    batch = max(1, int(round(average * 0.25)))
    pick = _stream(seed, "product_vendor", product.product_id).randrange(len(suppliers))
    vendor_name = suppliers[pick]
    vendor_id = f"V-{pick + 1:04d}"

    on_hand = 0
    for offset in range(months):
        m_start, _m_end = _month_start_end(fiscal_start, offset)
        issued = monthly_demand[offset]

        shortfall = issued + target_close - on_hand
        received = 0
        if shortfall > 0:
            received = math.ceil(shortfall / batch) * batch
            _receive_stock(
                seed, cycle, product, location, received, offset, m_start,
                vendor_id, vendor_name, counters,
            )
        on_hand += received - issued

        # Quarterly cycle count — the only inventory expense that is not a sale.
        if offset % 3 == 2 and on_hand > 0:
            on_hand += _count_stock(
                seed, cycle, product, location, on_hand, offset, fiscal_start, counters
            )

        cycle.positions.append(
            InventoryPosition(
                product_id=product.product_id,
                location_id=location,
                period=m_start.strftime("%Y-%m"),
                units_on_hand=on_hand,
                unit_cost=product.standard_cost,
                value=_quantize(product.standard_cost * on_hand),
            )
        )


def _receive_stock(
    seed: int,
    cycle: _InventoryCycle,
    product: Product,
    location: str,
    units: int,
    offset: int,
    m_start: date,
    vendor_id: str,
    vendor_name: str,
    counters: _Counters,
) -> None:
    """One month's replenishment, delivered on one or two vendor bills.

    Each delivery is a real payable: an ``Invoice`` row with ``category=goods``, a
    ``DR Inventory / CR AP`` entry, and — when the aging says so — a payment through
    the ordinary vendor-payment cycle. Splitting into deliveries is not decoration:
    one bill per product-month would make every goods payable land on the same day
    and DPO would measure the calendar rather than the firm.
    """
    plan = _stream(seed, "stock_receipt", product.product_id, location, offset)
    n_deliveries = plan.choices([1, 2], weights=[45, 55])[0]
    splits = [units] if n_deliveries == 1 else _split_units(plan, units)

    for i, delivery_units in enumerate(splits):
        if delivery_units <= 0:
            continue
        # The fiscal year's first receipt lands on day one: opening stock has to be
        # there before the first order ships out of it.
        if offset == 0 and i == 0:
            received_on = m_start
        else:
            received_on = m_start + timedelta(days=plan.randint(0, 3) + 12 * i)
        value = _quantize(product.standard_cost * delivery_units)

        invoice_id = counters.next_invoice()
        bill = _stream(seed, "goods_bill", invoice_id)
        terms = bill.choice(list(PaymentTerms))
        entry_id = counters.next_entry()
        cycle.invoices.append(
            Invoice(
                invoice_id=invoice_id,
                vendor_id=vendor_id,
                date=received_on,
                due_date=received_on + timedelta(days=_TERMS_DAYS[terms]),
                amount=value,
                status=InvoiceStatus.OPEN,  # set from the aging below
                payment_terms=terms,
                category=InvoiceCategory.GOODS,
                entry_id=entry_id,
            )
        )
        cycle.entries.append(
            JournalEntry(
                entry_id=entry_id,
                date=received_on,
                description=f"Goods receipt - {product.name} - {location} - {invoice_id}",
                status=JournalStatus.POSTED,
                created_by=bill.choice(USERS),
            )
        )
        cycle.lines.append(
            JournalLine(
                line_id=counters.next_line(), entry_id=entry_id,
                account_id=_INVENTORY_ACCOUNT, debit=value, credit=Decimal("0.00"),
            )
        )
        cycle.lines.append(
            JournalLine(
                line_id=counters.next_line(), entry_id=entry_id,
                account_id=bill.choice(_AP_ACCOUNTS), debit=Decimal("0.00"), credit=value,
            )
        )
        cycle.movements.append(
            StockMovement(
                movement_id="",
                product_id=product.product_id,
                location_id=location,
                date=received_on,
                movement_type=StockMovementType.RECEIPT,
                units=delivery_units,
                unit_cost=product.standard_cost,
                value=value,
                source_document=invoice_id,
                entry_id=entry_id,
            )
        )


def _split_units(draw: random.Random, units: int) -> list[int]:
    """Split a receipt across two deliveries, the larger one first."""
    first = max(1, int(round(units * draw.uniform(0.45, 0.75))))
    return [min(first, units), units - min(first, units)]


def _count_stock(
    seed: int,
    cycle: _InventoryCycle,
    product: Product,
    location: str,
    on_hand: int,
    offset: int,
    fiscal_start: date,
    counters: _Counters,
) -> int:
    """A quarterly physical count. Returns the unit delta it wrote to the books.

    Mostly shrinkage (DR 5150 / CR 1400); occasionally the count finds more than the
    book says, which posts the other way. Both directions matter: a corpus where the
    only adjustment sign is negative makes "is this an adjustment?" answerable from
    the sign alone.
    """
    count = _stream(seed, "stock_count", product.product_id, location, offset)
    if count.random() >= 0.35:
        return 0
    delta = -int(round(on_hand * count.uniform(0.002, 0.015)))
    if count.random() < 0.20:
        delta = -delta
    if delta == 0:
        return 0

    counted_on = _month_start_end(fiscal_start, offset)[1]
    value = _quantize(product.standard_cost * abs(delta))
    entry_id = counters.next_entry()
    document = f"CNT-{product.product_id}-{location}-{offset:02d}"
    cycle.entries.append(
        JournalEntry(
            entry_id=entry_id,
            date=counted_on,
            description=f"Cycle count adjustment - {product.name} - {location}",
            status=JournalStatus.POSTED,
            created_by=count.choice(USERS),
        )
    )
    shrink = delta < 0
    cycle.lines.append(
        JournalLine(
            line_id=counters.next_line(), entry_id=entry_id,
            account_id=_SHRINKAGE_ACCOUNT if shrink else _INVENTORY_ACCOUNT,
            debit=value, credit=Decimal("0.00"),
        )
    )
    cycle.lines.append(
        JournalLine(
            line_id=counters.next_line(), entry_id=entry_id,
            account_id=_INVENTORY_ACCOUNT if shrink else _SHRINKAGE_ACCOUNT,
            debit=Decimal("0.00"), credit=value,
        )
    )
    cycle.movements.append(
        StockMovement(
            movement_id="",
            product_id=product.product_id,
            location_id=location,
            date=counted_on,
            movement_type=StockMovementType.ADJUSTMENT,
            units=delta,
            unit_cost=product.standard_cost,
            value=value if delta > 0 else -value,
            source_document=document,
            entry_id=entry_id,
        )
    )
    return delta


def _generate_cash_receipts(
    seed: int,
    sales: list[_SaleRecord],
    fiscal_start: date,
    months: int,
    counters: _Counters,
    *,
    collection_rate: float = 0.85,
    applied: Sequence[_Applied] = (),
) -> tuple[list[JournalEntry], list[JournalLine], list[BankTransaction], list[Receipt]]:
    """Cash receipts for collected sales: DR Cash, CR AR + Bank Txn + the AR document.

    Keyed per SALE, not drawn from the sequential stream: whether order X
    is collected, when, and how much must not depend on how many other orders exist,
    or a volume lever would silently re-roll the collection outcome of every
    pre-existing sale and destroy the subset property the counterfactual rests on.

    A ``collection_lag`` lever scales the drawn lag for in-scope sales from
    ``period_k`` on. It is applied after the draw and *whether* a sale is collected is
    decided before it, so the levered and baseline runs collect exactly the same set
    of sales and differ only in when the cash lands.

    One honest caveat, recorded in ``intervention.yaml`` rather than left to be
    discovered: a receipt is clamped to the fiscal end, so late-year sales absorb
    part of a lag increase instead of moving by the full factor. DSO therefore moves
    by less than the factor near the year boundary, and the effect is not linear in
    it.
    """
    fiscal_end = _month_start_end(fiscal_start, months - 1)[1]
    lag_levers = [a for a in applied if a.lever.effective_driver == "collection_lag"]

    entries: list[JournalEntry] = []
    lines: list[JournalLine] = []
    bank_txns: list[BankTransaction] = []
    receipts: list[Receipt] = []

    for sale in sales:
        rng = _stream(seed, "receipt", sale.order_id or sale.entry_id)
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
        lag = rng.randint(5, 45)
        lag_scale = 1.0
        for a in lag_levers:
            if (
                _month_offset(fiscal_start, sale.sale_date) >= a.lever.period_k
                and a.scope.covers_customer(sale.customer_id)
            ):
                lag_scale *= a.factors.num(sale.customer_id)
        if lag_scale != 1.0:
            lag = max(0, int(round(lag * lag_scale)))
        receipt_date = sale.sale_date + timedelta(days=lag)
        if receipt_date > fiscal_end:
            receipt_date = fiscal_end

        # Partial collection (5% chance)
        if rng.random() < 0.05:
            amount = _quantize(sale.amount * Decimal(str(rng.uniform(0.5, 0.95))))
        else:
            amount = sale.amount

        entry_id = counters.next_entry()
        entries.append(
            JournalEntry(
                entry_id=entry_id,
                date=receipt_date,
                description=f"Cash receipt - {sale.customer}",
                status=JournalStatus.POSTED,
                created_by=rng.choice(USERS),
            )
        )

        # DR: Cash
        cash_account = rng.choice(_CASH_ACCOUNTS)
        lines.append(
            JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id=cash_account,
                debit=amount,
                credit=Decimal("0.00"),
            )
        )

        # CR: AR
        lines.append(
            JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id=rng.choice(_AR_ACCOUNTS),
                debit=Decimal("0.00"),
                credit=amount,
            )
        )

        # Bank transaction (positive = credit/inflow)
        bank_txns.append(
            BankTransaction(
                txn_id=counters.next_bank_txn(),
                account_id=cash_account,
                date=receipt_date,
                amount=amount,
                reference=f"RCV-{rng.randint(100000, 999999)}",
                counterparty=sale.customer,
                reconciled=rng.random() < 0.90,
            )
        )

        if sale.order_id:
            receipts.append(
                Receipt(
                    receipt_id=f"RC-{sale.order_id[3:]}",
                    ar_invoice_id=f"ARI-{sale.order_id[3:]}",
                    customer_id=sale.customer_id,
                    receipt_date=receipt_date,
                    amount=amount,
                    method=rng.choice(list(PaymentMethod)),
                )
            )

    return entries, lines, bank_txns, receipts


# --- Expenditure Cycle: Purchase Invoices → Vendor Payments ---


def _aged_invoice_status(
    draw: random.Random, days_since: int, terms_days: int
) -> InvoiceStatus:
    """Settlement status from the bill's age at fiscal close.

    Shared by both payable populations — the expense bills the ledger draws
    sequentially and the goods bills the stock subledger draws from entity-keyed
    streams — so the two age by the same rule and DPO is not an artifact of which
    cycle minted the invoice.
    """
    if days_since > terms_days + 30:
        return draw.choices(
            [InvoiceStatus.PAID, InvoiceStatus.CANCELLED],
            weights=[95, 5],
        )[0]
    if days_since > terms_days:
        return draw.choices(
            [InvoiceStatus.PAID, InvoiceStatus.OVERDUE, InvoiceStatus.PARTIAL],
            weights=[70, 20, 10],
        )[0]
    return draw.choices(
        [InvoiceStatus.OPEN, InvoiceStatus.PAID],
        weights=[60, 40],
    )[0]


def _generate_purchase_invoices(
    rng: random.Random,
    fiscal_start: date,
    months: int,
    counters: _Counters,
    *,
    count: int,
    suppliers: list[str],
    budget: Decimal,
) -> tuple[list[Invoice], list[JournalEntry], list[JournalLine]]:
    """Generate vendor/purchase invoices with GL entries: DR Expense, CR AP.

    ``budget`` is this cycle's share of the operating expense base, and the amounts
    are rescaled to hit it exactly. The count therefore sets *granularity* — how many
    bills the spend arrives on — instead of setting the spend itself, which is what
    made gross profit an artifact of a knob (§7): 3,000 invoices at a fixed 100–50,000
    band described a firm of no particular size, and the same number would have been
    absurd one profile up.

    Rescaling by a constant is deliberate rather than convenient. Amounts are drawn
    log-uniform, and multiplying a log-uniform variate by a constant shifts it in log
    space without changing the mantissa distribution — so the leading-digit (Benford)
    property the corpus relies on survives being budgeted.
    """
    leaf = _get_leaf_accounts()
    # COGS and shrinkage are EXCLUDED: both are derived from real events — cost of
    # sale from the order line (units x standard_cost), shrinkage from a physical
    # count that disagreed with the book. Letting random vendor invoices land there
    # too would make 5100 a mix of cost-of-sale and unrelated purchases (the reason
    # gross profit was ungradeable before) and would turn 5150 into a number that
    # says nothing about stock at all. Goods purchases reach the books as Inventory.
    derived_only = {_COGS_ACCOUNT, _SHRINKAGE_ACCOUNT}
    expense_accounts = [a for a in leaf.get(AccountType.EXPENSE, []) if a not in derived_only]

    invoices: list[Invoice] = []
    entries: list[JournalEntry] = []
    lines_out: list[JournalLine] = []

    fiscal_end = _month_start_end(fiscal_start, months - 1)[1]

    # 80/20 vendor concentration
    top_vendors = suppliers[:4]
    other_vendors = suppliers[4:] or suppliers
    vendor_index = {name: i for i, name in enumerate(suppliers)}

    terms_days = {
        PaymentTerms.NET_30: 30,
        PaymentTerms.NET_60: 60,
        PaymentTerms.NET_90: 90,
        PaymentTerms.DUE_ON_RECEIPT: 0,
    }

    # Drawn first, as a set, because the scale factor is a property of the whole
    # population and cannot be known one invoice at a time.
    raw = [_benford_amount(rng, 100, 50000) for _ in range(count)]
    drawn_total = sum(raw, Decimal("0"))
    factor = (budget / drawn_total) if drawn_total > 0 else Decimal("1")
    amounts = [max(_quantize(a * factor), Decimal("0.01")) for a in raw]

    for i in range(count):
        inv_date = _random_date(rng, fiscal_start, fiscal_end)
        terms = rng.choice(list(PaymentTerms))
        due = inv_date + timedelta(days=terms_days[terms])

        if rng.random() < 0.80:
            vendor = rng.choice(top_vendors)
        else:
            vendor = rng.choice(other_vendors)

        vendor_id = f"V-{vendor_index[vendor] + 1:04d}"
        amount = amounts[i]

        # Status depends on age relative to fiscal end
        days_since = (fiscal_end - inv_date).days
        status = _aged_invoice_status(rng, days_since, terms_days[terms])

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

            entries.append(
                JournalEntry(
                    entry_id=entry_id,
                    date=inv_date,
                    description=f"Vendor invoice - {vendor} - {invoice_id}",
                    status=JournalStatus.POSTED,
                    created_by=rng.choice(USERS),
                )
            )

            # DR: Expense (may split across 1-3 expense accounts)
            n_exp = rng.choices([1, 2, 3], weights=[70, 25, 5])[0]
            exp_amounts = _split_amount(rng, amount, n_exp)
            for exp_amt in exp_amounts:
                lines_out.append(
                    JournalLine(
                        line_id=counters.next_line(),
                        entry_id=entry_id,
                        account_id=rng.choice(expense_accounts),
                        debit=exp_amt,
                        credit=Decimal("0.00"),
                        cost_center=cost_center,
                    )
                )

            # CR: AP
            lines_out.append(
                JournalLine(
                    line_id=counters.next_line(),
                    entry_id=entry_id,
                    account_id=rng.choice(_AP_ACCOUNTS),
                    debit=Decimal("0.00"),
                    credit=amount,
                    cost_center=cost_center,
                )
            )

    return invoices, entries, lines_out


def _generate_vendor_payments(
    draw_for: Callable[[Invoice], random.Random],
    invoices: list[Invoice],
    counters: _Counters,
    suppliers: list[str],
) -> tuple[list[Payment], list[JournalEntry], list[JournalLine], list[BankTransaction]]:
    """Generate vendor payment events: DR AP, CR Cash + Bank Txn (debit).

    *draw_for* supplies the stream each invoice draws from, which is the whole reason
    this is a parameter. The expense cycle hands back the ledger's shared sequential
    ``rng``; the stock subledger hands back a stream keyed on the bill's own id, so a
    volume lever can add goods payables without shifting a single expense-side draw.
    Both populations otherwise settle by identical rules.
    """
    payments: list[Payment] = []
    entries: list[JournalEntry] = []
    lines_out: list[JournalLine] = []
    bank_txns: list[BankTransaction] = []

    methods = list(PaymentMethod)
    vendor_name_by_id = {f"V-{i + 1:04d}": name for i, name in enumerate(suppliers)}

    for inv in invoices:
        if inv.status not in (InvoiceStatus.PAID, InvoiceStatus.PARTIAL):
            continue

        rng = draw_for(inv)
        pay_date = inv.date + timedelta(days=rng.randint(1, 45))

        if inv.status == InvoiceStatus.PARTIAL:
            pay_amount = _quantize(inv.amount * Decimal(str(rng.uniform(0.3, 0.9))))
        else:
            pay_amount = inv.amount

        payment_id = counters.next_payment()
        payments.append(
            Payment(
                payment_id=payment_id,
                invoice_id=inv.invoice_id,
                date=pay_date,
                amount=pay_amount,
                currency=inv.currency,
                method=rng.choice(methods),
            )
        )

        # GL entry: DR AP, CR Cash
        entry_id = counters.next_entry()
        vendor_name = vendor_name_by_id.get(inv.vendor_id, inv.vendor_id)
        entries.append(
            JournalEntry(
                entry_id=entry_id,
                date=pay_date,
                description=f"Vendor payment - {vendor_name} - {inv.invoice_id}",
                status=JournalStatus.POSTED,
                created_by=rng.choice(USERS),
            )
        )

        # DR: AP
        lines_out.append(
            JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id=rng.choice(_AP_ACCOUNTS),
                debit=pay_amount,
                credit=Decimal("0.00"),
            )
        )

        # CR: Cash
        cash_account = rng.choice(_CASH_ACCOUNTS)
        lines_out.append(
            JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id=cash_account,
                debit=Decimal("0.00"),
                credit=pay_amount,
            )
        )

        # Bank transaction (negative = debit/outflow)
        bank_txns.append(
            BankTransaction(
                txn_id=counters.next_bank_txn(),
                account_id=cash_account,
                date=pay_date,
                amount=-pay_amount,
                reference=f"TXN-{rng.randint(100000, 999999)}",
                counterparty=vendor_name,
                reconciled=rng.random() < 0.90,
                payment_id=payment_id,
            )
        )

    return payments, entries, lines_out, bank_txns


# --- Operating Events: Payroll, Depreciation, Rent, Misc ---


def _generate_operating_events(
    rng: random.Random,
    fiscal_start: date,
    months: int,
    counters: _Counters,
    *,
    budget: Decimal,
    suppliers: list[str],
) -> tuple[list[JournalEntry], list[JournalLine], list[BankTransaction]]:
    """Generate recurring operating events: payroll, depreciation, rent, etc.

    Every line here is sized off ``budget`` — this cycle's share of what the firm
    contributes — rather than off a fixed band. Payroll of 30–75k a month described a
    company of one particular size and no other, which is precisely why the P&L sign
    could not be graded (§7).

    A month is not the annual average: each line carries a jitter, so period-over-period
    variance is real rather than a flat line a naive trend detector would ace.
    """
    entries: list[JournalEntry] = []
    lines_out: list[JournalLine] = []
    bank_txns: list[BankTransaction] = []

    def monthly(share: str, low: float = 0.88, high: float = 1.12) -> Decimal:
        target = budget * Decimal(str(_OPEX_ALLOCATION[share])) / Decimal(months)
        return max(_quantize(target * Decimal(str(rng.uniform(low, high)))), Decimal("0.01"))

    for month_offset in range(months):
        m_start, m_end = _month_start_end(fiscal_start, month_offset)
        month_label = m_start.strftime("%B %Y")

        # --- Payroll (last day of month, cash outflow) ---
        payroll_amount = monthly("payroll")
        entry_id = counters.next_entry()
        entries.append(
            JournalEntry(
                entry_id=entry_id,
                date=m_end,
                description=f"Payroll - {month_label}",
                status=JournalStatus.POSTED,
                created_by="system",
            )
        )

        # Split across salary/benefits/taxes
        salary = _quantize(payroll_amount * Decimal("0.70"))
        benefits = _quantize(payroll_amount * Decimal("0.20"))
        taxes = _quantize(payroll_amount - salary - benefits)

        for acct, amt in [("5210", salary), ("5220", benefits), ("5230", taxes)]:
            lines_out.append(
                JournalLine(
                    line_id=counters.next_line(),
                    entry_id=entry_id,
                    account_id=acct,
                    debit=amt,
                    credit=Decimal("0.00"),
                )
            )

        lines_out.append(
            JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id="1110",  # Operating account
                debit=Decimal("0.00"),
                credit=payroll_amount,
            )
        )

        bank_txns.append(
            BankTransaction(
                txn_id=counters.next_bank_txn(),
                account_id="1110",
                date=m_end,
                amount=-payroll_amount,
                reference=f"PAY-{m_start.strftime('%Y%m')}",
                counterparty="ADP Payroll",
                reconciled=True,
            )
        )

        # --- Rent (1st of month, cash outflow) ---
        rent_amount = monthly("rent", 0.98, 1.02)
        entry_id = counters.next_entry()
        entries.append(
            JournalEntry(
                entry_id=entry_id,
                date=m_start,
                description=f"Rent payment - {month_label}",
                status=JournalStatus.POSTED,
                created_by="system",
            )
        )

        lines_out.append(
            JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id="5310",  # Office Rent
                debit=rent_amount,
                credit=Decimal("0.00"),
            )
        )
        lines_out.append(
            JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id="1110",  # Operating account
                debit=Decimal("0.00"),
                credit=rent_amount,
            )
        )

        bank_txns.append(
            BankTransaction(
                txn_id=counters.next_bank_txn(),
                account_id="1110",
                date=m_start,
                amount=-rent_amount,
                reference=f"RENT-{m_start.strftime('%Y%m')}",
                counterparty="Building Management",
                reconciled=True,
            )
        )

        # --- Depreciation (end of month, non-cash) ---
        depr_amount = monthly("depreciation", 0.99, 1.01)
        entry_id = counters.next_entry()
        entries.append(
            JournalEntry(
                entry_id=entry_id,
                date=m_end,
                description=f"Depreciation - {month_label}",
                status=JournalStatus.POSTED,
                created_by="system",
            )
        )

        lines_out.append(
            JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id="5700",  # Depreciation expense
                debit=depr_amount,
                credit=Decimal("0.00"),
            )
        )
        lines_out.append(
            JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id="1590",  # Accumulated Depreciation
                debit=Decimal("0.00"),
                credit=depr_amount,
            )
        )

        # --- Insurance (1st of month, cash outflow) ---
        insurance = monthly("insurance", 0.97, 1.03)
        entry_id = counters.next_entry()
        entries.append(
            JournalEntry(
                entry_id=entry_id,
                date=m_start,
                description=f"Insurance premium - {month_label}",
                status=JournalStatus.POSTED,
                created_by="system",
            )
        )

        lines_out.append(
            JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id="5800",  # Insurance
                debit=insurance,
                credit=Decimal("0.00"),
            )
        )
        lines_out.append(
            JournalLine(
                line_id=counters.next_line(),
                entry_id=entry_id,
                account_id="1110",
                debit=Decimal("0.00"),
                credit=insurance,
            )
        )

        bank_txns.append(
            BankTransaction(
                txn_id=counters.next_bank_txn(),
                account_id="1110",
                date=m_start,
                amount=-insurance,
                reference=f"INS-{m_start.strftime('%Y%m')}",
                counterparty="Insurance Corp",
                reconciled=True,
            )
        )

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
        # Same budgeting as the vendor bills: draw the month's amounts log-uniformly,
        # then scale the set onto its share. The count stays a granularity choice.
        raw_misc = [_benford_amount(rng, 50, 5000) for _ in range(n_misc)]
        misc_total = sum(raw_misc, Decimal("0"))
        misc_target = budget * Decimal(str(_OPEX_ALLOCATION["misc"])) / Decimal(months)
        misc_factor = (misc_target / misc_total) if misc_total > 0 else Decimal("1")
        for k in range(n_misc):
            desc, account = rng.choice(misc_templates)
            misc_date = _random_date(rng, m_start, m_end)
            misc_amount = max(_quantize(raw_misc[k] * misc_factor), Decimal("0.01"))
            cost_center = rng.choice(COST_CENTERS) if rng.random() < 0.85 else None

            entry_id = counters.next_entry()
            entries.append(
                JournalEntry(
                    entry_id=entry_id,
                    date=misc_date,
                    description=f"{desc} - {month_label}",
                    status=JournalStatus.POSTED,
                    created_by=rng.choice(USERS),
                )
            )

            lines_out.append(
                JournalLine(
                    line_id=counters.next_line(),
                    entry_id=entry_id,
                    account_id=account,
                    debit=misc_amount,
                    credit=Decimal("0.00"),
                    cost_center=cost_center,
                )
            )
            lines_out.append(
                JournalLine(
                    line_id=counters.next_line(),
                    entry_id=entry_id,
                    account_id="1110",
                    debit=Decimal("0.00"),
                    credit=misc_amount,
                    cost_center=cost_center,
                )
            )

            # 70% are cash payments, 30% are accruals (no bank txn)
            if rng.random() < 0.70:
                bank_txns.append(
                    BankTransaction(
                        txn_id=counters.next_bank_txn(),
                        account_id="1110",
                        date=misc_date,
                        amount=-misc_amount,
                        reference=f"TXN-{rng.randint(100000, 999999)}",
                        counterparty=rng.choice(suppliers),
                        reconciled=rng.random() < 0.85,
                    )
                )

    return entries, lines_out, bank_txns


# --- Misc Bank Transactions (interest, transfers, unmatched) ---


def _generate_misc_bank_transactions(
    rng: random.Random,
    fiscal_start: date,
    months: int,
    counters: _Counters,
    *,
    suppliers: list[str],
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

            bank_txns.append(
                BankTransaction(
                    txn_id=counters.next_bank_txn(),
                    account_id=cash_account,
                    date=txn_date,
                    amount=amount,
                    reference=ref,
                    counterparty=counterparty,
                    reconciled=rng.random() < 0.80,
                )
            )

            # GL entry for the bank transaction
            entry_id = counters.next_entry()
            entries.append(
                JournalEntry(
                    entry_id=entry_id,
                    date=txn_date,
                    description=f"Bank {'credit' if amount > 0 else 'fee'} - {ref}",
                    status=JournalStatus.POSTED,
                    created_by="system",
                )
            )

            abs_amount = abs(amount)
            lines_out.append(
                JournalLine(
                    line_id=counters.next_line(),
                    entry_id=entry_id,
                    account_id=gl_debit_account,
                    debit=abs_amount,
                    credit=Decimal("0.00"),
                )
            )
            lines_out.append(
                JournalLine(
                    line_id=counters.next_line(),
                    entry_id=entry_id,
                    account_id=gl_credit_account,
                    debit=Decimal("0.00"),
                    credit=abs_amount,
                )
            )

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
                rates.append(
                    FXRate(
                        from_ccy=from_ccy,
                        to_ccy=to_ccy,
                        date=current,
                        rate=Decimal(str(round(rate_val, 6))),
                    )
                )
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
        result.append(
            TrialBalance(
                account_id=acct,
                period=period_str,
                debit_balance=_quantize(month_d),
                credit_balance=_quantize(month_c),
            )
        )

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
        c.account_id for c in chart if c.account_type in (AccountType.ASSET, AccountType.LIABILITY, AccountType.EQUITY)
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
            result.append(
                BalanceSheet(
                    account_id=acct,
                    period=period_str,
                    ending_balance=_quantize(running),
                )
            )

    return result


def _generate_measure_probes(months: int, fiscal_start: date, n_series: int) -> list[MeasureProbe]:
    """Skeleton ``(series_id, period)`` grain for the stock/flow probe table.

    ``n_series`` synthetic series × ``months`` periods — the grain that
    ``inject_stock_flow_probes`` fills with stock/flow measure columns. Rows are in
    ``(series, period)`` order so a stock column's per-series running total accumulates
    in row order. Empty when ``n_series == 0`` (no stock/flow strategy active).
    """
    if n_series <= 0:
        return []
    periods = [_month_start_end(fiscal_start, m)[0].strftime("%Y-%m") for m in range(months)]
    return [MeasureProbe(series_id=f"S{s:03d}", period=period) for s in range(n_series) for period in periods]


def _generate_formula_probes(n_rows: int) -> list[FormulaProbe]:
    """Skeleton ``probe_id`` grain for the formula-divergence probe table.

    ``n_rows`` synthetic records — the grain that ``inject_formula_divergence`` fills
    with labelled (source_a, source_b, target) formula groups for the derived_value
    witness calibration. Empty when ``n_rows == 0`` (no formula strategy active).
    """
    return [FormulaProbe(probe_id=f"FP{i:05d}") for i in range(n_rows)]


def _generate_relationship_probes(n_parents: int, n_children: int) -> tuple[list[RefEntity], list[RefActivity]]:
    """Skeleton grains for the relationship-pairs probe tables.

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


_ROLEPLAY_CITIES = [
    ("Zurich", "CH"), ("Geneva", "CH"), ("Berlin", "DE"), ("Munich", "DE"),
    ("Paris", "FR"), ("Lyon", "FR"), ("Milan", "IT"), ("Vienna", "AT"),
    ("Amsterdam", "NL"), ("Brussels", "BE"), ("Madrid", "ES"), ("Lisbon", "PT"),
]


def _generate_roleplay_probes(
    n_addresses: int, n_orders: int, n_deliveries: int, fiscal_start: date, months: int
) -> tuple[list[Address], list[Order], list[Delivery]]:
    """Skeleton grains for the role-playing-FK probe shape.

    A complete address dimension plus id×date grains for the two facts;
    ``inject_role_playing_fks`` fills the ROLE columns (orders.bill_to_addr /
    ship_to_addr, deliveries.delivery_addr). Deliveries reference orders
    round-robin with the delivery dated a few days after its order. Empty when
    any count is 0 (no role-play strategy active).
    """
    if n_addresses <= 0 or n_orders <= 0 or n_deliveries <= 0:
        return [], [], []
    addresses = [
        Address(
            address_id=f"ADDR-{i:04d}",
            street=f"{(i * 7) % 190 + 1} {'Main' if i % 3 else 'Station'} Street",
            city=_ROLEPLAY_CITIES[i % len(_ROLEPLAY_CITIES)][0],
            country=_ROLEPLAY_CITIES[i % len(_ROLEPLAY_CITIES)][1],
        )
        for i in range(n_addresses)
    ]
    span_days = max(months * 28, 1)
    orders = [
        Order(
            order_id=f"ORD-{i:05d}",
            order_date=fiscal_start + timedelta(days=(i * 13) % span_days),
        )
        for i in range(n_orders)
    ]
    deliveries = [
        Delivery(
            delivery_id=f"DLV-{i:05d}",
            order_id=orders[i % n_orders].order_id,
            delivery_date=orders[i % n_orders].order_date + timedelta(days=2 + i % 5),
        )
        for i in range(n_deliveries)
    ]
    return addresses, orders, deliveries


# --- Main Generator ---


def generate_finance_dataset(
    seed: int = 42,
    months: int = 12,
    fiscal_start: date | None = None,
    invoices_count: int | None = None,
    q4_seasonal_boost: float | None = None,
    probe_series: int = 0,
    formula_probe_rows: int = 0,
    relation_parents: int = 0,
    relation_children: int = 0,
    roleplay_addresses: int = 0,
    roleplay_orders: int = 0,
    roleplay_deliveries: int = 0,
    lever: Lever | None = None,
    levers: Sequence[Lever] | None = None,
    trend: Trend | Mapping[str, float] | None = None,
    merchants: int = 0,
    merchant_exponent: float = 1.05,
    profile: str | ScaleProfile | None = None,
) -> Corpus:
    """Generate a complete finance dataset with closed-loop accounting.

    Business events cascade across tables: sales create GL entries and AR,
    payments settle AP and create bank transactions, trial balance is derived
    from actual cumulative GL entries.

    Args:
        seed: Random seed for reproducibility.
        months: Number of months to generate (fiscal year).
        fiscal_start: First day of the fiscal year.
        invoices_count: Number of vendor/purchase invoices. Overrides the profile.
        q4_seasonal_boost: Fractional boost applied to Q4 revenue months.
        lever: Optional constructed intervention — a DGP parameter
            change at a known period with a known effect; see :class:`Lever`.
        levers: Several interventions in one run, which is how an interaction pair
            is built: the combined corpus is not the sum of the two single-lever
            ones, because the second lever acts on what the first already moved.
            Mutually exclusive with ``lever``.
        trend: Optional secular drift in prices and/or volumes — see :class:`Trend`.
            The control corpus for "did something happen": everything rises and
            nothing happened. Recorded in the corpus stamp, not as an intervention.
        merchants: Size of the opt-in high-cardinality payer dimension. 0 (the
            default) means no merchants table and no ``bank_transactions.merchant_id``
            column at all — an untouched corpus, byte for byte.
        merchant_exponent: The Zipf exponent that dimension's frequencies follow.
            Around 1.0 is the empirical law; higher concentrates the head.
        profile: Scale profile name or object (§9). Defaults to ``tiny``.

    Returns:
        Corpus with all 8 tables populated and numerically consistent.
    """
    rng = random.Random(seed)
    if fiscal_start is None:
        fiscal_start = date(2025, 1, 1)

    scale = get_profile(profile)
    q4_boost = q4_seasonal_boost if q4_seasonal_boost is not None else 0.3
    inv_count = invoices_count if invoices_count is not None else scale.vendor_invoices
    suppliers = supplier_names(scale.suppliers)

    counters = _Counters()

    # Static tables
    chart = generate_chart_of_accounts()
    fx_rates = _generate_fx_rates(rng, fiscal_start, months)

    # ── The operating chain ──
    # Drawn from entity-keyed streams only, NEVER the sequential `rng`, so the
    # ledger cycles below are independent of order volume and a volume lever stays
    # an exact counterfactual. Its GL entries are minted LAST (see below).
    active_levers = _normalise_levers(lever, levers)
    drift = trend if isinstance(trend, Trend) else (Trend(**trend) if trend else _NO_TREND)
    customers = generate_customers(seed, scale, fiscal_start, months)
    products = generate_products(seed, scale, fiscal_start, months)
    sales_orders, sales_order_lines = generate_sales_orders(
        seed,
        customers,
        products,
        fiscal_start,
        months,
        profile=scale,
        q4_seasonal_boost=q4_boost,
        levers=active_levers,
        trend=drift,
    )

    # ── The scale anchor ──
    # What the firm actually contributes, and therefore what it can afford to spend.
    # Computed at lever=None DELIBERATELY: a price or volume intervention must not
    # mechanically move payroll, or the counterfactual stops being attributable and
    # `intervention.yaml`'s "unaffected: the expenditure cycle" becomes a false claim.
    # The baseline chain is regenerated only when a lever is active, and it is pure
    # draws — no ledger, no ids — so the common path costs nothing.
    anchor_lines = sales_order_lines
    if active_levers:
        _, anchor_lines = generate_sales_orders(
            seed, customers, products, fiscal_start, months,
            profile=scale, q4_seasonal_boost=q4_boost, trend=drift,
        )
    opex_budget = _scale_anchor(anchor_lines) * Decimal(str(scale.opex_share_of_contribution))

    # ── Expenditure cycle ──
    # Purchase invoices → Invoice + GL (DR: Expense, CR: AP)
    invoices, inv_entries, inv_lines = _generate_purchase_invoices(
        rng,
        fiscal_start,
        months,
        counters,
        count=inv_count,
        suppliers=suppliers,
        budget=opex_budget * Decimal(str(_OPEX_ALLOCATION["vendor_bills"])),
    )

    # Vendor payments → Payment + GL + Bank (DR: AP, CR: Cash)
    payments, pay_entries, pay_lines, pay_bank = _generate_vendor_payments(
        lambda _inv: rng,
        invoices,
        counters,
        suppliers,
    )

    # ── Operating events ──
    # Payroll, depreciation, rent, misc expenses
    op_entries, op_lines, op_bank = _generate_operating_events(
        rng,
        fiscal_start,
        months,
        counters,
        budget=opex_budget,
        suppliers=suppliers,
    )

    # ── Misc bank transactions ──
    # Interest, fees, unmatched items (with corresponding GL)
    misc_entries, misc_lines, misc_bank = _generate_misc_bank_transactions(
        rng,
        fiscal_start,
        months,
        counters,
        suppliers=suppliers,
        count_per_month=20,
    )

    # ── The operating chain's GL, minted LAST ──
    # Ordering is load-bearing, not cosmetic: a volume lever changes how many sales
    # entries exist, so minting them mid-cascade would shift every later cycle's
    # entry_id/line_id and the expenditure cycle would stop being byte-identical
    # between a same-seed pair. Last means the extra ids land at the end and nothing
    # before them moves. (A separate id prefix would also work but puts two formats
    # in one column — a self-inflicted format-consistency false positive on clean.)
    sale_entries, sale_lines, sale_records, cogs_entry_by_order = _generate_revenue_entries(
        seed, sales_orders, sales_order_lines, customers, products, counters,
        fiscal_start,
    )
    receipt_entries, receipt_lines, receipt_bank, receipts = _generate_cash_receipts(
        seed, sale_records, fiscal_start, months, counters,
        applied=_resolve_levers(active_levers, customers, products),
    )
    ar_invoices = _generate_ar_invoices(
        sale_records, customers, _month_start_end(fiscal_start, months - 1)[1]
    )
    _settle_ar_invoices(ar_invoices, receipts)

    # ── The stock subledger, minted last of all ──
    # It consumes the cost-of-sale entries, so it cannot run earlier; and it mints
    # invoices, payments and bank transactions of its own, so running it last keeps
    # every id the expenditure cycle already handed out exactly where it was.
    stock = _generate_inventory_cycle(
        seed, sales_orders, sales_order_lines, products, cogs_entry_by_order,
        fiscal_start, months, counters, suppliers,
    )

    # ── Assembly ──
    all_entries = (
        inv_entries + pay_entries + op_entries + misc_entries
        + sale_entries + receipt_entries + stock.entries
    )
    all_lines = (
        inv_lines + pay_lines + op_lines + misc_lines
        + sale_lines + receipt_lines + stock.lines
    )
    all_bank = receipt_bank + pay_bank + op_bank + misc_bank + stock.bank_transactions
    invoices = invoices + stock.invoices
    payments = payments + stock.payments

    # Sort by date for chronological ordering
    all_entries.sort(key=lambda e: (e.date, e.entry_id))
    all_bank.sort(key=lambda t: (t.date, t.txn_id))

    # Derive trial balance (per-period movement, a flow) from actual GL entries
    trial_bal = _derive_trial_balance(all_entries, all_lines, fiscal_start, months)
    # Derive balance sheet (carry-forward ending balance, a stock) for BS accounts
    balance_sheet = _derive_balance_sheet(all_entries, all_lines, chart)
    # The opt-in payer dimension. Attached AFTER the chronological sort so the draw is
    # keyed by txn_id and not by position — turning it on must not renumber anything.
    merchant_rows = generate_merchants(seed, merchants) if merchants else []
    assign_merchants(seed, all_bank, merchant_rows, merchant_exponent)

    # Skeleton grain for the stock/flow probe table — empty unless a strategy needs it.
    measure_probes = _generate_measure_probes(months, fiscal_start, probe_series)
    # Skeleton grain for the formula-divergence probe table — same gating.
    formula_probes = _generate_formula_probes(formula_probe_rows)
    # Skeleton grains for the relationship probe tables — empty unless a strategy needs them.
    ref_entities, ref_activity = _generate_relationship_probes(relation_parents, relation_children)
    # Skeleton grains + dimension for the role-playing-FK shape — same gating.
    addresses, orders, deliveries = _generate_roleplay_probes(
        roleplay_addresses, roleplay_orders, roleplay_deliveries, fiscal_start, months
    )

    return Corpus(
        chart_of_accounts=chart,
        journal_entries=all_entries,
        journal_lines=all_lines,
        invoices=invoices,
        payments=payments,
        bank_transactions=all_bank,
        merchants=merchant_rows,
        fx_rates=fx_rates,
        trial_balance=trial_bal,
        balance_sheet=balance_sheet,
        measure_probes=measure_probes,
        formula_probes=formula_probes,
        ref_entities=ref_entities,
        ref_activity=ref_activity,
        addresses=addresses,
        orders=orders,
        deliveries=deliveries,
        customers=customers,
        products=products,
        sales_orders=sales_orders,
        sales_order_lines=sales_order_lines,
        ar_invoices=ar_invoices,
        receipts=receipts,
        stock_movements=stock.movements,
        inventory_positions=stock.positions,
    )
