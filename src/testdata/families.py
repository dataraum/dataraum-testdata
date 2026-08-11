"""The family registry — what a set of tables declares about itself.

A *family* is a cohesive group of tables: the ledger, the operating chain, the
inventory position, the probe shapes. Everything the rest of the package needs to
know about a table's schema is declared here once — which tables exist, which
column is a table's key, how a key spells as a natural key, how a column spells in
a legacy export — instead of being repeated in the exporter, the key transform and
the column-style transform.

The reason is not tidiness. Adding a family and forgetting one of those maps is a
silent defect, and it has already happened: the operating chain reached the
exporter but never the key or legacy maps, so `customer_id` and `product_id` were
quietly skipped by every key strategy. ``test_families_registry`` now fails if a
table exists without a declaration.

Adding a family: declare it here, generate its rows into ``Corpus``, and contribute
its truth. Nothing else should need to know it exists.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Merge:
    """A parent/child pair the ``partial`` normalization collapses into one table.

    The header/item split is what a real ERP presents; collapsing it is what the level
    exists to test a consumer against. ``spine`` is the table whose grain survives —
    ``journal_lines``, not ``journal_entries`` — and ``rename`` is applied to the joined
    side before the join, because a merge that silently drops a colliding column would
    lose data rather than reshape it.

    Declared here rather than in the transform because three other files need to know
    what the merge did: the exporter, the entropy map's table remap, and
    ``metadata_truth``, which used to carry its own hand-copied list of these renames.
    """

    name: str
    spine: str
    joined: str
    on: str
    rename: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Fold:
    """A dimension the ``flat`` normalization inlines into its facts.

    ``into`` maps each fact to the name it takes afterwards — the fold renames
    ``journal_data`` to ``general_ledger`` but leaves ``trial_balance`` alone, and both
    facts still end up sharing one account axis. ``attributes`` is what lands on the
    fact (post-rename, key excluded); ``concept`` is the conformed dimension the folded
    columns still identify, which is how two facts folding the SAME source are known to
    share ONE dimension rather than to have coincidentally similar columns.
    """

    concept: str
    dimension: str
    on: str
    into: Mapping[str, str]
    attributes: tuple[str, ...]
    rename: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Structure:
    """What a family declares about the *meaning* of its tables and columns.

    ``metadata_truth.yaml`` publishes this; the family owns it. §1's second rule — "a
    table that lands without its truth fragment is not done" — has no teeth while the
    truth lives in a different file from the tables, which is how the operating chain
    shipped a release with no FK topology and no semantic roles in the published truth.

    Table classification (``facts`` / ``dimensions`` / ``ambiguous``) must cover every
    table the family declares; ``tests/test_families_registry.py`` fails otherwise, so a
    new table cannot arrive truth-free.

    * ``facts`` / ``dimensions`` — HARD where structure decides: measure-bearing = fact,
      pure reference = dimension.
    * ``ambiguous`` — genuinely modelable either way (an event header carrying no
      measure; a reference row that does carry a rate). Reported, never asserted.
    * ``measures`` / ``timestamps`` — per-column semantic roles. key / dimension /
      attribute are convention-dependent and deliberately absent.
    * ``stock_flow`` — ``additive`` = per-period flow, ``point_in_time`` = level.
    * ``reconciles_structurally`` — measures that tie out against a finer event fact.
    * ``subledger_event_fact`` — the finer fact, when it is NOT the corpus event fact.
    * ``measured_in`` — the unit column a MONETARY measure is denominated in, or None
      when the table carries no unit source. Quantities are outside it entirely.
    * ``event_fact`` — the family's event-grain fact, if it owns the corpus's.
    """

    facts: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    ambiguous: tuple[str, ...] = ()
    measures: tuple[str, ...] = ()
    timestamps: tuple[str, ...] = ()
    stock_flow: Mapping[str, str] = field(default_factory=dict)
    reconciles_structurally: tuple[str, ...] = ()
    subledger_event_fact: Mapping[str, str] = field(default_factory=dict)
    measured_in: Mapping[str, str | None] = field(default_factory=dict)
    dimensionless: tuple[str, ...] = ()
    business_concepts: Mapping[str, str] = field(default_factory=dict)
    cycles: tuple[Mapping[str, Any], ...] = ()
    degenerate_ids: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    event_fact: str | None = None


@dataclass(frozen=True)
class Family:
    """One cohesive group of tables and what it declares about their schema."""

    name: str
    description: str
    tables: tuple[str, ...]
    # table -> the column that identifies one of its rows
    primary_keys: Mapping[str, str] = field(default_factory=dict)
    # key column -> the prefix it takes under the `natural` key strategy
    natural_prefixes: Mapping[str, str] = field(default_factory=dict)
    # canonical column -> its spelling in a `legacy` ERP export
    legacy_names: Mapping[str, str] = field(default_factory=dict)
    # (from, to) qualified columns — the family's TRUE foreign keys, including the
    # ones that point at another family's table. `metadata_truth` publishes these as
    # the FK topology, so a family that declares its tables here declares its joins
    # in the same breath. The operating chain shipped without them for exactly as
    # long as it shipped without its key maps.
    foreign_keys: tuple[tuple[str, str], ...] = ()
    # How this family's tables reshape under `partial` and `flat`. The transforms used
    # to name finance tables in their own bodies, so a new family's header/item pair
    # would simply never collapse — silently, and only visible as a table count.
    merges: tuple[Merge, ...] = ()
    folds: tuple[Fold, ...] = ()
    # What the family's tables MEAN — published as metadata_truth.yaml.
    structure: Structure = field(default_factory=Structure)
    # A probe family materializes only when a strategy injects into it.
    optional: bool = False


CORE_LEDGER = Family(
    name="core_ledger",
    description="Double-entry ledger: accounts, journal, AP subledger, bank, balances.",
    tables=(
        "chart_of_accounts",
        "journal_entries",
        "journal_lines",
        "invoices",
        "payments",
        "bank_transactions",
        "fx_rates",
        "trial_balance",
        "balance_sheet",
    ),
    primary_keys={
        "chart_of_accounts": "account_id",
        "journal_entries": "entry_id",
        "journal_lines": "line_id",
        "invoices": "invoice_id",
        "payments": "payment_id",
        "bank_transactions": "txn_id",
    },
    natural_prefixes={
        "account_id": "ACCT",
        "entry_id": "JE",
        "line_id": "JL",
        "invoice_id": "INV",
        "payment_id": "PAY",
        "txn_id": "BT",
        "vendor_id": "V",
    },
    legacy_names={
        "entry_id": "JRNL_ID",
        "line_id": "LN_ID",
        "account_id": "ACCT_NO",
        "account_name": "ACCT_NM",
        "account_type": "ACCT_TYP",
        "parent_id": "PRNT_ACCT",
        "parent_account_id": "PRNT_ACCT",
        "debit": "DR_AMT",
        "credit": "CR_AMT",
        "debit_balance": "DR_BAL",
        "credit_balance": "CR_BAL",
        "ending_balance": "END_BAL",
        "cost_center": "CC",
        "invoice_id": "INV_NO",
        "vendor_id": "VNDR_ID",
        "due_date": "DUE_DT",
        "payment_terms": "PMT_TERMS",
        "payment_id": "PMT_ID",
        "payment_date": "PMT_DT",
        "payment_amount": "PMT_AMT",
        "payment_currency": "PMT_CCY",
        "payment_method": "PMT_MTHD",
        "txn_id": "TXN_ID",
        "counterparty": "CNTRPRTY",
        "reconciled": "RECON_FLG",
        "from_ccy": "FROM_CCY",
        "to_ccy": "TO_CCY",
        "created_by": "CRTD_BY",
        "date": "TXN_DT",
        "amount": "AMT",
        "currency": "CCY",
        "description": "DESC",
        "status": "STAT",
        "reference": "REF",
        "period": "PRD",
        "rate": "FX_RATE",
        "source": "SRC",
        "method": "MTHD",
        "account_currency": "ACCT_CCY",
    },
    # Deliberately EXCLUDED, and the exclusions are the interesting half: `currency`
    # (Currency is an enum — no dimension table exists) and trial_balance.period ↔
    # balance_sheet.period (a shared conformed period, a fan trap, not an FK).
    foreign_keys=(
        ("journal_lines.entry_id", "journal_entries.entry_id"),
        ("journal_lines.account_id", "chart_of_accounts.account_id"),
        ("invoices.entry_id", "journal_entries.entry_id"),
        ("payments.invoice_id", "invoices.invoice_id"),
        ("bank_transactions.account_id", "chart_of_accounts.account_id"),
        ("bank_transactions.payment_id", "payments.payment_id"),
        ("trial_balance.account_id", "chart_of_accounts.account_id"),
        ("balance_sheet.account_id", "chart_of_accounts.account_id"),
        ("chart_of_accounts.parent_id", "chart_of_accounts.account_id"),
    ),
    merges=(
        Merge(name="journal_data", spine="journal_lines", joined="journal_entries", on="entry_id"),
        Merge(
            name="invoice_data",
            spine="invoices",
            joined="payments",
            on="invoice_id",
            rename={
                "date": "payment_date",
                "amount": "payment_amount",
                "currency": "payment_currency",
                "method": "payment_method",
            },
        ),
    ),
    folds=(
        # Every fact carrying `account_id` that has a dimension table to lose gets the
        # fold — that is what denormalizing a warehouse does, and it keeps the three
        # facts' account axes conformed to one concept. `balance_sheet` is included so
        # the conformed group has three members rather than two: a consumer needing >= 2
        # facts per shared dimension then has slack instead of sitting exactly on the
        # boundary where one missing axis silently zeroes the group.
        #
        # `bank_transactions` deliberately does NOT get the fold: its account_id stays a
        # key_only exposure (2 distinct values, invisible to any overlap measure). That
        # is a graded acceptance class, not an oversight.
        Fold(
            concept="account",
            dimension="chart_of_accounts",
            on="account_id",
            into={
                "journal_data": "general_ledger",
                "trial_balance": "trial_balance",
                "balance_sheet": "balance_sheet",
            },
            rename={"name": "account_name", "parent_id": "parent_account_id", "currency": "account_currency"},
            # opened_date is the coincidental-bijection case: unique per account, so on
            # the fact grain it is 1:1 with account_id and statistically identical to
            # the true account_name alias. It is a folded ATTRIBUTE of the account,
            # never an alias of it.
            attributes=("account_name", "account_type", "parent_account_id", "account_currency", "opened_date"),
        ),
    ),
    structure=Structure(
        facts=("journal_lines", "invoices", "payments", "bank_transactions", "trial_balance", "balance_sheet"),
        dimensions=("chart_of_accounts",),
        # journal_entries is an event header carrying no measure; fx_rates is a lookup
        # row that DOES carry a rate-like measure. Neither branch of the rule fits, so
        # both are reported rather than asserted.
        ambiguous=("journal_entries", "fx_rates"),
        measures=(
            "journal_lines.debit",
            "journal_lines.credit",
            "journal_lines.net_amount",
            "invoices.amount",
            "payments.amount",
            "bank_transactions.amount",
            "trial_balance.debit_balance",
            "trial_balance.credit_balance",
            "balance_sheet.ending_balance",
            "fx_rates.rate",
        ),
        timestamps=(
            "journal_entries.date",
            "invoices.date",
            "invoices.due_date",
            "payments.date",
            "bank_transactions.date",
            "fx_rates.date",
            "trial_balance.period",
            "balance_sheet.period",
        ),
        # TrialBalance is per-period movement (FLOW); BalanceSheet.ending_balance is a
        # carry-forward level (STOCK); fx_rates.rate is a price level (STOCK).
        stock_flow={
            "journal_lines.debit": "additive",
            "journal_lines.credit": "additive",
            "journal_lines.net_amount": "additive",
            "invoices.amount": "additive",
            "payments.amount": "additive",
            "bank_transactions.amount": "additive",
            "trial_balance.debit_balance": "additive",
            "trial_balance.credit_balance": "additive",
            "balance_sheet.ending_balance": "point_in_time",
            "fx_rates.rate": "point_in_time",
        },
        reconciles_structurally=(
            "trial_balance.debit_balance",
            "trial_balance.credit_balance",
            "balance_sheet.ending_balance",
        ),
        # Every monetary measure sits beside the Currency-typed column of its own table.
        # None = no same-table unit source, so NO measured_in edge may be projected:
        # fx_rates.rate is DIMENSIONLESS (a ratio BETWEEN two Currency columns), and the
        # balance tables are denominated by the ACCOUNT dimension, reachable only via FK
        # at canonical. The CoA fold changes that — see `metadata_truth._MEASURED_IN_FOLD`.
        measured_in={
            "journal_lines.debit": "journal_lines.currency",
            "journal_lines.credit": "journal_lines.currency",
            "journal_lines.net_amount": "journal_lines.currency",
            "invoices.amount": "invoices.currency",
            "payments.amount": "payments.currency",
            "bank_transactions.amount": "bank_transactions.currency",
            "fx_rates.rate": None,
            "trial_balance.debit_balance": None,
            "trial_balance.credit_balance": None,
            "balance_sheet.ending_balance": None,
        },
        dimensionless=("fx_rates.rate",),
        business_concepts={
            "journal_lines.debit": "debit",
            "journal_lines.credit": "credit",
            "trial_balance.debit_balance": "account_balance",
            "trial_balance.credit_balance": "account_balance",
            "balance_sheet.ending_balance": "account_balance",
            "invoices.amount": "transaction_amount",
            "payments.amount": "transaction_amount",
            "bank_transactions.amount": "transaction_amount",
        },
        cycles=(
            {
                "canonical_type": "journal_entry_cycle",
                "key_tables": ["journal_entries", "journal_lines"],
                "required": True,
            },
            # accounts_payable is the corpus's DIRECTED backbone cycle — family and
            # direction are DECLARED truth (vendor→invoice→payment settles OUTGOING),
            # never a truth patch. Undirected cycles carry neither.
            {
                "canonical_type": "accounts_payable",
                "key_tables": ["invoices", "payments"],
                "required": True,
                "family": "settlement",
                "direction": "outgoing",
            },
            {
                "canonical_type": "bank_reconciliation",
                "key_tables": ["bank_transactions", "payments"],
                "required": True,
            },
            # Real but weakly signalled → soft.
            {
                "canonical_type": "period_close",
                "key_tables": ["trial_balance", "balance_sheet"],
                "required": False,
            },
        ),
        # A fact's OWN operational primary key grounds to no dimension concept and
        # carries NO cross-table identity, so the right answer is to ABSTAIN. This is
        # the journal-line PK surviving the fold as the wide fact's own key — the exact
        # surface on which a distinct-count ratio wrongly asserts a hierarchy.
        degenerate_ids={"flat": ("general_ledger.line_id",), "single": ("mega_table.line_id",)},
        event_fact="journal_lines",
    ),
)

OPERATING_CHAIN = Family(
    name="operating_chain",
    description="Customer → sales order → order line → AR invoice → receipt, with products.",
    tables=(
        "customers",
        "products",
        "sales_orders",
        "sales_order_lines",
        "ar_invoices",
        "receipts",
    ),
    primary_keys={
        "customers": "customer_id",
        "products": "product_id",
        "sales_orders": "order_id",
        "sales_order_lines": "order_line_id",
        "ar_invoices": "ar_invoice_id",
        "receipts": "receipt_id",
    },
    natural_prefixes={
        "customer_id": "CUST",
        "product_id": "PROD",
        "order_id": "SO",
        "order_line_id": "SOL",
        "ar_invoice_id": "ARI",
        "receipt_id": "RCPT",
    },
    legacy_names={
        "customer_id": "CUST_ID",
        "product_id": "PROD_ID",
        "order_id": "ORD_NO",
        "order_line_id": "ORD_LN",
        "ar_invoice_id": "ARINV_NO",
        "receipt_id": "RCPT_ID",
        "product_group": "PROD_GRP",
        "standard_cost": "STD_COST",
        "list_price": "LST_PRC",
        "unit_price": "UNIT_PRC",
        "line_amount": "LN_AMT",
        "line_cost": "LN_COST",
        "created_date": "CRTD_DT",
        "churned_date": "CHRN_DT",
        "launched_date": "LNCH_DT",
        "discontinued_date": "DISC_DT",
        "order_date": "ORD_DT",
        "invoice_date": "INV_DT",
        "receipt_date": "RCPT_DT",
        "segment": "SEGMNT",
        "region": "REGN",
        "units": "QTY",
        "name": "NM",
    },
    foreign_keys=(
        ("sales_orders.customer_id", "customers.customer_id"),
        ("sales_order_lines.order_id", "sales_orders.order_id"),
        ("sales_order_lines.product_id", "products.product_id"),
        ("ar_invoices.order_id", "sales_orders.order_id"),
        ("ar_invoices.customer_id", "customers.customer_id"),
        ("receipts.ar_invoice_id", "ar_invoices.ar_invoice_id"),
        ("receipts.customer_id", "customers.customer_id"),
    ),
    # The chain's own header/item pair, folded exactly like journal_data. Customers and
    # products stay as lookups at this level: they are dimension masters, not the
    # order's parent.
    merges=(Merge(name="sales_data", spine="sales_order_lines", joined="sales_orders", on="order_id"),),
    structure=Structure(
        facts=("sales_order_lines", "ar_invoices", "receipts"),
        dimensions=("customers",),
        # sales_orders is an event header carrying no measure; products is a reference
        # row that DOES carry rate-like measures (standard_cost / list_price).
        ambiguous=("sales_orders", "products"),
        measures=(
            # `units` is a count and `unit_price` a rate — both are measures in the
            # graded sense (numeric, driver/slicing input); whether they SUM is the
            # additivity verdict's question, not this role's.
            "sales_order_lines.units",
            "sales_order_lines.unit_price",
            "sales_order_lines.line_amount",
            "sales_order_lines.line_cost",
            "ar_invoices.amount",
            "receipts.amount",
            # Prices on a product row: constant per entity, the same shape as fx_rates.rate.
            "products.standard_cost",
            "products.list_price",
        ),
        timestamps=(
            "sales_orders.order_date",
            "ar_invoices.invoice_date",
            "ar_invoices.due_date",
            "receipts.receipt_date",
            # Master-data validity windows (§9) — the birth/death evidence. A consumer
            # computing a prior-period comparison without them reads a customer's
            # non-existence as a collapse.
            "customers.created_date",
            "customers.churned_date",
            "products.launched_date",
            "products.discontinued_date",
        ),
        measured_in={
            # No in-table currency column on either — the chain is single-currency there
            # — so the unit source is undeclared, exactly like the derived balance
            # tables. Authored as None deliberately: inventing a currency column to give
            # these a unit source would be writing the schema to suit the truth file.
            "products.standard_cost": None,
            "products.list_price": None,
            "sales_order_lines.unit_price": None,
            "sales_order_lines.line_amount": None,
            "sales_order_lines.line_cost": None,
            # These two DO carry an in-table currency column.
            "ar_invoices.amount": "ar_invoices.currency",
            "receipts.amount": "receipts.currency",
        },
    ),
)

INVENTORY = Family(
    name="inventory",
    description="The stock subledger under GL 1400: movements at document grain, positions at period grain.",
    tables=("stock_movements", "inventory_positions"),
    primary_keys={
        "stock_movements": "movement_id",
        # inventory_positions is keyed by the compound (product, location, period).
    },
    natural_prefixes={"movement_id": "STK"},
    legacy_names={
        "movement_id": "STK_MVT",
        "location_id": "LOC_ID",
        "movement_type": "MVT_TYP",
        "unit_cost": "UNIT_CST",
        "units_on_hand": "QTY_OH",
        "value": "VAL",
        "source_document": "SRC_DOC",
    },
    foreign_keys=(
        ("stock_movements.product_id", "products.product_id"),
        ("stock_movements.entry_id", "journal_entries.entry_id"),
        ("inventory_positions.product_id", "products.product_id"),
    ),
    structure=Structure(
        # The movement is an event fact, the position a periodic snapshot fact — both
        # measure-bearing, neither a reference table.
        facts=("stock_movements", "inventory_positions"),
        measures=(
            "stock_movements.units",
            "stock_movements.unit_cost",
            "stock_movements.value",
            "inventory_positions.units_on_hand",
            "inventory_positions.unit_cost",
            "inventory_positions.value",
        ),
        # `inventory_positions.period` is the same shape as the balance tables' period:
        # a period label, not a date.
        timestamps=("stock_movements.date", "inventory_positions.period"),
        # The corpus's second stock/flow PAIR, and a sharper one than the balance
        # tables: the movement and the position sit in different tables over the same
        # key space, so nothing but meaning separates "how much moved" from "how much is
        # there". Signed movement units sum across time; on-hand does not.
        stock_flow={
            "stock_movements.units": "additive",
            "stock_movements.value": "additive",
            "inventory_positions.units_on_hand": "point_in_time",
            "inventory_positions.value": "point_in_time",
        },
        # The position reconciles `cumulative` against its movements, exactly as
        # balance_sheet does against journal_lines — the same witness over a second,
        # independent subledger.
        reconciles_structurally=("inventory_positions.units_on_hand", "inventory_positions.value"),
        # A position is the cumulative sum of stock_movements at its own (product,
        # location) key and only incidentally equal to a GL balance, so pointing its
        # lineage at the corpus event fact would name the wrong finer grain.
        subledger_event_fact={
            "inventory_positions.units_on_hand": "stock_movements",
            "inventory_positions.value": "stock_movements",
        },
        # Money columns only, and no in-table currency column to bind them to. The
        # quantity columns are outside this map entirely — it pairs MONETARY measures
        # with their denomination, and a count of pieces has no currency.
        measured_in={
            "stock_movements.unit_cost": None,
            "stock_movements.value": None,
            "inventory_positions.unit_cost": None,
            "inventory_positions.value": None,
        },
    ),
)

PROBES = Family(
    name="probes",
    description="Labelled shapes for structure detection: role-playing FKs, relationship pairs, stock/flow, formulas.",
    tables=(
        "measure_probes",
        "formula_probes",
        "ref_entities",
        "ref_activity",
        "addresses",
        "orders",
        "deliveries",
    ),
    primary_keys={
        "measure_probes": "series_id",
        "formula_probes": "probe_id",
        "ref_entities": "entity_seq",
        "ref_activity": "activity_seq",
        "addresses": "address_id",
        # `orders` claims `order_id` too — see AMBIGUOUS_KEY_COLUMNS below.
        "orders": "order_id",
        "deliveries": "delivery_id",
    },
    natural_prefixes={"address_id": "ADDR", "delivery_id": "DLV"},
    legacy_names={"address_id": "ADDR_ID", "delivery_id": "DLVRY_ID", "street": "STRT", "city": "CTY"},
    optional=True,
)

PAYER_DIMENSION = Family(
    name="payer_dimension",
    description="A high-cardinality Zipfian payer axis on the bank statement — thousands of merchants, half of them seen once.",
    tables=("merchants",),
    primary_keys={"merchants": "merchant_id"},
    natural_prefixes={"merchant_id": "MRC"},
    legacy_names={
        "merchant_id": "MRCH_ID",
        "merchant_name": "MRCH_NM",
        "category": "MRCH_CAT",
        "country": "CTRY",
    },
    foreign_keys=(("bank_transactions.merchant_id", "merchants.merchant_id"),),
    structure=Structure(
        # A pure reference row: an id, a name and two attributes, no measure. No
        # `business_concepts` — those key a MEASURE by `table.column`, and this table
        # has none; a table-keyed entry among them breaks the canonical remap.
        dimensions=("merchants",),
    ),
    optional=True,
)

FAMILIES: tuple[Family, ...] = (CORE_LEDGER, OPERATING_CHAIN, INVENTORY, PAYER_DIMENSION, PROBES)

# A column that exists only while its optional dimension does. Without this, adding
# the payer dimension would put an all-null `merchant_id` on every bank_transactions
# export ever produced — a bytes change for a feature the corpus never asked for.
OPTIONAL_DIMENSION_COLUMNS: Mapping[str, tuple[str, str]] = {
    "bank_transactions": ("merchant_id", "merchants"),
}


def all_tables() -> tuple[str, ...]:
    """Every declared table, in family order."""
    return tuple(table for fam in FAMILIES for table in fam.tables)


def default_tables() -> tuple[str, ...]:
    """The tables every corpus carries — everything but the optional probe families.

    This is the number a table-count assertion actually means. Spelling it as a
    literal makes every such test a maintenance tax on the next family, and the tax
    is paid in magic numbers that no longer say what they counted.
    """
    return tuple(table for fam in FAMILIES if not fam.optional for table in fam.tables)


def default_families() -> tuple[str, ...]:
    """The families every corpus carries — the family set in the corpus identity.

    Optional probe families are activated by the *strategy*, which is itself part of
    that identity, so they are not listed separately: the whole tuple stays knowable
    before generation runs, and therefore pinnable.
    """
    return tuple(fam.name for fam in FAMILIES if not fam.optional)


def _claims() -> dict[str, set[str]]:
    """Key column -> the set of tables claiming it as their primary key."""
    claims: dict[str, set[str]] = {}
    for fam in FAMILIES:
        for table, column in fam.primary_keys.items():
            claims.setdefault(column, set()).add(table)
    return claims


def ambiguous_key_columns() -> frozenset[str]:
    """Key columns claimed by more than one table.

    ``order_id`` is the live case: the operating chain's sales order and the
    role-play probe fact both use it for unrelated id spaces. Remapping such a
    column would fuse two populations into one, so key strategies leave it alone
    rather than guess — the ambiguity is real and belongs in the data.
    """
    return frozenset(col for col, tables in _claims().items() if len(tables) > 1)


def key_columns() -> dict[str, str]:
    """Unambiguous key column -> the table that owns it."""
    return {col: next(iter(tables)) for col, tables in _claims().items() if len(tables) == 1}


def natural_key_prefixes() -> dict[str, str]:
    """Key column -> natural-key prefix, merged across families."""
    merged: dict[str, str] = {}
    for fam in FAMILIES:
        merged.update(fam.natural_prefixes)
    return merged


def legacy_names() -> dict[str, str]:
    """Canonical column -> legacy spelling, merged across families."""
    merged: dict[str, str] = {}
    for fam in FAMILIES:
        merged.update(fam.legacy_names)
    return merged


def merges() -> tuple[Merge, ...]:
    """Every declared parent/child merge, in family order.

    Order is the contract: ``partial`` applies them in this sequence, and a fold later
    references a merged name (``journal_data`` → ``general_ledger``), so a family
    declaring a merge that another family folds must come first.
    """
    return tuple(merge for fam in FAMILIES for merge in fam.merges)


def folds() -> tuple[Fold, ...]:
    """Every declared dimension fold, in family order."""
    return tuple(fold for fam in FAMILIES for fold in fam.folds)


def merge_column_renames() -> dict[str, str]:
    """``joined_table.column -> new column name`` for every merge that renames.

    The remap ``metadata_truth`` needs to keep qualified references valid after a merge.
    It used to carry its own copy of this map, one file away from the transform that
    performed the rename.
    """
    return {
        f"{merge.joined}.{old}": new
        for fam in FAMILIES
        for merge in fam.merges
        for old, new in merge.rename.items()
    }


# --- the structural truth, merged across families ---------------------------
#
# Every accessor below merges in family order. `metadata_truth` publishes what it finds
# and authors nothing about tables — which is the point: it used to hold one canonical
# blob listing finance tables by name, so a family's roles, units and cycles were a
# second place to remember.


def table_roles() -> dict[str, list[str]]:
    """``{facts, dimensions, ambiguous}`` — the table classification, merged."""
    return {
        "facts": [t for fam in FAMILIES for t in fam.structure.facts],
        "dimensions": [t for fam in FAMILIES for t in fam.structure.dimensions],
        "ambiguous": [t for fam in FAMILIES for t in fam.structure.ambiguous],
    }


def semantic_roles() -> dict[str, list[str]]:
    """``{measure, timestamp}`` per qualified column, merged.

    key / dimension / attribute are deliberately absent — they are convention-dependent,
    so the oracle reports them rather than asserting them.
    """
    return {
        "measure": [c for fam in FAMILIES for c in fam.structure.measures],
        "timestamp": [c for fam in FAMILIES for c in fam.structure.timestamps],
    }


def stock_flow() -> dict[str, str]:
    merged: dict[str, str] = {}
    for fam in FAMILIES:
        merged.update(fam.structure.stock_flow)
    return merged


def reconciles_structurally() -> list[str]:
    return [c for fam in FAMILIES for c in fam.structure.reconciles_structurally]


def subledger_event_facts() -> dict[str, str]:
    merged: dict[str, str] = {}
    for fam in FAMILIES:
        merged.update(fam.structure.subledger_event_fact)
    return merged


def measured_in() -> dict[str, str | None]:
    merged: dict[str, str | None] = {}
    for fam in FAMILIES:
        merged.update(fam.structure.measured_in)
    return merged


def dimensionless_measures() -> frozenset[str]:
    return frozenset(c for fam in FAMILIES for c in fam.structure.dimensionless)


def business_concepts() -> dict[str, str]:
    merged: dict[str, str] = {}
    for fam in FAMILIES:
        merged.update(fam.structure.business_concepts)
    return merged


def cycles() -> list[dict[str, Any]]:
    return [dict(cycle) for fam in FAMILIES for cycle in fam.structure.cycles]


def degenerate_ids(level: str) -> list[str]:
    return [q for fam in FAMILIES for q in fam.structure.degenerate_ids.get(level, ())]


def event_fact() -> str | None:
    """The corpus's event-grain fact — the finest table a structural witness ties to.

    One family owns it (the ledger's ``journal_lines``); a subledger that reconciles
    against its OWN finer fact says so in ``subledger_event_fact`` instead.
    """
    for fam in FAMILIES:
        if fam.structure.event_fact:
            return fam.structure.event_fact
    return None


def folded_dimensions() -> frozenset[str]:
    """Dimension tables a fold REMOVES from the corpus at ``flat`` and below.

    Their content fans out into several facts, so no single ``old -> new`` rename can
    express what happened to them — which is why they need naming separately from
    ``table_mapping``.
    """
    return frozenset(fold.dimension for fold in folds())


def table_mapping(level: str) -> dict[str, str]:
    """The ``old -> new`` table rename *level* produces for a corpus of default tables.

    Name algebra over the same declarations ``apply_normalization`` executes over
    frames, so the two cannot disagree about what a level does. Callers with a live run
    still pass the real mapping; this is the answer for tests, offline truth builds, and
    anything reasoning about a level it is not currently generating.
    """
    if level == "full":
        return {}

    mapping: dict[str, str] = {}
    for merge in merges():
        mapping[merge.spine] = merge.name
        mapping[merge.joined] = merge.name
    if level == "partial":
        return mapping

    for fold in folds():
        for fact, result in fold.into.items():
            if result == fact:
                continue
            for old, new in list(mapping.items()):
                if new == fact:
                    mapping[old] = result
    if level == "flat":
        return mapping

    removed = folded_dimensions()
    return {table: "mega_table" for table in default_tables() if table not in removed}


def foreign_keys() -> tuple[tuple[str, str], ...]:
    """Every declared FK as ``(from_column, to_column)``, in family order.

    The source of the FK topology `metadata_truth` publishes. Probe families are
    included: their FKs are real when the shape materializes, and `metadata_truth`
    already gates the role-play edges on the frames being present.
    """
    return tuple(fk for fam in FAMILIES for fk in fam.foreign_keys)
