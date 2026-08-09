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

FAMILIES: tuple[Family, ...] = (CORE_LEDGER, OPERATING_CHAIN, INVENTORY, PROBES)


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


def foreign_keys() -> tuple[tuple[str, str], ...]:
    """Every declared FK as ``(from_column, to_column)``, in family order.

    The source of the FK topology `metadata_truth` publishes. Probe families are
    included: their FKs are real when the shape materializes, and `metadata_truth`
    already gates the role-play edges on the frames being present.
    """
    return tuple(fk for fam in FAMILIES for fk in fam.foreign_keys)
