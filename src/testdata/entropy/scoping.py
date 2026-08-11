"""Resolving an injection's target slice — which rows a scoped defect lands on.

An unscoped injector corrupts a column table-wide, which answers "is this column
broken". A *scoped* one lands a defect on a named slice of a named period, which is
what makes "a metric moved — artifact or real change, and which slice drove it" a
question with an answer key: the artifact's lineage is declared, not inferred.

The scope vocabulary is deliberately the same as a lever's — ``segment``, ``region``,
``customer_id``, ``product_group``, ``product_id`` — because the confusable pair needs
an artifact and an intervention aimed at *the same slice*, and two vocabularies would
make that a translation exercise. ``periods`` adds the time axis injections need and
levers get from ``period_k``.

A dimension is reached by a declared path, never by guessing at column names. Tables
that cannot reach a dimension simply cannot be scoped on it, and say so rather than
silently injecting table-wide — an injection recorded as scoped that actually hit
everything is a wrong answer key, which is worse than a missing one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl


CUSTOMER_DIMS = ("segment", "region", "customer_id")
PRODUCT_DIMS = ("product_group", "product_id")
SCOPE_DIMS = CUSTOMER_DIMS + PRODUCT_DIMS

# How each table reaches the customer it belongs to. A one-tuple is a column on the
# table itself; a three-tuple is (local fk, parent table, parent column).
_CUSTOMER_PATH: dict[str, tuple[str, ...]] = {
    "customers": ("customer_id",),
    "sales_orders": ("customer_id",),
    "ar_invoices": ("customer_id",),
    "receipts": ("customer_id",),
    "sales_order_lines": ("order_id", "sales_orders", "customer_id"),
}

_PRODUCT_PATH: dict[str, tuple[str, ...]] = {
    "products": ("product_id",),
    "sales_order_lines": ("product_id",),
    "stock_movements": ("product_id",),
    "inventory_positions": ("product_id",),
}

# The date each table is dated by, for period scoping. Same convention: a one-tuple
# is local, a three-tuple joins a parent that carries the date.
_DATE_PATH: dict[str, tuple[str, ...]] = {
    "sales_orders": ("order_date",),
    "sales_order_lines": ("order_id", "sales_orders", "order_date"),
    "ar_invoices": ("invoice_date",),
    "receipts": ("receipt_date",),
    "invoices": ("date",),
    "payments": ("date",),
    "bank_transactions": ("date",),
    "journal_entries": ("date",),
    "stock_movements": ("date",),
}

_ROW = "__row__"


class UnscopableTable(ValueError):
    """The table cannot reach the dimension asked for.

    Raised rather than silently widening: an injection recorded as scoped that
    actually landed table-wide publishes a lineage the data does not have.
    """


def _resolved(
    df: pl.DataFrame, table: str, path: tuple[str, ...], dataframes: Mapping[str, pl.DataFrame], what: str
) -> pl.Series:
    """The per-row value of a dimension or date, following the declared path."""
    if len(path) == 1:
        return df[path[0]]
    fk, parent_table, parent_col = path
    parent = dataframes[parent_table].select([fk, parent_col])
    joined = df.select([fk]).join(parent, on=fk, how="left")
    if joined.height != df.height:  # a duplicated parent key would silently fan out
        raise UnscopableTable(f"{table}.{fk} does not join {parent_table} one-to-one; cannot scope by {what}")
    return joined[parent_col]


def _members(
    df: pl.DataFrame,
    table: str,
    dataframes: Mapping[str, pl.DataFrame],
    dim: str,
    paths: dict[str, tuple[str, ...]],
    dimension_table: str,
    key: str,
) -> pl.Series:
    """The per-row value of one scope dimension on the target table."""
    path = paths.get(table)
    if path is None:
        raise UnscopableTable(
            f"table {table!r} cannot be scoped by {dim!r}: no declared path to {dimension_table}"
        )
    ids = _resolved(df, table, path, dataframes, dim)
    if dim == key:
        return ids
    lookup = dataframes[dimension_table].select([key, dim])
    return pl.DataFrame({key: ids}).join(lookup, on=key, how="left")[dim]


def slice_rows(
    df: pl.DataFrame,
    table: str,
    dataframes: Mapping[str, pl.DataFrame],
    *,
    scope: Mapping[str, Sequence[str]] | None = None,
    periods: Sequence[str] | None = None,
) -> list[int]:
    """Row indices of ``df`` inside the scope — every row when nothing is scoped.

    Dimensions INTERSECT, matching a lever's scope: ``segment=[Enterprise]`` with
    ``region=[DACH]`` is enterprise accounts in DACH. ``periods`` are ``YYYY-MM``
    strings and intersect with the rest.
    """
    if not scope and not periods:
        return list(range(df.height))

    keep = pl.Series("keep", [True] * df.height)
    for dim, members in (scope or {}).items():
        if dim not in SCOPE_DIMS:
            raise ValueError(f"unknown scope dimension {dim!r} (supported: {list(SCOPE_DIMS)})")
        if dim in CUSTOMER_DIMS:
            values = _members(df, table, dataframes, dim, _CUSTOMER_PATH, "customers", "customer_id")
        else:
            values = _members(df, table, dataframes, dim, _PRODUCT_PATH, "products", "product_id")
        keep = keep & values.is_in(list(members))

    if periods:
        path = _DATE_PATH.get(table)
        if path is None:
            raise UnscopableTable(f"table {table!r} carries no declared date; cannot scope by period")
        dates = _resolved(df, table, path, dataframes, "period")
        keep = keep & dates.cast(pl.Date).dt.strftime("%Y-%m").is_in(list(periods))

    return [i for i, flag in enumerate(keep.to_list()) if flag]


def scope_parameters(
    scope: Mapping[str, Sequence[str]] | None,
    periods: Sequence[str] | None,
    matched: int,
    total: int,
) -> dict[str, Any]:
    """The scope fragment recorded on every scoped injection.

    ``slice_share`` is carried because it is what a consumer needs to reason about
    magnitude and cannot recompute without re-deriving the join.

    Empty when nothing was scoped — the record grows only when the injection used the
    capability, so an unscoped injection's answer key reads exactly as it always did
    rather than carrying four fields that say "the whole table", which is what the
    absence of a scope already says.
    """
    if not scope and not periods:
        return {}
    return {
        "scope": {dim: list(members) for dim, members in scope.items()} if scope else None,
        "periods": list(periods) if periods else None,
        "slice_rows": matched,
        "slice_share": round(matched / total, 6) if total else 0.0,
    }
