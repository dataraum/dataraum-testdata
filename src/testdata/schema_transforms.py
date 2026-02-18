"""Schema transforms — reshape DataFrames to different normalization levels.

Entropy injections run *before* these transforms, so injection specs use
the original (fully-normalized) table names.  The transform returns a
table-name mapping that ``run_scenario`` uses to update registry entries.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

NormalizationLevel = Literal["full", "partial", "flat"]


def apply_normalization(
    dataframes: dict[str, pl.DataFrame],
    level: NormalizationLevel = "full",
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """Reshape *dataframes* according to *level*.

    Returns:
        (transformed_dataframes, old_name → new_name mapping)
        The mapping only contains entries for tables whose name changed.
    """
    if level == "full":
        return dataframes, {}

    out = dict(dataframes)
    mapping: dict[str, str] = {}

    # partial: merge parent→child pairs
    out, mapping = _merge_journal_data(out, mapping)
    out, mapping = _merge_invoice_data(out, mapping)

    if level == "partial":
        return out, mapping

    # flat: additionally inline lookups
    out, mapping = _inline_chart_of_accounts(out, mapping)

    return out, mapping


# ---------------------------------------------------------------------------
# partial helpers
# ---------------------------------------------------------------------------

def _merge_journal_data(
    dfs: dict[str, pl.DataFrame],
    mapping: dict[str, str],
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """journal_data = journal_lines LEFT JOIN journal_entries ON entry_id."""
    lines = dfs.pop("journal_lines")
    entries = dfs.pop("journal_entries")

    # No column conflicts — join directly.
    journal_data = lines.join(entries, on="entry_id", how="left")

    dfs["journal_data"] = journal_data
    mapping["journal_lines"] = "journal_data"
    mapping["journal_entries"] = "journal_data"
    return dfs, mapping


def _merge_invoice_data(
    dfs: dict[str, pl.DataFrame],
    mapping: dict[str, str],
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """invoice_data = invoices LEFT JOIN payments ON invoice_id.

    Conflicting payment columns are prefixed with ``payment_``.
    """
    invoices = dfs.pop("invoices")
    payments = dfs.pop("payments")

    # Rename conflicting + ambiguous payment columns before join.
    payments = payments.rename({
        "date": "payment_date",
        "amount": "payment_amount",
        "currency": "payment_currency",
        "method": "payment_method",
    })

    invoice_data = invoices.join(payments, on="invoice_id", how="left")

    dfs["invoice_data"] = invoice_data
    mapping["invoices"] = "invoice_data"
    mapping["payments"] = "invoice_data"
    return dfs, mapping


# ---------------------------------------------------------------------------
# flat helpers
# ---------------------------------------------------------------------------

def _inline_chart_of_accounts(
    dfs: dict[str, pl.DataFrame],
    mapping: dict[str, str],
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """Inline chart_of_accounts into journal_data → general_ledger, and
    enrich trial_balance with account metadata.
    """
    coa = dfs.pop("chart_of_accounts")

    # Prepare CoA columns: rename to avoid conflicts.
    coa_renamed = coa.rename({
        "name": "account_name",
        "parent_id": "parent_account_id",
        "currency": "account_currency",
    })

    # general_ledger = journal_data LEFT JOIN coa ON account_id
    journal_data = dfs.pop("journal_data")
    general_ledger = journal_data.join(coa_renamed, on="account_id", how="left")
    dfs["general_ledger"] = general_ledger

    # Update mapping: anything that pointed to journal_data now points to general_ledger
    for old, new in list(mapping.items()):
        if new == "journal_data":
            mapping[old] = "general_ledger"

    # trial_balance = trial_balance LEFT JOIN coa ON account_id
    tb = dfs.pop("trial_balance")
    tb_enriched = tb.join(coa_renamed, on="account_id", how="left")
    dfs["trial_balance"] = tb_enriched
    # trial_balance keeps its name, so no mapping entry needed.

    return dfs, mapping
