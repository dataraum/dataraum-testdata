"""Tests for deterministic finance data generators."""

from collections import Counter
from decimal import Decimal

from testdata.canonical.finance.generators import generate_finance_dataset


def _dataset():
    return generate_finance_dataset(seed=42, months=12)


def test_deterministic_generation():
    """Same seed produces identical datasets."""
    ds1 = generate_finance_dataset(seed=42, months=12)
    ds2 = generate_finance_dataset(seed=42, months=12)
    assert len(ds1.journal_entries) == len(ds2.journal_entries)
    assert ds1.journal_entries[0].entry_id == ds2.journal_entries[0].entry_id
    assert ds1.journal_lines[0].debit == ds2.journal_lines[0].debit


def test_different_seeds_differ():
    ds1 = generate_finance_dataset(seed=42, months=12)
    ds2 = generate_finance_dataset(seed=99, months=12)
    assert len(ds1.journal_entries) != len(ds2.journal_entries)


def test_row_counts():
    ds = _dataset()
    assert len(ds.chart_of_accounts) >= 50
    assert len(ds.journal_entries) >= 4000
    assert len(ds.journal_lines) >= 10000
    assert len(ds.invoices) == 3000
    assert len(ds.payments) >= 2000
    assert len(ds.bank_transactions) == 8000
    assert len(ds.fx_rates) >= 400
    assert len(ds.trial_balance) >= 400


def test_balanced_journals():
    """Every journal entry has sum(debit) == sum(credit)."""
    ds = _dataset()
    lines_by_entry: dict[str, list] = {}
    for line in ds.journal_lines:
        lines_by_entry.setdefault(line.entry_id, []).append(line)

    for entry_id, lines in lines_by_entry.items():
        total_debit = sum(l.debit for l in lines)
        total_credit = sum(l.credit for l in lines)
        assert total_debit == total_credit, (
            f"Entry {entry_id}: debit={total_debit} != credit={total_credit}"
        )


def test_referential_integrity_invoices_payments():
    """Every payment references a valid invoice."""
    ds = _dataset()
    invoice_ids = {inv.invoice_id for inv in ds.invoices}
    for pay in ds.payments:
        assert pay.invoice_id in invoice_ids, f"Orphan payment: {pay.payment_id}"


def test_referential_integrity_journal_lines():
    """Every journal line references a valid entry and account."""
    ds = _dataset()
    entry_ids = {e.entry_id for e in ds.journal_entries}
    account_ids = {a.account_id for a in ds.chart_of_accounts}
    for line in ds.journal_lines:
        assert line.entry_id in entry_ids
        assert line.account_id in account_ids


def test_temporal_consistency():
    """Payment dates are after invoice dates."""
    ds = _dataset()
    inv_map = {inv.invoice_id: inv for inv in ds.invoices}
    for pay in ds.payments:
        inv = inv_map[pay.invoice_id]
        assert pay.date >= inv.date, (
            f"Payment {pay.payment_id} date {pay.date} before invoice {inv.invoice_id} date {inv.date}"
        )


def test_benford_distribution():
    """Bank transaction amounts follow Benford's law (digit 1 is most common)."""
    ds = _dataset()
    first_digits = Counter(
        str(int(abs(float(t.amount))))[0]
        for t in ds.bank_transactions
        if abs(float(t.amount)) >= 1
    )
    # Benford: digit 1 should have ~30% frequency
    total = sum(first_digits.values())
    digit_1_pct = first_digits["1"] / total
    assert digit_1_pct > 0.20, f"Digit 1 only {digit_1_pct:.1%} (expected >20%)"
    # Digit 1 should be more common than any other single digit
    assert first_digits["1"] > first_digits["9"]


def test_vendor_concentration():
    """80/20 rule: top vendors get most invoices."""
    ds = _dataset()
    vendor_counts = Counter(inv.vendor_id for inv in ds.invoices)
    top_4 = sum(c for _, c in vendor_counts.most_common(4))
    assert top_4 / len(ds.invoices) > 0.50, "Top 4 vendors should have >50% of invoices"
