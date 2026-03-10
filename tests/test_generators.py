"""Tests for deterministic finance data generators."""

from collections import Counter
from decimal import Decimal

from testdata.canonical.finance.generators import generate_finance_dataset
from testdata.canonical.finance.models import JournalStatus


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
    assert len(ds.bank_transactions) >= 4000  # Event-driven: derived from business events
    assert len(ds.fx_rates) >= 400
    assert len(ds.trial_balance) >= 200  # accounts_used × months


def test_balanced_journals():
    """Every journal entry has sum(debit) == sum(credit)."""
    ds = _dataset()
    lines_by_entry: dict[str, list] = {}
    for line in ds.journal_lines:
        lines_by_entry.setdefault(line.entry_id, []).append(line)

    for entry_id, lines in lines_by_entry.items():
        total_debit = sum(line.debit for line in lines)
        total_credit = sum(line.credit for line in lines)
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


# --- Closed-loop accounting tests ---


def test_trial_balance_derived_from_gl():
    """Trial balance matches cumulative GL entries within the fiscal year."""
    ds = _dataset()
    from datetime import date

    fiscal_start = date(2025, 1, 1)
    fiscal_end = date(2025, 12, 31)

    # Only count entries within the fiscal year (some payments spill into next year)
    entry_info = {}
    for e in ds.journal_entries:
        if e.status == JournalStatus.POSTED and fiscal_start <= e.date <= fiscal_end:
            entry_info[e.entry_id] = e.date

    # Compute expected final-period balance for each account from fiscal-year GL
    account_debits: dict[str, Decimal] = {}
    account_credits: dict[str, Decimal] = {}
    for line in ds.journal_lines:
        if line.entry_id not in entry_info:
            continue
        account_debits[line.account_id] = account_debits.get(line.account_id, Decimal("0")) + line.debit
        account_credits[line.account_id] = account_credits.get(line.account_id, Decimal("0")) + line.credit

    # Check final period TB matches total GL within fiscal year
    final_period = "2025-12"
    tb_final = {tb.account_id: tb for tb in ds.trial_balance if tb.period == final_period}

    for acct in account_debits:
        if acct in tb_final:
            assert tb_final[acct].debit_balance == account_debits[acct].quantize(Decimal("0.01")), (
                f"Account {acct}: TB debit {tb_final[acct].debit_balance} != GL debit {account_debits[acct]}"
            )
            assert tb_final[acct].credit_balance == account_credits.get(acct, Decimal("0")).quantize(Decimal("0.01")), (
                f"Account {acct}: TB credit {tb_final[acct].credit_balance} != GL credit {account_credits.get(acct, Decimal('0'))}"
            )


def test_trial_balance_balanced():
    """Total debits equal total credits in every period."""
    ds = _dataset()
    periods: dict[str, tuple[Decimal, Decimal]] = {}
    for tb in ds.trial_balance:
        d, c = periods.get(tb.period, (Decimal("0"), Decimal("0")))
        periods[tb.period] = (d + tb.debit_balance, c + tb.credit_balance)

    for period, (total_d, total_c) in periods.items():
        assert total_d == total_c, (
            f"Period {period}: total debit {total_d} != total credit {total_c}"
        )


def test_invoice_gl_linkage():
    """Every non-cancelled invoice has a corresponding GL entry."""
    ds = _dataset()
    # GL entries for invoices have description containing the invoice_id
    gl_invoice_ids = set()
    for entry in ds.journal_entries:
        if "Vendor invoice" in entry.description:
            # Extract invoice ID from description: "Vendor invoice - Vendor - INV-000001"
            parts = entry.description.split(" - ")
            if len(parts) >= 3:
                gl_invoice_ids.add(parts[-1])

    non_cancelled = {inv.invoice_id for inv in ds.invoices if inv.status != "cancelled"}
    # Every non-cancelled invoice should have a GL entry
    assert gl_invoice_ids == non_cancelled, (
        f"Missing GL for {len(non_cancelled - gl_invoice_ids)} invoices"
    )


def test_payment_creates_bank_transaction():
    """Every vendor payment has a matching bank transaction (by amount)."""
    ds = _dataset()
    # Collect payment amounts (they become negative bank transactions)
    payment_amounts = Counter(float(p.amount) for p in ds.payments)
    # Collect negative (outflow) bank transactions to vendors
    bank_outflows = Counter(-float(bt.amount) for bt in ds.bank_transactions if float(bt.amount) < 0)

    # Every payment amount should appear in bank outflows
    for amount, count in payment_amounts.items():
        assert bank_outflows[amount] >= count, (
            f"Payment amount {amount} appears {count} times but only {bank_outflows[amount]} bank outflows"
        )


def test_revenue_creates_ar():
    """Revenue GL entries create AR debits."""
    ds = _dataset()
    ar_accounts = {"1210", "1220"}
    revenue_accounts = set()
    for entry in ds.journal_entries:
        if "Revenue recognition" in entry.description:
            revenue_accounts.add(entry.entry_id)

    # Every revenue entry should have AR debit lines
    lines_by_entry: dict[str, list] = {}
    for line in ds.journal_lines:
        lines_by_entry.setdefault(line.entry_id, []).append(line)

    for entry_id in revenue_accounts:
        entry_lines = lines_by_entry.get(entry_id, [])
        has_ar_debit = any(
            line.account_id in ar_accounts and line.debit > 0
            for line in entry_lines
        )
        assert has_ar_debit, f"Revenue entry {entry_id} has no AR debit"


def test_cash_receipts_reduce_ar():
    """Cash receipt GL entries credit AR and debit Cash."""
    ds = _dataset()
    ar_accounts = {"1210", "1220"}
    cash_accounts = {"1110", "1120"}

    for entry in ds.journal_entries:
        if "Cash receipt" not in entry.description:
            continue
        entry_lines = [l for l in ds.journal_lines if l.entry_id == entry.entry_id]
        has_cash_debit = any(l.account_id in cash_accounts and l.debit > 0 for l in entry_lines)
        has_ar_credit = any(l.account_id in ar_accounts and l.credit > 0 for l in entry_lines)
        assert has_cash_debit, f"Cash receipt {entry.entry_id} has no Cash debit"
        assert has_ar_credit, f"Cash receipt {entry.entry_id} has no AR credit"
