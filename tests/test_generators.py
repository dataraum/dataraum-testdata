"""Tests for deterministic finance data generators."""

import functools
import random
from collections import Counter
from datetime import date
from decimal import Decimal

import pytest

from testdata.canonical.finance.generators import generate_finance_dataset
from testdata.canonical.finance.models import AccountType, JournalStatus


@functools.lru_cache(maxsize=4)
def _dataset(seed: int = 42, months: int = 12):
    return generate_finance_dataset(seed=seed, months=months)


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
    # 3000 expense bills plus one per goods delivery — the stock subledger raises its
    # own payables, so the vendor-bill population is no longer a fixed count.
    assert sum(1 for i in ds.invoices if i.category == "expense") == 3000
    assert sum(1 for i in ds.invoices if i.category == "goods") > 0
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
        assert total_debit == total_credit, f"Entry {entry_id}: debit={total_debit} != credit={total_credit}"


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
    first_digits = Counter(str(int(abs(float(t.amount))))[0] for t in ds.bank_transactions if abs(float(t.amount)) >= 1)
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
    """Trial balance per-period activity sums to total GL entries."""
    ds = _dataset()

    # Compute total GL debits/credits per account (all POSTED entries)
    entry_status = {e.entry_id: e.status for e in ds.journal_entries}
    account_debits: dict[str, Decimal] = {}
    account_credits: dict[str, Decimal] = {}
    for line in ds.journal_lines:
        if entry_status.get(line.entry_id) != JournalStatus.POSTED:
            continue
        account_debits[line.account_id] = account_debits.get(line.account_id, Decimal("0")) + line.debit
        account_credits[line.account_id] = account_credits.get(line.account_id, Decimal("0")) + line.credit

    # Sum all TB periods per account — should match GL totals
    tb_debits: dict[str, Decimal] = {}
    tb_credits: dict[str, Decimal] = {}
    for tb in ds.trial_balance:
        tb_debits[tb.account_id] = tb_debits.get(tb.account_id, Decimal("0")) + tb.debit_balance
        tb_credits[tb.account_id] = tb_credits.get(tb.account_id, Decimal("0")) + tb.credit_balance

    for acct in account_debits:
        assert acct in tb_debits, f"Account {acct} in GL but missing from TB"
        assert tb_debits[acct] == account_debits[acct].quantize(Decimal("0.01")), (
            f"Account {acct}: TB debit {tb_debits[acct]} != GL debit {account_debits[acct]}"
        )
        assert tb_credits.get(acct, Decimal("0")) == account_credits.get(acct, Decimal("0")).quantize(
            Decimal("0.01")
        ), (
            f"Account {acct}: TB credit {tb_credits.get(acct, Decimal('0'))} != GL credit {account_credits.get(acct, Decimal('0'))}"
        )


def test_balance_sheet_is_carry_forward_stock():
    """balance_sheet.ending_balance is a STOCK: Δ per period == GL net movement.

    The defining property the temporal_behavior reconciliation witness keys on —
    the change in the carried-forward level equals that period's net movement
    (debit − credit), so the column reconciles as a stock (not a per-period flow).
    """
    ds = _dataset()
    assert ds.balance_sheet, "balance_sheet must be populated"

    # Only balance-sheet accounts (asset/liability/equity) appear.
    bs_accounts = {
        c.account_id
        for c in ds.chart_of_accounts
        if c.account_type in (AccountType.ASSET, AccountType.LIABILITY, AccountType.EQUITY)
    }
    assert {b.account_id for b in ds.balance_sheet} <= bs_accounts

    # Independent per-period net movement (debit − credit) from POSTED GL.
    entry_period = {e.entry_id: (e.date.strftime("%Y-%m"), e.status) for e in ds.journal_entries}
    gl_net: dict[tuple[str, str], Decimal] = {}
    for line in ds.journal_lines:
        info = entry_period.get(line.entry_id)
        if info is None or info[1] != JournalStatus.POSTED:
            continue
        period, _ = info
        key = (line.account_id, period)
        gl_net[key] = gl_net.get(key, Decimal("0")) + line.debit - line.credit

    # Per account, ending_balance must carry forward: ending[t] − ending[t-1] == net[t].
    by_acct: dict[str, list] = {}
    for b in ds.balance_sheet:
        by_acct.setdefault(b.account_id, []).append(b)
    checked = 0
    for acct, rows in by_acct.items():
        rows.sort(key=lambda b: b.period)
        prev = Decimal("0")
        for b in rows:
            delta = b.ending_balance - prev
            expected = gl_net.get((acct, b.period), Decimal("0")).quantize(Decimal("0.01"))
            assert delta == expected, f"{acct} {b.period}: Δending {delta} != GL net {expected} (not carry-forward)"
            prev = b.ending_balance
            checked += 1
    assert checked >= 50


def test_balance_sheet_persists_across_no_activity_periods():
    """A stock carries forward even when a period has no movement (dense periods)."""
    ds = _dataset()
    by_acct: dict[str, list] = {}
    for b in ds.balance_sheet:
        by_acct.setdefault(b.account_id, []).append(b)
    # At least one account must span contiguous periods with no gaps (dense carry-forward).
    spanned = max(len(rows) for rows in by_acct.values())
    assert spanned >= 6, "expected dense per-period balances for active BS accounts"


def test_trial_balance_balanced():
    """Total debits equal total credits in every period."""
    ds = _dataset()
    periods: dict[str, tuple[Decimal, Decimal]] = {}
    for tb in ds.trial_balance:
        d, c = periods.get(tb.period, (Decimal("0"), Decimal("0")))
        periods[tb.period] = (d + tb.debit_balance, c + tb.credit_balance)

    for period, (total_d, total_c) in periods.items():
        assert total_d == total_c, f"Period {period}: total debit {total_d} != total credit {total_c}"


def test_invoice_gl_linkage():
    """Every non-cancelled invoice carries an entry_id that resolves, and only those do.

    Checked on the FK rather than the entry description: two populations now raise
    payables (expense bills and goods receipts) and they describe themselves
    differently, so a description match would silently pass by only testing one.
    """
    ds = _dataset()
    entry_ids = {e.entry_id for e in ds.journal_entries}

    for inv in ds.invoices:
        if inv.status == "cancelled":
            assert inv.entry_id is None, f"{inv.invoice_id} was cancelled but posted"
            continue
        assert inv.entry_id in entry_ids, f"{inv.invoice_id} has no GL entry"

    # A goods bill is never cancelled: the pallet arrived and the movement is booked.
    assert all(i.status != "cancelled" for i in ds.invoices if i.category == "goods")
    assert {i.category for i in ds.invoices} == {"expense", "goods"}


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
        has_ar_debit = any(line.account_id in ar_accounts and line.debit > 0 for line in entry_lines)
        assert has_ar_debit, f"Revenue entry {entry_id} has no AR debit"


def test_cash_receipts_reduce_ar():
    """Cash receipt GL entries credit AR and debit Cash."""
    ds = _dataset()
    ar_accounts = {"1210", "1220"}
    cash_accounts = {"1110", "1120"}

    lines_by_entry: dict[str, list] = {}
    for line in ds.journal_lines:
        lines_by_entry.setdefault(line.entry_id, []).append(line)

    for entry in ds.journal_entries:
        if "Cash receipt" not in entry.description:
            continue
        entry_lines = lines_by_entry.get(entry.entry_id, [])
        has_cash_debit = any(line.account_id in cash_accounts and line.debit > 0 for line in entry_lines)
        has_ar_credit = any(line.account_id in ar_accounts and line.credit > 0 for line in entry_lines)
        assert has_cash_debit, f"Cash receipt {entry.entry_id} has no Cash debit"
        assert has_ar_credit, f"Cash receipt {entry.entry_id} has no AR credit"


# --- Non-January fiscal start ---


def test_non_january_fiscal_start():
    """October fiscal year produces entries spanning two calendar years."""
    ds = generate_finance_dataset(seed=42, months=12, fiscal_start=date(2025, 10, 1))

    # Should cover Oct 2025 through Sep 2026
    entry_dates = [e.date for e in ds.journal_entries]
    assert min(entry_dates).month == 10
    assert min(entry_dates).year == 2025
    assert max(entry_dates).year >= 2026

    # TB should have periods starting from 2025-10, may extend past fiscal year
    # due to late payments that spill into subsequent months
    tb_periods = sorted({tb.period for tb in ds.trial_balance})
    assert tb_periods[0] == "2025-10"
    assert len(tb_periods) >= 12

    # Balanced journals still hold
    lines_by_entry: dict[str, list] = {}
    for line in ds.journal_lines:
        lines_by_entry.setdefault(line.entry_id, []).append(line)

    for entry_id, lines in lines_by_entry.items():
        total_debit = sum(line.debit for line in lines)
        total_credit = sum(line.credit for line in lines)
        assert total_debit == total_credit, f"Entry {entry_id} unbalanced"


def test_formula_probes_skeleton_generated_only_when_requested():
    """The formula-divergence probe grain is opt-in — empty unless a strategy needs it."""
    assert generate_finance_dataset(seed=1, months=2).formula_probes == []
    probes = generate_finance_dataset(seed=1, months=2, formula_probe_rows=25).formula_probes
    ids = [p.probe_id for p in probes]
    assert len(ids) == 25 and len(set(ids)) == 25


def _monthly_revenue_credit(ds, month: int) -> Decimal:
    """Sum of revenue-account credits for one fiscal month (sales revenue only)."""
    entry_month = {e.entry_id: e.date.month for e in ds.journal_entries}
    return sum(
        (
            line.credit
            for line in ds.journal_lines
            if line.account_id.startswith(("41", "42")) and entry_month[line.entry_id] == month
        ),
        Decimal("0"),
    )


def test_price_level_lever_is_exact_counterfactual():
    """DAT-744: a same-seed (baseline, levered) pair differs ONLY in sale amounts
    from period_k on — identical event stream, pre-lever months untouched,
    post-lever revenue scaled by exactly the factor (up to per-sale cent rounding)."""
    from testdata.canonical.finance.generators import Lever

    base = generate_finance_dataset(seed=7, months=6)
    lev = generate_finance_dataset(seed=7, months=6, lever=Lever(period_k=3, factor=1.2))

    # RNG-stream identity: same events, same dates, same row counts
    assert len(base.journal_entries) == len(lev.journal_entries)
    assert len(base.journal_lines) == len(lev.journal_lines)
    assert all(a.date == b.date for a, b in zip(base.journal_entries, lev.journal_entries))

    # pre-lever months identical, post-lever months scaled by exactly the factor
    for month in (1, 2, 3):  # fiscal months before period_k=3 (offsets 0-2)
        assert _monthly_revenue_credit(base, month) == _monthly_revenue_credit(lev, month)
    for month in (4, 5, 6):  # offsets 3-5, lever active
        base_sum, lev_sum = _monthly_revenue_credit(base, month), _monthly_revenue_credit(lev, month)
        assert abs(float(lev_sum / base_sum) - 1.2) < 1e-4, f"month {month}: ratio {float(lev_sum / base_sum)}"

    # expenditure cycle untouched
    assert len(base.invoices) == len(lev.invoices)
    assert base.invoices[0].amount == lev.invoices[0].amount


def test_chart_of_accounts_opened_date_is_a_bijection() -> None:
    """opened_date is unique per account BY CONSTRUCTION.

    This is load-bearing: once CoA is inlined at `flat`, account_id <-> opened_date
    becomes a non-key bijection on the fact grain — the coincidental-bijection case
    the dimension-identity judge is tested against. A single collision would break the
    bijection and silently void that test, so pin uniqueness here at the source.
    """
    from testdata.canonical.finance.generators import generate_chart_of_accounts

    coa = generate_chart_of_accounts()
    opened = [a.opened_date for a in coa]
    assert len(set(opened)) == len(coa), "opened_date must be unique per account"
    assert len({a.account_id for a in coa}) == len(coa), "account_id must be unique"


# --- Entity-keyed RNG streams (DAT-884 slice 1) ---


def test_stream_is_deterministic_per_key() -> None:
    """Same (seed, key) → same draws, wherever and whenever it is called."""
    from testdata.canonical.finance.generators import _stream

    a = [_stream(7, "order", "C-0003", 4).random() for _ in range(3)]
    b = [_stream(7, "order", "C-0003", 4).random() for _ in range(3)]
    assert a == b


def test_stream_separates_entities_and_seeds() -> None:
    """A different entity, coordinate or seed is a different stream.

    Load-bearing: a key missing a coordinate would silently make two entities share a
    stream, and their draws would correlate for no modelled reason.
    """
    from testdata.canonical.finance.generators import _stream

    base = _stream(7, "order", "C-0003", 4).random()
    assert base != _stream(7, "order", "C-0004", 4).random()  # entity
    assert base != _stream(7, "order", "C-0003", 5).random()  # month
    assert base != _stream(8, "order", "C-0003", 4).random()  # seed
    assert base != _stream(7, "receipt", "C-0003", 4).random()  # kind


def test_stream_is_stable_under_a_count_change() -> None:
    """THE property a volume lever needs: adding events perturbs nothing existing.

    On one sequential stream, drawing 12 events instead of 10 shifts every later
    value — the two runs then differ everywhere and the difference is no longer
    attributable to the intervention. Keyed by identity, the first 10 are byte-identical
    and the baseline is a strict SUBSET of the levered run.
    """
    from testdata.canonical.finance.generators import _stream

    def draw(n: int) -> list[float]:
        return [_stream(7, "order", "C-0003", 4, i).random() for i in range(n)]

    baseline, levered = draw(10), draw(12)
    assert levered[:10] == baseline
    assert len(set(levered)) == 12  # the added draws are genuinely new, not repeats

    # …and the sequential alternative demonstrably does NOT have this property,
    # which is the whole reason this helper exists.
    def sequential(n: int) -> list[float]:
        rng = random.Random(7)
        return [rng.random() for _ in range(n)]

    assert sequential(12)[:10] == sequential(10)  # same prefix only because nothing

    # else consumed the stream; interleave a second event type and the prefix breaks:
    def interleaved(n_orders: int) -> list[float]:
        rng = random.Random(7)
        for _ in range(n_orders):
            rng.random()
        return [rng.random() for _ in range(5)]  # a later cycle's draws

    assert interleaved(12) != interleaved(10)


# --- The operating chain (DAT-884 slices 2-4) ---


@functools.lru_cache(maxsize=2)
def _chain_dataset(seed: int = 42):
    return generate_finance_dataset(seed=seed, months=12)


def test_order_line_arithmetic_holds_by_construction() -> None:
    """line_amount == units x unit_price and line_cost == units x standard_cost.

    Load-bearing: DB1 truth is computed from these two columns, so if the identity
    ever drifts the answer key silently stops being an answer key.
    """
    d = _chain_dataset()
    cost_of = {p.product_id: p.standard_cost for p in d.products}
    assert d.sales_order_lines
    for line in d.sales_order_lines:
        assert line.line_amount == (line.unit_price * line.units).quantize(Decimal("0.01"))
        assert line.line_cost == (cost_of[line.product_id] * line.units).quantize(Decimal("0.01"))


def test_every_order_line_resolves_to_a_customer_and_a_product() -> None:
    """No orphan in the chain — DB1 per entity would silently drop rows."""
    d = _chain_dataset()
    orders = {o.order_id for o in d.sales_orders}
    customers = {c.customer_id for c in d.customers}
    products = {p.product_id for p in d.products}
    assert all(o.customer_id in customers for o in d.sales_orders)
    assert all(line.order_id in orders for line in d.sales_order_lines)
    assert all(line.product_id in products for line in d.sales_order_lines)


def test_cost_of_sale_is_no_longer_a_random_slice_of_purchases() -> None:
    """COGS is EXACTLY the order lines' cost — the defect that made margins ungradeable.

    Before DAT-884 nothing targeted account 5100; vendor invoices landed there by
    `rng.choice(expense_accounts)`, so gross profit had no true value to compare to.
    Cost of goods sold must now equal the sum of line_cost to the cent.
    """
    d = _chain_dataset()
    cogs = sum(line.debit for line in d.journal_lines if line.account_id == "5100")
    expected = sum(line.line_cost for line in d.sales_order_lines)
    assert cogs == expected

    # …and nothing else posts there.
    credits = sum(line.credit for line in d.journal_lines if line.account_id == "5100")
    assert credits == Decimal("0.00")


def test_gross_profit_is_positive_and_realistic() -> None:
    """A corpus whose cost of sale exceeds its revenue would be a self-inflicted defect."""
    d = _chain_dataset()
    revenue = sum(line.line_amount for line in d.sales_order_lines)
    cost = sum(line.line_cost for line in d.sales_order_lines)
    margin = float((revenue - cost) / revenue * 100)
    assert 15.0 < margin < 60.0, f"gross margin {margin:.1f}% is not a plausible business"


def test_inventory_never_goes_negative() -> None:
    """Cost of sale credits Inventory, so replenishment must keep the stock positive.

    Without it the corpus would carry a negative asset we invented ourselves, and
    every detector firing on it would be our defect reported as theirs.
    """
    d = _chain_dataset()
    balance = Decimal("0.00")
    by_date = sorted(d.journal_entries, key=lambda e: (e.date, e.entry_id))
    lines_by_entry: dict[str, list] = {}
    for line in d.journal_lines:
        lines_by_entry.setdefault(line.entry_id, []).append(line)
    for entry in by_date:
        for line in lines_by_entry.get(entry.entry_id, []):
            if line.account_id == "1400":
                balance += line.debit - line.credit
    assert balance > 0, f"inventory ends at {balance}"


def test_db1_truth_ties_to_the_two_cuts_and_to_the_ledger() -> None:
    """DB1 per customer and per product group are two cuts of one true number."""
    from testdata.ground_truth import calculate_ground_truth

    d = _chain_dataset()
    gt = calculate_ground_truth(d)
    assert gt.db1_by_customer and gt.db1_by_product_group

    by_customer = sum(c.db1 for c in gt.db1_by_customer)
    by_group = sum(c.db1 for c in gt.db1_by_product_group)
    assert by_customer == by_group

    lines_total = sum(line.line_amount - line.line_cost for line in d.sales_order_lines)
    assert abs(by_customer - lines_total) < Decimal("1.00")


# --- The volume lever (DAT-884 slice 5) ---


def test_volume_lever_is_an_exact_counterfactual() -> None:
    """The baseline's orders are a strict SUBSET of the levered run's.

    This is the property entity-keyed streams exist for, and it is stronger than the
    price lever's ratio claim: every pre-existing order is byte-identical, so the
    difference between the two corpora IS the added volume and nothing else.
    """
    from testdata.canonical.finance.generators import Lever

    base = generate_finance_dataset(seed=11, months=6)
    lev = generate_finance_dataset(seed=11, months=6, lever=Lever(period_k=3, factor=1.4, type="volume"))

    base_orders = {o.order_id: o for o in base.sales_orders}
    lev_orders = {o.order_id: o for o in lev.sales_orders}
    assert set(base_orders) < set(lev_orders), "baseline orders must be a strict subset"
    for oid, order in base_orders.items():
        assert lev_orders[oid] == order, f"{oid} changed under the lever"

    base_lines = {line.order_line_id: line for line in base.sales_order_lines}
    lev_lines = {line.order_line_id: line for line in lev.sales_order_lines}
    assert set(base_lines) < set(lev_lines)
    for lid, line in base_lines.items():
        assert lev_lines[lid] == line, f"{lid} changed under the lever"

    # Pre-lever months are untouched; post-lever months gained volume.
    def units(dataset, before: bool) -> int:
        dates = {o.order_id: o.order_date for o in dataset.sales_orders}
        return sum(line.units for line in dataset.sales_order_lines if (dates[line.order_id].month <= 3) is before)

    assert units(base, True) == units(lev, True)
    assert units(lev, False) > units(base, False)

    # The EXPENSE cycle never draws from the chain's streams, so it is untouched.
    def expense_bills(dataset):
        return [i for i in dataset.invoices if i.category == "expense"]

    assert len(expense_bills(base)) == len(expense_bills(lev))
    assert [i.amount for i in expense_bills(base)] == [i.amount for i in expense_bills(lev)]
    expense_ids = {i.invoice_id for i in expense_bills(base)}
    assert [p.amount for p in base.payments if p.invoice_id in expense_ids] == [
        p.amount for p in lev.payments if p.invoice_id in expense_ids
    ]

    # The stock subledger, by contrast, MUST move: more orders issue more stock, which
    # has to be bought and paid for. That propagation is what a volume lever means,
    # and the goods payables are where it shows up on the cash side.
    def goods_value(dataset) -> int:
        return sum(i.amount for i in dataset.invoices if i.category == "goods")

    assert goods_value(lev) > goods_value(base)
    base_issues = sum(-m.units for m in base.stock_movements if m.movement_type == "issue")
    lev_issues = sum(-m.units for m in lev.stock_movements if m.movement_type == "issue")
    assert lev_issues > base_issues


def test_price_lever_leaves_volume_and_cost_untouched() -> None:
    """A price change moves revenue, not units and not standard cost."""
    from testdata.canonical.finance.generators import Lever

    base = generate_finance_dataset(seed=11, months=6)
    lev = generate_finance_dataset(seed=11, months=6, lever=Lever(period_k=3, factor=1.2, type="price_level"))

    assert len(base.sales_orders) == len(lev.sales_orders)
    assert sum(line.units for line in base.sales_order_lines) == sum(line.units for line in lev.sales_order_lines)
    base_cogs = sum(line.debit for line in base.journal_lines if line.account_id == "5100")
    lev_cogs = sum(line.debit for line in lev.journal_lines if line.account_id == "5100")
    assert base_cogs == lev_cogs


def test_price_lever_moves_the_order_lines_the_ledger_derives_from() -> None:
    """The lever acts at the draw site, so the lines and the GL move together.

    It used to scale the revenue credit and leave ``sales_order_lines`` at baseline.
    Two things broke silently: ``operating_revenue`` stopped reconstructing from the
    lines (the oracle contract pins it as reconstructible to the cent), and every
    entity-grain metric read IDENTICAL in the levered run and its baseline — so a
    price lever moved the aggregate and moved nothing at the grain "which slice drove
    it" is asked at.
    """
    from datetime import date

    from testdata.canonical.finance.generators import Lever
    from testdata.ground_truth import calculate_ground_truth

    base = generate_finance_dataset(seed=7, months=6)
    lev = generate_finance_dataset(seed=7, months=6, lever=Lever(period_k=0, factor=1.2))

    # The lines carry the lever, and the ledger still derives from them.
    lines_total = sum(line.line_amount for line in lev.sales_order_lines)
    base_lines = sum(line.line_amount for line in base.sales_order_lines)
    assert lines_total > base_lines
    operating_revenue = sum(
        line.credit - line.debit for line in lev.journal_lines if line.account_id.startswith(("41", "42"))
    )
    assert lines_total == operating_revenue
    assert sum(invoice.amount for invoice in lev.ar_invoices) == operating_revenue

    # `line_amount == units x unit_price` survives the lever: the factor lands on the
    # unit price and the extension follows, rather than being applied to the total.
    for line in lev.sales_order_lines:
        assert line.line_amount == (line.unit_price * line.units).quantize(Decimal("0.01"))

    # Entity grain moves — the property the old placement lost entirely.
    def db1_by_customer(ds: object) -> dict[str, Decimal]:
        truth = calculate_ground_truth(ds, fiscal_start=date(2025, 1, 1), months=6)
        return {c.entity: c.db1 for c in truth.db1_by_customer}

    base_db1, lev_db1 = db1_by_customer(base), db1_by_customer(lev)
    assert base_db1 and set(base_db1) == set(lev_db1)
    assert all(lev_db1[k] > base_db1[k] for k in base_db1 if base_db1[k] > 0)


# --- Lever scope (A1) ---


def test_scoped_price_lever_moves_the_slice_and_nothing_else() -> None:
    """THE property scope exists for: the aggregate moves, one named slice moved.

    Out-of-scope customers must be byte-identical to their baseline, not merely
    similar — otherwise "which slice drove it" has no answer key, only a correlation.
    """
    from testdata.canonical.finance.generators import Lever

    base = generate_finance_dataset(seed=7, months=6)
    lev = generate_finance_dataset(
        seed=7,
        months=6,
        lever=Lever(period_k=0, factor=1.3, scope={"segment": ["Enterprise"]}),
    )

    segment_of = {c.customer_id: c.segment for c in base.customers}
    order_segment = {o.order_id: segment_of[o.customer_id] for o in base.sales_orders}
    base_lines = {line.order_line_id: line for line in base.sales_order_lines}
    lev_lines = {line.order_line_id: line for line in lev.sales_order_lines}
    assert set(base_lines) == set(lev_lines), "a price lever adds no orders"

    moved = [lid for lid in base_lines if base_lines[lid].unit_price != lev_lines[lid].unit_price]
    assert moved, "the scoped slice must actually move"
    assert {order_segment[base_lines[lid].order_id] for lid in moved} == {"Enterprise"}

    # Every out-of-scope line is byte-identical, not approximately unchanged.
    for lid, line in base_lines.items():
        if order_segment[line.order_id] != "Enterprise":
            assert lev_lines[lid] == line, f"{lid} moved outside the scope"

    # The aggregate moved anyway — that is the confusable part of the question.
    def revenue(ds: object) -> Decimal:
        return sum((line.line_amount for line in ds.sales_order_lines), Decimal("0"))

    assert revenue(lev) > revenue(base)


def test_scope_dimensions_intersect_rather_than_union() -> None:
    """`segment + region` is enterprise accounts IN EMEA, not enterprise plus EMEA."""
    from testdata.canonical.finance.generators import Lever

    base = generate_finance_dataset(seed=7, months=6)
    lev = generate_finance_dataset(
        seed=7,
        months=6,
        lever=Lever(period_k=0, factor=1.3, scope={"segment": ["Enterprise"], "region": ["DACH"]}),
    )
    by_id = {c.customer_id: c for c in base.customers}
    order_customer = {o.order_id: by_id[o.customer_id] for o in base.sales_orders}
    base_lines = {line.order_line_id: line for line in base.sales_order_lines}
    lev_lines = {line.order_line_id: line for line in lev.sales_order_lines}

    moved = {line.order_id for lid, line in base_lines.items() if lev_lines[lid].unit_price != line.unit_price}
    assert moved
    for order_id in moved:
        customer = order_customer[order_id]
        assert customer.segment == "Enterprise" and customer.region == "DACH"


def test_product_scope_moves_only_its_group() -> None:
    """A price lever scoped to a product group acts per LINE, not per order.

    Orders mix groups, so scoping at order grain would move lines the intervention
    never named.
    """
    from testdata.canonical.finance.generators import Lever

    base = generate_finance_dataset(seed=7, months=6)
    lev = generate_finance_dataset(
        seed=7,
        months=6,
        lever=Lever(period_k=0, factor=1.3, scope={"product_group": ["Instruments"]}),
    )
    group_of = {p.product_id: p.product_group for p in base.products}
    base_lines = {line.order_line_id: line for line in base.sales_order_lines}
    lev_lines = {line.order_line_id: line for line in lev.sales_order_lines}

    moved = {lid for lid in base_lines if base_lines[lid].unit_price != lev_lines[lid].unit_price}
    assert moved
    assert {group_of[base_lines[lid].product_id] for lid in moved} == {"Instruments"}

    # A mixed order proves the per-line grain: same order, one line moved, one did not.
    by_order: dict[str, set[bool]] = {}
    for lid, line in base_lines.items():
        by_order.setdefault(line.order_id, set()).add(lid in moved)
    assert any(states == {True, False} for states in by_order.values()), "no mixed-group order to prove line grain"


def test_scoped_volume_lever_adds_orders_only_inside_the_scope() -> None:
    """Outside the scope the baseline's orders are not a subset — they are equal."""
    from testdata.canonical.finance.generators import Lever

    base = generate_finance_dataset(seed=11, months=6)
    lev = generate_finance_dataset(
        seed=11,
        months=6,
        lever=Lever(period_k=0, factor=1.5, type="volume", scope={"segment": ["SMB"]}),
    )
    segment_of = {c.customer_id: c.segment for c in base.customers}
    base_ids = {o.order_id for o in base.sales_orders}
    lev_ids = {o.order_id for o in lev.sales_orders}

    assert base_ids < lev_ids, "the scoped slice must gain orders"
    added = lev_ids - base_ids
    lev_orders = {o.order_id: o for o in lev.sales_orders}
    assert {segment_of[lev_orders[oid].customer_id] for oid in added} == {"SMB"}

    # Out-of-scope customers: identical order sets, byte-identical rows.
    base_orders = {o.order_id: o for o in base.sales_orders}
    for oid, order in base_orders.items():
        if segment_of[order.customer_id] != "SMB":
            assert lev_orders[oid] == order


def test_a_volume_lever_refuses_a_product_scope() -> None:
    """Order count is drawn before any product exists, so the scope is unanswerable.

    Ignoring it would put an intervention.yaml on disk claiming a narrower
    intervention than the one that ran.
    """
    import pytest

    from testdata.canonical.finance.generators import Lever

    with pytest.raises(ValueError, match="cannot scope on"):
        Lever(period_k=0, factor=1.5, type="volume", scope={"product_group": ["Instruments"]})

    for bad in ({"vertical": ["x"]}, {"segment": []}):
        with pytest.raises(ValueError):
            Lever(period_k=0, factor=1.2, scope=bad)


def test_an_unscoped_lever_is_unchanged_by_the_scope_machinery() -> None:
    """The opt-in guarantee: absent scope must reproduce the pre-A1 corpus exactly."""
    from testdata.canonical.finance.generators import Lever

    plain = generate_finance_dataset(seed=7, months=6, lever=Lever(period_k=3, factor=1.2))
    explicit = generate_finance_dataset(seed=7, months=6, lever=Lever(period_k=3, factor=1.2, scope=None))
    assert plain.sales_order_lines == explicit.sales_order_lines
    assert plain.journal_lines == explicit.journal_lines


# --- The mix lever (A2) and the typed rate lever (A3) ---


def _segment_share(ds: object, segment: str, period_k: int) -> tuple[float, int]:
    """The segment's share of order count from period_k on, and the window's size."""
    seg = {c.customer_id: c.segment for c in ds.customers}
    window = [o for o in ds.sales_orders if (o.order_date.year - 2025) * 12 + o.order_date.month - 1 >= period_k]
    return sum(1 for o in window if seg[o.customer_id] == segment) / len(window), len(window)


def test_mix_lever_moves_composition_and_holds_volume() -> None:
    """A mix lever that moved total volume would be a frequency change mislabelled.

    The complement factor is DERIVED to solve s0*ft + (1-s0)*fc = 1, so the scoped
    share lands on its target while total order count stays put — which makes any
    move in an aggregate metric compositional by construction.
    """
    from testdata.canonical.finance.generators import Lever

    kw = dict(seed=7, months=12, profile="mid")
    base = generate_finance_dataset(**kw)
    lev = generate_finance_dataset(
        **kw, lever=Lever(period_k=6, type="mix", target_share=0.45, scope={"segment": ["Enterprise"]})
    )

    base_share, base_n = _segment_share(base, "Enterprise", 6)
    lev_share, lev_n = _segment_share(lev, "Enterprise", 6)

    assert abs(lev_share - 0.45) < 0.01, f"share landed on {lev_share:.4f}, not the 0.45 target"
    assert lev_share > base_share
    assert abs(lev_n / base_n - 1.0) < 0.01, f"total order count moved {lev_n / base_n - 1:+.2%} — not a mix"

    # Pre-period months are untouched on both sides.
    assert _segment_share(base, "Enterprise", 0)[1] > base_n  # the window really is a subset
    pre_base = [o for o in base.sales_orders if o.order_date.month <= 6]
    pre_lev = [o for o in lev.sales_orders if o.order_date.month <= 6]
    assert {o.order_id for o in pre_base} == {o.order_id for o in pre_lev}


def test_mix_lever_holds_within_member_rates_fixed() -> None:
    """Composition moves; how each member behaves does not.

    If unit prices or order sizes moved too, an aggregate shift could no longer be
    attributed to composition, and the lever would answer a different question than
    the one it names.
    """
    from testdata.canonical.finance.generators import Lever

    kw = dict(seed=7, months=12, profile="mid")
    base = generate_finance_dataset(**kw)
    lev = generate_finance_dataset(
        **kw, lever=Lever(period_k=6, type="mix", target_share=0.45, scope={"segment": ["Enterprise"]})
    )
    base_lines = {line.order_line_id: line for line in base.sales_order_lines}
    lev_lines = {line.order_line_id: line for line in lev.sales_order_lines}

    shared = set(base_lines) & set(lev_lines)
    assert shared, "the two runs must share most of their lines"
    for lid in shared:
        assert base_lines[lid] == lev_lines[lid], f"{lid} changed under a mix lever"


def test_mix_refuses_a_share_it_cannot_shift() -> None:
    """An empty scope has no share to move, and s1/0 is not a truth to publish."""
    import pytest

    from testdata.canonical.finance.generators import Lever

    with pytest.raises(ValueError, match="cannot shift a share"):
        generate_finance_dataset(
            seed=7,
            months=6,
            lever=Lever(period_k=0, type="mix", target_share=0.4, scope={"customer_id": ["CU-NOBODY"]}),
        )
    with pytest.raises(ValueError, match="needs a scope"):
        Lever(period_k=0, type="mix", target_share=0.4)
    with pytest.raises(ValueError, match="target_share"):
        Lever(period_k=0, type="mix", target_share=1.5, scope={"segment": ["SMB"]})


def test_rate_lever_names_its_driver_and_matches_the_legacy_types() -> None:
    """`rate`/`price` IS `price_level`; one vocabulary, two spellings, one corpus."""
    from testdata.canonical.finance.generators import Lever

    legacy = generate_finance_dataset(seed=7, months=6, lever=Lever(period_k=2, factor=1.2))
    typed = generate_finance_dataset(seed=7, months=6, lever=Lever(period_k=2, factor=1.2, type="rate", driver="price"))
    assert legacy.sales_order_lines == typed.sales_order_lines

    legacy_vol = generate_finance_dataset(seed=7, months=6, lever=Lever(period_k=2, factor=1.4, type="volume"))
    typed_vol = generate_finance_dataset(
        seed=7, months=6, lever=Lever(period_k=2, factor=1.4, type="rate", driver="frequency")
    )
    assert legacy_vol.sales_orders == typed_vol.sales_orders

    import pytest

    with pytest.raises(ValueError, match="needs driver"):
        Lever(period_k=0, factor=1.2, type="rate")
    with pytest.raises(ValueError, match="driver is a `rate` field"):
        Lever(period_k=0, factor=1.2, type="price_level", driver="price")


def test_collection_lag_lever_moves_when_cash_lands_not_whether() -> None:
    """The same sales are collected in both runs; only the dates differ.

    Collection is decided before the lag is drawn, so a lag lever cannot silently
    change WHICH invoices settle — otherwise a DSO shift would be confounded with a
    collection-rate shift and neither would be attributable.
    """
    from testdata.canonical.finance.generators import Lever

    kw = dict(seed=7, months=12, profile="mid")
    base = generate_finance_dataset(**kw)
    lev = generate_finance_dataset(
        **kw,
        lever=Lever(period_k=0, factor=2.0, type="rate", driver="collection_lag", scope={"segment": ["Enterprise"]}),
    )

    base_receipts = {r.ar_invoice_id: r for r in base.receipts}
    lev_receipts = {r.ar_invoice_id: r for r in lev.receipts}
    assert set(base_receipts) == set(lev_receipts), "the same invoices collect"
    assert all(base_receipts[k].amount == lev_receipts[k].amount for k in base_receipts)

    segment_of = {c.customer_id: c.segment for c in base.customers}
    later = [k for k in base_receipts if lev_receipts[k].receipt_date > base_receipts[k].receipt_date]
    assert later, "the scoped slice must actually pay later"
    assert {segment_of[base_receipts[k].customer_id] for k in later} == {"Enterprise"}

    # Out of scope, the receipt date is identical — not merely close.
    for k, receipt in base_receipts.items():
        if segment_of[receipt.customer_id] != "Enterprise":
            assert lev_receipts[k].receipt_date == receipt.receipt_date


def test_ar_side_exists_and_reconciles() -> None:
    """The AR half of Capital — the corpus defect DAT-884 names.

    `invoices` is vendor-side only, so DSO had no receivable document to measure.
    One AR invoice per order, amount tied to the order's revenue posting.
    """
    d = _chain_dataset()
    assert d.ar_invoices and d.receipts

    invoiced = {i.ar_invoice_id: i for i in d.ar_invoices}
    assert len(invoiced) == len(d.sales_orders), "one AR invoice per order"

    order_total = {}
    for line in d.sales_order_lines:
        order_total[line.order_id] = order_total.get(line.order_id, Decimal("0")) + line.line_amount
    for inv in d.ar_invoices:
        assert inv.amount == order_total[inv.order_id].quantize(Decimal("0.01"))

    # Every receipt points at a real invoice, and never collects more than was billed.
    for receipt in d.receipts:
        assert receipt.ar_invoice_id in invoiced
        assert receipt.amount <= invoiced[receipt.ar_invoice_id].amount


def test_ar_invoice_status_is_derived_from_actual_receipts() -> None:
    """A `paid` invoice with no receipt behind it is exactly the cross-table
    inconsistency the engine's validation induction exists to catch — and it would
    be ours, not theirs. Status must follow the money."""
    d = _chain_dataset()
    collected: dict[str, Decimal] = {}
    for receipt in d.receipts:
        collected[receipt.ar_invoice_id] = collected.get(receipt.ar_invoice_id, Decimal("0.00")) + receipt.amount
    for inv in d.ar_invoices:
        got = collected.get(inv.ar_invoice_id, Decimal("0.00"))
        if inv.status.value == "paid":
            assert got >= inv.amount
        elif inv.status.value == "partial":
            assert Decimal("0") < got < inv.amount
        else:
            assert got == Decimal("0.00"), f"{inv.ar_invoice_id} is {inv.status} but collected {got}"


def test_ar_due_dates_follow_the_customer_terms() -> None:
    """DSO must be a property of the data, not a constant."""
    d = _chain_dataset()
    terms_of = {c.customer_id: c.payment_terms for c in d.customers}
    lags = {(inv.due_date - inv.invoice_date).days for inv in d.ar_invoices}
    assert len(lags) > 1, "every invoice shares one due lag — terms are not being honoured"
    for inv in d.ar_invoices[:200]:
        expected = {"net_30": 30, "net_60": 60, "net_90": 90, "due_on_receipt": 0}[terms_of[inv.customer_id].value]
        assert (inv.due_date - inv.invoice_date).days == expected


def _revenue_by_segment(ds):
    """Order-line revenue per customer segment — the grain attribution is claimed at."""
    from decimal import Decimal

    segment = {c.customer_id: c.segment for c in ds.customers}
    customer = {o.order_id: o.customer_id for o in ds.sales_orders}
    out: dict[str, Decimal] = {}
    for line in ds.sales_order_lines:
        seg = segment[customer[line.order_id]]
        out[seg] = out.get(seg, Decimal("0")) + line.line_amount
    return out


def test_a_per_member_factor_moves_each_member_by_its_own_amount():
    """The aggregate delta matches no single member's factor — that is the point."""
    from testdata.canonical.finance.generators import Lever, generate_finance_dataset

    kw = dict(seed=7, months=12, profile="mid")
    scope = {"segment": ["Enterprise", "Mid-Market"]}

    base = generate_finance_dataset(**kw)
    heterogeneous = generate_finance_dataset(
        **kw,
        lever=Lever(
            period_k=6,
            type="rate",
            driver="price",
            scope=scope,
            factor={"Enterprise": 1.20, "Mid-Market": 1.05},
        ),
    )
    # The same lever with ONE factor, to show the per-member one is not just a relabel.
    uniform = generate_finance_dataset(
        **kw, lever=Lever(period_k=6, type="rate", driver="price", scope=scope, factor=1.20)
    )

    b, h, u = (_revenue_by_segment(ds) for ds in (base, heterogeneous, uniform))
    lift = {seg: float(h[seg] / b[seg]) - 1.0 for seg in b}

    # Enterprise carries the full 20% (identical to the uniform run, so the map is
    # genuinely per-member and not an averaged fudge); Mid-Market carries only 5%.
    assert lift["Enterprise"] == pytest.approx(float(u["Enterprise"] / b["Enterprise"]) - 1.0, abs=1e-9)
    # 5%/20% = a quarter of the lift, to within the segments' differing H2 weight.
    assert lift["Mid-Market"] == pytest.approx(lift["Enterprise"] * 0.25, rel=0.05)
    assert lift["SMB"] == 0.0, "an unscoped segment is byte-identical to its baseline"

    total = float(sum(h.values()) / sum(b.values())) - 1.0
    assert min(lift["Mid-Market"], lift["Enterprise"]) < total < lift["Enterprise"], (
        "the aggregate is the member-weighted mix and matches no single member — so "
        "recovering the total proves nothing about who drove it"
    )


def test_a_per_member_factor_reaches_frequency_and_collection_too():
    """Heterogeneity is a property of the factor, not of one driver."""
    from testdata.canonical.finance.generators import Lever, generate_finance_dataset

    kw = dict(seed=7, months=12, profile="mid")
    scope = {"segment": ["Enterprise", "Mid-Market"]}
    factor = {"Enterprise": 1.5, "Mid-Market": 1.1}

    base = generate_finance_dataset(**kw)
    segment = {c.customer_id: c.segment for c in base.customers}

    volume = generate_finance_dataset(
        **kw, lever=Lever(period_k=6, type="rate", driver="frequency", scope=scope, factor=factor)
    )

    def late_orders(ds):
        counts: dict[str, int] = {}
        for order in ds.sales_orders:
            if order.order_date.month > 6:
                seg = segment[order.customer_id]
                counts[seg] = counts.get(seg, 0) + 1
        return counts

    b, v = late_orders(base), late_orders(volume)
    assert v["Enterprise"] / b["Enterprise"] == pytest.approx(1.5, rel=0.05)
    assert v["Mid-Market"] / b["Mid-Market"] == pytest.approx(1.1, rel=0.05)
    assert v["SMB"] == b["SMB"]

    lag = generate_finance_dataset(
        **kw, lever=Lever(period_k=6, type="rate", driver="collection_lag", scope=scope, factor=factor)
    )

    def mean_lag(ds):
        sale_date = {}
        for order in ds.sales_orders:
            sale_date[order.order_id] = order.order_date
        by_seg: dict[str, list[int]] = {}
        for inv in ds.ar_invoices:
            pass
        return by_seg

    # Receipts carry the customer directly, so the lag shift is readable per segment.
    def mean_days(ds):
        order_date = {o.order_id: o.order_date for o in ds.sales_orders}
        due = {i.ar_invoice_id: order_date.get(i.order_id) for i in ds.ar_invoices}
        buckets: dict[str, list[int]] = {}
        for r in ds.receipts:
            start = due.get(r.ar_invoice_id)
            if start is None or start.month <= 6:
                continue
            buckets.setdefault(segment[r.customer_id], []).append((r.receipt_date - start).days)
        return {seg: sum(days) / len(days) for seg, days in buckets.items()}

    b_days, l_days = mean_days(base), mean_days(lag)
    # Clamped at the fiscal end, so the realised shift is BELOW the factor — the
    # caveat intervention.yaml states. The ordering is what must hold.
    assert 1.0 < l_days["Mid-Market"] / b_days["Mid-Market"] < l_days["Enterprise"] / b_days["Enterprise"]
    assert l_days["SMB"] == b_days["SMB"]


def test_a_factor_map_must_name_exactly_one_scope_dimension():
    """The map's keys ARE a dimension's scope, so disagreement is a refusal."""
    from testdata.canonical.finance.generators import Lever

    ok = Lever(
        period_k=6,
        type="rate",
        driver="price",
        scope={"segment": ["Enterprise", "Mid-Market"]},
        factor={"Enterprise": 1.2, "Mid-Market": 1.05},
    )
    assert ok.factor_dimension == "segment"

    with pytest.raises(ValueError, match="needs a scope"):
        Lever(period_k=6, type="rate", driver="price", factor={"Enterprise": 1.2})

    with pytest.raises(ValueError, match="exactly one scope dimension"):
        Lever(
            period_k=6,
            type="rate",
            driver="price",
            scope={"segment": ["Enterprise", "Mid-Market"]},
            factor={"Enterprise": 1.2},  # omits a scoped member
        )

    with pytest.raises(ValueError, match="exactly one scope dimension"):
        Lever(
            period_k=6,
            type="rate",
            driver="price",
            scope={"segment": ["Enterprise"]},
            factor={"Enterprise": 1.2, "Mid-Market": 1.05},  # names one outside the scope
        )

    with pytest.raises(ValueError, match="non-positive"):
        Lever(
            period_k=6,
            type="rate",
            driver="price",
            scope={"segment": ["Enterprise", "Mid-Market"]},
            factor={"Enterprise": 1.2, "Mid-Market": 0.0},
        )

    with pytest.raises(ValueError, match="takes no factor at all"):
        Lever(
            period_k=6,
            type="mix",
            target_share=0.4,
            scope={"segment": ["Enterprise"]},
            factor={"Enterprise": 1.2},
        )


def test_a_scalar_factor_is_untouched_by_the_per_member_machinery():
    """Opt-in only: the scalar path must be the one that predates the map."""
    from testdata.canonical.finance.generators import Lever, generate_finance_dataset

    kw = dict(seed=7, months=12, profile="mid")
    scope = {"segment": ["Enterprise"]}
    scalar = generate_finance_dataset(
        **kw, lever=Lever(period_k=6, type="rate", driver="price", scope=scope, factor=1.15)
    )
    as_map = generate_finance_dataset(
        **kw, lever=Lever(period_k=6, type="rate", driver="price", scope=scope, factor={"Enterprise": 1.15})
    )
    # A one-member map is the scalar lever spelled differently, and must produce the
    # identical corpus — otherwise the two spellings are two DGPs.
    assert [line.line_amount for line in scalar.sales_order_lines] == [
        line.line_amount for line in as_map.sales_order_lines
    ]


def test_a_trend_is_drift_not_an_event():
    """The control corpus: everything rises and nothing happened."""
    from testdata.canonical.finance.generators import generate_finance_dataset

    kw = dict(seed=7, months=12, profile="mid")
    base = generate_finance_dataset(**kw)
    drifted = generate_finance_dataset(**kw, trend={"price": 0.06, "volume": 0.12})

    def revenue_by_month(ds):
        when = {o.order_id: o.order_date.month for o in ds.sales_orders}
        out: dict[int, Decimal] = {}
        for line in ds.sales_order_lines:
            m = when[line.order_id]
            out[m] = out.get(m, Decimal("0")) + line.line_amount
        return out

    b, d = revenue_by_month(base), revenue_by_month(drifted)
    lift = {m: float(d[m] / b[m]) for m in sorted(b)}

    # Month 0 is undrifted by construction — the drift accumulates, it does not start.
    assert lift[1] == 1.0
    # And it compounds: the year-end lift is the two annual rates, near enough that
    # seasonality and order-size noise are the only slack.
    assert lift[12] == pytest.approx(1.06 ** (11 / 12) * 1.12 ** (11 / 12), rel=0.05)
    # No step, anywhere. A consumer looking for "what changed in month k" must find
    # nothing, which is the entire point of having this corpus.
    steps = [lift[m + 1] / lift[m] for m in range(2, 12)]
    assert max(steps) / min(steps) < 1.15, f"a trend must not look like an event: {steps}"


def test_a_trend_cancels_in_a_lever_pair():
    """Drift is a property of the corpus, so it must not disturb a counterfactual."""
    from testdata.canonical.finance.generators import Lever, generate_finance_dataset

    kw = dict(seed=7, months=12, profile="mid", trend={"price": 0.04})
    lever = Lever(period_k=6, type="rate", driver="price", scope={"segment": ["Enterprise"]}, factor=1.2)

    base = generate_finance_dataset(**kw)
    levered = generate_finance_dataset(**kw, lever=lever)

    segment = {c.customer_id: c.segment for c in base.customers}
    owner = {o.order_id: o.customer_id for o in base.sales_orders}

    def revenue(ds, seg, second_half):
        when = {o.order_id: o.order_date.month for o in ds.sales_orders}
        return sum(
            line.line_amount
            for line in ds.sales_order_lines
            if segment[owner[line.order_id]] == seg and (when[line.order_id] > 6) == second_half
        )

    # The lever's effect is exactly its factor on top of whatever the trend did —
    # both runs carry the identical drift, so it divides out.
    assert float(revenue(levered, "Enterprise", True) / revenue(base, "Enterprise", True)) == pytest.approx(
        1.2, rel=1e-3
    )
    assert revenue(levered, "Enterprise", False) == revenue(base, "Enterprise", False)
    assert revenue(levered, "SMB", True) == revenue(base, "SMB", True)


def test_the_payer_dimension_is_zipfian_and_opt_in():
    """High cardinality with a real head and a real tail — and absent by default."""
    from collections import Counter

    from testdata.canonical.finance.generators import generate_finance_dataset

    kw = dict(seed=7, months=12, profile="mid")
    plain = generate_finance_dataset(**kw)
    assert plain.merchants == []
    assert all(txn.merchant_id is None for txn in plain.bank_transactions)

    zipfian = generate_finance_dataset(**kw, merchants=4000)
    assert len(zipfian.merchants) == 4000
    counts = Counter(txn.merchant_id for txn in zipfian.bank_transactions)
    total = sum(counts.values())
    ranked = counts.most_common()

    # Cardinality the corpus has nowhere else: the flat `counterparty` axis holds a
    # few hundred members with no head worth the name.
    flat = Counter(txn.counterparty for txn in plain.bank_transactions)
    assert len(counts) > 4 * len(flat)

    # A genuine power law, not merely a big dimension: a head that carries real mass,
    # and a tail the sample never fully reaches.
    assert ranked[0][1] / total > 0.05, "the top member alone is several percent"
    assert sum(c for _, c in ranked[:10]) / total > 0.30
    assert sum(1 for c in counts.values() if c == 1) > 0.25 * len(counts)
    assert len(counts) < 4000, "a power-law sample does not reach every member"

    # Rank is not id order — a head readable off the key is not an aggregation test.
    assert ranked[0][0] != zipfian.merchants[0].merchant_id


def test_turning_the_payer_dimension_on_does_not_move_anything_else():
    """Opt-in means opt-in: the ledger is the same corpus, merchants or not."""
    from testdata.canonical.finance.generators import generate_finance_dataset

    kw = dict(seed=7, months=12, profile="mid")
    plain = generate_finance_dataset(**kw)
    zipfian = generate_finance_dataset(**kw, merchants=2000)

    assert [line.debit for line in plain.journal_lines] == [line.debit for line in zipfian.journal_lines]
    assert [txn.amount for txn in plain.bank_transactions] == [txn.amount for txn in zipfian.bank_transactions]
    assert [o.order_id for o in plain.sales_orders] == [o.order_id for o in zipfian.sales_orders]
