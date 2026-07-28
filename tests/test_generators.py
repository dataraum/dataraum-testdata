"""Tests for deterministic finance data generators."""

import functools
import random
from collections import Counter
from datetime import date
from decimal import Decimal

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
    assert gl_invoice_ids == non_cancelled, f"Missing GL for {len(non_cancelled - gl_invoice_ids)} invoices"


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
    gt = calculate_ground_truth(d, seed=42)
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
    lev = generate_finance_dataset(
        seed=11, months=6, lever=Lever(period_k=3, factor=1.4, type="volume")
    )

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
        return sum(
            line.units for line in dataset.sales_order_lines
            if (dates[line.order_id].month <= 3) is before
        )

    assert units(base, True) == units(lev, True)
    assert units(lev, False) > units(base, False)

    # The ledger cycles never draw from the chain's streams, so they are untouched.
    assert len(base.invoices) == len(lev.invoices)
    assert base.invoices[0].amount == lev.invoices[0].amount
    assert [p.amount for p in base.payments] == [p.amount for p in lev.payments]


def test_price_lever_leaves_volume_and_cost_untouched() -> None:
    """A price change moves revenue, not units and not standard cost."""
    from testdata.canonical.finance.generators import Lever

    base = generate_finance_dataset(seed=11, months=6)
    lev = generate_finance_dataset(
        seed=11, months=6, lever=Lever(period_k=3, factor=1.2, type="price_level")
    )

    assert len(base.sales_orders) == len(lev.sales_orders)
    assert sum(line.units for line in base.sales_order_lines) == sum(
        line.units for line in lev.sales_order_lines
    )
    base_cogs = sum(line.debit for line in base.journal_lines if line.account_id == "5100")
    lev_cogs = sum(line.debit for line in lev.journal_lines if line.account_id == "5100")
    assert base_cogs == lev_cogs


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
        collected[receipt.ar_invoice_id] = (
            collected.get(receipt.ar_invoice_id, Decimal("0.00")) + receipt.amount
        )
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
        expected = {"net_30": 30, "net_60": 60, "net_90": 90, "due_on_receipt": 0}[
            terms_of[inv.customer_id].value
        ]
        assert (inv.due_date - inv.invoice_date).days == expected
