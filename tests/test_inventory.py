"""The stock subledger's contracts.

The inventory family exists to make CCC gradeable and to give the payables inventory
creates somewhere to go. Both claims are only worth as much as the ties below: a stock
table that does not roll forward, or does not reconcile to the ledger it claims to
detail, is a plausible table rather than an answer key.
"""

from __future__ import annotations

import functools
from decimal import Decimal

from testdata.canonical.finance.generators import generate_finance_dataset
from testdata.canonical.finance.models import FinanceDataset
from testdata.ground_truth import calculate_ground_truth

_INVENTORY_ACCOUNT = "1400"
_COGS_ACCOUNT = "5100"
_SHRINKAGE_ACCOUNT = "5150"
_AP_ACCOUNTS = {"2110", "2120"}


@functools.lru_cache(maxsize=2)
def _dataset(seed: int = 42, months: int = 12) -> FinanceDataset:
    return generate_finance_dataset(seed=seed, months=months)


def _gl_balance(dataset: FinanceDataset, account: str) -> Decimal:
    return sum(
        (line.debit - line.credit for line in dataset.journal_lines if line.account_id == account),
        Decimal("0"),
    )


def test_roll_forward_holds_per_product_location_and_period() -> None:
    """``opening + Σ movements = closing`` at every key, not merely in aggregate.

    Checked here independently of ``Invariants``, which computes the same thing: a
    truth file that grades itself proves nothing.
    """
    dataset = _dataset()
    periods = sorted({p.period for p in dataset.inventory_positions})

    moved: dict[tuple[str, str, str], int] = {}
    for movement in dataset.stock_movements:
        key = (movement.product_id, movement.location_id, movement.date.strftime("%Y-%m"))
        moved[key] = moved.get(key, 0) + movement.units

    closing = {(p.product_id, p.location_id, p.period): p.units_on_hand for p in dataset.inventory_positions}
    for (product, location, period), units in closing.items():
        index = periods.index(period)
        opening = 0 if index == 0 else closing[(product, location, periods[index - 1])]
        assert opening + moved.get((product, location, period), 0) == units, (
            f"{product}/{location}/{period}: {opening} + movements != {units}"
        )


def test_positions_tie_to_the_inventory_account() -> None:
    """Σ position value == the GL inventory balance, exactly.

    Both sides are valued at standard cost, so this is an equality and not an
    approximation — a tolerance here would be covering for a valuation the generator
    does not actually perform.
    """
    dataset = _dataset()
    last_period = max(p.period for p in dataset.inventory_positions)
    closing_value = sum((p.value for p in dataset.inventory_positions if p.period == last_period), Decimal("0"))
    assert closing_value == _gl_balance(dataset, _INVENTORY_ACCOUNT)
    assert closing_value > 0, "a firm that sells goods holds some"


def test_issues_are_exactly_the_cost_of_sale() -> None:
    """Every issue is the subledger detail of a cost-of-sale posting, to the cent."""
    dataset = _dataset()
    issued = sum((-m.value for m in dataset.stock_movements if m.movement_type == "issue"), Decimal("0"))
    assert issued == _gl_balance(dataset, _COGS_ACCOUNT)

    # And each issue names the entry it details.
    entry_ids = {e.entry_id for e in dataset.journal_entries}
    product_ids = {p.product_id for p in dataset.products}
    for movement in dataset.stock_movements:
        assert movement.entry_id in entry_ids
        assert movement.product_id in product_ids


def test_movement_signs_make_the_table_additive() -> None:
    """The sign convention is the movement type's arithmetic, so a plain SUM works."""
    dataset = _dataset()
    for movement in dataset.stock_movements:
        assert movement.unit_cost > 0, "a cost is a rate, never signed"
        assert (movement.units > 0) == (movement.value > 0)
        assert movement.value == movement.unit_cost * movement.units
        if movement.movement_type == "receipt":
            assert movement.units > 0
        elif movement.movement_type == "issue":
            assert movement.units < 0

    # Both adjustment directions occur — a corpus where shrinkage is the only sign
    # makes "is this an adjustment?" answerable from the sign alone.
    adjustments = [m.units for m in dataset.stock_movements if m.movement_type == "adjustment"]
    assert adjustments, "the cycle count must actually find something"
    assert min(adjustments) < 0 < max(adjustments)


def test_on_hand_never_goes_negative() -> None:
    """Running the movements in order never drives a location's stock below zero.

    The generator ships receipts before the issues they cover on the same day; without
    that ordering the fiscal year's first orders would sell stock that had not arrived,
    which is a defect we would have invented ourselves.
    """
    dataset = _dataset()
    running: dict[tuple[str, str], int] = {}
    for movement in dataset.stock_movements:
        key = (movement.product_id, movement.location_id)
        running[key] = running.get(key, 0) + movement.units
        assert running[key] >= 0, f"{key} went negative at {movement.movement_id}"


def test_goods_receipts_raise_payables_that_actually_settle() -> None:
    """The defect this family was built to fix.

    Every receipt is a vendor bill, every bill ages like any other, and the ones that
    age past their terms get paid. Before this, the replenishment credit to AP had no
    purchasing event behind it and nothing ever cleared it: annual DPO read 271 days
    and 95% of closing payables were permanently open.
    """
    dataset = _dataset()
    goods = [i for i in dataset.invoices if i.category == "goods"]
    assert goods

    receipts_value = sum((m.value for m in dataset.stock_movements if m.movement_type == "receipt"), Decimal("0"))
    assert sum((i.amount for i in goods), Decimal("0")) == receipts_value

    goods_ids = {i.invoice_id for i in goods}
    paid = {p.invoice_id for p in dataset.payments if p.invoice_id in goods_ids}
    settled = sum(i.amount for i in goods if i.invoice_id in paid)
    assert settled / receipts_value > 0.5, "most goods payables must actually clear"

    truth = calculate_ground_truth(dataset)
    assert 15 < truth.annual.annual_dpo < 120, truth.annual.annual_dpo


def test_shrinkage_account_carries_only_counted_differences() -> None:
    """5150 is derived from physical counts, never a bucket for random vendor bills.

    Letting purchase invoices land there would make the account say nothing about
    stock — the same ambiguity that made gross profit ungradeable when COGS mixed
    cost-of-sale with unrelated purchases.
    """
    dataset = _dataset()
    entry_desc = {e.entry_id: e.description for e in dataset.journal_entries}
    touching = {entry_desc[line.entry_id] for line in dataset.journal_lines if line.account_id == _SHRINKAGE_ACCOUNT}
    assert touching
    assert all(d.startswith("Cycle count adjustment") for d in touching)

    adjusted = sum((abs(m.value) for m in dataset.stock_movements if m.movement_type == "adjustment"), Decimal("0"))
    assert abs(_gl_balance(dataset, _SHRINKAGE_ACCOUNT)) <= adjusted


def test_cash_conversion_cycle_is_gradeable_monthly_and_annually() -> None:
    """CCC = DIO + DSO − DPO, at both grains, with its components published."""
    truth = calculate_ground_truth(_dataset())

    annual = truth.annual
    assert annual.annual_dio > 0 and annual.annual_dso > 0 and annual.annual_dpo > 0
    assert annual.annual_cash_conversion_cycle == round(annual.annual_dio + annual.annual_dso - annual.annual_dpo, 1)
    # The two denominators are named alternatives, not a mistake in one of them.
    assert annual.annual_dpo != annual.annual_dpo_on_expenses
    assert annual.total_purchases > annual.total_cogs > 0

    assert len(truth.monthly) == 12
    for month in truth.monthly:
        assert month.cash_conversion_cycle == round(month.dio + month.dso - month.dpo, 1)
        assert month.inventory_balance > 0
        assert month.dio > 0


def test_invariants_report_the_inventory_ties() -> None:
    truth = calculate_ground_truth(_dataset())
    assert truth.invariants.inventory_rollforward_holds
    assert truth.invariants.inventory_ties_to_gl


def test_ap_is_no_longer_dominated_by_unsettled_stock() -> None:
    """The payable inventory creates is a normal payable now.

    Measured directly on the ledger rather than through a metric: goods credits to AP
    are largely matched by debits, instead of the 46.6M credited and 0.00 debited the
    replenishment plug used to leave behind.
    """
    dataset = _dataset()
    ap_credits = sum((line.credit for line in dataset.journal_lines if line.account_id in _AP_ACCOUNTS), Decimal("0"))
    ap_debits = sum((line.debit for line in dataset.journal_lines if line.account_id in _AP_ACCOUNTS), Decimal("0"))
    assert ap_debits / ap_credits > 0.75, "most payables must be settled by fiscal close"
