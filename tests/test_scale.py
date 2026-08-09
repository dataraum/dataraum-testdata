"""Scale profiles — the populations, their shapes, and the expense base they size.

§9's claim is that a population without a shape is not more realistic than a handful.
These tests hold the generator to that: it is not enough for `mid` to have 400
customers, the concentration, the tail and the lifecycle have to be *there* and have to
be visible in the data a consumer actually receives.
"""

from __future__ import annotations

import functools
import statistics
from collections import Counter
from decimal import Decimal

from testdata.canonical.finance.generators import Lever, generate_finance_dataset
from testdata.canonical.finance.models import FinanceDataset
from testdata.ground_truth import calculate_ground_truth
from testdata.scale import PROFILES, get_profile


@functools.lru_cache(maxsize=4)
def _dataset(profile: str = "tiny", months: int = 12, seed: int = 42) -> FinanceDataset:
    return generate_finance_dataset(seed=seed, months=months, profile=profile)


def test_profiles_produce_the_populations_they_declare() -> None:
    """The table in §9 is a promise, not a description of intent."""
    for name in ("tiny", "mid"):
        profile = get_profile(name)
        dataset = _dataset(name)
        assert len(dataset.customers) == profile.customers
        assert len(dataset.products) == profile.products
        assert len({p.product_group for p in dataset.products}) == profile.product_groups
        assert len({i.vendor_id for i in dataset.invoices}) > 1


def test_unknown_profile_fails_loudly() -> None:
    """A typo in a scenario YAML must not silently generate the wrong firm."""
    try:
        get_profile("enormous")
    except ValueError as exc:
        assert "enormous" in str(exc) and "tiny" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an unknown profile must raise")


def test_revenue_per_customer_is_concentrated_not_uniform() -> None:
    """Pareto, so "the top 5% of customers" is a number rather than a shrug.

    A uniform book of 400 makes concentration risk exactly as unmeasurable as a book
    of 16 — which is why counts alone were never the point.
    """
    truth = calculate_ground_truth(_dataset("mid"), months=12)
    revenues = sorted((c.revenue for c in truth.db1_by_customer), reverse=True)
    assert len(revenues) > 300

    total = sum(revenues)
    top_fifth = sum(revenues[: len(revenues) // 5]) / total
    assert top_fifth > 0.40, top_fifth
    # ...and not so concentrated that one customer IS the firm — the cap's job.
    assert revenues[0] / total < 0.10


def test_order_value_is_log_normal() -> None:
    """Mean and median part company, so reporting one for the other is visibly wrong."""
    lines = _dataset("mid").sales_order_lines
    units = [line.units for line in lines]
    mean, median = statistics.mean(units), statistics.median(units)
    assert mean > median * 1.3, (mean, median)
    assert max(units) > median * 8, "a log-normal has a tail worth having"
    assert min(units) >= 1


def test_the_catalogue_has_a_tail_below_the_contribution_threshold() -> None:
    """Products that lose money once discounted — so the portfolio can be pruned.

    A catalogue where every item earns makes "what should we drop?" unanswerable, and
    that is a canonical Offer question.
    """
    dataset = _dataset("mid")
    profile = get_profile("mid")

    thin = [p for p in dataset.products if p.list_price < p.standard_cost * Decimal("1.15")]
    share = len(thin) / len(dataset.products)
    assert abs(share - profile.tail_product_fraction) < 0.05, share

    # The tail is not one group — otherwise "which products lose money" is answerable
    # from the group label alone, which is not the question.
    assert len({p.product_group for p in thin}) > 1

    # And it must actually show up as negative contribution on real lines.
    losing = [line for line in dataset.sales_order_lines if line.line_amount < line.line_cost]
    assert losing, "a thin catalogue that never sells at a loss is decoration"


def test_entities_are_born_and_die_inside_the_window() -> None:
    """Validity windows exist, and they gate the ORDERS rather than only the master row.

    This is the case a prior-period comparison fails on: a customer whose collapse is
    that they did not exist yet. A window nothing respects would not pose it.
    """
    dataset = _dataset("mid")
    assert any(c.churned_date for c in dataset.customers)
    assert any(c.created_date.year == 2025 for c in dataset.customers)
    assert any(p.discontinued_date for p in dataset.products)

    window = {c.customer_id: (c.created_date, c.churned_date) for c in dataset.customers}
    for order in dataset.sales_orders:
        created, churned = window[order.customer_id]
        assert order.order_date >= created, order.order_id
        if churned is not None:
            assert order.order_date <= churned, order.order_id


def test_the_customer_book_grows_by_appending() -> None:
    """C-0001 is the same customer at every scale.

    Names, segments and regions come from the index rather than a draw, so retuning a
    distribution never renames an entity. The catalogue is deliberately NOT stable the
    same way: a profile with 12 product groups is a different catalogue rather than a
    longer one, and pretending otherwise would mean the group count did not really
    change. So `products` is restructured across profiles while `customers` is
    extended — an asymmetry worth stating rather than smoothing over.
    """
    small = {c.customer_id: c.name for c in _dataset("tiny").customers}
    big = {c.customer_id: c.name for c in _dataset("mid").customers}
    assert small and all(big[cid] == name for cid, name in small.items())
    assert len(big) > len(small)

    groups = {p.product_group for p in _dataset("mid").products}
    assert groups > {p.product_group for p in _dataset("tiny").products}


def test_the_expense_base_follows_the_firm() -> None:
    """The §7 defect. Gross profit is a property of the business, not of a row count.

    Before this, 3,000 fixed vendor invoices plus fixed monthly payroll made the P&L
    sign an artifact: -3.66M at the small profile, and implausibly profitable one
    profile up. Operating expense is now a declared share of contribution.
    """
    for name in ("tiny", "mid"):
        truth = calculate_ground_truth(_dataset(name), months=12)
        annual = truth.annual
        assert annual.gross_profit > 0, (name, annual.gross_profit)

        margin = annual.gross_profit / annual.total_revenue
        assert 0.03 < margin < 0.15, (name, margin)

        contribution = annual.total_revenue - annual.total_cogs
        opex = annual.total_expenses - annual.total_cogs
        share = float(opex / contribution)
        assert abs(share - get_profile(name).opex_share_of_contribution) < 0.05, (name, share)


def test_the_scale_anchor_ignores_the_lever() -> None:
    """A price intervention must not mechanically move payroll.

    ``intervention.yaml`` declares the expenditure cycle unaffected. If the expense
    base tracked levered revenue, that claim would be false and the counterfactual's
    difference would no longer be attributable to the intervention alone.
    """
    base = generate_finance_dataset(seed=42, months=6, profile="tiny")
    levered = generate_finance_dataset(
        seed=42, months=6, profile="tiny", lever=Lever(type="volume", period_k=2, factor=1.5)
    )

    def expense_bills(dataset: FinanceDataset) -> list[tuple[str, Decimal]]:
        return [(i.invoice_id, i.amount) for i in dataset.invoices if i.category == "expense"]

    # A VOLUME lever is the case that bites: it changes the order lines the anchor is
    # computed from, so only recomputing the anchor at lever=None keeps this equal.
    assert len(levered.sales_order_lines) > len(base.sales_order_lines), "the lever must do something"
    assert expense_bills(base) == expense_bills(levered)


def test_budgeted_amounts_stay_benford() -> None:
    """Rescaling to a budget must not flatten the leading-digit distribution.

    Amounts are drawn log-uniform and multiplied by one constant, which shifts them in
    log space without touching the mantissa — so the property survives being budgeted.
    Asserted rather than assumed, because it is the reason the budget is applied as a
    scale factor instead of by redrawing inside a narrower band.
    """
    amounts = [i.amount for i in _dataset("mid").invoices if i.category == "expense"]
    assert len(amounts) > 1000
    leading = Counter(str(a).lstrip("0.")[0] for a in amounts)
    frequency = leading["1"] / sum(leading.values())
    assert 0.25 < frequency < 0.36, frequency
    assert leading["1"] > leading["9"] * 3


def test_a_bigger_profile_is_a_bigger_firm_not_a_different_one() -> None:
    """Scaling up changes magnitudes; it must not break any invariant."""
    small = calculate_ground_truth(_dataset("tiny"), months=12)
    big = calculate_ground_truth(_dataset("mid"), months=12)

    assert big.annual.total_revenue > small.annual.total_revenue * 4
    for truth in (small, big):
        invariants = truth.invariants
        assert invariants.journal_balanced
        assert invariants.trial_balance_balanced
        assert invariants.inventory_rollforward_holds
        assert invariants.inventory_ties_to_gl
        # Working-capital days stay in a comparable band: a profile is a size, not a
        # different business model.
        assert 30 < truth.annual.annual_dso < 130
        assert 15 < truth.annual.annual_dpo < 120
        assert 10 < truth.annual.annual_dio < 90


def test_every_profile_is_declared_completely() -> None:
    """A profile with a missing shape parameter would silently fall back to uniform."""
    for name, profile in PROFILES.items():
        assert profile.name == name
        assert profile.customers > 0 and profile.products > 0 and profile.suppliers > 0
        assert profile.customer_pareto_alpha > 1.0, "alpha <= 1 has no finite mean to normalise by"
        assert profile.customer_weight_cap > 1.0
        assert 0.0 < profile.tail_product_fraction < 0.5
        assert 0.0 <= profile.churn_fraction < 0.5
        assert 0.0 < profile.opex_share_of_contribution < 1.0
        assert profile.products >= profile.product_groups
