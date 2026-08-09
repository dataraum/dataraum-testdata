"""The oracle contract — every metric publishes the definition that produced it.

The point of these tests is not that the numbers are right (``test_ground_truth`` covers
that) but that no number ships without a definition, no definition ships without a
number, and the contract's own claims about how its figures relate hold in the published
values a consumer actually reads.
"""

import functools
from decimal import Decimal

import pytest

from testdata.canonical.finance.generators import generate_finance_dataset
from testdata.ground_truth import _IMPACT_RULES, calculate_ground_truth, metric_contract
from testdata.metadata_truth import canonical_metadata_truth
from testdata.oracle import (
    INTEGRITY_SURFACES,
    METRIC_IDS,
    METRICS,
    METRICS_BY_ID,
    VARIANT_IDS,
    build_contract,
)


@functools.lru_cache(maxsize=1)
def _truth():
    return calculate_ground_truth(generate_finance_dataset(seed=42, months=12), months=12)


@functools.lru_cache(maxsize=1)
def _contract() -> dict[str, dict]:
    return {entry["id"]: entry for entry in metric_contract(_truth())}


# --- the contract itself ----------------------------------------------------


def test_every_metric_publishes_a_definition_and_values() -> None:
    """The §5 exit criterion, stated directly."""
    contract = _contract()
    assert set(contract) == METRIC_IDS
    for metric_id, entry in contract.items():
        assert entry["definition"].strip(), metric_id
        assert entry["scope"].strip(), metric_id
        assert entry["unit"] and entry["kind"] and entry["window"], metric_id
        assert entry["basis"] == "derived", metric_id
        assert set(entry["values"]) == set(entry["grains"]), metric_id


def test_a_metric_cannot_ship_without_values() -> None:
    """The guard that makes the contract a contract rather than a convention."""
    with pytest.raises(ValueError, match="has a definition but no values"):
        build_contract({}, {})


def test_values_cannot_ship_without_a_definition() -> None:
    with pytest.raises(ValueError, match="unknown metrics"):
        build_contract({"days_to_kevin": {"year": {"2025": 1}}}, {})


def test_declared_grains_must_match_the_values() -> None:
    values = {m.id: {g: {} for g in m.grains} for m in METRICS}
    variants = {v.id: {"year": {}} for m in METRICS for v in m.variants}
    values["dpo"] = {"month": {}}  # drops the year grain the metric declares
    with pytest.raises(ValueError, match="declares grains"):
        build_contract(values, variants)


def test_the_window_rule_is_published_per_kind() -> None:
    """A consumer reading a quarter out of monthly values must not have to guess.

    A stock takes the window's last period; a ratio is recomputed on the window's own
    aggregates rather than averaged. Getting this wrong is the classic way a correct
    monthly series produces a wrong quarterly number.
    """
    expected = {"flow": "sum", "count": "sum", "stock": "last", "ratio": "recompute"}
    for entry in _contract().values():
        assert entry["window"] == expected[entry["kind"]], entry["id"]

    assert _contract()["ar_balance"]["window"] == "last"
    assert _contract()["dso"]["window"] == "recompute"


# --- variants ---------------------------------------------------------------


def test_dpo_publishes_its_expense_denominator_variant() -> None:
    """The worked example: two correct answers to different questions, both published."""
    dpo = _contract()["dpo"]
    variant = next(v for v in dpo["variants"] if v["id"] == "dpo_on_total_expenses")

    assert variant["definition"] != dpo["definition"]
    assert variant["rationale"].strip()
    assert set(variant["values"]) <= set(dpo["values"])
    assert variant["values"]["year"] != dpo["values"]["year"]


def test_every_declared_variant_carries_its_own_values() -> None:
    published = {v["id"] for e in _contract().values() for v in e.get("variants", [])}
    assert published == VARIANT_IDS
    for entry in _contract().values():
        for variant in entry.get("variants", []):
            assert variant["values"], variant["id"]
            assert set(variant["values"]) <= set(entry["grains"]), variant["id"]


def test_the_operating_revenue_variant_is_what_the_order_lines_reconstruct() -> None:
    """The contract's sharpest claim, and the reason the variant is not cosmetic.

    Entity-grain revenue comes from the order lines; period-grain revenue comes from the
    GL and carries 43xx other income on top. So the customer totals reconcile to the
    VARIANT, never to the pinned metric — and the contract says so rather than leaving
    a consumer to discover the gap as an unexplained delta.
    """
    revenue = _contract()["revenue"]
    year = next(iter(revenue["values"]["year"]))
    pinned = revenue["values"]["year"][year]
    variant = next(v for v in revenue["variants"] if v["id"] == "operating_revenue")
    operating = variant["values"]["year"][year]

    by_customer = sum(revenue["values"]["customer"].values(), Decimal("0"))
    by_group = sum(revenue["values"]["product_group"].values(), Decimal("0"))

    assert by_customer == by_group
    assert abs(by_customer - operating) < Decimal("1.00")
    assert pinned > operating  # other income belongs to no customer


def test_cogs_ties_the_two_cuts_to_the_ledger_exactly() -> None:
    """Both sides are units x standard_cost, so this is a tie, not an approximation."""
    cogs = _contract()["cogs"]
    year = next(iter(cogs["values"]["year"]))
    assert sum(cogs["values"]["customer"].values(), Decimal("0")) == cogs["values"]["year"][year]
    assert sum(cogs["values"]["product_group"].values(), Decimal("0")) == cogs["values"]["year"][year]


# --- self-consistency of the published values -------------------------------


def test_the_published_ccc_recombines_from_the_published_components() -> None:
    """A consumer recomputing CCC from the file must land on the file's own answer."""
    values = {mid: _contract()[mid]["values"] for mid in ("dio", "dso", "dpo", "cash_conversion_cycle")}
    for grain in ("month", "year"):
        for key in values["cash_conversion_cycle"][grain]:
            expected = round(values["dio"][grain][key] + values["dso"][grain][key] - values["dpo"][grain][key], 1)
            assert values["cash_conversion_cycle"][grain][key] == expected, (grain, key)

    ccc = _contract()["cash_conversion_cycle"]
    variant = next(v for v in ccc["variants"] if v["id"] == "cash_conversion_cycle_on_expense_dpo")
    dpo_variant = next(v for v in _contract()["dpo"]["variants"] if v["id"] == "dpo_on_total_expenses")
    year = next(iter(ccc["values"]["year"]))
    assert variant["values"]["year"][year] == round(
        values["dio"]["year"][year] + values["dso"]["year"][year] - dpo_variant["values"]["year"][year], 1
    )


def test_flows_sum_across_the_window_and_stocks_do_not() -> None:
    """The window rule, checked against the values rather than only declared."""
    for metric_id in ("revenue", "cogs", "expenses", "gross_profit", "free_cash_flow"):
        entry = _contract()[metric_id]
        year = next(iter(entry["values"]["year"]))
        monthly_sum = sum(entry["values"]["month"].values(), Decimal("0"))
        assert monthly_sum == entry["values"]["year"][year], metric_id

    for metric_id in ("ar_balance", "ap_balance", "inventory_balance"):
        entry = _contract()[metric_id]
        year = next(iter(entry["values"]["year"]))
        last_month = list(entry["values"]["month"])[-1]
        assert entry["values"]["year"][year] == entry["values"]["month"][last_month], metric_id


def test_gross_profit_and_operating_income_are_separate_published_metrics() -> None:
    """The mislabelling the contract was written to make impossible."""
    contract = _contract()
    year = next(iter(contract["revenue"]["values"]["year"]))
    revenue = contract["revenue"]["values"]["year"][year]
    cogs = contract["cogs"]["values"]["year"][year]
    expenses = contract["expenses"]["values"]["year"][year]

    assert contract["gross_profit"]["values"]["year"][year] == revenue - cogs
    assert contract["operating_income"]["values"]["year"][year] == revenue - expenses
    assert contract["gross_profit"]["definition"] == "revenue - cogs"
    assert contract["operating_income"]["definition"] == "revenue - expenses"


def test_per_entity_unit_metrics_are_first_class() -> None:
    """§5: a dimension without at least one graded per-entity metric is not lit."""
    for metric_id in ("db1", "db1_pct", "units_sold", "order_count"):
        entry = _contract()[metric_id]
        assert set(entry["grains"]) == {"customer", "product_group"}
        assert entry["values"]["customer"], metric_id
        assert entry["values"]["product_group"], metric_id

    db1 = _contract()["db1"]["values"]
    assert sum(db1["customer"].values(), Decimal("0")) == sum(db1["product_group"].values(), Decimal("0"))


def test_month_only_metrics_publish_no_year_value() -> None:
    """Growth has no annual figure — there is no prior year inside the corpus."""
    growth = _contract()["revenue_growth_pct"]
    assert growth["grains"] == ["month"]
    assert list(growth["values"]["month"].values())[0] is None


def test_counted_metrics_declare_their_breakdown_axis() -> None:
    invoices = _contract()["invoice_count"]
    assert invoices["breakdown"] == "status"
    year = next(iter(invoices["values"]["year"]))
    totals = invoices["values"]["year"][year]
    assert sum(totals.values()) == sum(sum(m.values()) for m in invoices["values"]["month"].values())
    assert _contract()["payment_count"]["breakdown"] == "method"


# --- the contract against its neighbours ------------------------------------


def test_injection_impact_reports_against_defined_targets_only() -> None:
    """A target that looks like a metric id and has no definition is what this removes."""
    targets = {metric for rules in _IMPACT_RULES.values() for metric, _ in rules}
    assert targets <= METRIC_IDS | INTEGRITY_SURFACES
    assert not METRIC_IDS & INTEGRITY_SURFACES


def test_the_additivity_verdict_agrees_with_the_metric_kind() -> None:
    """``metadata_truth`` publishes additivity per metric NAME; the kind fixes it.

    Two files naming the same metric is exactly the shape that drifts — ``metadata_truth``
    still described DPO as AP over COGS long after the pinned definition moved to
    purchases. Where the names overlap, the verdicts must follow from the kind.
    """
    by_kind = {
        "flow": (True, True),
        "count": (True, True),
        "stock": (True, False),
        "ratio": (False, False),
    }
    additivity = canonical_metadata_truth()["metric_additivity"]["metrics"]
    overlap = set(additivity) & METRIC_IDS
    assert overlap, "the two files stopped naming any metric in common — check the seam"
    for metric_id in sorted(overlap):
        categorical, temporal = by_kind[METRICS_BY_ID[metric_id].kind]
        assert additivity[metric_id]["categorical_additive"] is categorical, metric_id
        assert additivity[metric_id]["time_additive"] is temporal, metric_id
