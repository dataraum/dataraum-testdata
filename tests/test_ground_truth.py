"""Tests for ground truth calculator."""

import tempfile
from decimal import Decimal
from pathlib import Path

import yaml

from testdata.canonical.finance.generators import generate_finance_dataset
from testdata.ground_truth import (
    calculate_ground_truth,
    estimate_injection_impact,
    export_ground_truth,
)


def _truth():
    ds = generate_finance_dataset(seed=42, months=12)
    return calculate_ground_truth(ds, seed=42, strategy="clean", months=12)


def test_annual_revenue_positive():
    """Annual revenue should be a large positive number."""
    truth = _truth()
    assert truth.annual.total_revenue > 100_000


def test_annual_expenses_positive():
    truth = _truth()
    assert truth.annual.total_expenses > 100_000


def test_gross_profit_is_revenue_minus_expenses():
    truth = _truth()
    assert truth.annual.gross_profit == truth.annual.total_revenue - truth.annual.total_expenses


def test_monthly_periods_count():
    truth = _truth()
    assert len(truth.monthly) == 12
    assert truth.monthly[0].period == "2025-01"
    assert truth.monthly[-1].period == "2025-12"


def test_monthly_revenue_sums_to_annual():
    truth = _truth()
    monthly_sum = sum(m.revenue for m in truth.monthly)
    assert monthly_sum == truth.annual.total_revenue


def test_monthly_expenses_sums_to_annual():
    truth = _truth()
    monthly_sum = sum(m.expenses for m in truth.monthly)
    assert monthly_sum == truth.annual.total_expenses


def test_ending_balances_match_last_month():
    """Annual ending balances should match the last month's cumulative balances."""
    truth = _truth()
    last = truth.monthly[-1]
    assert truth.annual.ending_ar_balance == last.ar_balance
    assert truth.annual.ending_ap_balance == last.ap_balance
    assert truth.annual.ending_cash_balance == last.cash_balance


def test_dso_reasonable():
    """DSO should be between 0 and 365 days."""
    truth = _truth()
    assert 0 < truth.annual.annual_dso < 365
    for m in truth.monthly:
        assert m.dso >= 0


def test_dpo_reasonable():
    """DPO should be between 0 and 365 days."""
    truth = _truth()
    assert 0 < truth.annual.annual_dpo < 365


def test_invariants_clean_data():
    """Clean data should pass all invariants."""
    truth = _truth()
    assert truth.invariants.journal_balanced is True
    assert truth.invariants.trial_balance_balanced is True
    assert truth.invariants.invoice_payment_matched is True
    assert truth.invariants.bank_reconciliation_rate > 0.80


def test_revenue_growth_first_month_none():
    """First month has no revenue growth (no prior month)."""
    truth = _truth()
    assert truth.monthly[0].revenue_growth_pct is None


def test_revenue_growth_subsequent_months():
    """Subsequent months have revenue growth percentage."""
    truth = _truth()
    for m in truth.monthly[1:]:
        assert m.revenue_growth_pct is not None


def test_invoice_counts_present():
    """Monthly metrics include invoice counts by status."""
    truth = _truth()
    total_invoices = sum(
        sum(m.invoice_count.values()) for m in truth.monthly
    )
    assert total_invoices > 0


def test_free_cash_flow():
    """FCF should be non-zero — there are both inflows and outflows."""
    truth = _truth()
    # FCF can be positive or negative, just not exactly zero with 5500+ bank txns
    assert truth.annual.free_cash_flow != Decimal("0")


def test_deterministic():
    """Same seed produces identical ground truth."""
    t1 = _truth()
    t2 = _truth()
    assert t1.annual.total_revenue == t2.annual.total_revenue
    assert t1.annual.free_cash_flow == t2.annual.free_cash_flow
    assert t1.monthly[6].dso == t2.monthly[6].dso


def test_export_creates_yaml():
    """export_ground_truth writes a valid YAML file."""
    truth = _truth()
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir)
        export_ground_truth(truth, output)

        gt_path = output / "ground_truth.yaml"
        assert gt_path.exists()

        with open(gt_path) as f:
            data = yaml.safe_load(f)

        assert data["generator"] == "finance"
        assert data["seed"] == 42
        assert data["annual"]["total_revenue"] > 0
        assert len(data["monthly"]) == 12
        assert data["invariants"]["journal_balanced"] is True


def test_scenario_includes_ground_truth():
    """run_scenario returns ground truth in result dict."""
    from testdata.scenarios.month_end_close import run_scenario

    result = run_scenario(strategy_name="clean", seed=42, months=6)
    assert "ground_truth" in result
    gt = result["ground_truth"]
    assert gt.annual.total_revenue > 0
    assert len(gt.monthly) == 6


def test_scenario_export_includes_ground_truth_yaml():
    """Exported scenario includes ground_truth.yaml alongside manifest."""
    from testdata.scenarios.month_end_close import run_scenario

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "output"
        run_scenario(strategy_name="clean", seed=42, months=6, output_dir=output)
        assert (output / "ground_truth.yaml").exists()
        assert (output / "manifest.yaml").exists()


# --- Injection impact ---


def test_clean_has_no_injection_impact():
    """Clean strategy produces no injection impact."""
    from testdata.scenarios.month_end_close import run_scenario

    result = run_scenario(strategy_name="clean", seed=42, months=6)
    assert result["ground_truth"].injection_impact == []


def test_medium_has_injection_impact():
    """Medium strategy produces injection impact estimates."""
    from testdata.scenarios.month_end_close import run_scenario

    result = run_scenario(strategy_name="medium", seed=42, months=12)
    gt = result["ground_truth"]
    assert len(gt.injection_impact) > 0
    # Each impact should have metric, error_pct, and affected_by
    for impact in gt.injection_impact:
        assert impact.metric
        assert impact.expected_error_pct > 0
        assert len(impact.affected_by) > 0


def test_estimate_injection_impact_directly():
    """Direct call to estimate_injection_impact works."""
    injections = [
        {
            "injection_type": "corrupt_type",
            "target_file": "journal_lines.csv",
            "parameters": {"ratio": 0.03},
        },
        {
            "injection_type": "mix_units",
            "target_file": "invoices.csv",
            "parameters": {"ratio": 0.10, "fx_rate": 1.1},
        },
    ]
    impacts = estimate_injection_impact(injections)
    metrics = {i.metric for i in impacts}
    assert "revenue" in metrics
    assert "invoice_totals" in metrics


def test_injection_impact_in_exported_yaml():
    """Exported YAML includes injection_impact for non-clean strategies."""
    from testdata.scenarios.month_end_close import run_scenario

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "output"
        run_scenario(strategy_name="medium", seed=42, months=6, output_dir=output)

        with open(output / "ground_truth.yaml") as f:
            data = yaml.safe_load(f)

        assert len(data["injection_impact"]) > 0
        assert data["injection_impact"][0]["metric"]
        assert data["injection_impact"][0]["expected_error_pct"] > 0
