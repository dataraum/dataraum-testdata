"""Tests for scenario orchestration and end-to-end generation."""

import tempfile
from pathlib import Path

import yaml

from testdata.scenarios.month_end_close import run_scenario


def test_clean_scenario_no_injections():
    """Clean strategy produces data with zero injections."""
    result = run_scenario(strategy_name="clean", seed=42, months=6)
    assert len(result["registry"]) == 0
    assert len(result["dataframes"]["journal_entries"]) > 0


def test_medium_scenario_has_injections():
    """Medium strategy produces data with multiple injection types."""
    result = run_scenario(strategy_name="medium", seed=42, months=12)
    assert len(result["registry"]) >= 10
    summary = result["registry"].summary()
    # Should have injections across multiple layers
    assert len(summary["by_layer"]) >= 3


def test_high_scenario_has_more_injections():
    """High strategy has more injections than medium."""
    medium = run_scenario(strategy_name="medium", seed=42, months=12)
    high = run_scenario(strategy_name="high", seed=42, months=12)
    assert len(high["registry"]) > len(medium["registry"])


def test_export_creates_files():
    """Exported scenario creates all expected files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "output"
        run_scenario(strategy_name="medium", seed=42, months=6, output_dir=output)

        assert (output / "manifest.yaml").exists()
        assert (output / "entropy_map.yaml").exists()
        assert (output / "journal_lines.csv").exists()
        assert (output / "invoices.csv").exists()
        assert (output / "bank_transactions.csv").exists()
        assert (output / "payments.csv").exists()

        # Verify manifest structure
        with open(output / "manifest.yaml") as f:
            manifest = yaml.safe_load(f)
        assert manifest["generator"] == "dataraum-testdata"
        assert len(manifest["files"]) == 9  # 8 canonical tables + balance_sheet

        # Verify entropy map has injections
        with open(output / "entropy_map.yaml") as f:
            emap = yaml.safe_load(f)
        assert emap["total_injections"] > 0
        assert len(emap["injections"]) == emap["total_injections"]


def test_deterministic_scenario():
    """Same seed + strategy produces identical output."""
    r1 = run_scenario(strategy_name="medium", seed=42, months=12)
    r2 = run_scenario(strategy_name="medium", seed=42, months=12)

    for table in r1["dataframes"]:
        assert r1["dataframes"][table].shape == r2["dataframes"][table].shape

    assert len(r1["registry"]) == len(r2["registry"])


def test_low_strategy():
    """Low strategy produces fewer injections than medium."""
    low = run_scenario(strategy_name="low", seed=42, months=12)
    medium = run_scenario(strategy_name="medium", seed=42, months=12)
    assert len(low["registry"]) < len(medium["registry"])


_EVENTS_BACKED_STRATEGY = """\
name: events-backed-stockflow-test
level: high
description: events-backed stock/flow probes (DAT-491)
injections:
  - injector: inject_stock_flow_probes
    table: measure_probes
    detector_id: temporal_behavior
    params:
      seed: 7
      n_columns: [10, 10]
      backed_fraction: [1.0, 1.0]
      broken_fraction: [0.5, 0.5]
"""


def test_events_backed_stockflow_strategy_emits_probe_events():
    """The runner threads the dataframes dict into the injector, exports probe_events,
    and the entropy map carries the backed/reconciles ground truth (DAT-491)."""
    from testdata.scenarios.runner import run_scenario as run

    with tempfile.TemporaryDirectory() as tmpdir:
        strategy_path = Path(tmpdir) / "events_backed.yaml"
        strategy_path.write_text(_EVENTS_BACKED_STRATEGY)
        output = Path(tmpdir) / "output"
        result = run("month-end-close", strategy_file=strategy_path, seed=42, months=6, output_dir=output)

        probes = result["dataframes"]["measure_probes"]
        events = result["dataframes"]["probe_events"]
        # The shared slice dimension + strictly finer event grain (>= 2 events per cell).
        assert set(events["series_id"].to_list()) == set(probes["series_id"].to_list())
        assert len(events) >= 2 * len(probes)
        assert (output / "probe_events.csv").exists()
        assert (output / "measure_probes.csv").exists()

        with open(output / "entropy_map.yaml") as f:
            emap = yaml.safe_load(f)
        stockflow = [
            inj["parameters"] for inj in emap["injections"] if inj["injection_type"] == "inject_stock_flow_probes"
        ]
        backed = [p for p in stockflow if p["backed"]]
        assert backed and all(p["events_table"] == "probe_events" for p in backed)
        assert {p["reconciles"] for p in backed} == {True, False}  # both calibration strata
        for p in backed:
            assert p["events_column"] in events.columns
