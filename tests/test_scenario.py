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
