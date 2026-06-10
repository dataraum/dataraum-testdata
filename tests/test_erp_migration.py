"""Tests for ERP migration scenario."""

import tempfile
from pathlib import Path

import yaml

from testdata.scenarios.erp_migration import run_scenario


def test_erp_clean_no_injections():
    """Clean strategy produces data with zero injections."""
    result = run_scenario(strategy_name="clean", seed=7777, months=6)
    assert len(result["registry"]) == 0
    assert len(result["dataframes"]) > 0


def test_erp_high_has_injections():
    """High strategy (default for ERP migration) produces many injections."""
    result = run_scenario(strategy_name="high", seed=7777, months=6)
    assert len(result["registry"]) >= 15
    summary = result["registry"].summary()
    assert len(summary["by_layer"]) >= 3


def test_erp_uses_partial_normalization():
    """ERP migration defaults to partial normalization (merged tables)."""
    result = run_scenario(strategy_name="clean", seed=7777, months=6)
    tables = set(result["dataframes"].keys())
    # partial merges journal_lines+journal_entries -> journal_data
    # and invoices+payments -> invoice_data
    assert "journal_data" in tables
    assert "invoice_data" in tables
    assert "journal_lines" not in tables
    assert "journal_entries" not in tables
    assert "invoices" not in tables
    assert "payments" not in tables
    assert len(tables) == 7  # partial: 6 + balance_sheet (a standalone period table)


def test_erp_export_creates_files():
    """Exported ERP scenario creates expected files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "erp"
        run_scenario(strategy_name="medium", seed=7777, months=6, output_dir=output)

        assert (output / "manifest.yaml").exists()
        assert (output / "entropy_map.yaml").exists()
        # partial normalization tables
        assert (output / "journal_data.csv").exists()
        assert (output / "invoice_data.csv").exists()
        assert (output / "bank_transactions.csv").exists()

        with open(output / "manifest.yaml") as f:
            manifest = yaml.safe_load(f)
        assert manifest["parameters"]["scenario"] == "erp-migration"
        assert len(manifest["files"]) == 7  # partial normalization + balance_sheet


def test_erp_deterministic():
    """Same seed produces identical output."""
    r1 = run_scenario(strategy_name="high", seed=7777, months=6)
    r2 = run_scenario(strategy_name="high", seed=7777, months=6)

    for table in r1["dataframes"]:
        assert r1["dataframes"][table].shape == r2["dataframes"][table].shape
    assert len(r1["registry"]) == len(r2["registry"])


def test_erp_different_seed_from_month_end():
    """ERP migration uses different default seed than month-end-close."""
    from testdata.scenarios.month_end_close import run_scenario as mec_run

    erp = run_scenario(strategy_name="clean", months=6)
    mec = mec_run(strategy_name="clean", months=6)

    # Different seeds should produce different row counts
    # (different generator params: invoices_count, bank_transactions_count, etc.)
    erp_rows = sum(len(df) for df in erp["dataframes"].values())
    mec_rows = sum(len(df) for df in mec["dataframes"].values())
    assert erp_rows != mec_rows
