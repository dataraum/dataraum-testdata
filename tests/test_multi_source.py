"""Tests for multi-source scenario."""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import yaml

from testdata.scenarios.multi_system_recon import run_scenario


def test_multi_source_returns_all_tables():
    """run_scenario returns all 8 tables in dataframes dict."""
    result = run_scenario(strategy_name="clean", seed=42, months=3)
    assert len(result["dataframes"]) == 8


def test_multi_source_has_sources_config():
    """Config includes source definitions."""
    result = run_scenario(strategy_name="clean", seed=42, months=3)
    config = result["config"]
    assert config.sources is not None
    assert len(config.sources) == 3


def test_multi_source_export_creates_subdirectories():
    """Export creates one subdirectory per source."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "output"
        run_scenario(strategy_name="clean", seed=42, months=3, output_dir=output)

        assert (output / "erp_export").is_dir()
        assert (output / "banking_feed").is_dir()
        assert (output / "ap_system").is_dir()


def test_multi_source_erp_has_legacy_columns():
    """ERP source uses legacy column naming."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "output"
        run_scenario(strategy_name="clean", seed=42, months=3, output_dir=output)

        jl = pl.read_csv(output / "erp_export" / "journal_lines.csv")
        assert "DR_AMT" in jl.columns
        assert "CR_AMT" in jl.columns
        assert "ACCT_NO" in jl.columns


def test_multi_source_banking_has_pascal_columns():
    """Banking source uses PascalCase column naming."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "output"
        run_scenario(strategy_name="clean", seed=42, months=3, output_dir=output)

        bt = pl.read_csv(output / "banking_feed" / "bank_transactions.csv")
        assert "TxnId" in bt.columns
        assert "Amount" in bt.columns


def test_multi_source_ap_has_camel_columns():
    """AP source uses camelCase column naming."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "output"
        run_scenario(strategy_name="clean", seed=42, months=3, output_dir=output)

        inv = pl.read_csv(output / "ap_system" / "invoices.csv")
        assert "invoiceId" in inv.columns
        assert "vendorId" in inv.columns


def test_multi_source_has_sources_yaml():
    """Export produces a top-level sources.yaml index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "output"
        run_scenario(strategy_name="clean", seed=42, months=3, output_dir=output)

        sources_path = output / "sources.yaml"
        assert sources_path.exists()

        with open(sources_path) as f:
            data = yaml.safe_load(f)

        assert len(data["sources"]) == 3
        names = {s["name"] for s in data["sources"]}
        assert names == {"erp_export", "banking_feed", "ap_system"}


def test_multi_source_has_ground_truth():
    """Export includes ground_truth.yaml at top level."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "output"
        run_scenario(strategy_name="clean", seed=42, months=3, output_dir=output)

        assert (output / "ground_truth.yaml").exists()


def test_multi_source_each_source_has_manifest():
    """Each source subdirectory has its own manifest.yaml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "output"
        run_scenario(strategy_name="clean", seed=42, months=3, output_dir=output)

        for src in ("erp_export", "banking_feed", "ap_system"):
            assert (output / src / "manifest.yaml").exists()


def test_multi_source_table_distribution():
    """Each source gets the correct subset of tables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "output"
        run_scenario(strategy_name="clean", seed=42, months=3, output_dir=output)

        erp_files = {p.stem for p in (output / "erp_export").glob("*.csv")}
        assert erp_files == {"chart_of_accounts", "journal_entries", "journal_lines", "trial_balance"}

        bank_files = {p.stem for p in (output / "banking_feed").glob("*.csv")}
        assert bank_files == {"bank_transactions", "fx_rates"}

        ap_files = {p.stem for p in (output / "ap_system").glob("*.csv")}
        assert ap_files == {"invoices", "payments"}
