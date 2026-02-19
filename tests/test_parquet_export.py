"""Tests for Parquet export support."""

import tempfile
from pathlib import Path

import polars as pl
import yaml

from testdata.scenarios.month_end_close import run_scenario


def test_parquet_export():
    """Parquet format creates .parquet files instead of .csv."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "parquet_out"
        run_scenario(strategy_name="clean", seed=42, months=6, output_dir=output, fmt="parquet")

        assert (output / "journal_lines.parquet").exists()
        assert (output / "invoices.parquet").exists()
        assert not (output / "journal_lines.csv").exists()

        # Verify Parquet is readable
        df = pl.read_parquet(output / "journal_lines.parquet")
        assert len(df) > 0
        assert "entry_id" in df.columns


def test_both_format_export():
    """'both' format creates CSV and Parquet side by side."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "both_out"
        run_scenario(strategy_name="clean", seed=42, months=6, output_dir=output, fmt="both")

        assert (output / "journal_lines.csv").exists()
        assert (output / "journal_lines.parquet").exists()
        assert (output / "invoices.csv").exists()
        assert (output / "invoices.parquet").exists()

        # Manifest should list both formats
        with open(output / "manifest.yaml") as f:
            manifest = yaml.safe_load(f)
        # 8 tables × 2 formats = 16 file entries
        assert len(manifest["files"]) == 16


def test_parquet_manifest_structure():
    """Parquet manifest entries have correct file extensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "pq"
        run_scenario(strategy_name="clean", seed=42, months=6, output_dir=output, fmt="parquet")

        with open(output / "manifest.yaml") as f:
            manifest = yaml.safe_load(f)

        for entry in manifest["files"]:
            assert entry["file"].endswith(".parquet")
            assert entry["rows"] >= 0
            assert isinstance(entry["columns"], list)


def test_csv_default_unchanged():
    """Default CSV export still works as before."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "csv_out"
        run_scenario(strategy_name="clean", seed=42, months=6, output_dir=output)

        assert (output / "journal_lines.csv").exists()
        assert not (output / "journal_lines.parquet").exists()

        with open(output / "manifest.yaml") as f:
            manifest = yaml.safe_load(f)
        assert len(manifest["files"]) == 8
        for entry in manifest["files"]:
            assert entry["file"].endswith(".csv")


def test_parquet_with_injections():
    """Parquet export preserves entropy injections correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "injected_pq"
        run_scenario(strategy_name="medium", seed=42, months=6, output_dir=output, fmt="parquet")

        with open(output / "entropy_map.yaml") as f:
            emap = yaml.safe_load(f)
        assert emap["total_injections"] > 0

        # Verify Parquet files are readable after injection
        df = pl.read_parquet(output / "journal_lines.parquet")
        assert len(df) > 0
