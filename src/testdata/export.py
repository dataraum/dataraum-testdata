"""Export finance dataset to CSV files + manifest YAML."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from testdata.canonical.finance.models import FinanceDataset


# Table name -> model field name mapping
TABLE_NAMES: dict[str, str] = {
    "chart_of_accounts": "chart_of_accounts",
    "journal_entries": "journal_entries",
    "journal_lines": "journal_lines",
    "invoices": "invoices",
    "payments": "payments",
    "bank_transactions": "bank_transactions",
    "fx_rates": "fx_rates",
    "trial_balance": "trial_balance",
}


def _serialize_value(v: Any) -> Any:
    """Convert Pydantic-native types to CSV-friendly primitives."""
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    return str(v)


def dataset_to_dataframes(dataset: FinanceDataset) -> dict[str, pl.DataFrame]:
    """Convert a FinanceDataset into a dict of Polars DataFrames."""
    result: dict[str, pl.DataFrame] = {}

    for table_name, field_name in TABLE_NAMES.items():
        records = getattr(dataset, field_name)
        if not records:
            result[table_name] = pl.DataFrame()
            continue

        rows = [
            {k: _serialize_value(v) for k, v in rec.model_dump().items()}
            for rec in records
        ]
        result[table_name] = pl.DataFrame(rows)

    return result


def export_dataset(
    dataset: FinanceDataset,
    output_dir: Path,
    entropy_records: list[dict] | None = None,
    generation_params: dict | None = None,
) -> None:
    """Export dataset to CSV files with manifest and optional entropy map.

    Args:
        dataset: The finance dataset to export.
        output_dir: Directory to write files into.
        entropy_records: Optional list of entropy injection records for the map.
        generation_params: Optional dict of generation parameters for the manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframes = dataset_to_dataframes(dataset)
    file_manifest: list[dict[str, Any]] = []

    for table_name, df in dataframes.items():
        csv_path = output_dir / f"{table_name}.csv"
        df.write_csv(csv_path)
        file_manifest.append({
            "file": f"{table_name}.csv",
            "table": table_name,
            "rows": len(df),
            "columns": df.columns,
        })

    # Write manifest
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "generator": "dataraum-testdata",
        "version": "0.1.0",
        "parameters": generation_params or {},
        "files": file_manifest,
    }
    manifest_path = output_dir / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    # Write entropy map
    entropy_path = output_dir / "entropy_map.yaml"
    if entropy_records:
        with open(entropy_path, "w") as f:
            yaml.dump(
                {"injections": entropy_records, "total_injections": len(entropy_records)},
                f,
                default_flow_style=False,
                sort_keys=False,
            )
    else:
        with open(entropy_path, "w") as f:
            yaml.dump(
                {"injections": [], "total_injections": 0},
                f,
                default_flow_style=False,
                sort_keys=False,
            )


def export_dataframes(
    dataframes: dict[str, pl.DataFrame],
    output_dir: Path,
    entropy_records: list[dict] | None = None,
    generation_params: dict | None = None,
) -> None:
    """Export pre-built DataFrames (after injection) to CSV + manifest.

    This is the post-injection export path: the scenario has already
    mutated the DataFrames and recorded entropy injections.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    file_manifest: list[dict[str, Any]] = []

    for table_name, df in dataframes.items():
        csv_path = output_dir / f"{table_name}.csv"
        df.write_csv(csv_path)
        file_manifest.append({
            "file": f"{table_name}.csv",
            "table": table_name,
            "rows": len(df),
            "columns": df.columns,
        })

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "generator": "dataraum-testdata",
        "version": "0.1.0",
        "parameters": generation_params or {},
        "files": file_manifest,
    }
    with open(output_dir / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    entropy_path = output_dir / "entropy_map.yaml"
    entropy_data = {
        "injections": entropy_records or [],
        "total_injections": len(entropy_records) if entropy_records else 0,
    }
    with open(entropy_path, "w") as f:
        yaml.dump(entropy_data, f, default_flow_style=False, sort_keys=False)
