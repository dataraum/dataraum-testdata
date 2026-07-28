"""Export finance dataset to CSV/Parquet/JSON/JSONL files + manifest YAML."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import polars as pl
import yaml

from testdata.canonical.finance.models import FinanceDataset


ExportFormat = Literal["csv", "parquet", "json", "jsonl", "both"]

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
    "balance_sheet": "balance_sheet",
    "measure_probes": "measure_probes",
    "formula_probes": "formula_probes",
    "ref_entities": "ref_entities",
    "ref_activity": "ref_activity",
    "addresses": "addresses",
    "orders": "orders",
    "deliveries": "deliveries",
    # The operating chain (DAT-884) — always present, unlike the probe shapes above.
    "customers": "customers",
    "products": "products",
    "sales_orders": "sales_orders",
    "sales_order_lines": "sales_order_lines",
}


def _serialize_value(v: Any) -> Any:
    """Convert Pydantic-native types to export-friendly primitives."""
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
            # An empty table (e.g. measure_probes when no stock/flow strategy is
            # active) is omitted entirely rather than emitted as a headerless CSV.
            continue

        rows = [{k: _serialize_value(v) for k, v in rec.model_dump().items()} for rec in records]
        result[table_name] = pl.DataFrame(rows)

    return result


def _write_table(
    df: pl.DataFrame,
    output_dir: Path,
    table_name: str,
    fmt: ExportFormat,
) -> list[dict[str, Any]]:
    """Write a single table in the requested format(s). Returns manifest entries."""
    entries: list[dict[str, Any]] = []
    base = {"table": table_name, "rows": len(df), "columns": df.columns}

    if fmt in ("csv", "both"):
        csv_path = output_dir / f"{table_name}.csv"
        df.write_csv(csv_path)
        entries.append({"file": f"{table_name}.csv", **base})

    if fmt in ("parquet", "both"):
        parquet_path = output_dir / f"{table_name}.parquet"
        df.write_parquet(parquet_path)
        entries.append({"file": f"{table_name}.parquet", **base})

    if fmt == "json":
        json_path = output_dir / f"{table_name}.json"
        df.write_json(json_path)
        entries.append({"file": f"{table_name}.json", **base})

    if fmt == "jsonl":
        jsonl_path = output_dir / f"{table_name}.jsonl"
        df.write_ndjson(jsonl_path)
        entries.append({"file": f"{table_name}.jsonl", **base})

    return entries


def export_dataset(
    dataset: FinanceDataset,
    output_dir: Path,
    entropy_records: list[dict] | None = None,
    generation_params: dict | None = None,
    fmt: ExportFormat = "csv",
) -> None:
    """Export dataset to files with manifest and optional entropy map.

    Args:
        dataset: The finance dataset to export.
        output_dir: Directory to write files into.
        entropy_records: Optional list of entropy injection records for the map.
        generation_params: Optional dict of generation parameters for the manifest.
        fmt: Export format — "csv", "parquet", or "both".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframes = dataset_to_dataframes(dataset)

    file_manifest: list[dict[str, Any]] = []
    for table_name, df in dataframes.items():
        file_manifest.extend(_write_table(df, output_dir, table_name, fmt))

    _write_manifest(output_dir, file_manifest, generation_params)
    _write_entropy_map(output_dir, entropy_records)


def export_dataframes(
    dataframes: dict[str, pl.DataFrame],
    output_dir: Path,
    entropy_records: list[dict] | None = None,
    generation_params: dict | None = None,
    fmt: ExportFormat = "csv",
) -> None:
    """Export pre-built DataFrames (after injection) to files + manifest.

    This is the post-injection export path: the scenario has already
    mutated the DataFrames and recorded entropy injections.

    Args:
        dataframes: Table name → DataFrame mapping.
        output_dir: Directory to write files into.
        entropy_records: Optional list of entropy injection records.
        generation_params: Optional dict of generation parameters.
        fmt: Export format — "csv", "parquet", or "both".
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    file_manifest: list[dict[str, Any]] = []
    for table_name, df in dataframes.items():
        file_manifest.extend(_write_table(df, output_dir, table_name, fmt))

    _write_manifest(output_dir, file_manifest, generation_params)
    _write_entropy_map(output_dir, entropy_records)


def _write_manifest(
    output_dir: Path,
    file_manifest: list[dict[str, Any]],
    generation_params: dict | None,
) -> None:
    """Write manifest.yaml."""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "generator": "dataraum-testdata",
        "version": "0.1.0",
        "parameters": generation_params or {},
        "files": file_manifest,
    }
    with open(output_dir / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)


def _write_entropy_map(
    output_dir: Path,
    entropy_records: list[dict] | None,
) -> None:
    """Write entropy_map.yaml."""
    entropy_data = {
        "injections": entropy_records or [],
        "total_injections": len(entropy_records) if entropy_records else 0,
    }
    with open(output_dir / "entropy_map.yaml", "w") as f:
        yaml.dump(entropy_data, f, default_flow_style=False, sort_keys=False)
