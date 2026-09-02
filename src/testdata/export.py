"""Export finance dataset to CSV/Parquet/JSON/JSONL files + manifest YAML."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import polars as pl
import yaml

from testdata.canonical.finance.models import Corpus
from testdata.families import OPTIONAL_DIMENSION_COLUMNS
from testdata.identity import CorpusIdentity


ExportFormat = Literal["csv", "parquet", "json", "jsonl", "both"]


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


def dataset_to_dataframes(dataset: Corpus) -> dict[str, pl.DataFrame]:
    """Convert a Corpus into a dict of Polars DataFrames.

    Reads ``Corpus.tables`` — the family-ordered view — so the exporter never learns a
    table's name. A new family appears here by being declared, not by being added.
    """
    result: dict[str, pl.DataFrame] = {}

    for table_name, records in dataset.tables.items():
        if not records:
            # An empty table (e.g. measure_probes when no stock/flow strategy is
            # active) is omitted entirely rather than emitted as a headerless CSV.
            continue

        rows = [{k: _serialize_value(v) for k, v in rec.model_dump().items()} for rec in records]
        frame = pl.DataFrame(rows)

        # A column belonging to an optional dimension is dropped when that dimension
        # is absent — see OPTIONAL_DIMENSION_COLUMNS. The alternative is every corpus
        # carrying an all-null FK to a table it does not have.
        optional = OPTIONAL_DIMENSION_COLUMNS.get(table_name)
        if optional is not None:
            column, dimension = optional
            if not getattr(dataset, dimension, None) and column in frame.columns:
                frame = frame.drop(column)

        result[table_name] = frame

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
    dataset: Corpus,
    output_dir: Path,
    entropy_records: list[dict] | None = None,
    identity: CorpusIdentity | None = None,
    run_facts: dict | None = None,
    fmt: ExportFormat = "csv",
    truth_dir: Path | None = None,
) -> None:
    """Export dataset to files with manifest and optional entropy map.

    Args:
        dataset: The finance dataset to export.
        output_dir: Directory to write files into.
        entropy_records: Optional list of entropy injection records for the map.
        identity: The corpus identity to stamp into the manifest and entropy map.
        run_facts: Non-identity facts about this run (injection count, source name).
        fmt: Export format — "csv", "parquet", or "both".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframes = dataset_to_dataframes(dataset)

    file_manifest: list[dict[str, Any]] = []
    for table_name, df in dataframes.items():
        file_manifest.extend(_write_table(df, output_dir, table_name, fmt))

    # The manifest is the corpus's own packing list and stays with the
    # data. The entropy map names every injection coordinate — answer
    # key — so with a truth_dir it lands there, out of reach of
    # anything that mounts or serves the corpus.
    _write_manifest(output_dir, file_manifest, identity, run_facts)
    map_dir = truth_dir or output_dir
    map_dir.mkdir(parents=True, exist_ok=True)
    _write_entropy_map(map_dir, entropy_records, identity)


def export_dataframes(
    dataframes: dict[str, pl.DataFrame],
    output_dir: Path,
    entropy_records: list[dict] | None = None,
    identity: CorpusIdentity | None = None,
    run_facts: dict | None = None,
    fmt: ExportFormat = "csv",
    truth_dir: Path | None = None,
) -> None:
    """Export pre-built DataFrames (after injection) to files + manifest.

    This is the post-injection export path: the scenario has already
    mutated the DataFrames and recorded entropy injections.

    Args:
        dataframes: Table name → DataFrame mapping.
        output_dir: Directory to write files into.
        entropy_records: Optional list of entropy injection records.
        identity: The corpus identity to stamp into the manifest and entropy map.
        run_facts: Non-identity facts about this run (injection count, source name).
        fmt: Export format — "csv", "parquet", or "both".
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    file_manifest: list[dict[str, Any]] = []
    for table_name, df in dataframes.items():
        file_manifest.extend(_write_table(df, output_dir, table_name, fmt))

    # The manifest is the corpus's own packing list and stays with the
    # data. The entropy map names every injection coordinate — answer
    # key — so with a truth_dir it lands there, out of reach of
    # anything that mounts or serves the corpus.
    _write_manifest(output_dir, file_manifest, identity, run_facts)
    map_dir = truth_dir or output_dir
    map_dir.mkdir(parents=True, exist_ok=True)
    _write_entropy_map(map_dir, entropy_records, identity)


def _write_manifest(
    output_dir: Path,
    file_manifest: list[dict[str, Any]],
    identity: CorpusIdentity | None,
    run_facts: dict | None,
) -> None:
    """Write manifest.yaml.

    The run parameters live in ``corpus`` and nowhere else. They used to be repeated
    under ``parameters`` beside a hand-written generator version, which is how a file
    that claims to be an answer key starts disagreeing with itself. ``run`` carries
    what is *not* a parameter — counts and per-source facts that follow from the run.
    """
    manifest: dict[str, Any] = {"generated_at": datetime.now().isoformat()}
    if identity is not None:
        manifest["corpus"] = identity.as_dict()
    manifest["run"] = run_facts or {}
    manifest["files"] = file_manifest
    with open(output_dir / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)


def _write_entropy_map(
    output_dir: Path,
    entropy_records: list[dict] | None,
    identity: CorpusIdentity | None = None,
) -> None:
    """Write entropy_map.yaml."""
    entropy_data: dict[str, Any] = {}
    if identity is not None:
        entropy_data["corpus"] = identity.as_dict()
    entropy_data["injections"] = entropy_records or []
    entropy_data["total_injections"] = len(entropy_records) if entropy_records else 0
    with open(output_dir / "entropy_map.yaml", "w") as f:
        yaml.dump(entropy_data, f, default_flow_style=False, sort_keys=False)
