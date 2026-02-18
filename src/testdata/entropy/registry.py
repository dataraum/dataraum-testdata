"""Entropy injection registry — tracks all injections for ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EntropyInjection:
    """Record of a single entropy injection applied to the dataset."""

    injection_id: str
    target_file: str              # e.g., "journal_lines.csv"
    target_column: str
    target_rows: list[int]        # Row indices affected
    layer: str                    # structural | semantic | value | computational
    dimension: str
    sub_dimension: str
    detector_id: str              # Which dataraum detector should catch this
    injection_type: str           # e.g., "corrupt_type", "introduce_nulls"
    parameters: dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"      # low | medium | high | critical


class InjectionRegistry:
    """Accumulates injection records and exports them as YAML ground truth."""

    def __init__(self) -> None:
        self._injections: list[EntropyInjection] = []
        self._counter: int = 0

    def record(self, injection: EntropyInjection) -> None:
        self._injections.append(injection)

    def next_id(self, prefix: str = "INJ") -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:04d}"

    @property
    def injections(self) -> list[EntropyInjection]:
        return list(self._injections)

    def export_dicts(self) -> list[dict[str, Any]]:
        """Return all injections as plain dicts (for YAML export)."""
        return [asdict(inj) for inj in self._injections]

    def export_yaml(self, path: Path) -> None:
        data = {
            "injections": self.export_dicts(),
            "total_injections": len(self._injections),
            "summary": self.summary(),
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def summary(self) -> dict[str, dict[str, int]]:
        """Count injections by layer and detector."""
        by_layer: dict[str, int] = {}
        by_detector: dict[str, int] = {}
        for inj in self._injections:
            by_layer[inj.layer] = by_layer.get(inj.layer, 0) + 1
            by_detector[inj.detector_id] = by_detector.get(inj.detector_id, 0) + 1
        return {"by_layer": by_layer, "by_detector": by_detector}

    def remap_tables(self, mapping: dict[str, str]) -> None:
        """Update ``target_file`` for injections whose table was renamed by schema transforms."""
        for inj in self._injections:
            old_table = inj.target_file.removesuffix(".csv")
            if old_table in mapping:
                inj.target_file = f"{mapping[old_table]}.csv"

    def __len__(self) -> int:
        return len(self._injections)
