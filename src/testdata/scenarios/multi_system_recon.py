"""Multi-system reconciliation scenario — thin wrapper around shared runner.

All configuration lives in ``config/scenarios/multi_system_recon.yaml``.
This module provides a convenience ``run_scenario()`` for direct use in
tests and scripts.
"""

from __future__ import annotations

from pathlib import Path

from testdata.export import ExportFormat
from testdata.scenarios.runner import run_scenario as _run

SCENARIO_NAME = "multi-system-recon"


def run_scenario(
    strategy_name: str | None = None,
    seed: int | None = None,
    months: int | None = None,
    output_dir: Path | None = None,
    fmt: ExportFormat = "csv",
) -> dict:
    """Run the multi-system-recon scenario. See ``runner.run_scenario`` for details."""
    return _run(
        SCENARIO_NAME,
        strategy_name=strategy_name,
        seed=seed,
        months=months,
        output_dir=output_dir,
        fmt=fmt,
    )
