"""Tests for scenario orchestration and end-to-end generation."""

import tempfile
from pathlib import Path

import yaml

from testdata.families import default_tables
from testdata.scenarios.month_end_close import run_scenario


def _truth(corpus: Path) -> Path:
    """The answer key's sibling dir (run_scenario's --truth-output default)."""
    return corpus.parent / (corpus.name + "-truth")


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
        assert (_truth(output) / "entropy_map.yaml").exists()
        assert (output / "journal_lines.csv").exists()
        assert (output / "invoices.csv").exists()
        assert (output / "bank_transactions.csv").exists()
        assert (output / "payments.csv").exists()

        # Verify manifest structure
        with open(output / "manifest.yaml") as f:
            manifest = yaml.safe_load(f)
        assert manifest["corpus"]["generator"] == "dataraum-testdata"
        assert len(manifest["files"]) == len(default_tables())

        # Verify entropy map has injections
        with open(_truth(output) / "entropy_map.yaml") as f:
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
    consumer_hint: temporal_behavior
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

        with open(_truth(output) / "entropy_map.yaml") as f:
            emap = yaml.safe_load(f)
        stockflow = [
            inj["parameters"] for inj in emap["injections"] if inj["injection_type"] == "inject_stock_flow_probes"
        ]
        backed = [p for p in stockflow if p["backed"]]
        assert backed and all(p["events_table"] == "probe_events" for p in backed)
        assert {p["reconciles"] for p in backed} == {True, False}  # both calibration strata
        for p in backed:
            assert p["events_column"] in events.columns


def test_relationship_probe_strategy_dispatch(tmp_path: Path):
    """A strategy targeting the child probe table reaches inject_relationship_pairs
    end-to-end: skeleton grains generated, parent filled via the dataframes
    pass-through, pair ground truth exported in entropy_map.yaml."""
    from testdata.scenarios.runner import run_scenario as run_any

    strategy = {
        "name": "relationship-cal-test",
        "level": "high",
        "description": "relationship_pairs dispatch test",
        "injections": [
            {
                "injector": "inject_relationship_pairs",
                "table": "ref_activity",
                "params": {"seed": 7, "severity": "high"},
            }
        ],
    }
    strategy_file = tmp_path / "relationship_cal_test.yaml"
    with open(strategy_file, "w") as f:
        yaml.dump(strategy, f)

    output = tmp_path / "out"
    result = run_any("month-end-close", strategy_file=strategy_file, seed=7, months=3, output_dir=output)

    dfs = result["dataframes"]
    assert len(dfs["ref_entities"]) == 300 and len(dfs["ref_activity"]) == 1200
    records = [inj for inj in result["registry"].injections]
    assert records and all(inj.defect == "relationships" for inj in records)
    # Every sampled pair landed as columns on BOTH probe tables.
    for inj in records:
        assert inj.parameters["parent_column"] in dfs["ref_entities"].columns
        assert inj.parameters["child_column"] in dfs["ref_activity"].columns
    assert {inj.parameters["stratum"] for inj in records} == {
        "genuine_clean",
        "genuine_broken",
        "spurious_overlap",
    }

    # The exported ground truth carries the pair-level labels the rig reads.
    assert (output / "ref_entities.csv").exists() and (output / "ref_activity.csv").exists()
    with open(_truth(output) / "entropy_map.yaml") as f:
        emap = yaml.safe_load(f)
    pairs = [i for i in emap["injections"] if i["injection_type"] == "inject_relationship_pairs"]
    assert len(pairs) == len(records)
    assert all(i["parameters"]["label"] in ("genuine", "spurious") for i in pairs)


def test_baseline_strategies_skip_relationship_probes():
    """Without a relationship stanza the probe tables do not exist at all."""
    result = run_scenario(strategy_name="medium", seed=42, months=3)
    assert "ref_entities" not in result["dataframes"]
    assert "ref_activity" not in result["dataframes"]


_OVERRIDE_STRATEGY = """\
name: override-multi-record-test
level: high
description: consumer_hint must label EVERY record of a multi-record injection
injections:
  - injector: introduce_nulls
    table: journal_lines
    consumer_hint: null_ratio
    params:
      col: cost_center
      ratio: 0.2
  - injector: inject_stock_flow_probes
    table: measure_probes
    consumer_hint: custom_temporal_check
    params:
      seed: 7
      n_columns: [8, 8]
"""


def test_consumer_hint_labels_every_record_of_the_injection():
    """The strategy-YAML override applies to ALL records the injection produced —
    the old [-1] patch labelled only the last probe column (lane F2 finding)."""
    from testdata.scenarios.runner import run_scenario as run_any

    with tempfile.TemporaryDirectory() as tmpdir:
        strategy_path = Path(tmpdir) / "override.yaml"
        strategy_path.write_text(_OVERRIDE_STRATEGY)
        result = run_any("month-end-close", strategy_file=strategy_path, seed=7, months=3)

    by_type: dict[str, set[str]] = {}
    for inj in result["registry"].injections:
        by_type.setdefault(inj.injection_type, set()).add(inj.consumer_hint)
    assert by_type["inject_stock_flow_probes"] == {"custom_temporal_check"}
    assert len([i for i in result["registry"].injections if i.injection_type == "inject_stock_flow_probes"]) == 8
    # The earlier single-record injection keeps its own label untouched.
    assert by_type["introduce_nulls"] == {"null_ratio"}
