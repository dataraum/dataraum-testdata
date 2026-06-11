# Lane F1 — events-backed stock/flow family (DAT-450/DAT-491)

Branch `lane/f1-events-stockflow`. Data-side prerequisite for re-measuring the
temporal_behavior `structural_reconciliation` reliability (the 0.85 in
reliabilities.yaml is an uncalibrated placeholder because "probe tables have no
events").

## Shared-infrastructure changes (integrator: reconcile with other lanes)

1. **`scenarios/runner.py` — `_apply_injection` now passes `dataframes`** (the run's
   full table dict) to injectors that declare the parameter. The existing
   signature-filter makes this invisible to every other injector. Needed so
   `inject_stock_flow_probes` can emit the companion `probe_events` table; any lane
   touching the injector-dispatch contract collides here.
2. **`schema_transforms.py` — `probe_events` added to the `single`-normalization
   drop list** (one line, same treatment as `measure_probes`).
3. **Format drift on `main`**: the locked ruff (0.15.1 via uv.lock) reformats 17
   files on `main` — `ruff format .` (the documented gate) dirties files no lane
   touched. This lane committed formatter output only for the six files it edits and
   restored the rest. Recommend one repo-wide `ruff format` commit on main before/at
   integration.

## Witness visibility (answered from engine code — NO engine change needed)

The structural witness reads `MeasureAggregationLineage` rows
(`entropy/detectors/loaders.py:686`), written by `discover_aggregation_lineage`
(`analysis/lineage/processor.py:175`) at the begin_session `aggregation_lineage`
phase. Discovery is pure arithmetic over the slice substrate — **no relationship, no
name convention, no ontology binding** between events and measures. Preconditions
the data satisfies:

- **Shared slice dimension**: `series_id` exists with identical values in
  `measure_probes` and `probe_events` (processor pairs facts only on a dimension
  with SliceDefinitions on ≥2 tables, processor.py:221).
- **Time axis per table**: `measure_probes.period` ("YYYY-MM") types to DATE via the
  `period_yyyy_mm` pattern (dataraum-config phases/typing.yaml:59) — same shape as
  the proven trial_balance.period; `probe_events.event_date` is a plain ISO date.
- **Direction gate** (processor.py:290): events strictly finer-grained — enforced by
  `events_per_cell` lower bound ≥ 2 (param guard in the family).
- **Convention cap** (processor.py:48, MAX_CONVENTION_COLUMNS=8): sampler caps the
  backed set at 8 so every backed label is measurable.
- **dispose() gates** (analysis/lineage/reconcile.py): MIN_PERIODS=4 (run ≥4 months),
  FIRE_RESIDUAL_MAX=0.5, MIN_ENTITIES_FIRED=2, AGREEMENT_MIN=0.8. `break_magnitude`
  is calibrated in these units: per-entity R_stock ≈ break_magnitude; the default
  range (0.3, 1.2) straddles the 0.5 abstain gate.

**Residual e2e dependency (not an engine change, an LLM judgment):** the slicing
agent must recommend `series_id` as a slice dimension on BOTH probe tables
(plain-column recommendations do NOT propagate — only enriched `fk__col` dims do,
slicing_phase.py:343). The prompt anti-patterns "high-cardinality identifiers", but
15 values + "shared dimension partitioning most tables" (criterion 5) is the same
case the canonical corpus passes with `account_id`. If a live run shows the agent
skipping `series_id`, the data-side mitigation is renaming the probe key to
something more dimension-flavoured (e.g. `business_unit`) — a skeleton change shared
with the existing corpus, so it was NOT done unilaterally here.

**Witness fires at begin_session `session_detect` only** (lineage rows are
exact-run, loaders.py:702): the rig must read a session run, the eval runner already
drives one.

## Scope note

Flows stay unbacked (per mission). Consequence: this family measures the witness's
*stock*-confirmation reliability + abstain behaviour (broken/unbacked); its
flow-confirmation (`per_period`) reliability needs a backed-flow stratum — a small
follow-up (events whose per-period sums equal the flow value itself).

## Ground truth contract (entropy_map.yaml parameters, per probe column)

`backed: bool`, `reconciles: bool` (backed ∧ unbroken), `break_ratio: float`
(fraction of series broken, 0.0 when clean), `break_magnitude: float` (≈ injected
per-entity R_stock, 0.0 when clean), `events_table: "probe_events" | null`,
`events_column: "<name>_delta" | null`. Existing fields (`true_behavior`,
`ambiguous`, …) unchanged. Recorded seeds reproduce the pre-change surface
bit-exactly (verified against main for seeds 42/7/20260610 incl. the
detection-stockflow-cal-v1 shape).

## Strategy stanza (eval side)

```yaml
injections:
  - injector: inject_stock_flow_probes
    table: measure_probes
    detector_id: temporal_behavior
    params:
      seed: 20260611
      n_columns: [16, 24]
      backed_fraction: [0.6, 0.9]   # fraction of stocks events-backed
      broken_fraction: [0.3, 0.5]   # fraction of backed columns broken
      # optional overrides (defaults shown):
      # break_ratio: [0.5, 1.0]
      # break_magnitude: [0.3, 1.2]
      # events_per_cell: [2, 6]
```


---

# Lane F3 — formula_divergence family (DAT-442, ADR-0009 derived-value)

Branch: `lane/f3-formula-divergence`. Corpus for calibrating the derived_value
witnesses (`formula_discovery` 0.9 / `llm_hypothesis` 0.6 placeholders in
`reliabilities.yaml`).

## Shared-infrastructure touches (integrator: merge order / conflicts)

Per lane protocol, family + injector code is APPENDED at the END of the shared
files; the only mid-file edits are imports and four small probe-table hooks.

| File | Change | Conflict risk with F1/F2 |
|---|---|---|
| `src/testdata/entropy/families.py` | formula_divergence family appended at EOF | low — append-only; F1/F2 also append at EOF → trivial both-sides merge |
| `src/testdata/entropy/injectors.py` | `inject_formula_divergence` appended at EOF; 4 names added to the `from .families import (...)` block (alphabetical) | import block is the one mid-file hotspot — merge is mechanical (union the names) |
| `src/testdata/canonical/finance/models.py` | `FormulaProbe` model after `MeasureProbe`; `FinanceDataset.formula_probes: list[FormulaProbe] = []` | low |
| `src/testdata/canonical/finance/generators.py` | `_generate_formula_probes` after `_generate_measure_probes`; `formula_probe_rows: int = 0` kwarg + assembly in `generate_finance_dataset` | low |
| `src/testdata/export.py` | `"formula_probes"` entry at end of `TABLE_NAMES` | low |
| `src/testdata/scenarios/runner.py` | grain gating: `formula_probe_rows = 300 if any(s.table == "formula_probes" ...)` next to the existing `measure_probes` gate | low |
| `tests/test_families.py` | imports + formula_divergence tests appended at EOF | same import-block note |
| `tests/test_generators.py` | one gating test appended at EOF | low |

## Pre-existing format drift on main (integrator heads-up)

`main` is NOT format-clean under the locked ruff 0.15.1: `ruff format --check .`
on a pristine main checkout reports **16 files would be reformatted**. The
mandated gate (`ruff format .`) therefore sweeps whole files. This branch
commits the sweep ONLY for the 8 files it actually edits (the big hunks in
`generators.py` / `injectors.py` are that sweep, not logic); the other 9
drifted files were restored to main state and left uncommitted. Lanes running
the same gate produce IDENTICAL sweep hunks → merges stay mechanical. Consider
landing one repo-wide format commit on main after the wave.

## Design notes for the integrator / eval rig

- Probe table `formula_probes` (grain `probe_id`, 300 rows) mirrors the
  `measure_probes` opt-in gating — generated only when a strategy injects into it.
- Registry labels per TARGET column: `named_formula` (canonical identity, engine
  spelling: lowercased, commutative operands sorted), `actual_formula`,
  `named_op`/`actual_op`, `factor`, `divergence_mode` (agree|wholesale|partial),
  `divergence_ratio`, `discoverable`, `source_columns`. `target_rows` = the rows
  violating the named formula (empty for agree, all for wholesale).
- `discoverable: false` marks the scaled stress kind (values = named formula × a
  sampled factor) — deliberately OUTSIDE the engine's binary-op discovery space;
  the rig must stratify on it.
- Engine dedup caveat the rig must expect: the discovery sweep dedups algebraic
  equivalences per column TRIPLE preferring sum/product — for a difference-named
  target in the agree stratum the discovered annotation can land on the MINUEND
  source column (as a sum), and the target's data witness then comes from the
  hypothesis-grading leg instead. Labels are per target column, so scoring is
  unaffected, but "discovery found it on the target" is not the right assert.
