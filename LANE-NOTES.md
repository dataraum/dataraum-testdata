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
