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
