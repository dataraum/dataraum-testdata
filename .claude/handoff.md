# Testdata Handoff

Changes in dataraum-testdata that need attention in other repos (dataraum-eval
calibration, dataraum-context engine). Newest first.

## 2026-06-09: generative injection families — the family framework (DAT-450)

ADR-0009: "fixed fixtures are dead." A strategy now declares a **family** + a
recorded **seed**, and a parameterized generator SAMPLES the corruption — so
different seeds vary the surface (tokens, values, rates) while the recorded seed
reproduces exactly, and a detector that memorized one fixed token proves nothing.

New module **`entropy/families.py`** — the reusable framework. First-wave family
implemented: **null_tokens** (feeds the `null_semantics` adjudication). It samples
two ground-truth-LABELLED classes:

- **markers** (`is-null`) — sentinel shapes (status words / error codes /
  punctuation runs), a small set repeated → they cluster in the quarantine.
  `vocab_coverage` controls how many are in the curated null vocabulary vs novel.
- **decoys** (`is-value`) — genuine amounts that merely fail a numeric cast
  (locale numbers, currency, unit-suffixed, annotated), distinct per row → smear.

**Live-run constraint (DAT-450, learned from a real pipeline run):** marker + decoy
together are the column's cast-failure rate, and the typing phase only infers a
numeric type — and thus quarantines the tokens for `null_semantics` to adjudicate —
when `parse_success ≥ min_confidence` (0.85, `phases/typing.yaml`). At ~16%
corruption `journal_lines.debit` fell to **VARCHAR**, never quarantined, and the
detector was correctly skipped. So the family's combined ratio is capped at ≤0.10
(`marker_ratio (0.05,0.075)` + `decoy_ratio (0.015,0.025)`) — high enough to create
a clear conflict, low enough to keep the column numeric. The isolated rig can't see
this (it assumes the column quarantines); only the live run surfaced it.

New injector **`inject_null_token_family(col, seed, …)`** (the generative successor
to `inject_null_tokens`): its own seed-derived RNG (order-independent, reproducible),
and it records the labels in `entropy_map` `parameters` — `markers`,
`in_vocab_markers`, `decoys`, `marker_rows`, `decoy_rows`, `seed`, `vocab_coverage`,
`decoy_style`. That label set is what the eval calibration rig scores witnesses
against.

**eval:** `strategies/detection-null-v1.yaml` migrated to `inject_null_token_family`
(seeds 20260609 / 20260824); `scripts/calibrate_reliabilities.py` +
`calibration/reliability_rig.py` consume the family to measure witness reliabilities.

**Adding a family:** add a `sample_<family>` + its grammar to `families.py` and an
`inject_<family>` injector recording labelled ground truth. The other first-wave
families (garbage names DAT-446, mixed units DAT-428, stock/flow DAT-445,
heterogeneity DAT-473) are framework-ready but deferred until their witnesses land.

## Notes

`uv run` in this repo currently rebuilds polars from source (local rust-toolchain
breakage) — run tests via the dataraum-eval venv, which has testdata installed
editable: `.venv/bin/python -m pytest vendor/dataraum-testdata/tests/...`.
