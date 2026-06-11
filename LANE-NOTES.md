# Lane F2 — relationship_pairs family (DAT-408 / DAT-442)

Generative RELATIONSHIP family: labelled column pairs across
genuine_clean / genuine_broken / spurious_overlap strata, the corpus for
measuring the relationship_discovery witnesses' reliabilities (DAT-450 rig).

## Shared-infrastructure changes (integrator: review on merge)

Other lanes (F1/F3) append families to `families.py` / `injectors.py`; this lane
appended at the END of both, but also touched these shared spots:

1. **`src/testdata/scenarios/runner.py` — `_apply_injection`**: now puts the live
   `dataframes` dict into `all_kwargs`. The existing signature filter means only
   injectors that DECLARE a `dataframes` parameter receive it — all existing
   injectors are untouched. Needed because the relationship family fills a
   PARENT probe table and a CHILD probe table from one injector call (the
   single-table dispatch can't represent a pair corpus).
2. **`src/testdata/scenarios/runner.py` — `run_scenario`**: skeleton-generation
   gate for the relationship probe grains (`ref_entities` 300 rows /
   `ref_activity` 1200 rows), mirroring the `measure_probes` `probe_series`
   precedent. Active only when a strategy injects into `ref_activity`.
3. **`src/testdata/canonical/finance/models.py`**: `RefEntity` / `RefActivity`
   models + two `FinanceDataset` fields (default `[]`, omitted from export when
   empty — same as `measure_probes`).
4. **`src/testdata/export.py`**: `TABLE_NAMES` entries for both probe tables.
5. **`src/testdata/schema_transforms.py`**: both probe tables added to the
   "single"-normalization drop list (next to `measure_probes`).
6. **Import blocks** of `injectors.py` / `tests/test_families.py` gained the new
   family symbols — the only mid-file edits in files F1/F3 also append to; merge
   is mechanical (union of import names).

## Format-gate caveat (deliberate deviation, flagged not hidden)

`main` is NOT ruff-format-clean under the locked ruff 0.15.1: a repo-wide
`uv run python -m ruff format .` rewrites ~17 files in PRE-EXISTING regions
(the repo's hooks only ever formatted files as they were edited). Committing
that rewrite would destroy the mechanical merge for F1/F3, so this lane
committed semantic changes only:

- `ruff check`, `mypy`, `pytest` — all green (see commit).
- All NEW code in this lane is format-stable (`ruff format --diff` reports no
  changes inside any lane-F2 region).
- Recommendation: one repo-wide `ruff format .` commit AFTER all lanes merge.

## Strategy-contract caveats for the eval side

- Do **not** set `detector_id:` on the `inject_relationship_pairs` stanza: the
  runner's override patches only the LAST registry record, and this injector
  records one entry PER PAIR. The injector's default (`relationship_discovery`)
  is already correct. (Same latent issue exists for `inject_stock_flow_probes`
  and `obscure_column_names` — pre-existing, not addressed here.)
- The probe tables exist only under full/partial/flat normalization; the
  "single" mega-table fold drops them. month-end-close defaults to `full`.
- Keep the family in its OWN strategy (e.g. detection-relationship-cal-v1), not
  mixed into detection-v1 — the DAT-405 lesson: extra confirmed-FK candidates in
  a mixed run can steal relationship_entropy's recall target
  (payments.invoice_id). Probe pool values can never equal a canonical id:
  prefixes are blocklisted against the dash-4-digit namespaces (V/RE/RA plus
  JE/INV/TXN/CC for safety), and every other canonical id uses 6-7 digits while
  probe values use 4 — exact-match overlap is structurally zero, so no
  accidental candidates against canonical tables either way.
