# dataraum-testdata

Synthetic operating-model data with an **answer key** — a corpus whose metrics, structure
and injected defects are all known, so anything computed over it can be graded rather than
believed.

[`docs/operating-model.md`](docs/operating-model.md) is the plan: the six performance
dimensions, the family framework, the oracle contract, and the order of work.

## Architecture

The generator uses an **event-driven cascade model** where business events produce numerically consistent data across all tables:

- **Operating chain**: Customer → sales order → order line (units × price) → AR invoice → receipt
- **Revenue cycle**: Order lines → revenue and COGS journal entries → cash receipts → bank transactions
- **Expenditure cycle**: Purchase invoices → AP journal entries → vendor payments → bank transactions
- **Operating events**: Monthly payroll, rent, depreciation, insurance, misc expenses
- **Trial balance**: Derived from actual cumulative GL entries (not approximated)

This produces **closed-loop accounting** — GL entries, invoices, payments, bank transactions, and trial balance are all numerically consistent and traceable back to the originating business event. Because revenue and cost of sale both derive from the order line (`units × unit_price`, `units × standard_cost`), contribution margin per customer and per product group is exact, not estimated.

## Quick Start

```bash
# Generate clean baseline data
testdata generate --scenario month-end-close --strategy clean --output ./output/clean --seed 42

# Generate data with realistic entropy injections
testdata generate --scenario month-end-close --strategy medium --output ./output/medium --seed 42

# List available scenarios
testdata list-scenarios

# Describe a scenario
testdata describe --scenario month-end-close
```

## Strategies

| Strategy | Description |
|----------|-------------|
| `clean`  | No injections — baseline data |
| `low`    | Subtle issues (2-5% rates) |
| `medium` | Realistic problems (~11 injection types) |
| `high`   | Severe quality issues across all layers |

## Normalization Levels

The `normalization` setting in the scenario YAML controls table structure:

| Level | Tables | Analogue |
|-------|--------|----------|
| `full` | 8 (default) | ERP schema export |
| `partial` | 6 | Reporting views — merges parent-child pairs |
| `flat` | 5 | Analyst spreadsheet — inlines lookup tables |

Set via `generator.normalization` in `config/scenarios/month_end_close.yaml`.

## Output

Each generation produces:
- **CSV files** — one per table (varies by normalization level)
- **manifest.yaml** — file list, row counts, generation parameters
- **entropy_map.yaml** — defect ground truth: every injection with its target rows, layer, defect class and severity
- **ground_truth.yaml** — known-correct financial metrics (see [Ground Truth](#ground-truth))
- **metadata_truth.yaml** — structural ground truth (see [Metadata Truth](#metadata-truth))

### Defect labels are consumer-agnostic

An injection records **what was broken**, never which detector should catch it:
`layer` (structural / semantic / value / computational), `defect` (the class, e.g.
`referential_integrity`), `defect_detail` (the form it took) and `injection_type` (the
injector that produced it). Mapping a defect onto the machinery meant to catch it is the
consumer's job. A strategy may set `consumer_hint:` on an injection to carry its own
label through; the generator never reads it.

## Metadata Truth

`metadata_truth.yaml` is the **structural** ground truth: the FK topology, table and
column roles, stock vs flow, metric additivity, cycles and the conformed-dimension
matrix. The generator knows every one of these answers from the models and the generator
design, so anything that recovers structure from the data can be graded against it, the
way `entropy_map.yaml` grades defect detection. Authored in
`src/testdata/metadata_truth.py`; **generated — do not hand-edit**.

| Section | Keyed by | Content |
|---|---|---|
| `metric_additivity` | metric/measure name | drill additivity verdict (`categorical_additive` / `time_additive` + `reason`) with a `determinism` tag (`function_symmetry` = assertable, `label_dependent` = diagnostic) |
| `stock_flow` | `table.column` | `additive` (per-period flow) vs `point_in_time` (stock/level) |
| `reconciles_structurally` | `table.column` | measures that reconcile against a finer event fact |
| `relationships` | — | the true FK topology (`{from, to}` qualified names) |
| `table_roles` | — | `facts` / `dimensions` / `ambiguous` table lists |
| `semantic_roles` | role | `measure` / `timestamp` column lists |
| `business_concepts` | `table.column` | required measure→concept bindings |
| `cycles` | — | business cycles the corpus supports (`canonical_type`, `key_tables`, `required`) |

**Remap-safety.** The truth is authored at canonical (`full` / snake_case) names and
rewritten to match each run's exported schema: table names follow the normalization
`table_mapping` (like `InjectionRegistry.remap_tables`), column names follow the
`column_style`. A cross-table FK that a merge collapses into one table is dropped (no
longer discoverable); a genuine self-FK (`chart_of_accounts.parent_id`) is kept. As with
`entropy_map.yaml`, column renames introduced by normalization *merges* (e.g.
`payments.amount → invoice_data.payment_amount`) are not reflected — merged columns keep
canonical names. Multi-source runs write one canonical top-level file, mirroring the
top-level `entropy_map.yaml`.

## Finance Vertical Tables

| Table | ~Rows | Description |
|-------|-------|-------------|
| chart_of_accounts | 60 | Account hierarchy (60 accounts, 5 types) |
| journal_entries | 12K | General ledger entries (event-driven) |
| journal_lines | 25K | Debit/credit lines (balanced per entry) |
| invoices | 3K | Vendor/purchase invoices |
| payments | 2.5K | Invoice payments (paid + partial) |
| bank_transactions | 5.5K | Bank statement (derived from cash events) |
| fx_rates | 470 | Weekly exchange rates (8 currency pairs) |
| trial_balance | 324 | Monthly cumulative balances (27 accounts × 12 months) |

## Development

```bash
uv sync
uv run pytest tests/ -v
```

## Schema Variants

Beyond normalization levels, the library provides additional transforms:

**Column naming styles** (`apply_column_style`):
- `snake_case` — default (identity)
- `camelCase` — JavaScript/API style
- `PascalCase` — C#/.NET style
- `legacy` — abbreviated uppercase (ERP-style: `DR_AMT`, `ACCT_NO`, `CC`)

**Key strategies** (`apply_key_strategy`):
- `surrogate` — default (identity, e.g. `JE-0001`)
- `natural` — prefix-based (e.g. `JE-00001`)
- `uuid` — deterministic UUIDs (seeded)
- `composite` — table-prefixed (`journal_entries::JE-0001`)

**Pivots** (standalone functions):
- `pivot_trial_balance_wide` — accounts as rows, periods as columns
- `pivot_journal_lines_wide` — single `amount` + `side` column instead of separate debit/credit

## Scenarios

| Scenario | Sources | Description |
|----------|---------|-------------|
| `month-end-close` | 1 | 12-month fiscal year, 8 tables, standard ERP export |
| `erp-migration` | 1 | 6-month migration window, high entropy, partial normalization |
| `multi-system-recon` | 3 | Same events exported through ERP (legacy), banking (PascalCase), AP system (camelCase) |

### Multi-Source Scenarios

Multi-source scenarios split tables across separate "data sources" with different schema conventions. Each source gets its own subdirectory, manifest, and column naming.

```bash
testdata generate --scenario multi-system-recon --strategy clean --output ./output/multi --seed 42
```

Output:
```
output/
├── erp_export/          # chart_of_accounts, journal_*, trial_balance (legacy columns)
├── banking_feed/        # bank_transactions, fx_rates (PascalCase columns)
├── ap_system/           # invoices, payments (camelCase columns)
├── sources.yaml         # source index
├── entropy_map.yaml     # injection ground truth
└── ground_truth.yaml    # financial ground truth
```

Define sources in scenario YAML:
```yaml
sources:
  erp_export:
    tables: [chart_of_accounts, journal_entries, journal_lines, trial_balance]
    column_style: legacy
    key_strategy: surrogate
    format: csv
```

## Ground Truth

Each scenario run computes `ground_truth.yaml` with known-correct financial metrics:
- **Annual**: revenue, expenses, gross profit, AR/AP/cash balances, DSO, DPO, FCF
- **Monthly**: same metrics per period plus revenue growth MoM
- **Invariants**: journal balanced, TB balanced, invoice-payment matched, bank reconciliation rate
- **Injection impact**: estimated metric deviations from known injection parameters

## Backlog

- Format profiles (DATEV, SAP, Salesforce, HubSpot) via YAML config + OpenAPI specs
- Additional verticals (supply chain, sales/CRM)
