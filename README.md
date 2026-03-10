# dataraum-testdata

Synthetic test data generator with **known entropy injections** for calibrating [dataraum-context](https://github.com/...) entropy detectors.

## Architecture

The generator uses an **event-driven cascade model** where business events produce numerically consistent data across all tables:

- **Revenue cycle**: Sales → AR journal entries → cash receipts → bank transactions
- **Expenditure cycle**: Purchase invoices → AP journal entries → vendor payments → bank transactions
- **Operating events**: Monthly payroll, rent, depreciation, insurance, misc expenses
- **Trial balance**: Derived from actual cumulative GL entries (not approximated)

This produces **closed-loop accounting** — GL entries, invoices, payments, bank transactions, and trial balance are all numerically consistent and traceable back to the originating business event.

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
- **entropy_map.yaml** — ground truth: every injection with target rows, detector ID, layer, severity

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
