# dataraum-testdata

Synthetic test data generator with **known entropy injections** for calibrating [dataraum-context](https://github.com/...) entropy detectors.

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

## Output

Each generation produces:
- **CSV files** — one per table (8 tables for finance)
- **manifest.yaml** — file list, row counts, generation parameters
- **entropy_map.yaml** — ground truth: every injection with target rows, detector ID, layer, severity

## Finance Vertical Tables

| Table | ~Rows | Description |
|-------|-------|-------------|
| chart_of_accounts | 60 | Account hierarchy |
| journal_entries | 5K | General ledger entries |
| journal_lines | 15K | Debit/credit lines |
| invoices | 3K | Vendor invoices |
| payments | 2.5K | Invoice payments |
| bank_transactions | 8K | Bank statement |
| fx_rates | 500 | Exchange rates |
| trial_balance | 500 | Monthly trial balance |

## Development

```bash
uv sync
uv run pytest tests/ -v
```
