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
- **Stock subledger**: Order line issues stock → replenishment receives it on a vendor bill → the bill settles like any other payable
- **Expenditure cycle**: Purchase invoices → AP journal entries → vendor payments → bank transactions
- **Operating events**: Monthly payroll, rent, depreciation, insurance, misc expenses
- **Trial balance**: Derived from actual cumulative GL entries (not approximated)

This produces **closed-loop accounting** — GL entries, invoices, payments, bank transactions, and trial balance are all numerically consistent and traceable back to the originating business event. Because revenue and cost of sale both derive from the order line (`units × unit_price`, `units × standard_cost`), contribution margin per customer and per product group is exact, not estimated.

The stock subledger closes the same loop on the asset side: `opening + receipts − issues ± adjustments = closing` holds per product, location and period, and Σ position value equals the GL inventory balance exactly. That is what makes **CCC = DIO + DSO − DPO** an answer key rather than a plausible number.

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
| `full` | 17 (default) | ERP schema export |
| `partial` | 14 | Reporting views — merges three parent-child pairs |
| `flat` | 13 | Analyst spreadsheet — inlines lookup tables |
| `single` | 1 | One mega-table |

Counts follow the family registry (`default_tables()`), so they grow with the corpus.

Set via `generator.normalization` in `config/scenarios/month_end_close.yaml`.

## Output

Each generation produces:
- **CSV files** — one per table (varies by normalization level)
- **manifest.yaml** — corpus identity, file list, row counts
- **entropy_map.yaml** — defect ground truth: every injection with its target rows, layer, defect class and severity
- **ground_truth.yaml** — known-correct financial metrics (see [Ground Truth](#ground-truth))
- **metadata_truth.yaml** — structural ground truth (see [Metadata Truth](#metadata-truth))

### Corpus identity

`output/` is gitignored and stays that way. A corpus is not an artifact to preserve; it is
a **function of its parameters**, and every output file carries them as a `corpus:` block:

```yaml
corpus:
  id: 524369963d65
  generator: dataraum-testdata
  version: 0.2.0
  scenario: month-end-close
  strategy: clean
  seed: 42
  months: 12
  fiscal_start: '2025-01-01'
  normalization: full
  families: [core_ledger, operating_chain, inventory]
  lever: null
```

Pin the `id`, regenerate when you want the bytes, and assert which corpus you graded
against. The digest is sha256 over exactly the fields shown, so you can **recompute it**
rather than trust it. Two directories with the same id hold the same data; a different id
means something upstream moved — most often a new family, which changes the corpus under
an unchanged seed by design. `intervention.yaml` additionally names
`counterfactual_corpus_id`, so a lever's baseline pair can be verified rather than assumed.

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

## Tables

Every corpus carries these; the probe tables (`addresses`, `orders`, `deliveries`,
`ref_entities`, `ref_activity`, `measure_probes`, `formula_probes`) materialize only when
a strategy injects into them. Row counts are for a 12-month `clean` run at seed 42.

| Table | ~Rows | Description |
|-------|-------|-------------|
| chart_of_accounts | 61 | Account hierarchy (5 types) |
| journal_entries | 16.6K | General ledger entries (event-driven) |
| journal_lines | 36.4K | Debit/credit lines (balanced per entry) |
| invoices | 3.3K | Vendor bills — `category` splits `expense` from `goods` |
| payments | 2.9K | Invoice payments (paid + partial) |
| bank_transactions | 6.0K | Bank statement (derived from cash events) |
| fx_rates | 472 | Weekly exchange rates (8 currency pairs) |
| trial_balance | 336 | Per-period movement (a flow) |
| balance_sheet | 112 | Carry-forward ending balance (a stock) |
| customers / products | 16 / 9 | Master data — the Demand and Offer ladders |
| sales_orders / sales_order_lines | 3.6K / 5.8K | The operating chain's event grain |
| ar_invoices / receipts | 3.6K / 2.8K | The AR side — what DSO measures |
| stock_movements | 6.1K | Stock subledger — signed receipts, issues, adjustments |
| inventory_positions | 216 | Closing stock at (product, location, period) — a **stock** |

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

Every source carries the **same** corpus id — they are one set of events exported through
different conventions, which is the premise a reconciliation scenario rests on.

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
- **Annual**: revenue, expenses, gross profit, COGS, purchases, AR/AP/cash/inventory balances, DSO, DPO, DIO, CCC, FCF
- **Monthly**: same metrics per period plus revenue growth MoM
- **Contribution margin**: true DB1 per customer and per product group, exact from the order lines
- **Invariants**: journal balanced, TB balanced, invoice-payment matched, bank reconciliation rate, inventory roll-forward, inventory-to-GL tie
- **Injection impact**: estimated metric deviations from known injection parameters

The file holds metrics and nothing else — *which* corpus they are true of is the
`corpus:` stamp's job, so seed, strategy, months and fiscal start appear there once.

**Two DPOs, both correct.** `dpo` divides the payable by *purchases* (vendor-bill credits
to AP) — the textbook definition, computable only since goods bills became separable from
expense bills. `dpo_on_expenses` carries the total-expense denominator, which is what a
consumer without a separable purchases figure necessarily computes. They are named
alternatives, not one right and one wrong answer; `cash_conversion_cycle` uses the first.
CCC is composed from the *published, rounded* DIO/DSO/DPO, so recombining them reproduces
it exactly.

**Not yet graded.** `gross_profit` and `free_cash_flow` are computed but should be treated
as ungraded: the operating expense base is a fixed 3,000 vendor invoices plus fixed monthly
payroll and rent, sized independently of the firm, so their sign is an artifact of a knob.
See `docs/operating-model.md` §7 and §9.

## Backlog

The ordered plan is [`docs/operating-model.md`](docs/operating-model.md) §10 — next up are
scale profiles (which also add `profile` to the corpus identity), then the Supply, Capacity
and Throughput families. Independent of that:

- Format profiles (DATEV, SAP, Salesforce, HubSpot) via YAML config + OpenAPI specs
