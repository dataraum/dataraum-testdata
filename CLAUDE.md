# CLAUDE.md

## Project overview

Synthetic operating-model data with an answer key — a corpus whose metrics, structure and injected defects are all known, so anything computed over it can be graded rather than believed. Produces a closed-loop operating model (customers, products, sales orders and order lines cascading into GL entries, invoices, payments, bank transactions and trial balance) where every table is numerically consistent and traceable to the originating business event.

The plan for growing it across the six performance dimensions — the family framework, the oracle contract, the order of work — is `docs/operating-model.md`. Consumers bind to this generator; it references none of them.

## Setup

```bash
uv sync
```

Requires Python 3.14+. Uses `uv` as package manager.

## Common commands

```bash
# Run all tests
uv run pytest tests/ -v

# Lint
uv run python -m ruff check .

# Format
uv run python -m ruff format .

# Type-check
uv run python -m mypy -i src --no-error-summary

# Generate test data
testdata generate --scenario month-end-close --strategy medium --output ./output/medium --seed 42

# A constructed intervention and its exact same-seed baseline (the counterfactual pair)
testdata generate --strategy clean --output ./output/levered --seed 42 \
  --lever '{"type":"rate","driver":"price","period_k":6,"scope":{"segment":["Enterprise"]},"factor":1.15}'
testdata generate --strategy clean --output ./output/baseline --seed 42

# The confusable pair — same apparent magnitude, one real and one an artifact
testdata generate --strategy unit-mix-artifact --output ./output/artifact --seed 42

# A trended control corpus, and the high-cardinality payer dimension
testdata generate --strategy clean --output ./output/trend --trend-price 0.03 --merchants 4000
```

## Code layout

- `src/testdata/` — main package
  - `cli.py` — Typer CLI entry point (`testdata` command)
  - `scenarios/runner.py` — orchestrates generation → injection → normalization → export
  - `canonical/finance/models.py` — Pydantic domain models (7 table types)
  - `canonical/finance/generators.py` — event-driven data generation
  - `entropy/injectors.py` — ~10 injector functions (corrupt_types, introduce_nulls, etc.)
  - `entropy/scoping.py` — resolves an injection's target slice through declared joins
  - `entropy/registry.py` — tracks injections with metadata (layer, dimension, detector_id)
  - `entropy/strategies.py` — strategy loading from YAML
  - `schema_transforms.py` — normalization levels, column styles, key strategies
  - `export.py` — CSV/Parquet export with manifest
  - `ground_truth.py` — financial metrics, dimension profiles, injection impact estimation
  - `identity.py` — the corpus stamp; new parameters enter the digest only when used
- `config/scenarios/` — scenario YAML definitions
- `config/strategies/` — injection strategy YAML definitions (clean/low/medium/high, plus
  single-defect falsification corpora: unit-mix-artifact, fk-repeated-orphan, fk-shuffled)
- `tests/` — pytest tests (self-contained, no conftest fixtures)

## Quality gates

A hook at `.claude/hooks/end-of-turn-check.sh` runs after every turn and blocks on failure. It checks, in order: ruff lint, mypy, pytest. All three must pass.

A PostToolUse hook auto-formats with `ruff format` after every Edit/Write.

## Code style

- Line length: 120 (configured in pyproject.toml)
- Target: Python 3.14
- Type annotations are required — mypy is enforced
- Tests use `functools.lru_cache` for shared dataset fixtures, not pytest conftest
