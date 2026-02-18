"""Pre-built injection strategy profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class StrategyLevel(StrEnum):
    CLEAN = "clean"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class InjectionSpec:
    """Specification for a single injection to apply."""

    injector: str           # Function name from injectors module
    table: str              # Target table name
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Strategy:
    """A named collection of injection specs."""

    name: str
    level: StrategyLevel
    description: str
    injections: list[InjectionSpec] = field(default_factory=list)


# --- Strategy Definitions ---

CLEAN = Strategy(
    name="clean",
    level=StrategyLevel.CLEAN,
    description="No injections — baseline clean data",
    injections=[],
)

LOW_ENTROPY = Strategy(
    name="low",
    level=StrategyLevel.LOW,
    description="Subtle issues (2-5% rates), hard to detect",
    injections=[
        InjectionSpec("corrupt_types", "journal_lines", {"col": "debit", "ratio": 0.02, "severity": "low"}),
        InjectionSpec("introduce_nulls", "journal_lines", {"col": "cost_center", "ratio": 0.05, "severity": "low"}),
        InjectionSpec("inject_outliers", "journal_lines", {"col": "credit", "ratio": 0.02, "factor": 5.0, "severity": "low"}),
        InjectionSpec("corrupt_dates", "payments", {
            "col": "date",
            "formats": ["MM/DD/YYYY", "YYYYMMDD"],
            "severity": "low",
        }),
    ],
)

MEDIUM_ENTROPY = Strategy(
    name="medium",
    level=StrategyLevel.MEDIUM,
    description="Realistic problems — ~11 injection types across all layers",
    injections=[
        # Value layer
        InjectionSpec("corrupt_types", "journal_lines", {"col": "debit", "ratio": 0.03, "severity": "medium"}),
        InjectionSpec("introduce_nulls", "journal_lines", {"col": "cost_center", "ratio": 0.15, "severity": "medium"}),
        InjectionSpec("inject_outliers", "journal_lines", {"col": "credit", "ratio": 0.05, "factor": 10.0, "severity": "medium"}),
        InjectionSpec("break_benford", "bank_transactions", {"col": "amount", "method": "round_numbers", "severity": "medium"}),
        InjectionSpec("inject_temporal_drift", "bank_transactions", {
            "value_col": "amount",
            "time_col": "date",
            "shift_date": "2025-09-01",
            "shift_factor": 1.35,
            "severity": "medium",
        }),

        # Semantic layer
        InjectionSpec("mix_units", "invoices", {
            "col": "amount",
            "alt_currency": "EUR",
            "ratio": 0.10,
            "fx_rate": 1.1,
            "severity": "medium",
        }),
        InjectionSpec("obscure_column_names", "invoices", {
            "mapping": {"vendor_id": "vid", "payment_terms": "pt"},
            "severity": "medium",
        }),

        # Structural layer
        InjectionSpec("corrupt_dates", "payments", {
            "col": "date",
            "formats": ["MM/DD/YYYY", "DD/MM/YYYY", "DD-Mon-YY"],
            "severity": "medium",
        }),
        InjectionSpec("break_referential_integrity", "payments", {
            "fk_col": "invoice_id",
            "ratio": 0.05,
            "severity": "high",
        }),
        InjectionSpec("create_mutual_exclusivity", "journal_lines", {
            "col_a": "debit",
            "col_b": "credit",
            "severity": "low",
        }),

        # Computational layer
        InjectionSpec("drift_formula", "trial_balance", {
            "derived_col": "debit_balance",
            "source_cols": ["account_id", "period"],
            "error_ratio": 0.02,
            "severity": "medium",
        }),
    ],
)

HIGH_ENTROPY = Strategy(
    name="high",
    level=StrategyLevel.HIGH,
    description="Severe quality issues — high rates across all layers",
    injections=[
        # Value layer (aggressive)
        InjectionSpec("corrupt_types", "journal_lines", {"col": "debit", "ratio": 0.10, "severity": "high"}),
        InjectionSpec("corrupt_types", "journal_lines", {"col": "credit", "ratio": 0.08, "severity": "high"}),
        InjectionSpec("introduce_nulls", "journal_lines", {"col": "cost_center", "ratio": 0.40, "severity": "critical"}),
        InjectionSpec("introduce_nulls", "invoices", {"col": "amount", "ratio": 0.10, "severity": "high"}),
        InjectionSpec("inject_outliers", "journal_lines", {"col": "credit", "ratio": 0.15, "factor": 50.0, "severity": "critical"}),
        InjectionSpec("break_benford", "bank_transactions", {"col": "amount", "method": "uniform", "severity": "high"}),
        InjectionSpec("inject_temporal_drift", "bank_transactions", {
            "value_col": "amount",
            "time_col": "date",
            "shift_date": "2025-07-01",
            "shift_factor": 2.0,
            "severity": "high",
        }),

        # Semantic layer (aggressive)
        InjectionSpec("mix_units", "invoices", {
            "col": "amount",
            "alt_currency": "EUR",
            "ratio": 0.25,
            "fx_rate": 1.1,
            "severity": "high",
        }),
        InjectionSpec("mix_units", "bank_transactions", {
            "col": "amount",
            "alt_currency": "GBP",
            "ratio": 0.15,
            "fx_rate": 1.27,
            "severity": "high",
        }),
        InjectionSpec("obscure_column_names", "invoices", {
            "mapping": {"vendor_id": "vid", "payment_terms": "pt", "due_date": "dd", "status": "st"},
            "severity": "high",
        }),
        InjectionSpec("obscure_column_names", "bank_transactions", {
            "mapping": {"counterparty": "cp", "reconciled": "rc", "reference": "ref"},
            "severity": "high",
        }),

        # Structural layer (aggressive)
        InjectionSpec("corrupt_dates", "payments", {
            "col": "date",
            "formats": ["MM/DD/YYYY", "DD/MM/YYYY", "DD-Mon-YY", "Mon DD, YYYY", "YYYYMMDD"],
            "severity": "high",
        }),
        InjectionSpec("corrupt_dates", "invoices", {
            "col": "date",
            "formats": ["DD/MM/YYYY", "YYYYMMDD"],
            "severity": "high",
        }),
        InjectionSpec("break_referential_integrity", "payments", {
            "fk_col": "invoice_id",
            "ratio": 0.15,
            "severity": "critical",
        }),
        InjectionSpec("break_referential_integrity", "journal_lines", {
            "fk_col": "account_id",
            "ratio": 0.08,
            "severity": "critical",
        }),
        InjectionSpec("add_duplicate_fk_paths", "journal_lines", {
            "existing_fk_col": "entry_id",
            "new_col_name": "je_ref",
            "noise_ratio": 0.10,
            "severity": "medium",
        }),
        InjectionSpec("create_mutual_exclusivity", "journal_lines", {
            "col_a": "debit",
            "col_b": "credit",
            "severity": "low",
        }),

        # Computational layer (aggressive)
        InjectionSpec("drift_formula", "trial_balance", {
            "derived_col": "debit_balance",
            "source_cols": ["account_id", "period"],
            "error_ratio": 0.10,
            "severity": "high",
        }),
        InjectionSpec("drift_formula", "trial_balance", {
            "derived_col": "credit_balance",
            "source_cols": ["account_id", "period"],
            "error_ratio": 0.08,
            "severity": "high",
        }),
    ],
)


STRATEGIES: dict[str, Strategy] = {
    "clean": CLEAN,
    "low": LOW_ENTROPY,
    "medium": MEDIUM_ENTROPY,
    "high": HIGH_ENTROPY,
}


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name!r}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]
