"""Pre-built injection strategy profiles, loaded from YAML config."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from testdata.config import get_config_dir


class StrategyLevel(StrEnum):
    CLEAN = "clean"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class InjectionSpec:
    """Specification for a single injection to apply."""

    injector: str  # Function name from injectors module
    table: str  # Target table name
    kwargs: dict[str, Any] = field(default_factory=dict)
    consumer_hint: str | None = None  # Free-text label stamped onto every record


@dataclass
class Strategy:
    """A named collection of injection specs."""

    name: str
    level: StrategyLevel
    description: str
    injections: list[InjectionSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def load_strategy(path: Path) -> Strategy:
    """Parse a single strategy YAML file into a Strategy dataclass."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    injections = [
        InjectionSpec(
            injector=entry["injector"],
            table=entry["table"],
            kwargs=entry.get("params", {}),
            consumer_hint=entry.get("consumer_hint"),
        )
        for entry in (raw.get("injections") or [])
    ]

    return Strategy(
        name=raw["name"],
        level=StrategyLevel(raw["level"]),
        description=raw.get("description", ""),
        injections=injections,
    )


def load_all_strategies(config_dir: Path | None = None) -> dict[str, Strategy]:
    """Load all ``*.yaml`` files from the strategies config directory."""
    if config_dir is None:
        config_dir = get_config_dir() / "strategies"
    strategies: dict[str, Strategy] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        strategy = load_strategy(path)
        strategies[strategy.name] = strategy
    return strategies


# Module-level cache — populated on first call to get_strategy()
_STRATEGIES: dict[str, Strategy] | None = None


def get_strategy(name: str) -> Strategy:
    """Return a named strategy, loading from YAML on first access."""
    global _STRATEGIES  # noqa: PLW0603
    if _STRATEGIES is None:
        _STRATEGIES = load_all_strategies()
    if name not in _STRATEGIES:
        raise ValueError(f"Unknown strategy: {name!r}. Available: {list(_STRATEGIES.keys())}")
    return _STRATEGIES[name]
