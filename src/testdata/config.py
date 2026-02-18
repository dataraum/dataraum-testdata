"""Project-root and config directory discovery."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _find_project_root() -> Path:
    """Walk up from this package directory to find the project root (contains ``config/``)."""
    current = Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        if (parent / "config").is_dir():
            return parent
    raise FileNotFoundError(
        "Cannot locate project root — no 'config/' directory found "
        f"above {current}"
    )


def get_config_dir() -> Path:
    """Return the ``config/`` directory at the project root."""
    return _find_project_root() / "config"
