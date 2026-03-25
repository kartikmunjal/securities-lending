"""Configuration loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file, defaulting to configs/pipeline.yaml."""
    if path is None:
        path = _REPO_ROOT / "configs" / "pipeline.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    logger.debug("Loaded config from %s", path)
    return cfg


def load_universe(path: str | Path | None = None) -> dict[str, Any]:
    """Load the universe config (tickers, date ranges, data sources)."""
    if path is None:
        path = _REPO_ROOT / "configs" / "universe.yaml"
    return load_config(path)
