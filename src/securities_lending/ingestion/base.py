"""
Abstract base class for all data ingesters.

All concrete ingesters must implement `download()` and `load()`.  The base
provides retry-with-backoff HTTP fetching and idempotent local caching so
that re-running the pipeline never over-fetches already-fresh data.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30  # seconds
_RETRY_DELAYS = (1, 3, 10)  # exponential-ish back-off


class BaseIngester(ABC):
    """Abstract ingester with HTTP retry and local file caching."""

    def __init__(self, cache_dir: str | Path = "data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "securities-lending-research/0.1"})

    # ── Abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def download(self, *args, **kwargs) -> None:
        """Download raw data to the local cache directory."""

    @abstractmethod
    def load(self, *args, **kwargs):
        """Load previously downloaded data from cache into a DataFrame."""

    # ── HTTP helpers ─────────────────────────────────────────────────────────

    def _fetch_url(self, url: str, timeout: int = _DEFAULT_TIMEOUT) -> bytes | None:
        """
        Fetch *url* with retry-on-failure.

        Returns raw bytes on success, or None if the resource does not exist
        (HTTP 404).  Raises on other HTTP errors after exhausting retries.
        """
        for attempt, delay in enumerate([0] + list(_RETRY_DELAYS)):
            if delay:
                logger.debug("Retry %d/%d for %s, sleeping %ds", attempt, len(_RETRY_DELAYS), url, delay)
                time.sleep(delay)
            try:
                resp = self._session.get(url, timeout=timeout)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.content
            except requests.exceptions.ConnectionError:
                if attempt == len(_RETRY_DELAYS):
                    raise
        return None  # unreachable but satisfies type-checker

    def _is_cached(self, path: Path) -> bool:
        """Return True if *path* exists and has non-zero size."""
        return path.exists() and path.stat().st_size > 0

    def _write_cache(self, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        logger.debug("Cached %d bytes → %s", len(content), path)
