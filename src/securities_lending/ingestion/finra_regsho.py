"""
FINRA Reg SHO Consolidated Daily Short Sale Volume Ingester.

Data source
-----------
FINRA publishes consolidated short sale volume for all NMS securities on a
daily basis at:

    https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt

File format (pipe-delimited, header on first line):

    Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market

Key columns
-----------
ShortVolume         — Reported short sales (shares)
ShortExemptVolume   — Exempt short sales (market-maker hedging; excluded from SVR)
TotalVolume         — Total consolidated NMS volume (shares)

Derived field
-------------
SVR (short volume ratio) = (ShortVolume − ShortExemptVolume) / TotalVolume
Excludes exempt shorts because they reflect market-maker activity, not
directional short interest from investors.

References
----------
Engelberg, Reed & Ringgenberg (2012) "How are Shorts Informed?" JFE.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .base import BaseIngester
from ..utils.calendar import trading_dates

logger = logging.getLogger(__name__)

_BASE_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"

# FINRA changed the column layout in early 2014; both formats are handled.
_COLS_V1 = ["Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume", "Market"]
_COLS_V2 = _COLS_V1  # same as of 2021+

_DTYPE_MAP = {
    "Symbol": str,
    "ShortVolume": "int64",
    "ShortExemptVolume": "int64",
    "TotalVolume": "int64",
    "Market": str,
}


class FINRARegSHOIngester(BaseIngester):
    """Download and cache FINRA Reg SHO daily short sale volume files."""

    def __init__(
        self,
        cache_dir: str | Path = "data/raw/finra_regsho",
        tickers: list[str] | None = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.tickers = set(tickers) if tickers else None

    # ── Download ─────────────────────────────────────────────────────────────

    def download(
        self,
        start: str | date,
        end: str | date | None = None,
        show_progress: bool = True,
    ) -> None:
        """
        Download raw FINRA Reg SHO files for every trading day in [start, end].

        Files are stored as ``{cache_dir}/{YYYYMMDD}.txt``.  Already-cached
        files are skipped (idempotent).
        """
        if end is None:
            end = date.today()
        dates = trading_dates(start, end)
        if not dates:
            logger.warning("No trading dates found between %s and %s", start, end)
            return

        missing = [d for d in dates if not self._is_cached(self._raw_path(d))]
        logger.info("Downloading %d / %d FINRA Reg SHO files", len(missing), len(dates))

        for d in tqdm(missing, desc="FINRA RegSHO", disable=not show_progress):
            url = _BASE_URL.format(date=d.strftime("%Y%m%d"))
            content = self._fetch_url(url)
            if content is None:
                logger.debug("No file for %s (holiday / not yet published)", d)
                continue
            self._write_cache(self._raw_path(d), content)

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(
        self,
        start: str | date,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """
        Load all cached Reg SHO files into a single tidy DataFrame.

        Returns
        -------
        DataFrame with columns: date, symbol, short_volume, short_exempt_volume,
        total_volume, svr (short volume ratio), market.
        Indexed by (date, symbol).
        """
        if end is None:
            end = date.today()
        dates = trading_dates(start, end)
        frames = []
        for d in dates:
            path = self._raw_path(d)
            if not self._is_cached(path):
                continue
            try:
                df = self._parse_file(path, d)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", path, exc)
                continue
            if self.tickers is not None:
                df = df[df["symbol"].isin(self.tickers)]
            frames.append(df)

        if not frames:
            logger.warning("No Reg SHO data loaded for %s – %s", start, end)
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
        logger.info("Loaded %d Reg SHO rows (%d tickers)", len(out), out["symbol"].nunique())
        return out

    def load_panel(
        self,
        start: str | date,
        end: str | date | None = None,
        column: str = "svr",
    ) -> pd.DataFrame:
        """
        Convenience method: return a (date × ticker) pivot table for *column*.

        The short volume ratio (SVR) panel is the primary input to signal
        construction downstream.
        """
        long = self.load(start, end)
        if long.empty:
            return pd.DataFrame()
        return long.pivot(index="date", columns="symbol", values=column)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _raw_path(self, d: date) -> Path:
        return self.cache_dir / f"{d.strftime('%Y%m%d')}.txt"

    @staticmethod
    def _parse_file(path: Path, d: date) -> pd.DataFrame:
        """Parse a single FINRA Reg SHO text file into a tidy DataFrame."""
        df = pd.read_csv(
            path,
            sep="|",
            dtype=str,
            on_bad_lines="skip",
        )

        # Normalise column names (FINRA occasionally changes casing)
        df.columns = [c.strip() for c in df.columns]

        # Drop summary / footer rows (non-ticker lines)
        df = df[df["Symbol"].str.match(r"^[A-Z]{1,5}$", na=False)].copy()

        # Cast numeric columns; coerce bad values to NaN then drop
        for col in ("ShortVolume", "ShortExemptVolume", "TotalVolume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["ShortVolume", "TotalVolume"])
        df = df[df["TotalVolume"] > 0]

        # Compute short volume ratio: exclude exempt (market-maker) shorts
        df["svr"] = (df["ShortVolume"] - df["ShortExemptVolume"].fillna(0)) / df["TotalVolume"]
        df["svr"] = df["svr"].clip(0, 1)

        out = pd.DataFrame(
            {
                "date": d,
                "symbol": df["Symbol"].str.strip(),
                "short_volume": df["ShortVolume"].astype("int64"),
                "short_exempt_volume": df["ShortExemptVolume"].fillna(0).astype("int64"),
                "total_volume": df["TotalVolume"].astype("int64"),
                "svr": df["svr"],
            }
        )
        return out
