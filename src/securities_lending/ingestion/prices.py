"""
Price, volume, and float data ingester via yfinance.

yfinance provides adjusted OHLCV going back ~20 years for most liquid names,
as well as current-snapshot fundamental data (market cap, shares outstanding,
short interest ratio, float).

Limitations
-----------
* Float shares from yfinance are point-in-time (current), not historical.
  For a proper survivorship-bias-free panel, replace with Compustat quarterly
  shares outstanding.  This limitation is noted where float data is consumed.
* Historical short interest from yfinance (shortRatio, sharesShort) reflects
  the most recent FINRA filing, not a time series.  We use FINRA's own files
  for historical SI.
* yfinance rate-limits aggressively; the loader uses jittered delays and
  batch downloads to avoid 429 errors.
"""

from __future__ import annotations

import logging
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from .base import BaseIngester

logger = logging.getLogger(__name__)

_CACHE_FILE = "prices.parquet"
_FLOAT_CACHE_FILE = "float_shares.parquet"


class PriceIngester(BaseIngester):
    """Download and cache adjusted OHLCV and fundamental snapshots."""

    def __init__(self, cache_dir: str | Path = "data/raw/prices"):
        super().__init__(cache_dir=cache_dir)

    # ── Download / load prices ───────────────────────────────────────────────

    def download(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        chunk_size: int = 50,
        show_progress: bool = True,
    ) -> None:
        """
        Download adjusted daily OHLCV for *tickers* and cache as parquet.

        Downloads in chunks of *chunk_size* to avoid yfinance rate limits.
        """
        out_path = self.cache_dir / _CACHE_FILE
        existing: pd.DataFrame | None = None
        if self._is_cached(out_path):
            existing = pd.read_parquet(out_path)
            logger.info("Loaded existing price cache (%d rows)", len(existing))

        all_frames: list[pd.DataFrame] = []
        chunks = [tickers[i : i + chunk_size] for i in range(0, len(tickers), chunk_size)]

        for chunk in tqdm(chunks, desc="Downloading prices", disable=not show_progress):
            df = yf.download(
                chunk,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                continue
            # yfinance returns MultiIndex columns (OHLCV, Ticker) for multi-ticker downloads
            if isinstance(df.columns, pd.MultiIndex):
                df = df.stack(level=1, future_stack=True)
                df.index.names = ["date", "symbol"]
            else:
                df.index.name = "date"
                df = df.assign(symbol=chunk[0]).set_index("symbol", append=True)
            df.columns = [c.lower() for c in df.columns]
            all_frames.append(df.reset_index())
            time.sleep(random.uniform(0.3, 1.0))  # jitter to avoid rate-limit

        if not all_frames:
            logger.warning("No price data downloaded for any ticker")
            return

        new_data = pd.concat(all_frames, ignore_index=True)
        if existing is not None:
            combined = pd.concat([existing, new_data]).drop_duplicates(
                subset=["date", "symbol"], keep="last"
            )
        else:
            combined = new_data

        combined.to_parquet(out_path, index=False)
        logger.info(
            "Saved %d price rows for %d tickers to %s",
            len(combined),
            combined["symbol"].nunique(),
            out_path,
        )

    def load(
        self,
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Load cached prices.  Optionally filter by ticker and date range.

        Returns
        -------
        DataFrame with columns: date, symbol, open, high, low, close, volume.
        Sorted by (date, symbol).
        """
        out_path = self.cache_dir / _CACHE_FILE
        if not self._is_cached(out_path):
            raise FileNotFoundError(
                f"Price cache not found at {out_path}. Run download() first."
            )
        df = pd.read_parquet(out_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        if tickers:
            df = df[df["symbol"].isin(tickers)]
        if start:
            df = df[df["date"] >= pd.Timestamp(start).date()]
        if end:
            df = df[df["date"] <= pd.Timestamp(end).date()]

        return df.sort_values(["date", "symbol"]).reset_index(drop=True)

    def load_panel(
        self,
        column: str = "close",
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return a (date × ticker) pivot table for the given price column."""
        df = self.load(tickers=tickers, start=start, end=end)
        if df.empty:
            return pd.DataFrame()
        return df.pivot(index="date", columns="symbol", values=column)

    # ── Float / fundamental snapshot ─────────────────────────────────────────

    def download_float_snapshot(
        self,
        tickers: list[str],
        show_progress: bool = True,
    ) -> None:
        """
        Fetch current float shares (and short interest snapshot) from yfinance.

        WARNING: This is point-in-time data (today's values), not historical.
        Use only as a denominator for SI% of float; do not use for backtesting
        unless you accept the look-ahead bias this introduces.
        """
        records = []
        for ticker in tqdm(tickers, desc="Float snapshot", disable=not show_progress):
            try:
                info = yf.Ticker(ticker).info
                records.append(
                    {
                        "symbol": ticker,
                        "float_shares": info.get("floatShares"),
                        "shares_outstanding": info.get("sharesOutstanding"),
                        "market_cap": info.get("marketCap"),
                        "shares_short": info.get("sharesShort"),
                        "short_ratio": info.get("shortRatio"),
                        "short_pct_float": info.get("shortPercentOfFloat"),
                    }
                )
            except Exception as exc:
                logger.warning("Failed to fetch info for %s: %s", ticker, exc)
            time.sleep(random.uniform(0.2, 0.6))

        if not records:
            return

        df = pd.DataFrame(records)
        df.to_parquet(self.cache_dir / _FLOAT_CACHE_FILE, index=False)
        logger.info("Saved float snapshot for %d tickers", len(df))

    def load_float_snapshot(self) -> pd.DataFrame:
        """Load the cached float/fundamental snapshot."""
        path = self.cache_dir / _FLOAT_CACHE_FILE
        if not self._is_cached(path):
            raise FileNotFoundError(
                f"Float cache not found at {path}. Run download_float_snapshot() first."
            )
        return pd.read_parquet(path)

    # ── Derived daily panel helpers ──────────────────────────────────────────

    def compute_returns(
        self,
        prices: pd.DataFrame | None = None,
        tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compute log returns from the price panel.

        Returns a (date × ticker) DataFrame of daily log returns.
        """
        if prices is None:
            prices = self.load_panel(column="close", tickers=tickers)
        log_ret = np.log(prices / prices.shift(1))
        return log_ret

    def compute_dollar_volume(
        self,
        tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return (date × ticker) panel of daily dollar volume (close × volume)."""
        prices = self.load(tickers=tickers)
        prices["dollar_volume"] = prices["close"] * prices["volume"]
        return prices.pivot(index="date", columns="symbol", values="dollar_volume")
