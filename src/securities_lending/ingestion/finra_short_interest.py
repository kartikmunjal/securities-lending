"""
FINRA Biweekly Short Interest Ingester.

Data source
-----------
FINRA publishes aggregate short interest positions twice monthly for all
exchange-listed securities.  Settlement dates are typically around the 15th
and the last business day of each month.

The consolidated file URL follows the pattern:

    https://cdn.finra.org/equity/regsho/biweekly/FNRAshvol{YYYYMMDD}.txt

File format (pipe-delimited):

    SettlementDate|Symbol|ShortInterest|AvgDailyShareVolume|DaysToCover

Notes on data quality
---------------------
* Positions are as of the settlement date (T+2 from trade date), introducing
  a 2-day look-ahead lag relative to trade date.
* FINRA does not publish float shares; we merge float from yfinance to derive
  SI% of float.  Float data from yfinance is approximate and point-in-time;
  for production use, replace with Compustat shares-outstanding series.
* The biweekly cadence means the level signal has up to ~10 trading-day
  staleness.  Downstream code interpolates linearly between publication dates.

References
----------
Asquith, Pathak & Ritter (2005) "Short Interest, Institutional Ownership,
and Stock Returns". Journal of Financial Economics.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .base import BaseIngester

logger = logging.getLogger(__name__)

_BASE_URL = "https://cdn.finra.org/equity/regsho/biweekly/FNRAshvol{date}.txt"

# FINRA publishes on approximately these day-of-month targets; the actual
# settlement date is the closest NYSE business day.
_APPROX_SETTLEMENT_DAYS = (15, 31)


class FINRAShortInterestIngester(BaseIngester):
    """Download and cache FINRA biweekly short interest files."""

    def __init__(
        self,
        cache_dir: str | Path = "data/raw/finra_short_interest",
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
        Download biweekly short interest files for every settlement date in
        [start, end].  Already-cached files are skipped.
        """
        if end is None:
            end = date.today()
        candidates = self._settlement_candidates(start, end)
        missing = [d for d in candidates if not self._is_cached(self._raw_path(d))]
        logger.info("Downloading %d / %d FINRA short interest files", len(missing), len(candidates))

        for d in tqdm(missing, desc="FINRA ShortInterest", disable=not show_progress):
            url = _BASE_URL.format(date=d.strftime("%Y%m%d"))
            content = self._fetch_url(url)
            if content is None:
                # Try ±1 business day — FINRA occasionally shifts the date
                for offset in (-1, 1, -2, 2):
                    alt = d + timedelta(days=offset)
                    if self._is_cached(self._raw_path(alt)):
                        break
                    alt_url = _BASE_URL.format(date=alt.strftime("%Y%m%d"))
                    content = self._fetch_url(alt_url)
                    if content is not None:
                        d = alt
                        break
            if content is not None:
                self._write_cache(self._raw_path(d), content)
            else:
                logger.debug("No short interest file found near %s", d)

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(
        self,
        start: str | date,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """
        Load all cached short interest files into a tidy DataFrame.

        Returns
        -------
        DataFrame with columns:
            settlement_date, symbol, short_interest, avg_daily_volume, days_to_cover
        """
        if end is None:
            end = date.today()
        candidates = self._settlement_candidates(start, end)
        frames = []
        for d in candidates:
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
            logger.warning("No short interest data loaded for %s – %s", start, end)
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["settlement_date", "symbol"]).reset_index(drop=True)
        logger.info(
            "Loaded short interest: %d rows, %d unique tickers, %d settlement dates",
            len(out),
            out["symbol"].nunique(),
            out["settlement_date"].nunique(),
        )
        return out

    def load_daily_interpolated(
        self,
        start: str | date,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """
        Return a daily-frequency DataFrame with short interest linearly
        interpolated between biweekly settlement dates.

        This is the standard representation consumed by downstream features
        because the rest of the pipeline operates at daily frequency.

        Caveats
        -------
        Linear interpolation between biweekly observations assumes constant
        change between report dates.  In practice, short positions change
        daily.  This introduces measurement error that should be acknowledged
        in any predictive analysis.
        """
        si = self.load(start, end)
        if si.empty:
            return pd.DataFrame()

        # Build a daily index from the first to last settlement date
        all_dates = pd.date_range(
            start=si["settlement_date"].min(),
            end=si["settlement_date"].max(),
            freq="B",
        )
        tickers = si["symbol"].unique()

        # Pivot: settlement_date × ticker
        pivot = si.pivot(index="settlement_date", columns="symbol", values="short_interest")
        pivot.index = pd.DatetimeIndex(pivot.index)

        # Reindex to daily, then interpolate
        pivot = pivot.reindex(all_dates).interpolate(method="linear", limit_direction="forward")
        pivot.index.name = "date"
        pivot.columns.name = "symbol"

        dtc_pivot = (
            si.pivot(index="settlement_date", columns="symbol", values="days_to_cover")
        )
        dtc_pivot.index = pd.DatetimeIndex(dtc_pivot.index)
        dtc_pivot = dtc_pivot.reindex(all_dates).interpolate(method="linear", limit_direction="forward")

        # Convert to long format
        si_long = pivot.stack(future_stack=True).rename("short_interest").reset_index()
        dtc_long = dtc_pivot.stack(future_stack=True).rename("days_to_cover").reset_index()
        dtc_long.columns = ["date", "symbol", "days_to_cover"]

        out = si_long.merge(dtc_long, on=["date", "symbol"], how="left")
        return out

    # ── Private helpers ──────────────────────────────────────────────────────

    def _raw_path(self, d: date) -> Path:
        return self.cache_dir / f"{d.strftime('%Y%m%d')}.txt"

    @staticmethod
    def _settlement_candidates(
        start: str | date,
        end: str | date,
    ) -> list[date]:
        """Generate candidate settlement dates (around 15th and EOM) in range."""
        start = pd.Timestamp(start).date()
        end = pd.Timestamp(end).date()

        candidates: list[date] = []
        year, month = start.year, start.month
        while date(year, month, 1) <= end:
            for dom in _APPROX_SETTLEMENT_DAYS:
                try:
                    d = date(year, month, min(dom, 28))  # clamp for Feb
                    if start <= d <= end:
                        candidates.append(d)
                except ValueError:
                    pass
            month += 1
            if month > 12:
                month = 1
                year += 1
        return sorted(set(candidates))

    @staticmethod
    def _parse_file(path: Path, d: date) -> pd.DataFrame:
        df = pd.read_csv(path, sep="|", dtype=str, on_bad_lines="skip")
        df.columns = [c.strip() for c in df.columns]

        # Identify expected columns flexibly (FINRA format has varied)
        col_map = {}
        for col in df.columns:
            cl = col.lower().replace(" ", "_")
            if "symbol" in cl or cl == "issuesymbol":
                col_map[col] = "symbol"
            elif "shortinterest" in cl or cl == "currentshortinterest":
                col_map[col] = "short_interest"
            elif "avgdaily" in cl or "avgsharevol" in cl:
                col_map[col] = "avg_daily_volume"
            elif "daystocov" in cl or "daystocovershort" in cl:
                col_map[col] = "days_to_cover"
        df = df.rename(columns=col_map)

        required = {"symbol", "short_interest"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing required columns in {path}: found {list(df.columns)}")

        df = df[df["symbol"].str.match(r"^[A-Z]{1,5}$", na=False)].copy()
        df["short_interest"] = pd.to_numeric(df["short_interest"], errors="coerce")
        df["days_to_cover"] = pd.to_numeric(df.get("days_to_cover", pd.Series()), errors="coerce")
        df["avg_daily_volume"] = pd.to_numeric(df.get("avg_daily_volume", pd.Series()), errors="coerce")
        df = df.dropna(subset=["short_interest"])

        return pd.DataFrame(
            {
                "settlement_date": d,
                "symbol": df["symbol"].str.strip(),
                "short_interest": df["short_interest"].astype("int64"),
                "avg_daily_volume": df.get("avg_daily_volume"),
                "days_to_cover": df.get("days_to_cover"),
            }
        )
