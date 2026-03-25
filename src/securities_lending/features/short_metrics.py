"""
Short interest and short-flow metrics.

This module ingests the outputs of the two FINRA ingesters and the price
loader to produce a unified daily feature panel with the following signals:

SVR signals (daily, from Reg SHO)
----------------------------------
svr             — Short volume ratio on day t
svr_ma5/10/20   — Rolling mean SVR (5, 10, 20-day)
svr_z20         — SVR normalised as z-score vs trailing 20-day history
svr_trend5      — OLS slope of SVR over last 5 days (units: SVR per day)
svr_percentile  — Cross-sectional percentile of svr_z20 on each date

SI signals (biweekly, interpolated daily from FINRA short interest)
-------------------------------------------------------------------
si_level        — Short interest in shares (interpolated)
si_pct_float    — SI as % of float (requires float snapshot)
si_dtc          — Days-to-cover from FINRA, or computed if missing
si_chg          — Period-over-period change in SI (shares)
si_chg_pct      — Period-over-period % change
si_high_flag    — 1 if si_pct_float exceeds sector 90th percentile

Composite
---------
short_pressure  — Rank-weighted combination of svr_z20 and si_pct_float
squeeze_setup   — Binary flag: high DTC + recent positive price momentum
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..utils.winsorize import winsorize_series, standardize_cross_section

logger = logging.getLogger(__name__)


class ShortMetricsBuilder:
    """
    Build the short-side feature panel from raw FINRA and price inputs.

    Parameters
    ----------
    svr_panel : pd.DataFrame
        (date × ticker) short volume ratio panel from FINRARegSHOIngester.
    si_daily : pd.DataFrame
        Long-format daily-interpolated short interest from
        FINRAShortInterestIngester.load_daily_interpolated().
    price_panel : pd.DataFrame
        (date × ticker) adjusted close prices.
    float_snapshot : pd.DataFrame | None
        Point-in-time float shares from PriceIngester.load_float_snapshot().
    """

    def __init__(
        self,
        svr_panel: pd.DataFrame,
        si_daily: pd.DataFrame,
        price_panel: pd.DataFrame,
        float_snapshot: pd.DataFrame | None = None,
    ):
        self.svr = svr_panel
        self.si_daily = si_daily
        self.prices = price_panel
        self.float_snap = float_snapshot

    # ── Public entry point ───────────────────────────────────────────────────

    def build(self) -> pd.DataFrame:
        """
        Compute all short metrics and return a long-format feature DataFrame.

        Returns
        -------
        DataFrame indexed by (date, symbol) with all short metrics as columns.
        """
        logger.info("Building short metrics panel…")
        svr_features = self._build_svr_features()
        si_features = self._build_si_features()

        # Merge on (date, symbol)
        features = svr_features.merge(si_features, on=["date", "symbol"], how="outer")

        # Composite signals
        features = self._add_composite_signals(features)

        logger.info(
            "Short metrics panel: %d rows, %d tickers, %d features",
            len(features),
            features["symbol"].nunique(),
            features.shape[1] - 2,
        )
        return features.sort_values(["date", "symbol"]).reset_index(drop=True)

    # ── SVR features ─────────────────────────────────────────────────────────

    def _build_svr_features(self) -> pd.DataFrame:
        svr = self.svr.copy()

        # Rolling statistics per ticker
        svr_ma5 = svr.rolling(5, min_periods=3).mean()
        svr_ma10 = svr.rolling(10, min_periods=5).mean()
        svr_ma20 = svr.rolling(20, min_periods=10).mean()
        svr_std20 = svr.rolling(20, min_periods=10).std()
        svr_z20 = (svr - svr_ma20) / svr_std20.replace(0, np.nan)

        # Short-term SVR trend (OLS slope over last 5 days, vectorised)
        svr_trend5 = svr.rolling(5, min_periods=3).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
        )

        # Cross-sectional percentile rank (row-wise across tickers)
        svr_pct = svr_z20.rank(axis=1, pct=True)

        # Stack to long format
        frames = {
            "svr": svr,
            "svr_ma5": svr_ma5,
            "svr_ma10": svr_ma10,
            "svr_ma20": svr_ma20,
            "svr_z20": svr_z20,
            "svr_trend5": svr_trend5,
            "svr_percentile": svr_pct,
        }
        long = pd.concat(
            {k: v.stack(future_stack=True) for k, v in frames.items()},
            axis=1,
        ).reset_index()
        long.columns = ["date", "symbol"] + list(frames.keys())
        return long

    # ── SI features ──────────────────────────────────────────────────────────

    def _build_si_features(self) -> pd.DataFrame:
        si = self.si_daily.copy()
        si["date"] = pd.to_datetime(si["date"]).dt.date

        # Merge float shares if available
        if self.float_snap is not None:
            si = si.merge(
                self.float_snap[["symbol", "float_shares"]],
                on="symbol",
                how="left",
            )
            si["si_pct_float"] = si["short_interest"] / si["float_shares"].replace(0, np.nan)
            si["si_pct_float"] = winsorize_series(si["si_pct_float"], 0.005, 0.995)
        else:
            si["si_pct_float"] = np.nan
            logger.warning("No float snapshot provided; si_pct_float will be NaN")

        # Period-over-period SI change (by symbol, sorted by date)
        si = si.sort_values(["symbol", "date"])
        si["si_chg"] = si.groupby("symbol")["short_interest"].diff()
        si["si_chg_pct"] = si["si_chg"] / si["short_interest"].shift(1).replace(0, np.nan)

        # DTC — use FINRA's if available, else estimate from prices + SI
        if "days_to_cover" not in si.columns or si["days_to_cover"].isna().all():
            vol_panel = self._daily_volume_panel()
            if vol_panel is not None:
                si = self._estimate_dtc(si, vol_panel)
        else:
            si = si.rename(columns={"days_to_cover": "si_dtc"})

        # Cross-sectional flag: SI% > 90th percentile by date
        si["si_high_flag"] = si.groupby("date")["si_pct_float"].transform(
            lambda x: (x >= x.quantile(0.9)).astype(int)
        )

        keep = [
            "date", "symbol", "short_interest", "si_pct_float",
            "si_chg", "si_chg_pct", "si_dtc", "si_high_flag",
        ]
        keep = [c for c in keep if c in si.columns]
        return si[keep]

    # ── Composite signals ────────────────────────────────────────────────────

    def _add_composite_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add composite short pressure and squeeze-setup flags."""
        # short_pressure: equal-weight rank combination of SVR and SI signals
        rank_svr = df.groupby("date")["svr_z20"].rank(pct=True)
        rank_si = df.groupby("date")["si_pct_float"].rank(pct=True)
        df["short_pressure"] = 0.5 * rank_svr.fillna(0.5) + 0.5 * rank_si.fillna(0.5)

        # squeeze_setup: high DTC + recent positive price return
        # (squeeze risk: stock has gone up recently while heavily shorted)
        if "si_dtc" in df.columns:
            high_dtc = df["si_dtc"] > 5
            # Merge in 5-day price return
            price_ret = (
                np.log(self.prices / self.prices.shift(5))
                .stack(future_stack=True)
                .reset_index()
            )
            price_ret.columns = ["date", "symbol", "ret_5d"]
            price_ret["date"] = pd.to_datetime(price_ret["date"]).dt.date
            df = df.merge(price_ret, on=["date", "symbol"], how="left")
            df["squeeze_setup"] = (
                high_dtc & (df["ret_5d"] > 0.02)
            ).astype(int)
        else:
            df["squeeze_setup"] = np.nan

        return df

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _daily_volume_panel(self) -> pd.DataFrame | None:
        """Return a (date × ticker) panel of average daily share volume."""
        if self.prices is None:
            return None
        # We don't have raw share volume in the price panel unless loaded via prices.py
        # This is a placeholder; real implementation would load from PriceIngester
        return None

    @staticmethod
    def _estimate_dtc(si: pd.DataFrame, vol_panel: pd.DataFrame) -> pd.DataFrame:
        """Estimate DTC from SI / average_daily_volume when FINRA DTC is missing."""
        vol_long = vol_panel.stack(future_stack=True).reset_index()
        vol_long.columns = ["date", "symbol", "avg_daily_volume"]
        vol_long["date"] = pd.to_datetime(vol_long["date"]).dt.date

        si = si.merge(vol_long, on=["date", "symbol"], how="left")
        si["si_dtc"] = si["short_interest"] / si["avg_daily_volume"].replace(0, np.nan)
        si["si_dtc"] = si["si_dtc"].clip(0, 100)
        return si
