"""
Market microstructure feature computation.

Microstructure features capture the cost and friction of trading, which are
especially relevant in the context of short selling:

  * High borrow costs tend to concentrate in illiquid, hard-to-trade stocks.
  * Amihud (2002) illiquidity is correlated with short-selling constraints
    (few lenders → wide bid-ask spreads → high market impact).
  * Turnover rate interacts with DTC: a stock that turns over 0.5% per day
    and has 20% of float short will take 40 days to cover — very different
    from a stock with 3% daily turnover.

Features computed
-----------------
amihud_illiquidity  — |return| / dollar_volume, rolling 20-day mean
turnover_rate       — volume / shares_outstanding (proxy from float)
range_pct           — (high - low) / close, intraday volatility proxy
volume_zscore       — z-score of log(volume) vs 20-day history
rel_volume          — today's dollar volume / 20-day moving avg (volume surge)

References
----------
Amihud (2002) "Illiquidity and stock returns: cross-section and time-series
    effects". JFMR 5(1).
Kyle (1985) "Continuous auctions and insider trading". Econometrica 53(6).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MicrostructureFeatures:
    """Compute market microstructure features from OHLCV price data."""

    def __init__(self, price_df: pd.DataFrame):
        """
        Parameters
        ----------
        price_df : pd.DataFrame
            Long-format DataFrame with columns:
            date, symbol, open, high, low, close, volume.
        """
        self.prices = price_df.copy()
        self.prices["date"] = pd.to_datetime(self.prices["date"]).dt.date

    def build(self, vol_window: int = 20) -> pd.DataFrame:
        """
        Compute all microstructure features and return a long-format DataFrame.

        Returns
        -------
        DataFrame indexed by (date, symbol) with microstructure feature columns.
        """
        df = self.prices.copy()
        df = df.sort_values(["symbol", "date"])

        # Dollar volume
        df["dollar_volume"] = df["close"] * df["volume"]

        # Log returns (for Amihud)
        df["log_ret"] = df.groupby("symbol")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )
        df["abs_ret"] = df["log_ret"].abs()

        # Amihud illiquidity ratio: |r| / $volume, scaled to per-million
        df["amihud_raw"] = df["abs_ret"] / (df["dollar_volume"] / 1e6).replace(0, np.nan)
        df["amihud_illiquidity"] = (
            df.groupby("symbol")["amihud_raw"]
            .transform(lambda x: x.rolling(vol_window, min_periods=vol_window // 2).mean())
        )

        # Intraday range (high-low) as proxy for bid-ask + intraday volatility
        df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

        # Volume z-score vs recent history
        log_vol = np.log(df["volume"].replace(0, np.nan))
        df["log_vol"] = log_vol
        df["vol_ma20"] = df.groupby("symbol")["log_vol"].transform(
            lambda x: x.rolling(vol_window, min_periods=vol_window // 2).mean()
        )
        df["vol_std20"] = df.groupby("symbol")["log_vol"].transform(
            lambda x: x.rolling(vol_window, min_periods=vol_window // 2).std()
        )
        df["volume_zscore"] = (df["log_vol"] - df["vol_ma20"]) / df["vol_std20"].replace(0, np.nan)

        # Relative volume (today / trailing avg) — used in squeeze detection
        vol_20d_avg = df.groupby("symbol")["dollar_volume"].transform(
            lambda x: x.rolling(vol_window, min_periods=vol_window // 2).mean()
        )
        df["rel_volume"] = df["dollar_volume"] / vol_20d_avg.replace(0, np.nan)

        # Realised volatility (20-day)
        df["realized_vol_20d"] = df.groupby("symbol")["log_ret"].transform(
            lambda x: x.rolling(vol_window, min_periods=vol_window // 2).std() * np.sqrt(252)
        )

        keep = [
            "date", "symbol",
            "dollar_volume",
            "amihud_illiquidity",
            "range_pct",
            "volume_zscore",
            "rel_volume",
            "realized_vol_20d",
            "log_ret",
        ]
        out = df[keep].copy()

        # Cross-sectional log market cap proxy (close × shares, using rel_volume denominator)
        logger.info(
            "Microstructure features built for %d rows (%d tickers)",
            len(out),
            out["symbol"].nunique(),
        )
        return out.sort_values(["date", "symbol"]).reset_index(drop=True)
