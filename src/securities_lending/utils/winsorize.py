"""Cross-sectional winsorization and ranking utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_cross_section(
    df: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """
    Winsorize each column of *df* cross-sectionally (per row across columns)
    using row-wise quantiles.

    If *df* is a (dates × tickers) panel, each row is a cross-section and
    winsorization is applied across tickers for every date.
    """
    def _winsorize_row(row: pd.Series) -> pd.Series:
        lo = row.quantile(lower)
        hi = row.quantile(upper)
        return row.clip(lower=lo, upper=hi)

    return df.apply(_winsorize_row, axis=1)


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize a single series at the given quantiles."""
    return s.clip(lower=s.quantile(lower), upper=s.quantile(upper))


def rank_cross_section(df: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """
    Rank each column within each row (cross-sectional rank by date).

    Parameters
    ----------
    pct : bool
        If True, return percentile ranks in [0, 1].  Default True.
    """
    return df.rank(axis=1, pct=pct, na_option="keep")


def standardize_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract cross-sectional mean and divide by cross-sectional std (row-wise).
    Produces z-scores within each cross-section (date).
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.subtract(mean, axis=0).divide(std, axis=0)
