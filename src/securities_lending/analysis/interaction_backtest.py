"""Simple backtests for short-crowding x retail-attention interactions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InteractionBacktestResult:
    signal: str
    horizon: int
    n_periods: int
    mean_spread: float
    ann_spread: float
    spread_vol: float
    sharpe: float
    hit_rate: float
    top_bucket_mean: float
    bottom_bucket_mean: float
    event_hit_rate: float

    def as_dict(self) -> dict[str, float | int | str]:
        return self.__dict__.copy()


def backtest_interaction_signal(
    features: pd.DataFrame,
    *,
    signal_col: str = "borrow_stress_x_wsb_attention",
    return_col: str = "ret_fwd_5d",
    date_col: str = "date",
    symbol_col: str = "symbol",
    horizon: int = 5,
    n_quantiles: int = 5,
    min_names: int = 20,
) -> InteractionBacktestResult:
    """Backtest a cross-sectional interaction signal with equal-weight quantiles."""
    required = {date_col, symbol_col, signal_col, return_col}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"features missing required columns: {sorted(missing)}")

    frame = features[[date_col, symbol_col, signal_col, return_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.dropna(subset=[signal_col, return_col])

    rows = []
    for date, cross_section in frame.groupby(date_col):
        if len(cross_section) < min_names:
            continue
        ranks = cross_section[signal_col].rank(method="first")
        labels = pd.qcut(ranks, n_quantiles, labels=False) + 1
        by_q = cross_section.assign(q=labels).groupby("q")[return_col].mean()
        if 1 not in by_q.index or n_quantiles not in by_q.index:
            continue
        rows.append(
            {
                "date": date,
                "bottom": float(by_q.loc[1]),
                "top": float(by_q.loc[n_quantiles]),
                "spread": float(by_q.loc[n_quantiles] - by_q.loc[1]),
            }
        )

    results = pd.DataFrame(rows).sort_values("date")
    if results.empty:
        raise ValueError("no valid cross-sections for interaction backtest")

    periods_per_year = 252 / horizon
    mean_spread = float(results["spread"].mean())
    spread_vol = float(results["spread"].std(ddof=1))
    sharpe = mean_spread / spread_vol * np.sqrt(periods_per_year) if spread_vol > 0 else np.nan

    return InteractionBacktestResult(
        signal=signal_col,
        horizon=horizon,
        n_periods=len(results),
        mean_spread=mean_spread,
        ann_spread=mean_spread * periods_per_year,
        spread_vol=spread_vol,
        sharpe=float(sharpe),
        hit_rate=float((results["spread"] > 0).mean()),
        top_bucket_mean=float(results["top"].mean()),
        bottom_bucket_mean=float(results["bottom"].mean()),
        event_hit_rate=float((results["top"] > 0).mean()),
    )
