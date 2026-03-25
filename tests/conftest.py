"""
Shared pytest fixtures.

All fixtures use synthetic data so the test suite runs without downloading
any real FINRA or price data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_dates(n: int = 100) -> list:
    from datetime import date, timedelta
    start = date(2023, 1, 3)
    dates = []
    i = 0
    while len(dates) < n:
        d = start + timedelta(days=i)
        if d.weekday() < 5:  # Mon–Fri
            dates.append(d)
        i += 1
    return dates


TICKERS = ["AAPL", "MSFT", "GME", "AMC", "TSLA", "NVDA", "SPY"]


@pytest.fixture
def sample_dates():
    return make_dates(100)


@pytest.fixture
def svr_panel(sample_dates):
    """Synthetic (date × ticker) SVR panel with realistic values."""
    rng = np.random.default_rng(42)
    data = rng.uniform(0.3, 0.7, size=(len(sample_dates), len(TICKERS)))
    # GME gets higher SVR
    gme_idx = TICKERS.index("GME")
    data[:, gme_idx] = rng.uniform(0.6, 0.9, size=len(sample_dates))
    return pd.DataFrame(data, index=sample_dates, columns=TICKERS)


@pytest.fixture
def price_panel(sample_dates):
    """Synthetic (date × ticker) adjusted close price panel."""
    rng = np.random.default_rng(0)
    prices = {}
    for t in TICKERS:
        log_ret = rng.normal(0.0003, 0.015, len(sample_dates))
        prices[t] = 100 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(prices, index=sample_dates)


@pytest.fixture
def si_daily(sample_dates):
    """Synthetic daily short interest data."""
    rng = np.random.default_rng(7)
    rows = []
    for d in sample_dates:
        for t in TICKERS:
            rows.append(
                {
                    "date": d,
                    "symbol": t,
                    "short_interest": int(rng.integers(1_000_000, 50_000_000)),
                    "days_to_cover": rng.uniform(0.5, 20.0),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def float_snapshot():
    """Synthetic float snapshot."""
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "symbol": TICKERS,
            "float_shares": [int(rng.integers(50_000_000, 5_000_000_000)) for _ in TICKERS],
            "market_cap": [int(rng.integers(1_000_000_000, 3_000_000_000_000)) for _ in TICKERS],
        }
    )
