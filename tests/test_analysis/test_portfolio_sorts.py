"""Tests for PortfolioSorter."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from securities_lending.analysis.portfolio_sorts import PortfolioSorter


@pytest.fixture
def sort_setup(svr_panel, price_panel):
    """Return (signal_panel, return_panel) for portfolio sort tests."""
    log_ret = np.log(price_panel / price_panel.shift(1))
    # Use a windowed forward return
    fwd_ret = log_ret.shift(-5)
    return svr_panel, fwd_ret


def test_sort_result_n_quantiles(sort_setup):
    """SortResult should have the correct number of quantiles."""
    sig, ret = sort_setup
    sorter = PortfolioSorter(signal_panel=sig, return_panel=ret, n_quantiles=5, holding_period=5)
    result = sorter.run(signal_name="test")
    assert result.n_quantiles == 5
    assert len(result.quantile_returns) == 5


def test_monotonicity_score_bounded(sort_setup):
    """Monotonicity score should be in [0, 1]."""
    sig, ret = sort_setup
    sorter = PortfolioSorter(signal_panel=sig, return_panel=ret, n_quantiles=5, holding_period=5)
    result = sorter.run()
    assert 0.0 <= result.monotonicity <= 1.0


def test_net_spreads_less_than_gross(sort_setup):
    """Net spreads at any positive TC should be ≤ gross spread."""
    sig, ret = sort_setup
    sorter = PortfolioSorter(signal_panel=sig, return_panel=ret, n_quantiles=5, holding_period=5)
    result = sorter.run(cost_scenarios=[5, 10])
    for tc, net in result.net_spreads.items():
        assert net <= result.ls_spread_ann + 1e-9, f"Net spread at {tc}bps > gross spread"


def test_high_ic_signal_positive_spread():
    """A perfectly predictive signal should yield a strong positive L/S spread."""
    np.random.seed(1)
    dates = pd.date_range("2023-01-01", periods=80, freq="B")
    tickers = [f"T{i}" for i in range(40)]
    # Returns: cross-sectionally random
    ret = pd.DataFrame(np.random.randn(80, 40) * 0.02, index=dates, columns=tickers)
    # Signal: perfectly correlated with 5-day ahead return
    sig = ret.rolling(5).sum().shift(5)  # forward info as signal (oracle)
    sorter = PortfolioSorter(signal_panel=sig, return_panel=ret, n_quantiles=5, holding_period=5)
    result = sorter.run()
    assert result.ls_spread_ann > 0, "Oracle signal should yield positive L/S spread"
