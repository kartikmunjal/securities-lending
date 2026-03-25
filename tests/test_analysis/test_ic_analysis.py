"""Tests for ICAnalyzer."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from securities_lending.analysis.ic_analysis import ICAnalyzer


@pytest.fixture
def ic_setup(svr_panel, price_panel):
    """Return (signal_panel, return_panel) ready for IC computation."""
    # Use price_panel to construct a real return panel
    log_ret = np.log(price_panel / price_panel.shift(1))
    return svr_panel, log_ret


def test_ic_result_has_expected_fields(ic_setup):
    """ICResult dataclass should contain all expected fields."""
    sig, ret = ic_setup
    analyzer = ICAnalyzer(signal_panel=sig, return_panel=ret, min_stocks=3)
    results = analyzer.run(horizons=[1, 5], signal_name="svr")
    assert 1 in results and 5 in results
    r = results[1]
    assert hasattr(r, "mean_ic")
    assert hasattr(r, "icir")
    assert hasattr(r, "t_stat")
    assert hasattr(r, "pct_positive")


def test_positive_signal_has_positive_ic():
    """A signal perfectly correlated with returns should have IC near +1."""
    np.random.seed(0)
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    tickers = [f"T{i}" for i in range(30)]
    # Return panel: random
    ret = pd.DataFrame(
        np.random.randn(60, 30) * 0.01,
        index=dates,
        columns=tickers,
    )
    # Signal: return itself (perfect oracle) shifted by 1 day
    sig = ret.shift(1)
    analyzer = ICAnalyzer(signal_panel=sig, return_panel=ret, min_stocks=10)
    results = analyzer.run(horizons=[1], signal_name="perfect")
    # Mean IC should be strongly positive
    assert results[1].mean_ic > 0.3, f"Perfect oracle IC should be >0.3, got {results[1].mean_ic}"


def test_random_signal_ic_near_zero():
    """A random signal should have IC near zero (null hypothesis)."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    tickers = [f"T{i}" for i in range(50)]
    ret = pd.DataFrame(np.random.randn(100, 50) * 0.01, index=dates, columns=tickers)
    sig = pd.DataFrame(np.random.randn(100, 50), index=dates, columns=tickers)
    analyzer = ICAnalyzer(signal_panel=sig, return_panel=ret, min_stocks=20)
    results = analyzer.run(horizons=[1], signal_name="noise")
    # t-stat should not be extreme for random signal
    assert abs(results[1].mean_ic) < 0.2, f"Random IC magnitude should be <0.2, got {results[1].mean_ic}"


def test_run_multiple_bh_correction():
    """run_multiple should output BH-corrected p-values."""
    np.random.seed(9)
    dates = pd.date_range("2023-01-01", periods=80, freq="B")
    tickers = [f"T{i}" for i in range(40)]
    ret = pd.DataFrame(np.random.randn(80, 40) * 0.01, index=dates, columns=tickers)
    signals = {f"sig_{k}": pd.DataFrame(np.random.randn(80, 40), index=dates, columns=tickers)
               for k in range(3)}
    first = next(iter(signals.values()))
    analyzer = ICAnalyzer(signal_panel=first, return_panel=ret, min_stocks=10)
    table = analyzer.run_multiple(signals, horizons=[1, 5])
    assert "p_value_bh" in table.columns
    assert "significant_bh" in table.columns
    # BH p-values should be >= raw p-values
    assert (table["p_value_bh"] >= table["p_value"] - 1e-9).all()
