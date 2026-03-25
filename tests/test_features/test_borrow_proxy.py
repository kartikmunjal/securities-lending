"""Tests for BorrowRateProxy."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from securities_lending.features.borrow_proxy import BorrowRateProxy


@pytest.fixture
def proxy():
    return BorrowRateProxy(lendable_fraction=0.20)


def test_rate_schedule_monotone(proxy):
    """Borrow rate must be non-decreasing with utilisation."""
    utils = np.linspace(0, 1, 100)
    rates = [proxy._rate_from_utilisation(u) for u in utils]
    for i in range(1, len(rates)):
        assert rates[i] >= rates[i - 1], f"Rate decreased at u={utils[i]:.3f}"


def test_general_collateral_rate(proxy):
    """At zero utilisation, rate should equal the GC rate (~25 bps)."""
    rate = proxy._rate_from_utilisation(0.0)
    assert abs(rate - 25.0) < 5.0, f"GC rate should be ~25 bps, got {rate}"


def test_htb_threshold(proxy):
    """At 90%+ utilisation, rate should be substantially elevated (>500 bps)."""
    rate_htb = proxy._rate_from_utilisation(0.92)
    assert rate_htb > 500, f"HTB rate at 92% utilisation should be >500 bps, got {rate_htb}"


def test_nan_propagation(proxy):
    """NaN utilisation should produce NaN rate."""
    assert np.isnan(proxy._rate_from_utilisation(np.nan))


def test_compute_returns_expected_columns(si_daily, float_snapshot):
    """compute() should return required columns."""
    proxy = BorrowRateProxy(lendable_fraction=0.20)
    result = proxy.compute(si_daily=si_daily, float_snapshot=float_snapshot)
    required = {"date", "symbol", "utilisation", "borrow_rate_bps", "borrow_stress"}
    assert required.issubset(result.columns), f"Missing columns: {required - set(result.columns)}"


def test_borrow_stress_is_binary(si_daily, float_snapshot):
    """borrow_stress should be 0 or 1."""
    proxy = BorrowRateProxy()
    result = proxy.compute(si_daily=si_daily, float_snapshot=float_snapshot)
    assert result["borrow_stress"].dropna().isin([0, 1]).all()


def test_utilisation_bounded(si_daily, float_snapshot):
    """Utilisation should be in [0, 1]."""
    proxy = BorrowRateProxy()
    result = proxy.compute(si_daily=si_daily, float_snapshot=float_snapshot)
    valid = result["utilisation"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all()
