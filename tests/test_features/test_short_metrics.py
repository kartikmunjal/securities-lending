"""Tests for ShortMetricsBuilder."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from securities_lending.features.short_metrics import ShortMetricsBuilder


def test_svr_z20_computed(svr_panel, price_panel, si_daily, float_snapshot):
    """svr_z20 should be computed and have the correct shape."""
    builder = ShortMetricsBuilder(
        svr_panel=svr_panel,
        si_daily=si_daily,
        price_panel=price_panel,
        float_snapshot=float_snapshot,
    )
    features = builder.build()
    assert "svr_z20" in features.columns
    assert len(features) > 0
    # svr_z20 should be roughly mean-zero (cross-sectionally normalised)
    assert abs(features["svr_z20"].mean()) < 1.0


def test_si_pct_float_bounded(svr_panel, price_panel, si_daily, float_snapshot):
    """si_pct_float should be in [0, 1] after winsorization."""
    builder = ShortMetricsBuilder(
        svr_panel=svr_panel,
        si_daily=si_daily,
        price_panel=price_panel,
        float_snapshot=float_snapshot,
    )
    features = builder.build()
    if "si_pct_float" in features.columns:
        valid = features["si_pct_float"].dropna()
        assert (valid >= 0).all(), "si_pct_float should be non-negative"
        assert (valid <= 1.0).all(), "si_pct_float should be ≤ 1 after winsorization"


def test_no_lookahead_in_svr_ma(svr_panel, price_panel, si_daily):
    """Rolling statistics must not use future data."""
    builder = ShortMetricsBuilder(
        svr_panel=svr_panel,
        si_daily=si_daily,
        price_panel=price_panel,
    )
    features = builder.build()
    # The first few rows of svr_ma20 should be NaN (insufficient history)
    first_dates = features["date"].unique()[:15]
    early = features[features["date"].isin(first_dates)]["svr_ma20"].dropna()
    # With min_periods=10, some early dates may be valid, but very early ones should be NaN
    # This is a smoke test — if all early values are non-NaN, rolling is likely wrong
    all_features_early = features[features["date"] == features["date"].min()]["svr_ma20"]
    # min_periods=10 means day 1 should be NaN
    assert all_features_early.isna().all(), "svr_ma20 on day 1 should be NaN (insufficient history)"


def test_short_pressure_bounded(svr_panel, price_panel, si_daily, float_snapshot):
    """short_pressure should be in [0, 1] (rank-based composite)."""
    builder = ShortMetricsBuilder(
        svr_panel=svr_panel,
        si_daily=si_daily,
        price_panel=price_panel,
        float_snapshot=float_snapshot,
    )
    features = builder.build()
    if "short_pressure" in features.columns:
        valid = features["short_pressure"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1.01).all()  # small tolerance for floating point
