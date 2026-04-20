import numpy as np
import pandas as pd

from securities_lending.analysis.interaction_backtest import backtest_interaction_signal


def test_interaction_backtest_reports_positive_spread_for_predictive_signal():
    dates = pd.bdate_range("2024-01-01", periods=12)
    symbols = [f"T{i}" for i in range(30)]
    rows = []
    for date in dates:
        signal = np.linspace(-1, 1, len(symbols))
        for sym, score in zip(symbols, signal):
            rows.append(
                {
                    "date": date,
                    "symbol": sym,
                    "borrow_stress_x_wsb_attention": score,
                    "ret_fwd_5d": 0.01 * score,
                }
            )
    features = pd.DataFrame(rows)

    result = backtest_interaction_signal(features, min_names=20)

    assert result.mean_spread > 0
    assert result.hit_rate == 1.0
    assert result.top_bucket_mean > result.bottom_bucket_mean
