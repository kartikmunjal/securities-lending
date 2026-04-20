import pandas as pd

from securities_lending.features.retail_attention import (
    load_retail_attention_features,
    merge_retail_attention,
)


def test_load_retail_attention_features_from_factor_panels(tmp_path):
    panel = pd.DataFrame(
        [[1.0, -1.0], [0.5, -0.5]],
        index=pd.bdate_range("2024-01-01", periods=2),
        columns=["gme", "amc"],
    )
    panel.to_parquet(tmp_path / "WSB_ATTENTION_SHOCK_Z.parquet")

    features = load_retail_attention_features(tmp_path)

    assert set(features.columns) == {"date", "symbol", "wsb_attention_shock_z"}
    assert set(features["symbol"]) == {"GME", "AMC"}
    assert len(features) == 4


def test_merge_retail_attention_adds_interactions():
    base = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["GME"],
            "borrow_stress": [2.0],
            "si_dtc": [4.0],
            "short_pressure": [3.0],
        }
    )
    retail = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["GME"],
            "wsb_attention_shock_z": [1.5],
            "wsb_sentiment_z": [-0.5],
        }
    )

    merged = merge_retail_attention(base, retail)

    assert merged.loc[0, "borrow_stress_x_wsb_attention"] == 3.0
    assert merged.loc[0, "dtc_x_wsb_attention"] == 6.0
    assert merged.loc[0, "short_pressure_x_wsb_sentiment"] == -1.5
