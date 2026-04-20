"""Retail-attention alt-data features from exported WSB factor panels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RETAIL_SIGNAL_COLS = [
    "wsb_mention_z",
    "wsb_sentiment_z",
    "wsb_attention_shock_z",
]

RETAIL_INTERACTION_COLS = [
    "borrow_stress_x_wsb_attention",
    "dtc_x_wsb_attention",
    "short_pressure_x_wsb_sentiment",
]


def load_retail_attention_features(factor_dir: str | Path) -> pd.DataFrame:
    """Load alt-data factor panels and return long-format features.

    The expected input is the `factor_panels/` directory produced by
    `alt-data-equity-signals`, containing files such as `WSB_MENTION_Z.parquet`.
    """
    factor_dir = Path(factor_dir)
    if not factor_dir.exists():
        raise FileNotFoundError(factor_dir)

    frames = []
    for path in sorted(factor_dir.glob("WSB_*.parquet")):
        col = path.stem.lower()
        panel = pd.read_parquet(path)
        panel.index = pd.to_datetime(panel.index).tz_localize(None)
        panel.columns = [str(c).upper() for c in panel.columns]
        long = (
            panel.rename_axis(index="date", columns="symbol")
            .reset_index()
            .melt(id_vars="date", var_name="symbol", value_name=col)
        )
        frames.append(long)

    if not frames:
        return pd.DataFrame(columns=["date", "symbol", *RETAIL_SIGNAL_COLS])

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["date", "symbol"], how="outer")
    merged["date"] = pd.to_datetime(merged["date"])
    merged["symbol"] = merged["symbol"].astype(str).str.upper()
    return merged.sort_values(["date", "symbol"]).reset_index(drop=True)


def merge_retail_attention(
    features: pd.DataFrame,
    retail_features: pd.DataFrame,
    *,
    add_interactions: bool = True,
) -> pd.DataFrame:
    """Merge WSB retail-attention features into the securities-lending panel."""
    if retail_features.empty:
        return features

    merged = features.copy()
    merged["date"] = pd.to_datetime(merged["date"])
    merged["symbol"] = merged["symbol"].astype(str).str.upper()

    retail = retail_features.copy()
    retail["date"] = pd.to_datetime(retail["date"])
    retail["symbol"] = retail["symbol"].astype(str).str.upper()

    merged = merged.merge(retail, on=["date", "symbol"], how="left")

    if add_interactions:
        if {"borrow_stress", "wsb_attention_shock_z"}.issubset(merged.columns):
            merged["borrow_stress_x_wsb_attention"] = (
                merged["borrow_stress"] * merged["wsb_attention_shock_z"]
            )
        if {"si_dtc", "wsb_attention_shock_z"}.issubset(merged.columns):
            merged["dtc_x_wsb_attention"] = merged["si_dtc"] * merged["wsb_attention_shock_z"]
        if {"short_pressure", "wsb_sentiment_z"}.issubset(merged.columns):
            merged["short_pressure_x_wsb_sentiment"] = (
                merged["short_pressure"] * merged["wsb_sentiment_z"]
            )

    return merged.sort_values(["date", "symbol"]).reset_index(drop=True)
