#!/usr/bin/env python
"""Run a simple crowded-short x retail-attention interaction backtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from securities_lending.analysis.interaction_backtest import backtest_interaction_signal
from securities_lending.features.retail_attention import (
    load_retail_attention_features,
    merge_retail_attention,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest retail-attention squeeze interactions")
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--alt-factor-dir", required=True)
    parser.add_argument("--signal", default="borrow_stress_x_wsb_attention")
    parser.add_argument("--return-col", default="ret_fwd_5d")
    parser.add_argument("--out", default="data/results/retail_squeeze_backtest.csv")
    args = parser.parse_args()

    features = pd.read_parquet(args.features)
    retail = load_retail_attention_features(args.alt_factor_dir)
    merged = merge_retail_attention(features, retail)

    result = backtest_interaction_signal(
        merged,
        signal_col=args.signal,
        return_col=args.return_col,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.as_dict()]).to_csv(out, index=False)
    fig_path = out.with_suffix(".png")
    _plot_result(pd.DataFrame([result.as_dict()]), fig_path)

    print(pd.DataFrame([result.as_dict()]).round(4).to_string(index=False))
    print(f"\nSaved: {out}")
    print(f"Figure: {fig_path}")


def _plot_result(result: pd.DataFrame, path: Path) -> None:
    metrics = result.iloc[0]
    labels = ["Ann. spread", "Sharpe", "Hit rate", "Top bucket hit"]
    values = [
        metrics["ann_spread"],
        metrics["sharpe"],
        metrics["hit_rate"],
        metrics["event_hit_rate"],
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#337a5b", "#3b6f9e", "#64748b", "#64748b"]
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Retail Attention x Short Crowding Backtest: {metrics['signal']}")
    ax.set_ylabel("Metric value")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
