#!/usr/bin/env python
"""Run a simple crowded-short x retail-attention interaction backtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

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

    print(pd.DataFrame([result.as_dict()]).round(4).to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
