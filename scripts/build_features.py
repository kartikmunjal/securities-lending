#!/usr/bin/env python
"""
Build the feature panel from raw data.

Reads FINRA Reg SHO, FINRA short interest, price data, and float snapshot
from the local cache and outputs a merged feature parquet file.

Output: data/processed/features.parquet

Usage
-----
    python scripts/build_features.py
    python scripts/build_features.py --start 2022-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from securities_lending.features import (
    BorrowRateProxy,
    MicrostructureFeatures,
    ShortMetricsBuilder,
)
from securities_lending.ingestion import (
    FINRARegSHOIngester,
    FINRAShortInterestIngester,
    PriceIngester,
)
from securities_lending.utils.config import load_universe, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build feature panel")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--output", default="data/processed/features.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    uni = load_universe()
    cfg = load_config()

    start = args.start or uni["data"]["start_date"]
    end = args.end
    tickers = uni["universe"]["tickers"]

    # ── Load raw data ─────────────────────────────────────────────────────────
    logger.info("Loading FINRA Reg SHO…")
    regsho = FINRARegSHOIngester(cache_dir="data/raw/finra_regsho", tickers=tickers)
    svr_panel = regsho.load_panel(start=start, end=end)

    logger.info("Loading FINRA short interest…")
    si_ingester = FINRAShortInterestIngester(
        cache_dir="data/raw/finra_short_interest", tickers=tickers
    )
    si_daily = si_ingester.load_daily_interpolated(start=start, end=end)

    logger.info("Loading prices…")
    price_ingester = PriceIngester(cache_dir="data/raw/prices")
    prices = price_ingester.load(tickers=tickers, start=start, end=end)
    price_panel = price_ingester.load_panel(column="close", tickers=tickers, start=start, end=end)

    try:
        float_snap = price_ingester.load_float_snapshot()
    except FileNotFoundError:
        logger.warning("No float snapshot found; si_pct_float will be NaN")
        float_snap = None

    # ── Build short metrics ───────────────────────────────────────────────────
    logger.info("Building short metrics…")
    short_builder = ShortMetricsBuilder(
        svr_panel=svr_panel,
        si_daily=si_daily,
        price_panel=price_panel,
        float_snapshot=float_snap,
    )
    short_features = short_builder.build()

    # ── Build borrow rate proxy ───────────────────────────────────────────────
    borrow_proxy_cfg = cfg.get("borrow_proxy", {})
    proxy = BorrowRateProxy(
        lendable_fraction=borrow_proxy_cfg.get("lendable_fraction", 0.20)
    )

    if float_snap is not None and not si_daily.empty:
        logger.info("Computing borrow rate proxy…")
        borrow_features = proxy.compute(si_daily=si_daily, float_snapshot=float_snap)
    else:
        logger.warning("Skipping borrow proxy (missing float or SI data)")
        borrow_features = pd.DataFrame()

    # ── Build microstructure features ────────────────────────────────────────
    logger.info("Building microstructure features…")
    ms = MicrostructureFeatures(price_df=prices)
    ms_features = ms.build()

    # ── Merge all features ───────────────────────────────────────────────────
    logger.info("Merging feature panels…")
    features = short_features.copy()
    features["date"] = pd.to_datetime(features["date"])

    if not ms_features.empty:
        ms_features["date"] = pd.to_datetime(ms_features["date"])
        features = features.merge(ms_features, on=["date", "symbol"], how="left")

    if not borrow_features.empty:
        borrow_features["date"] = pd.to_datetime(borrow_features["date"])
        features = features.merge(borrow_features, on=["date", "symbol"], how="left")

    # ── Add forward returns ───────────────────────────────────────────────────
    logger.info("Computing forward returns…")
    for h in [1, 5, 10, 21]:
        fwd = (
            price_panel.shift(-h)
            .apply(lambda col: (col / price_panel[col.name]).apply(lambda x: x) if col.name in price_panel else col)
        )
        # Use log returns
        import numpy as np
        log_fwd = (
            (price_panel.shift(-h) / price_panel)
            .apply(np.log)
            .stack(future_stack=True)
            .reset_index()
        )
        log_fwd.columns = ["date", "symbol", f"ret_fwd_{h}d"]
        log_fwd["date"] = pd.to_datetime(log_fwd["date"])
        features = features.merge(log_fwd, on=["date", "symbol"], how="left")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    logger.info(
        "Feature panel saved: %d rows × %d cols → %s",
        len(features), features.shape[1], out_path,
    )


if __name__ == "__main__":
    main()
