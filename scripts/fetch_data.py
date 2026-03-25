#!/usr/bin/env python
"""
Download raw data: FINRA Reg SHO short sale volume, FINRA short interest,
price/volume data, and float snapshots.

Usage
-----
    python scripts/fetch_data.py                       # uses configs/universe.yaml defaults
    python scripts/fetch_data.py --start 2022-01-01    # custom start date
    python scripts/fetch_data.py --tickers GME AMC     # subset of tickers
    python scripts/fetch_data.py --skip-prices         # FINRA data only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from securities_lending.ingestion import (
    FINRARegSHOIngester,
    FINRAShortInterestIngester,
    PriceIngester,
)
from securities_lending.utils.config import load_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch all raw data for the securities-lending pipeline")
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD); default from universe.yaml")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD); default today")
    p.add_argument("--tickers", nargs="+", default=None, help="Override ticker list")
    p.add_argument("--skip-regsho", action="store_true", help="Skip FINRA Reg SHO download")
    p.add_argument("--skip-si", action="store_true", help="Skip FINRA short interest download")
    p.add_argument("--skip-prices", action="store_true", help="Skip price data download")
    p.add_argument("--skip-float", action="store_true", help="Skip float snapshot download")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    uni = load_universe()

    start = args.start or uni["data"]["start_date"]
    end = args.end  # None → today
    tickers = args.tickers or uni["universe"]["tickers"]

    logger.info("Fetching data for %d tickers from %s to %s", len(tickers), start, end or "today")

    # ── FINRA Reg SHO daily short sale volume ─────────────────────────────────
    if not args.skip_regsho:
        logger.info("── FINRA Reg SHO daily short sale volume ──")
        regsho = FINRARegSHOIngester(
            cache_dir="data/raw/finra_regsho",
            tickers=tickers,
        )
        regsho.download(start=start, end=end)

    # ── FINRA biweekly short interest ─────────────────────────────────────────
    if not args.skip_si:
        logger.info("── FINRA biweekly short interest ──")
        si_ingester = FINRAShortInterestIngester(
            cache_dir="data/raw/finra_short_interest",
            tickers=tickers,
        )
        si_ingester.download(start=start, end=end)

    # ── Price / volume data ──────────────────────────────────────────────────
    if not args.skip_prices:
        logger.info("── yfinance prices ──")
        price_ingester = PriceIngester(cache_dir="data/raw/prices")
        price_ingester.download(
            tickers=tickers,
            start=start,
            end=end,
        )

    # ── Float snapshot ────────────────────────────────────────────────────────
    if not args.skip_float:
        logger.info("── Float / fundamental snapshot ──")
        price_ingester = PriceIngester(cache_dir="data/raw/prices")
        price_ingester.download_float_snapshot(tickers=tickers)

    logger.info("Data fetch complete.")


if __name__ == "__main__":
    main()
