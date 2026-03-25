"""
Borrow Rate Data Interface.

Production data sources (not free)
------------------------------------
Actual borrow rates (daily fee rates charged to short sellers) are sourced from:

  * **S3 Partners** — leading independent short-side data vendor; provides
    daily borrow fees per stock via Bloomberg terminal (BBG: S3 <GO>) or API.
  * **DataLend / EquiLend** — institutional prime brokerage analytics platform.
  * **IBKR Securities Lending Dashboard** — Interactive Brokers' public tool
    showing indicative borrow rates for IBKR's lending pool
    (https://ibkr.com/seclending).  Rates are specific to IBKR's inventory,
    not market-wide.
  * **Markit Securities Finance (IHS Markit)** — industry standard for
    institutional borrow analytics.

This module provides:
  1. An ``IBKRBorrowScraper`` that downloads indicative rates from IBKR's
     public lending dashboard (no auth required; limited to current snapshot).
  2. A ``BorrowRateProxy`` class that estimates historical borrow rates from
     observable FINRA short interest data (see features/borrow_proxy.py for
     the full implementation).

The proxy is the primary data path for this open-source pipeline.  It is
explicitly documented as a proxy, not actual transaction rates.

References
----------
D'Avolio (2002) "The Market for Borrowing Stock". JFE.
Drechsler & Drechsler (2016) "The Shorting Premium and Asset Pricing Anomalies".
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# IBKR's publicly accessible securities lending dashboard
_IBKR_API_URL = (
    "https://www.interactivebrokers.com/en/trading/securities-financing.php"
)


class IBKRBorrowScraper:
    """
    Fetch indicative borrow rates from IBKR's Securities Lending dashboard.

    IBKR publishes indicative (not firm) borrow fee rates for their lending
    inventory via a public-facing tool.  This provides a real-data sanity check
    on the proxy model for currently traded names.

    Note: IBKR's rates reflect their internal inventory, which may differ from
    market-wide rates.  For research purposes this is a directional check only.
    """

    def __init__(self, cache_dir: str | Path = "data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (research)",
                "Accept": "application/json",
            }
        )

    def fetch_snapshot(self, tickers: list[str]) -> pd.DataFrame:
        """
        Query IBKR for current indicative borrow rates for *tickers*.

        Returns a DataFrame with columns: symbol, ibkr_fee_rate_bps,
        ibkr_availability.  Returns an empty DataFrame if the API is
        unavailable (to keep the pipeline runnable without IBKR access).
        """
        # IBKR does not provide a simple open REST endpoint for batch queries.
        # In practice this would be implemented against their Trader Workstation
        # (TWS) API or the IBKR Client Portal API.
        # For now, we return an empty frame and note the integration point.
        logger.info(
            "IBKR borrow rate API not implemented in this open-source version.  "
            "Integrate via IBKR TWS API (ib_insync) or Client Portal API."
        )
        return pd.DataFrame(columns=["symbol", "ibkr_fee_rate_bps", "ibkr_availability"])
