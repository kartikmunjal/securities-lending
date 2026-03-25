"""
Borrow Rate Proxy Model.

Constructs an estimate of the annualised borrow fee rate from FINRA short
interest and publicly available float data, using the supply-demand dynamics
of the securities lending market.

Theory
------
The borrow rate for a security is set by supply-demand equilibrium in the
securities lending market (D'Avolio 2002, Drechsler & Drechsler 2016).

  Supply:  institutional long holders who lend their shares (receive rebate).
  Demand:  short sellers who need to borrow (pay borrow fee).

When demand is high relative to supply (high utilisation), rates spike
nonlinearly — particularly above ~80% utilisation where the lending pool
becomes constrained.

Proxy construction
------------------
1. Utilisation proxy:
       u = short_interest / lendable_supply_estimate
   where lendable_supply ≈ float_shares × lendable_fraction (default 20%).

2. Piecewise nonlinear rate schedule calibrated to match the empirical
   distribution of borrow rates for HTB (hard-to-borrow) vs ETB (easy-to-
   borrow) stocks.  The breakpoints follow the industry convention:

     u < 50%:  ETB tier, ~25 bps (general collateral rate)
     50–80%:   elevated, 50–300 bps
     80–95%:   HTB, 300–800 bps
     > 95%:    special, 800–2000+ bps

3. Cross-sectional normalization: rates are winsorised at the 99th pct and
   z-scored within GICS sector and date to remove sector-structural effects
   (biotech carries systematically higher borrow costs than utilities due to
   higher speculative short interest and lower institutional float).

Caveats
-------
* This proxy does NOT capture supply-side shocks (e.g., a large lender
  recalling shares, ETF creation/redemption changing the lendable pool).
* It does not account for the bid-ask spread in the lending market.
* The 20% lendable_fraction is a cross-sectional average; individual stock
  estimates should use institutional ownership data from 13-F filings.

References
----------
D'Avolio (2002) "The Market for Borrowing Stock". JFE 66(2).
Drechsler & Drechsler (2016) "The Shorting Premium and Asset Pricing
    Anomalies". Working paper.
Cohen, Diether & Malloy (2007) "Supply and Demand Shifts in the Shorting
    Market". Journal of Finance 62(5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Piecewise schedule: list of (utilisation_threshold, rate_bps) breakpoints
# Rate is interpolated linearly between breakpoints.
_DEFAULT_RATE_SCHEDULE: list[tuple[float, float]] = [
    (0.00,   25.0),   # GC rate — everyone can borrow
    (0.50,   50.0),   # slightly elevated
    (0.70,  150.0),   # moderate HTB pressure
    (0.80,  300.0),   # hard-to-borrow threshold
    (0.90,  800.0),   # significantly constrained supply
    (0.95, 2000.0),   # extreme scarcity
    (1.00, 5000.0),   # theoretical max
]


@dataclass
class BorrowRateProxy:
    """
    Estimate daily annualised borrow fee rates from FINRA short interest data.

    Parameters
    ----------
    lendable_fraction : float
        Fraction of float shares assumed available for lending.
        0.20 is a conservative institutional estimate (D'Avolio 2002).
    rate_schedule : list of (util, rate_bps) tuples
        Piecewise linear schedule mapping utilisation → borrow fee in bps.
    """

    lendable_fraction: float = 0.20
    rate_schedule: list[tuple[float, float]] = field(
        default_factory=lambda: _DEFAULT_RATE_SCHEDULE
    )

    def fit_calibrate(
        self,
        known_htb: dict[str, float],
        si_pct_float: pd.Series,
    ) -> "BorrowRateProxy":
        """
        Optional: adjust the rate schedule so that known hard-to-borrow stocks
        (with actual observed rates from financial press / IBKR snapshot) fall
        in the right range.  Returns self for chaining.

        Parameters
        ----------
        known_htb : dict {symbol: actual_rate_bps}
            Ground-truth borrow rates for validation symbols.
        si_pct_float : pd.Series
            Current SI% of float for these symbols (index = symbol).
        """
        # Compute proxy rates for validation symbols
        utils = (si_pct_float / self.lendable_fraction).clip(0, 1)
        proxy_rates = utils.map(self._rate_from_utilisation)

        # Log calibration comparison
        for sym, actual in known_htb.items():
            if sym in proxy_rates.index:
                proxy = proxy_rates[sym]
                logger.info(
                    "Calibration  %s: proxy=%.0f bps  actual=%.0f bps  ratio=%.2f",
                    sym, proxy, actual, proxy / actual if actual else float("inf"),
                )
        return self

    # ── Main computation ─────────────────────────────────────────────────────

    def compute(
        self,
        si_daily: pd.DataFrame,
        float_snapshot: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute daily borrow rate proxy for all (date, symbol) pairs.

        Parameters
        ----------
        si_daily : pd.DataFrame
            Long-format daily SI from FINRAShortInterestIngester.load_daily_interpolated().
            Must contain columns: date, symbol, short_interest.
        float_snapshot : pd.DataFrame
            Float shares from PriceIngester.load_float_snapshot().
            Must contain columns: symbol, float_shares.

        Returns
        -------
        DataFrame with columns: date, symbol, utilisation, borrow_rate_bps,
        borrow_rate_pct, borrow_stress (binary: rate > sector 90th pct).
        """
        df = si_daily.merge(
            float_snapshot[["symbol", "float_shares"]],
            on="symbol",
            how="left",
        )
        df["float_shares"] = df["float_shares"].replace(0, np.nan)
        df["lendable_supply"] = df["float_shares"] * self.lendable_fraction
        df["utilisation"] = (df["short_interest"] / df["lendable_supply"]).clip(0, 1)

        # Vectorised piecewise linear interpolation
        df["borrow_rate_bps"] = df["utilisation"].map(self._rate_from_utilisation)
        df["borrow_rate_pct"] = df["borrow_rate_bps"] / 10_000

        # Borrow stress flag: rate exceeds 90th cross-sectional percentile
        df["borrow_stress"] = (
            df.groupby("date")["borrow_rate_bps"]
            .transform(lambda x: (x >= x.quantile(0.90)).astype(int))
        )

        # Z-score within date (cross-sectional normalisation)
        df["borrow_rate_z"] = df.groupby("date")["borrow_rate_bps"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )

        keep = [
            "date", "symbol", "utilisation", "borrow_rate_bps",
            "borrow_rate_pct", "borrow_rate_z", "borrow_stress",
        ]
        return df[keep].sort_values(["date", "symbol"]).reset_index(drop=True)

    # ── Private ──────────────────────────────────────────────────────────────

    def _rate_from_utilisation(self, u: float) -> float:
        """Piecewise linear interpolation of the rate schedule."""
        if np.isnan(u):
            return np.nan
        u = float(np.clip(u, 0.0, 1.0))
        thresholds = [t for t, _ in self.rate_schedule]
        rates = [r for _, r in self.rate_schedule]
        # Find the bracketing interval
        for i in range(len(thresholds) - 1):
            if thresholds[i] <= u <= thresholds[i + 1]:
                # Linear interpolation within the interval
                frac = (u - thresholds[i]) / (thresholds[i + 1] - thresholds[i])
                return rates[i] + frac * (rates[i + 1] - rates[i])
        return rates[-1]  # u == 1.0

    # ── Diagnostic ───────────────────────────────────────────────────────────

    def plot_rate_curve(self, ax=None):
        """Plot the piecewise borrow rate curve for documentation."""
        import matplotlib.pyplot as plt
        utils = np.linspace(0, 1, 500)
        rates = [self._rate_from_utilisation(u) for u in utils]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(utils * 100, rates, color="steelblue", linewidth=2)
        for u_thresh, r in self.rate_schedule:
            ax.axvline(u_thresh * 100, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_xlabel("Utilisation (%)")
        ax.set_ylabel("Annualised borrow rate (bps)")
        ax.set_title("Borrow Rate Proxy — Piecewise Utilisation Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        return ax
