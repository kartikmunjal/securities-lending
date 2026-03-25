"""
Information Coefficient (IC) Analysis.

The IC is the central diagnostic for signal quality in systematic equity
research.  It measures the cross-sectional correlation between the signal
and subsequent realised returns.

Definition
----------
    IC_t = SpearmanCorr_i[ signal_{i,t},  r_{i,t→t+h} ]

We use Spearman (rank) correlation rather than Pearson because:
  1. Robust to return outliers (earnings gaps, M&A, squeezes).
  2. Consistent with rank-based portfolio construction downstream.
  3. Less sensitive to distributional misspecification.

Key statistics
--------------
mean IC          — Expected per-period signal power
ICIR             — IC / std(IC):  signal-to-noise ratio (target > 0.5)
t-statistic      — mean IC / (std(IC)/√T), test H0: IC=0
IC decay curve   — IC at h=1,5,10,21 days: how quickly the signal decays
Multiple-testing — Benjamini-Hochberg FDR correction across signals/horizons

References
----------
Grinold & Kahn (2000) "Active Portfolio Management" Ch. 6.
Fama & MacBeth (1973) "Risk, Return, and Equilibrium". JPE 81(3).
Benjamini & Hochberg (1995) "Controlling the false discovery rate". JRSS-B.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ICResult:
    """Container for IC analysis results for a single signal × horizon."""

    signal_name: str
    horizon: int           # forward return horizon in days
    ic_series: pd.Series   # time series of per-date ICs
    mean_ic: float
    std_ic: float
    icir: float            # mean_ic / std_ic
    t_stat: float          # two-sided t-test vs H0: IC=0
    p_value: float
    n_obs: int             # number of dates with valid IC
    pct_positive: float    # fraction of dates with IC > 0

    def __str__(self) -> str:
        return (
            f"IC({self.signal_name}, h={self.horizon}d): "
            f"mean={self.mean_ic:.4f}  ICIR={self.icir:.2f}  "
            f"t={self.t_stat:.2f}  p={self.p_value:.4f}  "
            f"N={self.n_obs}  pct+={self.pct_positive:.1%}"
        )


class ICAnalyzer:
    """
    Compute IC analysis for one or more signals against forward returns.

    Parameters
    ----------
    signal_panel : pd.DataFrame
        (date × ticker) signal values.  NaN indicates missing data.
    return_panel : pd.DataFrame
        (date × ticker) forward returns.  This must already be shifted so that
        return_panel[t] contains the return from t to t+h.  Use
        `compute_forward_returns()` to prepare this.
    winsor_pct : float
        Percentile for cross-sectional winsorization before IC computation.
        Default 0.01 (1%/99%).
    min_stocks : int
        Minimum number of stocks in a cross-section for the IC to be valid.
        Dates with fewer observations are dropped.
    """

    def __init__(
        self,
        signal_panel: pd.DataFrame,
        return_panel: pd.DataFrame,
        winsor_pct: float = 0.01,
        min_stocks: int = 20,
    ):
        self.signal = signal_panel
        self.returns = return_panel
        self.winsor_pct = winsor_pct
        self.min_stocks = min_stocks

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        horizons: Sequence[int] = (1, 5, 10, 21),
        signal_name: str = "signal",
    ) -> dict[int, ICResult]:
        """
        Compute IC for a single signal at multiple forward-return horizons.

        Parameters
        ----------
        horizons : sequence of int
            Forward return lookforward periods in days.  A separate IC series
            is computed for each.
        signal_name : str
            Label used in ICResult and logging.

        Returns
        -------
        dict mapping horizon → ICResult.
        """
        results: dict[int, ICResult] = {}
        for h in horizons:
            fwd_ret = self._forward_returns(h)
            ic_series = self._compute_ic_series(self.signal, fwd_ret)
            results[h] = self._summarise(ic_series, signal_name, h)
            logger.info(results[h])
        return results

    def run_multiple(
        self,
        signal_panels: dict[str, pd.DataFrame],
        horizons: Sequence[int] = (1, 5, 10, 21),
        fdr_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """
        Run IC analysis for multiple signals and apply FDR correction.

        Parameters
        ----------
        signal_panels : dict {signal_name: (date × ticker) DataFrame}
        horizons : sequence of int
        fdr_threshold : float
            Benjamini-Hochberg false discovery rate threshold.

        Returns
        -------
        DataFrame with one row per (signal, horizon) and columns:
        signal, horizon, mean_ic, std_ic, icir, t_stat, p_value,
        p_value_bh (FDR-corrected), significant_bh.
        """
        records = []
        for name, panel in signal_panels.items():
            analyzer = ICAnalyzer(
                signal_panel=panel,
                return_panel=self.returns,
                winsor_pct=self.winsor_pct,
                min_stocks=self.min_stocks,
            )
            for h, result in analyzer.run(horizons=horizons, signal_name=name).items():
                records.append(
                    {
                        "signal": name,
                        "horizon": h,
                        "mean_ic": result.mean_ic,
                        "std_ic": result.std_ic,
                        "icir": result.icir,
                        "t_stat": result.t_stat,
                        "p_value": result.p_value,
                        "n_obs": result.n_obs,
                        "pct_positive": result.pct_positive,
                    }
                )

        df = pd.DataFrame(records)
        # Benjamini-Hochberg multiple-testing correction
        df = df.sort_values("p_value").reset_index(drop=True)
        m = len(df)
        df["p_value_bh"] = df["p_value"] * m / (df.index + 1)
        df["p_value_bh"] = df["p_value_bh"].clip(upper=1.0)
        df["significant_bh"] = df["p_value_bh"] < fdr_threshold

        logger.info(
            "IC analysis: %d tests, %d significant at BH q=%.2f",
            m, df["significant_bh"].sum(), fdr_threshold,
        )
        return df.sort_values(["signal", "horizon"]).reset_index(drop=True)

    # ── Forward-return helpers ────────────────────────────────────────────────

    def _forward_returns(self, h: int) -> pd.DataFrame:
        """Shift the return panel so row t contains r_{t→t+h}."""
        return self.returns.shift(-h)

    # ── IC computation ────────────────────────────────────────────────────────

    def _compute_ic_series(
        self,
        signal: pd.DataFrame,
        fwd_ret: pd.DataFrame,
    ) -> pd.Series:
        """Compute cross-sectional Spearman IC for every date."""
        # Align indices
        common_dates = signal.index.intersection(fwd_ret.index)
        sig = signal.loc[common_dates]
        ret = fwd_ret.loc[common_dates]

        ic_values: list[float] = []
        valid_dates: list = []

        for date in common_dates:
            s = sig.loc[date].dropna()
            r = ret.loc[date].dropna()
            common_tickers = s.index.intersection(r.index)
            if len(common_tickers) < self.min_stocks:
                continue

            s_clean = s[common_tickers]
            r_clean = r[common_tickers]

            # Winsorize cross-sectionally
            s_clean = self._winsorize(s_clean)
            r_clean = self._winsorize(r_clean)

            rho, _ = stats.spearmanr(s_clean.values, r_clean.values)
            ic_values.append(rho)
            valid_dates.append(date)

        return pd.Series(ic_values, index=valid_dates, name="ic")

    def _winsorize(self, s: pd.Series) -> pd.Series:
        lo = s.quantile(self.winsor_pct)
        hi = s.quantile(1 - self.winsor_pct)
        return s.clip(lower=lo, upper=hi)

    # ── Summarise ─────────────────────────────────────────────────────────────

    @staticmethod
    def _summarise(ic_series: pd.Series, name: str, h: int) -> ICResult:
        clean = ic_series.dropna()
        if len(clean) < 2:
            return ICResult(
                signal_name=name, horizon=h, ic_series=ic_series,
                mean_ic=np.nan, std_ic=np.nan, icir=np.nan,
                t_stat=np.nan, p_value=np.nan, n_obs=len(clean),
                pct_positive=np.nan,
            )
        mean_ic = clean.mean()
        std_ic = clean.std(ddof=1)
        icir = mean_ic / std_ic if std_ic > 0 else np.nan
        t_stat, p_value = stats.ttest_1samp(clean.values, popmean=0)
        pct_pos = (clean > 0).mean()
        return ICResult(
            signal_name=name, horizon=h, ic_series=ic_series,
            mean_ic=mean_ic, std_ic=std_ic, icir=icir,
            t_stat=t_stat, p_value=p_value, n_obs=len(clean),
            pct_positive=pct_pos,
        )

    # ── Static convenience ────────────────────────────────────────────────────

    @staticmethod
    def compute_forward_returns(
        close_panel: pd.DataFrame,
        horizons: Sequence[int] = (1, 5, 10, 21),
    ) -> dict[int, pd.DataFrame]:
        """
        Compute forward log-return panels for multiple horizons.

        Parameters
        ----------
        close_panel : pd.DataFrame
            (date × ticker) adjusted close prices.

        Returns
        -------
        dict {horizon: (date × ticker) forward return DataFrame}.
        The return at row t is log(price_{t+h} / price_t).
        """
        log_price = np.log(close_panel)
        return {h: log_price.shift(-h) - log_price for h in horizons}
