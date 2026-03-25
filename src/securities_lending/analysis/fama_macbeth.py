"""
Fama-MacBeth (1973) Cross-Sectional Regression.

Procedure
---------
Step 1 — For each date t, run a cross-sectional OLS regression:

    r_{i,t→t+h} = α_t + β_t × signal_{i,t} + γ_t × controls_{i,t} + ε_{i,t}

Step 2 — Average the time series of coefficient estimates:

    β̄ = (1/T) Σ_t β_t

Step 3 — Compute standard errors:

    FM SE:         se_{FM}(β̄) = std(β_t) / √T
    Newey-West SE: se_{NW}(β̄) accounts for autocorrelation in β_t series.

Newey-West SEs are important when the signal is slow-moving (e.g. biweekly
SI) because β_t will be positively autocorrelated — naive FM SEs under-
estimate uncertainty in this case.

Coefficient interpretation
--------------------------
All signals are standardised cross-sectionally (mean 0, std 1) before each
regression so that coefficients represent the expected return per one cross-
sectional standard deviation move in the signal.  This makes coefficients
across different signals directly comparable.

Incremental contribution test
------------------------------
We fit two models:
  M0 — forward return on controls only (Fama-French 3 factors proxied)
  M1 — forward return on controls + signal
Then compare mean cross-sectional R² and test whether signal adds incremental
explanatory power (paired t-test on Δ R²_t = R²_{M1,t} − R²_{M0,t}).

References
----------
Fama & MacBeth (1973) "Risk, Return, and Equilibrium: Empirical Tests". JPE.
Cochrane (2005) "Asset Pricing" Ch. 12.
Newey & West (1987) "A simple, positive semi-definite, heteroskedasticity
    and autocorrelation consistent covariance matrix". Econometrica.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

logger = logging.getLogger(__name__)


@dataclass
class FMResult:
    """Results from a Fama-MacBeth regression."""

    signal_name: str
    horizon: int
    # Time-series of per-date coefficients (T × K+1 DataFrame, incl. intercept)
    beta_series: pd.DataFrame
    # Aggregated estimates
    beta_mean: pd.Series
    fm_tstat: pd.Series     # Fama-MacBeth t-stats (se = std/√T)
    nw_tstat: pd.Series     # Newey-West t-stats
    fm_pvalue: pd.Series
    nw_pvalue: pd.Series
    # Mean cross-sectional R²
    mean_r2: float
    n_dates: int
    n_stocks_avg: float

    def summary(self) -> pd.DataFrame:
        """Return a tidy summary table."""
        return pd.DataFrame(
            {
                "beta": self.beta_mean,
                "t_FM": self.fm_tstat,
                "t_NW": self.nw_tstat,
                "p_FM": self.fm_pvalue,
                "p_NW": self.nw_pvalue,
            }
        )


class FamaMacBeth:
    """
    Run Fama-MacBeth regression for a signal with optional control variables.

    Parameters
    ----------
    forward_return_panel : pd.DataFrame
        (date × ticker) forward returns (already shifted).
    signal_panel : pd.DataFrame
        (date × ticker) signal values.
    control_panels : dict {name: (date × ticker) DataFrame}
        Control variable panels.  Typical controls: log_mktcap, ret_1m,
        ret_12m_skip1m, realized_vol_20d.
    nw_lags : int
        Newey-West lag truncation (default 4 for weekly holding periods).
    min_stocks : int
        Minimum cross-section size to run a regression.
    """

    def __init__(
        self,
        forward_return_panel: pd.DataFrame,
        signal_panel: pd.DataFrame,
        control_panels: dict[str, pd.DataFrame] | None = None,
        nw_lags: int = 4,
        min_stocks: int = 30,
    ):
        self.returns = forward_return_panel
        self.signal = signal_panel
        self.controls = control_panels or {}
        self.nw_lags = nw_lags
        self.min_stocks = min_stocks

    # ── Main regression ───────────────────────────────────────────────────────

    def run(
        self,
        signal_name: str = "signal",
        horizon: int = 5,
    ) -> FMResult:
        """
        Run the full FM regression and return a FMResult.

        The signal is cross-sectionally standardised before each regression.
        Controls are also standardised.
        """
        common_dates = self.returns.index.intersection(self.signal.index)
        betas: dict = {}
        r2s: list[float] = []
        n_stocks: list[int] = []

        for date in common_dates:
            df = self._build_cross_section(date)
            if df is None or len(df) < self.min_stocks:
                continue

            y = df["ret"].values
            X = df.drop(columns=["ret"]).values
            X = sm.add_constant(X, has_constant="add")
            try:
                ols = sm.OLS(y, X).fit()
                betas[date] = ols.params
                r2s.append(ols.rsquared)
                n_stocks.append(len(df))
            except Exception:
                continue

        if not betas:
            raise RuntimeError(f"No FM regressions succeeded for signal '{signal_name}'")

        col_names = ["const"] + list(self.controls.keys()) + [signal_name]
        beta_df = pd.DataFrame(betas, index=col_names).T
        beta_df.index = pd.DatetimeIndex(beta_df.index)

        beta_mean = beta_df.mean()
        beta_std = beta_df.std(ddof=1)
        T = len(beta_df)

        # Fama-MacBeth standard errors
        fm_se = beta_std / np.sqrt(T)
        fm_tstat = beta_mean / fm_se
        fm_pvalue = pd.Series(
            2 * stats.t.sf(np.abs(fm_tstat.values), df=T - 1),
            index=fm_tstat.index,
        )

        # Newey-West standard errors for each coefficient
        nw_tstat, nw_pvalue = self._newey_west_tstats(beta_df, self.nw_lags)

        result = FMResult(
            signal_name=signal_name,
            horizon=horizon,
            beta_series=beta_df,
            beta_mean=beta_mean,
            fm_tstat=fm_tstat,
            nw_tstat=nw_tstat,
            fm_pvalue=fm_pvalue,
            nw_pvalue=nw_pvalue,
            mean_r2=float(np.mean(r2s)) if r2s else np.nan,
            n_dates=T,
            n_stocks_avg=float(np.mean(n_stocks)) if n_stocks else np.nan,
        )
        logger.info(
            "FM (%s, h=%dd): β̄=%+.4f  t_FM=%.2f  t_NW=%.2f  mean_R²=%.4f  T=%d",
            signal_name, horizon,
            float(beta_mean.get(signal_name, np.nan)),
            float(fm_tstat.get(signal_name, np.nan)),
            float(nw_tstat.get(signal_name, np.nan)),
            result.mean_r2, T,
        )
        return result

    def compare_incremental(
        self,
        signal_name: str = "signal",
        horizon: int = 5,
    ) -> dict[str, float]:
        """
        Test whether the signal adds incremental explanatory power beyond controls.

        Fits M0 (controls only) and M1 (controls + signal) and reports:
          * Δ mean R²
          * Paired t-test p-value on Δ R²_t = R²_{M1,t} − R²_{M0,t}
        """
        common_dates = self.returns.index.intersection(self.signal.index)
        delta_r2: list[float] = []

        for date in common_dates:
            df = self._build_cross_section(date)
            if df is None or len(df) < self.min_stocks:
                continue

            y = df["ret"].values
            # M0: controls only
            X0 = sm.add_constant(df.drop(columns=["ret", signal_name], errors="ignore").values)
            # M1: controls + signal
            X1 = sm.add_constant(df.drop(columns=["ret"]).values)

            try:
                r2_m0 = sm.OLS(y, X0).fit().rsquared
                r2_m1 = sm.OLS(y, X1).fit().rsquared
                delta_r2.append(r2_m1 - r2_m0)
            except Exception:
                continue

        if not delta_r2:
            return {"delta_r2": np.nan, "t_stat": np.nan, "p_value": np.nan}

        t_stat, p_value = stats.ttest_1samp(delta_r2, popmean=0)
        return {
            "delta_r2_mean": float(np.mean(delta_r2)),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "n_dates": len(delta_r2),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_cross_section(self, date) -> pd.DataFrame | None:
        """Build a single cross-sectional regression DataFrame for *date*."""
        ret = self.returns.loc[date].dropna()
        sig = self.signal.loc[date].dropna()
        common = ret.index.intersection(sig.index)
        if len(common) < self.min_stocks:
            return None

        data = {"ret": ret[common], signal_name: self._standardize(sig[common])}

        for ctrl_name, ctrl_panel in self.controls.items():
            if date not in ctrl_panel.index:
                continue
            ctrl = ctrl_panel.loc[date].reindex(common).dropna()
            if len(ctrl) < self.min_stocks:
                continue
            data[ctrl_name] = self._standardize(ctrl.reindex(common))

        df = pd.DataFrame(data, index=common).dropna()
        signal_name_key = [k for k in data if k != "ret"][0]
        # Keep only rows where all columns are present
        return df if len(df) >= self.min_stocks else None

    @staticmethod
    def _standardize(s: pd.Series) -> pd.Series:
        """Cross-sectional z-score."""
        mu, sigma = s.mean(), s.std(ddof=1)
        return (s - mu) / sigma if sigma > 0 else s - mu

    @staticmethod
    def _newey_west_tstats(
        beta_df: pd.DataFrame,
        n_lags: int,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Compute Newey-West t-statistics for each column of beta_df.

        Uses statsmodels OLS with HAC covariance to extract NW standard errors.
        Each β_t series is regressed on a constant; the NW SE for the constant
        is the NW SE for the mean.
        """
        tstats, pvalues = {}, {}
        T = len(beta_df)
        for col in beta_df.columns:
            y = beta_df[col].dropna().values
            X = np.ones((len(y), 1))
            try:
                res = sm.OLS(y, X).fit(
                    cov_type="HAC",
                    cov_kwds={"maxlags": n_lags, "use_correction": True},
                )
                tstats[col] = float(res.tvalues[0])
                pvalues[col] = float(res.pvalues[0])
            except Exception:
                tstats[col] = np.nan
                pvalues[col] = np.nan
        return pd.Series(tstats), pd.Series(pvalues)
