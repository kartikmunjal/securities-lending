"""
Portfolio sorts for signal evaluation.

Univariate and bivariate cross-sectional portfolio sorts, following the
methodology of Fama & French (1992, 1993) and the empirical asset pricing
literature on short selling (Asquith, Pathak & Ritter 2005).

The long-short portfolio constructed here is *hypothetical* — it does not
account for execution, short availability constraints, or portfolio-level
leverage limits.  The spread is a diagnostic for whether the signal contains
return-predictive information, not a trading P&L estimate.

Net-of-cost analysis
--------------------
For the short leg, the full cost stack includes:
  * Borrow rate (from proxy model, in bps per year)
  * Market impact (estimated via sqrt-law: impact ∝ sqrt(participation))
  * Commission (modelled as constant bps)
We compute both gross spread and net spread at several cost scenarios.

References
----------
Fama & French (1992) "The cross-section of expected stock returns". JF.
Asquith, Pathak & Ritter (2005) "Short interest, institutional ownership,
    and stock returns". JFE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SortResult:
    """Quantile sort results for a single signal."""

    signal_name: str
    n_quantiles: int
    # Mean return per quantile (annualised, %)
    quantile_returns: pd.Series
    # Sharpe per quantile
    quantile_sharpe: pd.Series
    # L/S spread (Q_top − Q_bottom)
    ls_spread_ann: float
    ls_sharpe: float
    ls_t_stat: float
    ls_p_value: float
    # Net spreads at different cost scenarios (bps one-way)
    net_spreads: dict[int, float]
    # Monotonicity score: fraction of adjacent quantile pairs where return increases
    monotonicity: float


class PortfolioSorter:
    """
    Compute quantile portfolio sorts for a given signal.

    Parameters
    ----------
    signal_panel : pd.DataFrame
        (date × ticker) signal values.  Sorted on each date; NaN excluded.
    return_panel : pd.DataFrame
        (date × ticker) forward returns.
    borrow_rate_panel : pd.DataFrame | None
        (date × ticker) annualised borrow rate in bps.  Used to compute
        net-of-borrow-cost short-leg returns.
    n_quantiles : int
        Number of quantile portfolios (5 = quintiles).
    holding_period : int
        Days the portfolio is held before rebalancing.
    rebalance_freq : str
        Pandas offset alias for rebalancing frequency ('W', 'M', 'B').
    tc_bps : float
        One-way transaction cost in bps (commission + half-spread estimate).
    """

    def __init__(
        self,
        signal_panel: pd.DataFrame,
        return_panel: pd.DataFrame,
        borrow_rate_panel: pd.DataFrame | None = None,
        n_quantiles: int = 5,
        holding_period: int = 5,
        rebalance_freq: str = "W",
        tc_bps: float = 5.0,
    ):
        self.signal = signal_panel
        self.returns = return_panel
        self.borrow_rate = borrow_rate_panel
        self.n = n_quantiles
        self.holding_period = holding_period
        self.rebalance_freq = rebalance_freq
        self.tc_bps = tc_bps / 10_000

    # ── Main computation ─────────────────────────────────────────────────────

    def run(
        self,
        signal_name: str = "signal",
        cost_scenarios: list[int] | None = None,
    ) -> SortResult:
        """
        Run the portfolio sort and return a SortResult.

        Parameters
        ----------
        signal_name : str
            Label for the result.
        cost_scenarios : list of int
            One-way TC scenarios in bps for net-spread computation.
        """
        if cost_scenarios is None:
            cost_scenarios = [5, 10, 20]

        # Compute quantile assignment for each (date, ticker)
        quant_panel = self._assign_quantiles()

        # Compute forward returns
        fwd_ret = self.returns.shift(-self.holding_period)

        # Equal-weighted return per quantile per date
        quantile_ret_series: dict[int, pd.Series] = {}
        for q in range(1, self.n + 1):
            # Mask: 1 where quant == q, else NaN
            mask = quant_panel.where(quant_panel == q)
            # Equal-weight within quantile: mean return across tickers in the quantile
            quantile_ret_series[q] = fwd_ret.where(mask.notna()).mean(axis=1).dropna()

        # Build return DataFrame indexed by date
        ret_df = pd.DataFrame(quantile_ret_series)
        ret_df.columns = [f"Q{q}" for q in range(1, self.n + 1)]
        ret_df = ret_df.dropna(how="all")

        # Annualise (holding_period-day returns)
        ann_factor = 252 / self.holding_period

        qret_ann = ret_df.mean() * ann_factor
        qsharpe = (ret_df.mean() / ret_df.std(ddof=1)) * np.sqrt(ann_factor)

        # Long-short: long highest quantile, short lowest
        ls_ret = ret_df[f"Q{self.n}"] - ret_df["Q1"]
        ls_ann = ls_ret.mean() * ann_factor
        ls_sharpe = (ls_ret.mean() / ls_ret.std(ddof=1)) * np.sqrt(ann_factor)
        t_stat, p_value = stats.ttest_1samp(ls_ret.dropna().values, popmean=0)

        # Net spreads: subtract round-trip TC from both legs
        # Short leg additionally bears borrow cost
        net_spreads: dict[int, float] = {}
        for tc in cost_scenarios:
            tc_frac = tc / 10_000
            # Round-trip cost applied to both legs (enter + exit)
            gross = ls_ann
            tc_drag = 2 * tc_frac * ann_factor  # enter + exit per rebalance
            borrow_drag = self._estimate_avg_borrow_cost(quant_panel)
            net_spreads[tc] = gross - tc_drag - borrow_drag

        # Monotonicity: fraction of adjacent pairs where Q_{k+1} > Q_k
        adjacent_pairs = self.n - 1
        n_mono = sum(
            1 for k in range(1, self.n)
            if qret_ann[f"Q{k+1}"] > qret_ann[f"Q{k}"]
        )
        monotonicity = n_mono / adjacent_pairs

        return SortResult(
            signal_name=signal_name,
            n_quantiles=self.n,
            quantile_returns=qret_ann,
            quantile_sharpe=qsharpe,
            ls_spread_ann=ls_ann,
            ls_sharpe=ls_sharpe,
            ls_t_stat=t_stat,
            ls_p_value=p_value,
            net_spreads=net_spreads,
            monotonicity=monotonicity,
        )

    # ── Bivariate sort ───────────────────────────────────────────────────────

    def run_bivariate(
        self,
        control_panel: pd.DataFrame,
        control_name: str = "size",
        signal_name: str = "signal",
    ) -> pd.DataFrame:
        """
        Double sort: first on control, then on signal within control buckets.

        Returns a (n_control × n_signal) DataFrame of annualised mean returns.
        This tests whether the signal adds value independently of the control.
        """
        fwd_ret = self.returns.shift(-self.holding_period)

        # Rank control (e.g. market cap) into n quintiles
        ctrl_quant = self._rank_to_quantiles(control_panel, self.n)
        sig_quant = self._assign_quantiles()

        results = {}
        for cq in range(1, self.n + 1):
            row = {}
            for sq in range(1, self.n + 1):
                in_cell = (ctrl_quant == cq) & (sig_quant == sq)
                cell_ret = fwd_ret.where(in_cell).stack(future_stack=True)
                ann = cell_ret.mean() * 252 / self.holding_period if len(cell_ret) > 0 else np.nan
                row[f"sig_Q{sq}"] = ann
            results[f"{control_name}_Q{cq}"] = row

        df = pd.DataFrame(results).T
        df.index.name = control_name
        df.columns.name = signal_name
        return df

    # ── Private helpers ──────────────────────────────────────────────────────

    def _assign_quantiles(self) -> pd.DataFrame:
        """Assign quantile labels (1=lowest, n=highest) cross-sectionally per date."""
        return self.signal.apply(
            lambda row: pd.qcut(row.dropna().rank(method="first"), self.n,
                                labels=False) + 1,
            axis=1,
            result_type="broadcast",
        )

    @staticmethod
    def _rank_to_quantiles(panel: pd.DataFrame, n: int) -> pd.DataFrame:
        return panel.apply(
            lambda row: pd.qcut(row.dropna().rank(method="first"), n,
                                labels=False) + 1,
            axis=1,
            result_type="broadcast",
        )

    def _estimate_avg_borrow_cost(self, quant_panel: pd.DataFrame) -> float:
        """Estimate average annualised borrow drag on the short leg (Q1)."""
        if self.borrow_rate is None:
            return 0.0
        # Average borrow rate for stocks in Q1 (short leg)
        short_mask = quant_panel == 1
        short_borrow = self.borrow_rate.where(short_mask)
        avg_bps = short_borrow.stack(future_stack=True).mean()
        if np.isnan(avg_bps):
            return 0.0
        return avg_bps / 10_000  # convert bps to fraction, already annualised
