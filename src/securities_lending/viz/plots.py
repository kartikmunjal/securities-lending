"""
Visualization utilities for signal tear sheets and model diagnostics.

All plot functions return matplotlib Figure objects so the caller controls
saving behaviour (avoid side effects in library code).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from ..analysis.ic_analysis import ICResult
from ..analysis.portfolio_sorts import SortResult

# Consistent style
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F9FA",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)
_BLUE = "#2B6CB0"
_RED = "#C53030"
_GREEN = "#276749"
_GREY = "#718096"


def plot_ic_tearsheet(
    ic_results: dict[int, ICResult],
    signal_name: str = "Signal",
) -> plt.Figure:
    """
    Three-panel IC tear sheet:
      (1) IC time series with rolling mean
      (2) IC distribution histogram
      (3) IC decay curve across horizons
    """
    horizons = sorted(ic_results.keys())
    primary_h = horizons[0]
    ic_series = ic_results[primary_h].ic_series

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    ax_ts = fig.add_subplot(gs[0, :])    # top row: full width
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_decay = fig.add_subplot(gs[1, 1])

    # ── IC Time Series ────────────────────────────────────────────────────────
    ic_series.plot(ax=ax_ts, color=_GREY, alpha=0.5, linewidth=0.8, label="Daily IC")
    ic_series.rolling(21, min_periods=10).mean().plot(
        ax=ax_ts, color=_BLUE, linewidth=1.8, label="21-day rolling mean"
    )
    ax_ts.axhline(0, color="black", linewidth=0.8, linestyle="--")
    mean_ic = ic_results[primary_h].mean_ic
    ax_ts.axhline(mean_ic, color=_RED, linewidth=1, linestyle=":", label=f"Mean IC={mean_ic:.4f}")
    ax_ts.set_title(f"{signal_name} — IC Time Series (h={primary_h}d)  "
                    f"ICIR={ic_results[primary_h].icir:.2f}  "
                    f"t={ic_results[primary_h].t_stat:.2f}")
    ax_ts.legend(fontsize=9)
    ax_ts.set_ylabel("Spearman IC")

    # ── IC Distribution ───────────────────────────────────────────────────────
    ic_clean = ic_series.dropna()
    ax_hist.hist(ic_clean, bins=40, color=_BLUE, alpha=0.7, edgecolor="white")
    ax_hist.axvline(ic_clean.mean(), color=_RED, linewidth=1.5, label=f"mean={ic_clean.mean():.4f}")
    ax_hist.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax_hist.set_title(f"IC Distribution (h={primary_h}d)")
    ax_hist.set_xlabel("Spearman IC")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend(fontsize=9)

    # ── IC Decay Curve ────────────────────────────────────────────────────────
    decay_means = [ic_results[h].mean_ic for h in horizons]
    decay_icirs = [ic_results[h].icir for h in horizons]

    ax_decay.bar(
        [str(h) for h in horizons], decay_means,
        color=[_BLUE if m > 0 else _RED for m in decay_means],
        alpha=0.8, edgecolor="white",
    )
    ax_decay.axhline(0, color="black", linewidth=0.8)
    ax_decay2 = ax_decay.twinx()
    ax_decay2.plot(
        [str(h) for h in horizons], decay_icirs,
        "o--", color=_GREY, linewidth=1.5, markersize=6, label="ICIR"
    )
    ax_decay2.set_ylabel("ICIR", color=_GREY)
    ax_decay.set_title("IC Decay vs Forward Horizon")
    ax_decay.set_xlabel("Forward Horizon (days)")
    ax_decay.set_ylabel("Mean IC")

    fig.suptitle(f"{signal_name} — IC Tear Sheet", fontsize=13, fontweight="bold", y=1.01)
    return fig


def plot_portfolio_quintiles(
    result: SortResult,
    show_net_costs: bool = True,
) -> plt.Figure:
    """
    Two-panel: (1) annualised returns by quintile; (2) net spread vs cost scenarios.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Quintile Returns ──────────────────────────────────────────────────────
    ax = axes[0]
    labels = [f"Q{i+1}" for i in range(result.n_quantiles)]
    colors = [_RED if i == 0 else _GREEN if i == result.n_quantiles - 1 else _BLUE
              for i in range(result.n_quantiles)]
    bars = ax.bar(labels, result.quantile_returns.values * 100, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{result.signal_name} — Quintile Returns\n"
                 f"L/S Spread: {result.ls_spread_ann*100:.1f}%/yr  "
                 f"Sharpe={result.ls_sharpe:.2f}  t={result.ls_t_stat:.2f}")
    ax.set_ylabel("Annualised Return (%)")
    ax.set_xlabel("Quantile (Q1=low signal, Q5=high)")

    # Annotate monotonicity
    ax.text(0.02, 0.97, f"Monotonicity: {result.monotonicity:.0%}",
            transform=ax.transAxes, fontsize=9, va="top", color=_GREY)

    # ── Net Spread vs TC ──────────────────────────────────────────────────────
    if show_net_costs and result.net_spreads:
        ax2 = axes[1]
        tc_bps = list(result.net_spreads.keys())
        net_vals = [v * 100 for v in result.net_spreads.values()]
        colors_net = [_GREEN if v > 0 else _RED for v in net_vals]
        ax2.bar([f"{tc}bps" for tc in tc_bps], net_vals, color=colors_net, alpha=0.85, edgecolor="white")
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.axhline(result.ls_spread_ann * 100, color=_BLUE, linewidth=1, linestyle="--", label="Gross spread")
        ax2.set_title("Net L/S Spread vs Transaction Cost Scenario")
        ax2.set_xlabel("One-way TC (bps)")
        ax2.set_ylabel("Net Annualised Spread (%)")
        ax2.legend(fontsize=9)

    fig.tight_layout()
    return fig


def plot_borrow_proxy_calibration(
    proxy_df: pd.DataFrame,
    known_htb: dict[str, float] | None = None,
) -> plt.Figure:
    """
    Scatter: proxy borrow rate vs utilisation, with known HTB tickers annotated.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Distribution by utilisation bucket ───────────────────────────────────
    ax = axes[0]
    latest = proxy_df.sort_values("date").groupby("symbol").last().reset_index()
    sc = ax.scatter(
        latest["utilisation"] * 100,
        latest["borrow_rate_bps"],
        c=latest["borrow_rate_bps"],
        cmap="RdYlGn_r",
        s=30,
        alpha=0.6,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Borrow rate (bps)")
    if known_htb:
        for sym, rate in known_htb.items():
            row = latest[latest["symbol"] == sym]
            if not row.empty:
                ax.annotate(
                    sym, (row["utilisation"].values[0] * 100, row["borrow_rate_bps"].values[0]),
                    fontsize=8, color="black",
                    xytext=(5, 5), textcoords="offset points",
                )
    ax.set_xlabel("Estimated Utilisation (%)")
    ax.set_ylabel("Proxy Borrow Rate (bps/yr)")
    ax.set_yscale("log")
    ax.set_title("Borrow Rate Proxy vs Utilisation (latest cross-section)")

    # ── Cross-sectional distribution of borrow rates ──────────────────────────
    ax2 = axes[1]
    latest["borrow_tier"] = pd.cut(
        latest["borrow_rate_bps"],
        bins=[0, 50, 150, 500, 2000, 10000],
        labels=["ETB (<50)", "Moderate (50–150)", "Elevated (150–500)", "HTB (500–2000)", "Special (>2000)"],
    )
    latest["borrow_tier"].value_counts().sort_index().plot(
        kind="barh", ax=ax2, color=_BLUE, alpha=0.8, edgecolor="white"
    )
    ax2.set_title("Universe by Borrow Tier (latest)")
    ax2.set_xlabel("Number of Stocks")

    fig.tight_layout()
    return fig


def plot_fm_coefficients(
    fm_results: dict[str, "FMResult"],
    use_nw: bool = True,
) -> plt.Figure:
    """
    Bar chart of FM coefficients with confidence intervals for each signal.
    """
    from ..analysis.fama_macbeth import FMResult

    records = []
    for name, res in fm_results.items():
        # Extract coefficient for the signal itself (last column, not controls)
        if name in res.beta_mean.index:
            b = res.beta_mean[name]
            t = res.nw_tstat[name] if use_nw else res.fm_tstat[name]
            se = abs(b / t) if t != 0 else np.nan
            records.append({"signal": name, "beta": b, "se": se, "t": t})

    if not records:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No results to plot", ha="center")
        return fig

    df = pd.DataFrame(records).sort_values("beta", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.6)))
    colors = [_GREEN if b > 0 else _RED for b in df["beta"]]
    bars = ax.barh(df["signal"], df["beta"], color=colors, alpha=0.8, edgecolor="white")
    ax.errorbar(df["beta"], df["signal"], xerr=1.96 * df["se"],
                fmt="none", color="black", linewidth=1.5, capsize=4)
    ax.axvline(0, color="black", linewidth=0.8)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["beta"] + (0.0002 if row["beta"] >= 0 else -0.0002),
                i, f"t={row['t']:.2f}", va="center",
                ha="left" if row["beta"] >= 0 else "right", fontsize=8)
    se_label = "Newey-West" if use_nw else "Fama-MacBeth"
    ax.set_title(f"FM Signal Coefficients (±1.96 SE, {se_label})")
    ax.set_xlabel("Standardised coefficient (return per 1σ signal move)")
    fig.tight_layout()
    return fig


def plot_squeeze_shap(shap_values, feature_names: list[str]) -> plt.Figure:
    """
    SHAP beeswarm summary plot for the squeeze detector.
    Requires shap>=0.42.
    """
    try:
        import shap
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        fig.suptitle("Squeeze Detector — SHAP Feature Importance", fontsize=12)
        return plt.gcf()
    except ImportError:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Install shap>=0.42 for SHAP plots", ha="center")
        return fig
