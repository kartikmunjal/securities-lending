#!/usr/bin/env python
"""
Run the full signal analysis pipeline:
  1. IC analysis for all short interest signals
  2. Portfolio quintile sorts (gross and net of estimated costs)
  3. Fama-MacBeth regression with multiple-testing correction
  4. Save result tables and tear-sheet plots to data/results/

Usage
-----
    python scripts/run_analysis.py
    python scripts/run_analysis.py --features data/processed/features.parquet
    python scripts/run_analysis.py --output-dir data/results/run_01
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt

from securities_lending.analysis import ICAnalyzer, PortfolioSorter, FamaMacBeth
from securities_lending.features.retail_attention import (
    RETAIL_INTERACTION_COLS,
    RETAIL_SIGNAL_COLS,
    load_retail_attention_features,
    merge_retail_attention,
)
from securities_lending.utils.config import load_config
from securities_lending.viz import (
    plot_ic_tearsheet,
    plot_portfolio_quintiles,
    plot_fm_coefficients,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Signals we analyse in IC / portfolio sort
_SIGNAL_COLS = ["svr_z20", "svr_trend5", "si_pct_float", "si_dtc", "short_pressure",
                "borrow_rate_bps", "borrow_stress", *RETAIL_SIGNAL_COLS,
                *RETAIL_INTERACTION_COLS]

# Control variables for Fama-MacBeth
_CONTROL_COLS = ["log_mktcap_proxy", "ret_fwd_21d", "realized_vol_20d", *RETAIL_SIGNAL_COLS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run signal analysis pipeline")
    p.add_argument("--features", default="data/processed/features.parquet")
    p.add_argument("--output-dir", default="data/results")
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 21])
    p.add_argument("--n-quantiles", type=int, default=5)
    p.add_argument("--alt-factor-dir", default=None,
                   help="Directory of WSB_*.parquet factor panels from alt-data-equity-signals")
    p.add_argument("--no-retail-interactions", action="store_true",
                   help="Disable borrow/crowding x retail-attention interaction features")
    return p.parse_args()


def pivot_signal(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Pivot a long-format feature DataFrame to (date × ticker)."""
    return df.pivot(index="date", columns="symbol", values=col)


def main() -> None:
    args = parse_args()
    cfg = load_config()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load feature panel ────────────────────────────────────────────────────
    logger.info("Loading features from %s…", args.features)
    features = pd.read_parquet(args.features)
    features["date"] = pd.to_datetime(features["date"])
    features = features.sort_values(["date", "symbol"])

    if args.alt_factor_dir:
        retail = load_retail_attention_features(args.alt_factor_dir)
        features = merge_retail_attention(
            features,
            retail,
            add_interactions=not args.no_retail_interactions,
        )
        logger.info(
            "Merged retail-attention alt-data from %s: %s",
            args.alt_factor_dir,
            [c for c in RETAIL_SIGNAL_COLS + RETAIL_INTERACTION_COLS if c in features.columns],
        )

    # ── Build panels ──────────────────────────────────────────────────────────
    # Primary return: 5-day forward log return (h=5)
    h_primary = 5
    ret_panel = pivot_signal(features, f"ret_fwd_{h_primary}d")
    all_fwd_panels = {
        h: pivot_signal(features, f"ret_fwd_{h}d") for h in args.horizons
        if f"ret_fwd_{h}d" in features.columns
    }

    signal_panels = {}
    for col in _SIGNAL_COLS:
        if col in features.columns:
            panel = pivot_signal(features, col)
            # Drop all-NaN columns and dates
            panel = panel.dropna(how="all", axis=1).dropna(how="all", axis=0)
            if not panel.empty:
                signal_panels[col] = panel

    borrow_panel = pivot_signal(features, "borrow_rate_bps") if "borrow_rate_bps" in features.columns else None

    logger.info(
        "Panels: %d dates, %d tickers, %d signals available",
        len(ret_panel), ret_panel.shape[1], len(signal_panels),
    )

    # ── IC Analysis ───────────────────────────────────────────────────────────
    logger.info("Running IC analysis…")
    ic_summary_rows = []

    for sig_name, sig_panel in signal_panels.items():
        common_ret = ICAnalyzer.compute_forward_returns(
            pivot_signal(features[features["symbol"].isin(sig_panel.columns)], "ret_fwd_5d").reindex(sig_panel.index)
            if False else ret_panel,
            horizons=args.horizons,
        )
        analyzer = ICAnalyzer(signal_panel=sig_panel, return_panel=ret_panel)
        results = analyzer.run(horizons=args.horizons, signal_name=sig_name)

        # Save tear sheet
        fig = plot_ic_tearsheet(results, signal_name=sig_name)
        fig.savefig(out_dir / f"ic_tearsheet_{sig_name}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        for h, res in results.items():
            ic_summary_rows.append(
                {
                    "signal": sig_name, "horizon": h,
                    "mean_ic": res.mean_ic, "icir": res.icir,
                    "t_stat": res.t_stat, "p_value": res.p_value,
                    "n_obs": res.n_obs, "pct_positive": res.pct_positive,
                }
            )

    ic_summary = pd.DataFrame(ic_summary_rows)
    if not ic_summary.empty:
        # BH multiple-testing correction
        m = len(ic_summary)
        ic_summary = ic_summary.sort_values("p_value").reset_index(drop=True)
        ic_summary["p_bh"] = (ic_summary["p_value"] * m / (ic_summary.index + 1)).clip(upper=1.0)
        ic_summary["sig_bh"] = ic_summary["p_bh"] < cfg.get("ic_analysis", {}).get("fdr_threshold", 0.05)
        ic_summary.to_csv(out_dir / "ic_summary.csv", index=False)
        logger.info("IC summary saved. Significant signals (BH): %d/%d",
                    ic_summary["sig_bh"].sum(), len(ic_summary))

    # ── Portfolio Sorts ───────────────────────────────────────────────────────
    logger.info("Running portfolio sorts…")
    sort_results = {}
    for sig_name, sig_panel in signal_panels.items():
        try:
            sorter = PortfolioSorter(
                signal_panel=sig_panel,
                return_panel=ret_panel,
                borrow_rate_panel=borrow_panel,
                n_quantiles=args.n_quantiles,
                holding_period=h_primary,
                tc_bps=cfg.get("portfolio_sorts", {}).get("tc_bps", 5),
            )
            result = sorter.run(
                signal_name=sig_name,
                cost_scenarios=cfg.get("portfolio_sorts", {}).get("cost_scenarios", [5, 10, 20]),
            )
            sort_results[sig_name] = result
            fig = plot_portfolio_quintiles(result)
            fig.savefig(out_dir / f"quintile_sort_{sig_name}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info(
                "Sort (%s): L/S spread=%.2f%% Sharpe=%.2f t=%.2f",
                sig_name, result.ls_spread_ann * 100, result.ls_sharpe, result.ls_t_stat,
            )
        except Exception as exc:
            logger.warning("Portfolio sort failed for %s: %s", sig_name, exc)

    # ── Fama-MacBeth Regression ───────────────────────────────────────────────
    logger.info("Running Fama-MacBeth regressions…")
    fm_results = {}
    available_controls = {
        c: pivot_signal(features, c)
        for c in _CONTROL_COLS
        if c in features.columns
    }
    fm_summary_rows = []
    for sig_name, sig_panel in signal_panels.items():
        try:
            fm = FamaMacBeth(
                forward_return_panel=ret_panel,
                signal_panel=sig_panel,
                control_panels=available_controls,
                nw_lags=cfg.get("fama_macbeth", {}).get("nw_lags", 4),
            )
            res = fm.run(signal_name=sig_name, horizon=h_primary)
            fm_results[sig_name] = res
            incr = fm.compare_incremental(signal_name=sig_name, horizon=h_primary)

            # Signal row
            if sig_name in res.beta_mean.index:
                fm_summary_rows.append(
                    {
                        "signal": sig_name,
                        "beta": res.beta_mean[sig_name],
                        "t_FM": res.fm_tstat.get(sig_name),
                        "t_NW": res.nw_tstat.get(sig_name),
                        "p_NW": res.nw_pvalue.get(sig_name),
                        "mean_R2": res.mean_r2,
                        "delta_R2": incr.get("delta_r2_mean"),
                        "incr_p": incr.get("p_value"),
                        "n_dates": res.n_dates,
                    }
                )
        except Exception as exc:
            logger.warning("FM regression failed for %s: %s", sig_name, exc)

    if fm_results:
        fig = plot_fm_coefficients(fm_results)
        fig.savefig(out_dir / "fm_coefficients.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    if fm_summary_rows:
        pd.DataFrame(fm_summary_rows).to_csv(out_dir / "fm_summary.csv", index=False)

    logger.info("Analysis complete. Results saved to %s", out_dir)

    # ── Print summary table ───────────────────────────────────────────────────
    if not ic_summary.empty:
        print("\n── IC Summary (h=5d, sorted by |ICIR|) ──")
        subset = ic_summary[ic_summary["horizon"] == 5].sort_values("icir", key=abs, ascending=False)
        print(subset[["signal", "mean_ic", "icir", "t_stat", "p_value", "sig_bh"]].to_string(index=False))


if __name__ == "__main__":
    main()
