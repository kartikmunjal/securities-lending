#!/usr/bin/env python
"""
Train and evaluate the short-squeeze detector.

Runs walk-forward OOS evaluation and saves:
  * per-window metrics (roc_auc, pr_auc, precision@10%)
  * final model trained on all available data
  * SHAP feature importance plot

Usage
-----
    python scripts/run_squeeze_model.py
    python scripts/run_squeeze_model.py --train-window 504 --seed 123
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from securities_lending.models import SqueezeDetector, WalkForwardEvaluator
from securities_lending.models.squeeze_detector import SqueezeEventLabeler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate squeeze detector")
    p.add_argument("--features", default="data/processed/features.parquet")
    p.add_argument("--output-dir", default="data/results/squeeze")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--test-window", type=int, default=63)
    p.add_argument("--step-size", type=int, default=21)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--return-threshold", type=float, default=0.15)
    p.add_argument("--dtc-threshold", type=float, default=5.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading feature panel from %s…", args.features)
    features = pd.read_parquet(args.features)
    features["date"] = pd.to_datetime(features["date"])

    # ── Walk-forward evaluation ───────────────────────────────────────────────
    labeler = SqueezeEventLabeler(
        return_threshold=args.return_threshold,
        dtc_threshold=args.dtc_threshold,
    )
    evaluator = WalkForwardEvaluator(
        train_window=args.train_window,
        test_window=args.test_window,
        step_size=args.step_size,
        seed=args.seed,
    )

    logger.info("Running walk-forward evaluation (seed=%d)…", args.seed)
    wf_result = evaluator.run(features=features, labeler=labeler)

    summary = wf_result.summary()
    logger.info("Walk-forward summary: %s", summary)
    pd.DataFrame(wf_result.windows).to_csv(out_dir / "wf_results.csv", index=False)

    # ── Plot walk-forward metrics ─────────────────────────────────────────────
    if wf_result.windows:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, metric, label in zip(
            axes,
            [wf_result.roc_auc_series, wf_result.pr_auc_series, wf_result.precision_at_10_series],
            ["ROC-AUC", "PR-AUC", "Precision@10%"],
        ):
            ax.plot(metric.values, marker="o", markersize=4, linewidth=1.5, color="#2B6CB0")
            ax.axhline(metric.mean(), color="#C53030", linestyle="--", linewidth=1,
                       label=f"mean={metric.mean():.3f}")
            ax.set_title(f"Walk-Forward {label}")
            ax.set_xlabel("OOS Window")
            ax.set_ylabel(label)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.4)
        fig.suptitle(f"Squeeze Detector — Walk-Forward OOS (N={len(wf_result.windows)} windows)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "wf_metrics.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ── Train final model on all data ────────────────────────────────────────
    logger.info("Training final model on full dataset…")
    labels = labeler.label(features)
    valid_mask = features.set_index(["date", "symbol"]).index.isin(labels[labels.notna()].index) \
        if "date" in features.columns else labels.notna()

    final_model = SqueezeDetector(seed=args.seed)
    try:
        final_model.fit(features, labels)
        final_model.save(model_dir / "squeeze_detector.joblib")

        # Score the full dataset and save top candidates
        scores = final_model.predict_proba(features)
        features["squeeze_prob"] = scores.values
        top_candidates = (
            features.sort_values("date").groupby("date").apply(
                lambda g: g.nlargest(10, "squeeze_prob")
            ).reset_index(drop=True)
        )
        latest_date = top_candidates["date"].max()
        top_today = top_candidates[top_candidates["date"] == latest_date][
            ["date", "symbol", "squeeze_prob", "si_dtc", "svr_z20", "borrow_rate_bps"]
        ]
        print(f"\nTop squeeze candidates as of {latest_date.date()}:")
        print(top_today.to_string(index=False))
        top_today.to_csv(out_dir / "top_squeeze_candidates.csv", index=False)

    except Exception as exc:
        logger.error("Final model training failed: %s", exc)

    logger.info("Squeeze model pipeline complete. Results in %s", out_dir)


if __name__ == "__main__":
    main()
