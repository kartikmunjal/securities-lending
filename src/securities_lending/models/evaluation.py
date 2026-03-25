"""Walk-forward model evaluation with precision@K and PR-AUC metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .squeeze_detector import SqueezeDetector, SqueezeEventLabeler

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Per-window and aggregate walk-forward evaluation results."""

    windows: list[dict] = field(default_factory=list)

    @property
    def roc_auc_series(self) -> pd.Series:
        return pd.Series([w["roc_auc"] for w in self.windows if not np.isnan(w["roc_auc"])])

    @property
    def pr_auc_series(self) -> pd.Series:
        return pd.Series([w["pr_auc"] for w in self.windows if not np.isnan(w["pr_auc"])])

    @property
    def precision_at_10_series(self) -> pd.Series:
        return pd.Series([w["precision_at_10"] for w in self.windows if not np.isnan(w["precision_at_10"])])

    def summary(self) -> dict:
        return {
            "n_windows": len(self.windows),
            "mean_roc_auc": self.roc_auc_series.mean(),
            "mean_pr_auc": self.pr_auc_series.mean(),
            "mean_precision_at_10": self.precision_at_10_series.mean(),
            "std_roc_auc": self.roc_auc_series.std(),
            "std_pr_auc": self.pr_auc_series.std(),
        }


class WalkForwardEvaluator:
    """
    Walk-forward cross-validation for the SqueezeDetector.

    Parameters
    ----------
    train_window : int
        Training window in days (default 252 = 1 year).
    test_window : int
        Out-of-sample test window in days (default 63 = 1 quarter).
    step_size : int
        Days between successive train/test splits (default 21 = monthly retrain).
    """

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 63,
        step_size: int = 21,
        seed: int = 42,
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.seed = seed

    def run(
        self,
        features: pd.DataFrame,
        labeler: SqueezeEventLabeler | None = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward evaluation.

        Parameters
        ----------
        features : pd.DataFrame
            Multi-index (date, symbol) or flat with 'date' column.
            Must contain all columns in SqueezeDetector._FEATURE_COLS.
        labeler : SqueezeEventLabeler | None
            Event labeler.  Default parameters used if None.
        """
        if labeler is None:
            labeler = SqueezeEventLabeler()

        if "date" in features.columns:
            features = features.set_index(["date", "symbol"])

        all_dates = sorted(features.index.get_level_values("date").unique())
        labels = labeler.label(features.reset_index())
        labels.index = features.index

        result = WalkForwardResult()
        n_windows = (len(all_dates) - self.train_window - self.test_window) // self.step_size + 1

        for i in range(0, n_windows * self.step_size, self.step_size):
            train_start_idx = i
            train_end_idx = i + self.train_window
            test_end_idx = train_end_idx + self.test_window

            if test_end_idx > len(all_dates):
                break

            train_dates = all_dates[train_start_idx:train_end_idx]
            test_dates = all_dates[train_end_idx:test_end_idx]

            X_train = features.loc[pd.IndexSlice[train_dates, :], :]
            y_train = labels.loc[pd.IndexSlice[train_dates, :]]
            X_test = features.loc[pd.IndexSlice[test_dates, :], :]
            y_test = labels.loc[pd.IndexSlice[test_dates, :]]

            if y_train.sum() < 10 or y_test.sum() < 2:
                logger.debug("Skipping window %d: too few positive events", i)
                continue

            detector = SqueezeDetector(seed=self.seed)
            try:
                detector.fit(X_train.reset_index(), y_train.reset_index(drop=True))
                probs = detector.predict_proba(X_test.reset_index())
                y_true = y_test.values

                roc = roc_auc_score(y_true, probs.values) if y_true.sum() > 0 else np.nan
                pr = average_precision_score(y_true, probs.values) if y_true.sum() > 0 else np.nan
                pat10 = self._precision_at_k(y_true, probs.values, 0.10)

                result.windows.append(
                    {
                        "train_start": train_dates[0],
                        "train_end": train_dates[-1],
                        "test_start": test_dates[0],
                        "test_end": test_dates[-1],
                        "roc_auc": roc,
                        "pr_auc": pr,
                        "precision_at_10": pat10,
                        "n_test": len(y_test),
                        "n_positive_test": int(y_true.sum()),
                    }
                )
                logger.info(
                    "Window %d/%d  ROC-AUC=%.3f  PR-AUC=%.3f  P@10%%=%.3f",
                    i // self.step_size + 1, n_windows, roc, pr, pat10,
                )
            except Exception as exc:
                logger.warning("Walk-forward window %d failed: %s", i, exc)

        logger.info("Walk-forward complete: %s", result.summary())
        return result

    @staticmethod
    def _precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: float) -> float:
        """Precision among the top-k% predicted positives."""
        n_top = max(1, int(len(scores) * k))
        top_idx = np.argsort(scores)[-n_top:]
        return float(y_true[top_idx].mean())
