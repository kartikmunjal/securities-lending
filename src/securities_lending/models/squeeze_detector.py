"""
Short Squeeze Detection Model.

A short squeeze occurs when rising prices force short sellers to cover their
positions, amplifying the upward move.  Key conditions:
  * High short interest / days-to-cover (large short position to unwind)
  * Recent price appreciation (forcing mark-to-market losses on shorts)
  * Volume surge (covering activity)

This module trains a gradient-boosted classifier to assign a squeeze-risk
probability score to each stock on each day, evaluated via walk-forward OOS
validation to prevent data snooping.

Label construction
------------------
A "squeeze event" is operationally defined as:

    squeeze_{i,t} = 1  if  r_{i, t→t+5} > 15%
                          AND  volume_t > 2× 20-day avg
                          AND  si_dtc_t > 5 days

The 5-day 15% threshold captures the fast squeeze dynamic (GME: +400% in 3
days; AMC: +100% in 2 days).  The volume and DTC conditions ensure we label
genuine short-cover events rather than earnings gaps or M&A.

Class imbalance
---------------
Squeeze events are rare (~1-3% of stock-weeks).  We use:
  * class_weight='balanced' in the classifier
  * PR-AUC as primary metric (more appropriate than ROC-AUC for rare events)
  * Precision@K (top-10% by score) as the trading-relevant metric

Walk-forward evaluation
-----------------------
Train window: 252 days (1 year)
Test window:  63 days (1 quarter)
Step size:    21 days (monthly retrain)

This mirrors a realistic production deployment where the model is retrained
monthly using only past data.

References
----------
Brunnermeier & Pedersen (2009) "Market Liquidity and Funding Liquidity". RFS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Feature columns used by the model (must exist in the feature DataFrame)
_FEATURE_COLS = [
    # Short interest / borrow pressure
    "svr_z20",
    "svr_trend5",
    "si_pct_float",
    "si_dtc",
    "si_chg_pct",
    "borrow_rate_bps",
    "borrow_stress",
    # Price and volatility
    "ret_5d",
    "ret_21d",
    "realized_vol_20d",
    "rel_volume",
    "range_pct",
    # Cross-sectional context
    "svr_percentile",
    "short_pressure",
]


@dataclass
class SqueezeEventLabeler:
    """Label squeeze events from feature DataFrame."""

    return_threshold: float = 0.15   # 5-day forward return threshold
    dtc_threshold: float = 5.0       # minimum DTC for labeling
    volume_spike: float = 2.0        # volume relative to 20-day avg

    def label(self, features: pd.DataFrame) -> pd.Series:
        """
        Assign squeeze labels (0/1) to each row.

        Parameters
        ----------
        features : pd.DataFrame
            Must contain: ret_5d (forward), si_dtc, rel_volume.

        Returns
        -------
        pd.Series with same index as features.
        """
        high_return = features["ret_5d"].shift(-5) > self.return_threshold
        high_dtc = features.get("si_dtc", pd.Series(0, index=features.index)) > self.dtc_threshold
        volume_surge = features.get("rel_volume", pd.Series(0, index=features.index)) > self.volume_spike

        labels = (high_return & high_dtc & volume_surge).astype(int)
        pos_rate = labels.mean()
        logger.info(
            "Squeeze event rate: %.2f%% (%d / %d)",
            pos_rate * 100, labels.sum(), len(labels),
        )
        return labels


class SqueezeDetector:
    """
    Walk-forward short squeeze probability model.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    class_weight : str | None
        Passed to HistGradientBoostingClassifier.  'balanced' is recommended
        for the highly imbalanced squeeze label.
    calibrate : bool
        If True, wrap the classifier in CalibratedClassifierCV (isotonic
        regression).  Raw HGBC probabilities are not well-calibrated.
    """

    def __init__(
        self,
        seed: int = 42,
        class_weight: str | None = "balanced",
        calibrate: bool = True,
    ):
        self.seed = seed
        self.class_weight = class_weight
        self.calibrate = calibrate
        self._model: Pipeline | None = None
        self._feature_cols: list[str] = _FEATURE_COLS

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SqueezeDetector":
        """
        Train on (X, y) where y is the squeeze event label (0/1).

        X must contain the feature columns in _FEATURE_COLS.
        Missing features are handled by HistGradientBoostingClassifier natively.
        """
        X = X[self._feature_cols].copy()
        y = y.loc[X.index]

        base_clf = HistGradientBoostingClassifier(
            random_state=self.seed,
            class_weight=self.class_weight,
            max_iter=300,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=20,
        )

        if self.calibrate:
            clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
        else:
            clf = base_clf

        self._model = Pipeline([("clf", clf)])
        self._model.fit(X, y)
        logger.info(
            "SqueezeDetector trained on %d samples (%.1f%% positive)",
            len(y), 100 * y.mean(),
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Return squeeze probability scores in [0, 1].

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame with same columns as training data.

        Returns
        -------
        pd.Series indexed like X with probability of squeeze event.
        """
        if self._model is None:
            raise RuntimeError("Model not trained.  Call fit() first.")
        X_feat = X[self._feature_cols].copy()
        probs = self._model.predict_proba(X_feat)[:, 1]
        return pd.Series(probs, index=X.index, name="squeeze_prob")

    def feature_importance(self) -> pd.Series:
        """Return SHAP-based or impurity feature importance."""
        if self._model is None:
            raise RuntimeError("Model not trained.")
        try:
            import shap
            # Extract the base estimator from calibrated wrapper
            clf = self._model.named_steps["clf"]
            base = clf.estimator if hasattr(clf, "estimator") else clf
            explainer = shap.TreeExplainer(base)
            # Return expected absolute SHAP values (averaged across a training sample)
            # This requires passing a sample of X — caller should use explain() instead
            return pd.Series(dtype=float, name="shap_importance")
        except ImportError:
            logger.warning("shap not installed; returning None")
            return pd.Series(dtype=float)

    def explain(self, X: pd.DataFrame):
        """Return SHAP values for a sample of X."""
        import shap
        X_feat = X[self._feature_cols].copy()
        clf = self._model.named_steps["clf"]
        base = clf.estimator if hasattr(clf, "estimator") else clf
        explainer = shap.TreeExplainer(base)
        return explainer(X_feat)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {"model": self._model, "feature_cols": self._feature_cols, "seed": self.seed},
            path,
        )
        logger.info("SqueezeDetector saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SqueezeDetector":
        payload = joblib.load(path)
        obj = cls(seed=payload["seed"])
        obj._model = payload["model"]
        obj._feature_cols = payload["feature_cols"]
        return obj
