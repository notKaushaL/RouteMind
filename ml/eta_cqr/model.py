"""
ml/eta_cqr/model.py
====================
Phase 2 — Conformalized Quantile Regression ETA Model

Architecture:
  1. Three LightGBM models trained with quantile loss at α ∈ {0.10, 0.50, 0.90}
  2. Split Conformal Prediction on the held-out calibration set adjusts
     the raw quantile outputs to guarantee exact ≥90% empirical coverage:

       P(y ∈ [ŷ_0.10 − Δ, ŷ_0.90 + Δ]) ≥ 0.90

     where Δ = (1−α) quantile of the calibration non-conformity scores
              non_conformity_i = max(ŷ_0.10 − y_i,  y_i − ŷ_0.90)

  3. The calibrated P50 prediction is used for standard stops in OR-Tools.
     The calibrated P90 upper bound is used for high-priority SLA stops.

Interface:
    model = CQRModel()
    model.fit(X_train, y_train, X_cal, y_cal)
    preds = model.predict(X)   # returns dict with keys p10, p50, p90, p90_cal
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    log.warning("lightgbm not installed — CQRModel will raise on fit().")


@dataclass
class CQRConfig:
    """Hyperparameters for the LightGBM quantile models."""
    n_estimators: int = 800
    max_depth: int = 6
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    n_jobs: int = -1
    verbose: int = -1
    random_state: int = 42


class CQRModel:
    """
    Conformalized Quantile Regression model wrapping three LightGBM regressors.

    After calling fit(), the model exposes:
      self.delta        — calibrated conformal adjustment (scalar)
      self.coverage_    — empirical coverage on calibration set
    """

    ALPHAS = [0.10, 0.50, 0.90]

    def __init__(self, config: CQRConfig | None = None):
        if not _LGB_AVAILABLE:
            raise ImportError("lightgbm is required. Run: pip install lightgbm")
        self.config = config or CQRConfig()
        self._models: dict[float, Any] = {}
        self.delta: float = 0.0        # conformal adjustment
        self.coverage_: float = 0.0    # empirical coverage on calibration set
        self.mpiw_: float = 0.0        # mean prediction interval width on calibration
        self.feature_names_: list[str] = []

    def _lgb_params(self, alpha: float) -> dict[str, Any]:
        cfg = self.config
        return {
            "objective": "quantile",
            "alpha": alpha,
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "num_leaves": cfg.num_leaves,
            "min_child_samples": cfg.min_child_samples,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "reg_alpha": cfg.reg_alpha,
            "reg_lambda": cfg.reg_lambda,
            "n_jobs": cfg.n_jobs,
            "verbose": cfg.verbose,
            "random_state": cfg.random_state,
        }

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> "CQRModel":
        """
        1. Train LightGBM at α = 0.10, 0.50, 0.90.
        2. Compute calibration non-conformity scores.
        3. Set self.delta to (1-α) quantile of scores.
        """
        self.feature_names_ = list(X_train.columns)
        log.info("Training CQR model on %d samples, %d features …", len(X_train), X_train.shape[1])

        for alpha in self.ALPHAS:
            log.info("  Fitting LightGBM quantile α=%.2f …", alpha)
            m = lgb.LGBMRegressor(**self._lgb_params(alpha))
            m.fit(X_train.values, y_train.values)
            self._models[alpha] = m

        # ── Conformal calibration ─────────────────────────────────────────────
        log.info("Computing conformal calibration on %d samples …", len(X_cal))
        preds_cal = self._raw_predict(X_cal)
        q10_cal = preds_cal["p10"]
        q90_cal = preds_cal["p90"]
        y_cal_arr = y_cal.values

        # Non-conformity score: max of lower and upper violation
        scores = np.maximum(q10_cal - y_cal_arr, y_cal_arr - q90_cal)

        # (1 - α) quantile of scores with finite-sample correction
        alpha_conf = 0.10
        n_cal = len(scores)
        quantile_level = np.ceil((1 - alpha_conf) * (n_cal + 1)) / n_cal
        quantile_level = min(quantile_level, 1.0)
        self.delta = float(np.quantile(scores, quantile_level))

        log.info("Conformal delta (Δ) = %.4f min", self.delta)

        # Empirical coverage
        covered = ((q10_cal - self.delta <= y_cal_arr) & (y_cal_arr <= q90_cal + self.delta))
        self.coverage_ = float(covered.mean())
        self.mpiw_ = float(np.mean((q90_cal + self.delta) - (q10_cal - self.delta)))

        log.info(
            "Empirical coverage: %.1f%% (target ≥90%%) | MPIW: %.2f min",
            self.coverage_ * 100, self.mpiw_,
        )
        return self

    def _raw_predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Raw quantile predictions before conformal adjustment."""
        X_arr = X[self.feature_names_].values if hasattr(X, "columns") else X
        return {
            "p10": self._models[0.10].predict(X_arr),
            "p50": self._models[0.50].predict(X_arr),
            "p90": self._models[0.90].predict(X_arr),
        }

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Returns calibrated prediction intervals and point estimates.

        Keys:
            p10     : raw 10th percentile (lower bound, pre-conformal)
            p50     : median point estimate
            p90     : raw 90th percentile (pre-conformal)
            p10_cal : calibrated lower bound = p10 − Δ
            p90_cal : calibrated upper bound = p90 + Δ  ← use for SLA-critical stops
        """
        raw = self._raw_predict(X)
        return {
            "p10": raw["p10"],
            "p50": raw["p50"],
            "p90": raw["p90"],
            "p10_cal": raw["p10"] - self.delta,
            "p90_cal": raw["p90"] + self.delta,
        }

    def predict_single(self, feature_dict: dict[str, float]) -> dict[str, float]:
        """
        Predict for a single stop (used by OR-Tools callback and SHAP explainer).
        Returns scalar floats instead of arrays.
        """
        from ml.feature_store.store import FEATURE_COLS
        X = pd.DataFrame([{col: feature_dict.get(col, 0.0) for col in FEATURE_COLS}])
        preds = self.predict(X)
        return {k: float(v[0]) for k, v in preds.items()}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save model using joblib."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(path))
        log.info("CQR model saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "CQRModel":
        import joblib
        m = joblib.load(str(path))
        log.info("CQR model loaded from %s (coverage=%.1f%%)", path, m.coverage_ * 100)
        return m

    def metrics_dict(self) -> dict[str, float]:
        """Return tracking metrics for MLflow."""
        return {
            "empirical_coverage_pct": round(self.coverage_ * 100, 2),
            "conformal_delta_min": round(self.delta, 4),
            "mpiw_min": round(self.mpiw_, 3),
        }
