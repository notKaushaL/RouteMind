"""
ml/eta_cqr/train.py
====================
Phase 2 — CQR Model Training & MLflow Logging

Runs the full training pipeline:
  1. Build feature store (if needed)
  2. Load train / calibration / test splits
  3. Fit CQRModel (LightGBM × 3 + conformal calibration)
  4. Evaluate on test set: MAE, MedianAE, empirical coverage, MPIW
  5. Log everything to MLflow (params, metrics, artifacts)
  6. Save trained model to ml/eta_cqr/models/eta_cqr_pipeline.joblib

Usage:
    python ml/eta_cqr/train.py
    python ml/eta_cqr/train.py --n-estimators 1000 --mlflow-uri http://localhost:5001
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "eta_cqr_pipeline.joblib"


def compute_test_metrics(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Compute comprehensive test-set evaluation metrics."""
    preds = model.predict(X_test)
    y = y_test.values

    # Point estimate (P50) accuracy
    mae_p50 = float(np.mean(np.abs(preds["p50"] - y)))
    medae_p50 = float(np.median(np.abs(preds["p50"] - y)))

    # P90 prediction error (how far P90 is above actual — should be small positive)
    p90_error = float(np.mean(preds["p90"] - y))

    # Calibrated interval metrics
    covered = (preds["p10_cal"] <= y) & (y <= preds["p90_cal"])
    coverage = float(covered.mean())
    mpiw = float(np.mean(preds["p90_cal"] - preds["p10_cal"]))

    # Over-prediction bias (how often P90_cal > actual by large margin)
    excess_min = float(np.mean(np.maximum(0, preds["p90_cal"] - y)))

    return {
        "test_mae_p50_min": round(mae_p50, 4),
        "test_medae_p50_min": round(medae_p50, 4),
        "test_p90_pred_error_min": round(p90_error, 4),
        "test_coverage_pct": round(coverage * 100, 2),
        "test_mpiw_min": round(mpiw, 4),
        "test_excess_buffer_min": round(excess_min, 4),
        "test_n_samples": len(y),
    }


def train(
    n_estimators: int = 800,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    mlflow_uri: str = "http://localhost:5001",
    experiment_name: str = "routemind/eta_cqr",
    force_rebuild: bool = False,
) -> None:
    # ── Feature store ─────────────────────────────────────────────────────────
    from ml.feature_store.store import FeatureStore, TARGET_TOTAL
    store = FeatureStore()
    store.build(force=force_rebuild)

    X_train, y_train = store.get_training_data(target=TARGET_TOTAL)
    X_cal,   y_cal   = store.get_calibration_data(target=TARGET_TOTAL)
    X_test,  y_test  = store.get_test_data(target=TARGET_TOTAL)

    log.info(
        "Data splits: train=%d  cal=%d  test=%d", len(X_train), len(X_cal), len(X_test)
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    from ml.eta_cqr.model import CQRConfig, CQRModel
    config = CQRConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    model = CQRModel(config=config)
    model.fit(X_train, y_train, X_cal, y_cal)

    # ── Test evaluation ────────────────────────────────────────────────────────
    test_metrics = compute_test_metrics(model, X_test, y_test)
    for k, v in test_metrics.items():
        log.info("  %s = %s", k, v)

    # ── MLflow logging ─────────────────────────────────────────────────────────
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="cqr_lgbm"):
            # Parameters
            mlflow.log_params({
                "model_type": "LightGBM-CQR",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "alphas": "[0.10, 0.50, 0.90]",
                "conformal_method": "split_conformal",
                "train_days": "0-62",
                "cal_days": "63-74",
                "test_days": "75-89",
                "n_train": len(X_train),
                "n_cal": len(X_cal),
                "n_test": len(X_test),
            })

            # Calibration metrics
            mlflow.log_metrics(model.metrics_dict())

            # Test metrics
            mlflow.log_metrics(test_metrics)

            # Save and log artifact
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save(MODEL_PATH)
            mlflow.log_artifact(str(MODEL_PATH))

            # Log calibration delta as separate metric for monitoring
            mlflow.log_metric("conformal_delta_min", model.delta)

            run_id = mlflow.active_run().info.run_id
            log.info("MLflow run logged: %s (experiment: %s)", run_id, experiment_name)

    except Exception as e:
        log.warning("MLflow logging failed (%s) — saving model locally only.", e)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_PATH)

    log.info("✅ CQR model training complete.")
    log.info("   Empirical coverage: %.1f%%", test_metrics["test_coverage_pct"])
    log.info("   P50 MAE: %.2f min", test_metrics["test_mae_p50_min"])
    log.info("   MPIW: %.2f min", test_metrics["test_mpiw_min"])
    log.info("   Model saved → %s", MODEL_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CQR ETA model.")
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5001")
    parser.add_argument("--experiment", type=str, default="routemind/eta_cqr")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
