"""
ml/evaluation/drift_report.py
==============================
Phase 2 — Evidently AI Drift Monitoring

Compares a simulated "new week" of delivery data (incorporating seasonal
slowdowns) against the training baseline to detect:

  1. Data Drift: covariate shift in feature distributions
     - Wasserstein distance on continuous features
     - Chi-squared on categorical/boolean features
  2. Target Drift: shift in actual_total_leg_time_min distribution

Saves a drift report to reports/drift_report.html and prints a
summary table of drifted features.

Usage:
    python ml/evaluation/drift_report.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def generate_drift_report(
    feature_path: Path | None = None,
    output_html: Path | None = None,
) -> dict[str, float]:
    """
    Generate a drift report comparing train baseline vs a simulated new week.

    Returns a dict of {feature: p_value or distance} for drifted features.
    """
    from ml.feature_store.store import FeatureStore, FEATURE_COLS, TARGET_TOTAL

    if feature_path is None:
        store = FeatureStore()
        feature_path = store.feature_path

    if output_html is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_html = REPORTS_DIR / "drift_report.html"

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature store not found at {feature_path}. Run store.build() first.")

    # Load reference (training) and current (new week = last 7 days) data
    store = FeatureStore()
    full_df = store.get_full_dataframe()

    reference_df = full_df[full_df["day"] <= 62].copy()
    current_df = full_df[full_df["day"] >= 83].copy()   # last 7 days of 90-day dataset

    if len(current_df) == 0:
        log.warning("No data for days ≥83. Using days 75–89 as current window.")
        current_df = full_df[full_df["day"] >= 75].copy()

    log.info(
        "Drift report: reference=%d rows | current=%d rows",
        len(reference_df), len(current_df)
    )

    # Columns to monitor
    drift_cols = FEATURE_COLS + [TARGET_TOTAL]

    try:
        try:
            from evidently.legacy.pipeline.column_mapping import ColumnMapping
            from evidently.legacy.report import Report
            from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
        except ImportError:
            from evidently import ColumnMapping
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

        column_mapping = ColumnMapping(
            target=TARGET_TOTAL,
            numerical_features=[c for c in FEATURE_COLS if c not in ("peak_hour_flag", "is_priority", "vehicle_type_enc")],
            categorical_features=["peak_hour_flag", "is_priority", "vehicle_type_enc"],
        )

        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
        ])

        ref = reference_df[drift_cols].copy()
        cur = current_df[drift_cols].copy()

        report.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(output_html))
        log.info("Drift report saved → %s", output_html)

        # Extract summary
        result = report.as_dict()
        drift_summary: dict[str, float] = {}
        try:
            for metric in result.get("metrics", []):
                if "result" in metric:
                    res = metric["result"]
                    if "drift_by_columns" in res:
                        for col, info in res["drift_by_columns"].items():
                            drift_summary[col] = info.get("stattest_threshold", 0.0)
        except Exception:
            pass

        return drift_summary

    except ImportError:
        log.warning("evidently not installed — running manual drift checks with scipy.")
        return _manual_drift_check(reference_df, current_df, drift_cols)


def _manual_drift_check(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    cols: list[str],
) -> dict[str, float]:
    """
    Fallback: Kolmogorov-Smirnov test for continuous features,
    Chi-squared for categorical.
    Returns {col: p_value}.
    """
    from scipy import stats

    results: dict[str, float] = {}
    categorical = {"peak_hour_flag", "is_priority", "vehicle_type_enc"}

    for col in cols:
        if col not in ref.columns or col not in cur.columns:
            continue
        try:
            if col in categorical:
                # Chi-squared
                ref_counts = ref[col].value_counts()
                cur_counts = cur[col].value_counts()
                all_cats = sorted(set(ref_counts.index) | set(cur_counts.index))
                ref_arr = np.array([ref_counts.get(c, 0) for c in all_cats])
                cur_arr = np.array([cur_counts.get(c, 0) for c in all_cats])
                if ref_arr.sum() > 0 and cur_arr.sum() > 0:
                    # Normalize
                    ref_arr = ref_arr / ref_arr.sum() * cur_arr.sum()
                    chi2, p = stats.chisquare(cur_arr, f_exp=ref_arr + 1e-10)
                    results[col] = round(float(p), 4)
            else:
                ks_stat, p = stats.ks_2samp(
                    ref[col].dropna().values,
                    cur[col].dropna().values,
                )
                results[col] = round(float(p), 4)
        except Exception as e:
            log.debug("Could not test drift for %s: %s", col, e)

    # Print summary
    drifted = {k: v for k, v in results.items() if v < 0.05}
    log.info("Drift detected (p < 0.05) in %d / %d features:", len(drifted), len(results))
    for feat, p in sorted(drifted.items(), key=lambda x: x[1]):
        log.info("  %-40s p=%.4f  *** DRIFT ***", feat, p)

    return results


def main() -> None:
    drift_summary = generate_drift_report()

    if drift_summary:
        drifted = {k: v for k, v in drift_summary.items() if v < 0.05}
        print(f"\n{'='*60}")
        print(f"DRIFT SUMMARY: {len(drifted)} drifted features (p < 0.05)")
        print(f"{'='*60}")
        for feat, val in sorted(drifted.items(), key=lambda x: x[1])[:10]:
            print(f"  {feat:<40} = {val:.4f}")
    else:
        print("No drift summary available.")


if __name__ == "__main__":
    main()
