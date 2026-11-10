"""
ml/eta_cqr/explain.py
======================
Phase 2 — TreeSHAP Explainability for CQR ETA Model

Provides human-readable delay explanations for dispatchers, e.g.:
  "Alkapuri H3 zone parking friction (+6.2 min)"
  "Evening peak hour transit (+4.8 min)"
  "Van congestion penalty (+2.1 min)"

Usage:
    from ml.eta_cqr.explain import CQRExplainer
    explainer = CQRExplainer(model)
    factors = explainer.explain_delivery_delay(feature_dict, top_k=3)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    log.warning("shap not installed — CQRExplainer will raise on use.")

# ── Human-readable feature name mapping ───────────────────────────────────────
FEATURE_LABELS: dict[str, str] = {
    "sin_hour":                   "Time-of-day cyclical (sin)",
    "cos_hour":                   "Time-of-day cyclical (cos)",
    "sin_dow":                    "Day-of-week cyclical (sin)",
    "cos_dow":                    "Day-of-week cyclical (cos)",
    "peak_hour_flag":             "Peak hour flag",
    "hour_of_day":                "Hour of departure",
    "osrm_baseline_duration_min": "OSRM free-flow transit (min)",
    "osrm_baseline_distance_km":  "Route distance (km)",
    "weight_kg":                  "Package weight (kg)",
    "volume_l":                   "Package volume (L)",
    "is_priority":                "Priority SLA order",
    "vehicle_type_enc":           "Vehicle type",
    "h3r8_mean_dwell_min":        "Zone mean dwell time (R8, min)",
    "h3r8_std_dwell_min":         "Zone dwell variability (R8)",
    "h3r9_mean_dwell_min":        "Zone mean dwell time (R9, min)",
    "h3r9_std_dwell_min":         "Zone dwell variability (R9)",
}

# Thematic grouping for dispatcher-friendly messages
def _shap_to_message(feature: str, shap_val: float) -> str:
    """Convert a (feature, SHAP value) pair to a dispatcher-readable string."""
    direction = "+" if shap_val > 0 else "−"
    abs_val = abs(shap_val)

    if feature in ("h3r8_mean_dwell_min", "h3r9_mean_dwell_min"):
        return f"Zone parking friction ({direction}{abs_val:.1f} min)"
    elif feature == "peak_hour_flag" and shap_val > 0:
        return f"Peak-hour congestion ({direction}{abs_val:.1f} min)"
    elif feature == "hour_of_day":
        return f"Departure time effect ({direction}{abs_val:.1f} min)"
    elif feature in ("sin_hour", "cos_hour"):
        return f"Time-of-day pattern ({direction}{abs_val:.1f} min)"
    elif feature == "vehicle_type_enc":
        return f"Vehicle type congestion penalty ({direction}{abs_val:.1f} min)"
    elif feature == "osrm_baseline_duration_min":
        return f"Route distance ({direction}{abs_val:.1f} min)"
    elif feature in ("weight_kg", "volume_l"):
        return f"Cargo handling overhead ({direction}{abs_val:.1f} min)"
    elif feature == "is_priority":
        return f"Priority order handling ({direction}{abs_val:.1f} min)"
    elif feature in ("h3r8_std_dwell_min", "h3r9_std_dwell_min"):
        return f"Zone dwell uncertainty ({direction}{abs_val:.1f} min)"
    else:
        label = FEATURE_LABELS.get(feature, feature)
        return f"{label} ({direction}{abs_val:.1f} min)"


class CQRExplainer:
    """
    TreeSHAP-based delay explainer for the CQR P50 and P90 models.

    Builds SHAP TreeExplainer objects lazily on first use.
    """

    def __init__(self, cqr_model: Any):
        """
        Args:
            cqr_model: fitted CQRModel instance
        """
        if not _SHAP_AVAILABLE:
            raise ImportError("shap is required. Run: pip install shap")
        self.model = cqr_model
        self._explainer_p50: Any = None
        self._explainer_p90: Any = None

    def _get_explainer(self, alpha: float) -> Any:
        if alpha == 0.50:
            if self._explainer_p50 is None:
                log.info("Building SHAP TreeExplainer for P50 model …")
                self._explainer_p50 = shap.TreeExplainer(self.model._models[0.50])
            return self._explainer_p50
        elif alpha == 0.90:
            if self._explainer_p90 is None:
                log.info("Building SHAP TreeExplainer for P90 model …")
                self._explainer_p90 = shap.TreeExplainer(self.model._models[0.90])
            return self._explainer_p90
        raise ValueError(f"Unsupported alpha={alpha}. Use 0.50 or 0.90.")

    def explain_delivery_delay(
        self,
        feature_dict: dict[str, float],
        top_k: int = 3,
        alpha: float = 0.50,
    ) -> list[dict[str, Any]]:
        """
        Return the top-k SHAP feature contributions for a single delivery stop.

        Args:
            feature_dict: dict of {feature_name: value} (same keys as FEATURE_COLS)
            top_k: number of top contributing factors to return
            alpha: which model to explain (0.50 = P50 median, 0.90 = P90 upper bound)

        Returns:
            List of dicts, each with:
              feature:  raw feature name
              shap_value: float (minutes of predicted delay contributed)
              message:  human-readable string for dispatcher UI
        """
        from ml.feature_store.store import FEATURE_COLS

        # Build single-row DataFrame
        X = pd.DataFrame([{col: feature_dict.get(col, 0.0) for col in FEATURE_COLS}])

        explainer = self._get_explainer(alpha)
        shap_vals = explainer.shap_values(X.values)[0]  # shape (n_features,)

        # Sort by absolute SHAP value descending
        feature_names = FEATURE_COLS
        pairs = sorted(
            zip(feature_names, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        results = []
        for feat, sv in pairs[:top_k]:
            results.append({
                "feature": feat,
                "shap_value": round(float(sv), 3),
                "message": _shap_to_message(feat, sv),
            })

        return results

    def batch_shap_values(
        self,
        X: pd.DataFrame,
        alpha: float = 0.50,
    ) -> np.ndarray:
        """
        Compute SHAP values for a batch of stops.
        Returns array of shape (n_samples, n_features).
        """
        explainer = self._get_explainer(alpha)
        return explainer.shap_values(X.values)
