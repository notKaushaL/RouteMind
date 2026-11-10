"""
backend/app/schemas/mlops.py
============================
Pydantic schemas for Explainability (TreeSHAP) and MLOps benchmarking.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExplainDelayRequest(BaseModel):
    h3_cell: str = Field("88619b4563fffff", description="Uber H3 index (Res 8)")
    cluster_id: int = Field(0, description="Spatial cluster ID (0..7)")
    osrm_base_time_s: float = Field(420.0, description="OSRM nominal baseline travel time in seconds")
    hour: int = Field(17, ge=0, le=23, description="Dispatch hour of day (0..23)")
    vehicle_class: str = Field("three_wheeler", description="Vehicle type: 'three_wheeler' | 'van' | 'ev_two_wheeler'")
    package_weight_kg: float = Field(8.5, description="Package weight in kg")
    is_weekend: int = Field(0, description="Weekend flag (0 or 1)")


class SHAPFactor(BaseModel):
    feature: str
    impact_min: float
    direction: str  # "increase" or "decrease"
    description: str


class ExplainDelayResponse(BaseModel):
    osrm_nominal_min: float
    cqr_p50_predicted_min: float
    cqr_p90_robust_min: float
    conformal_margin_delta_min: float
    net_delay_risk_min: float
    top_factors: List[SHAPFactor]
    narrative: str


class BenchmarkTableRow(BaseModel):
    method_or_config: str
    metrics: Dict[str, Any]


class MLOpsMetricsResponse(BaseModel):
    phase1_deterministic_baselines: List[Dict[str, Any]]
    phase2_cqr_ablation: List[Dict[str, Any]]
    phase3_gnn_warmstart_ablation: List[Dict[str, Any]]
    phase4_rl_meta_ablation: List[Dict[str, Any]]
    cqr_calibration: Dict[str, Any]
    drift_health: Dict[str, Any]
