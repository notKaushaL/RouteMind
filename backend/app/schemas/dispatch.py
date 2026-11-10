"""
backend/app/schemas/dispatch.py
===============================
Pydantic schemas for dispatch planning and route schedules.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ObjectiveWeightsSchema(BaseModel):
    w1_distance: float = Field(1.0, description="Transit distance weight")
    w2_lateness: float = Field(5.0, description="SLA window lateness weight")
    w3_overtime: float = Field(3.0, description="Driver shift overtime weight")
    w4_unassigned: float = Field(100.0, description="Unassigned order penalty")
    w5_stability: float = Field(2.0, description="Kendall-Tau route stability weight")
    w6_workload_var: float = Field(0.5, description="Fleet workload variance weight")


class PlanRequest(BaseModel):
    workday_id: str = Field("DAY_01", description="Identifier for dispatch schedule")
    day_seed: int = Field(42, description="Seed for deterministic stop & traffic generation")
    n_stops: int = Field(30, ge=5, le=100, description="Number of customer delivery stops")
    solver_type: str = Field(
        "ortools_gatv2_warm",
        description="Solver engine: 'greedy' | 'ortools_cold' | 'ortools_gatv2_warm'",
    )
    risk_mode: str = Field(
        "p90_robust",
        description="Risk tolerance: 'osrm_static' | 'p50_median' | 'p90_robust'",
    )
    weights: Optional[ObjectiveWeightsSchema] = Field(default_factory=ObjectiveWeightsSchema)


class ScheduleStop(BaseModel):
    stop_id: str
    vehicle_id: str
    sequence_idx: int
    lat: float
    lon: float
    h3_cell: str
    cluster_id: int
    demand_weight_kg: float
    demand_volume_l: float
    tw_open_time: str
    tw_close_time: str
    arrival_time: str
    departure_time: str
    lateness_min: float
    is_sla_breached: bool
    is_priority: bool


class VehicleSummary(BaseModel):
    vehicle_id: str
    vtype: str
    color: str
    total_stops: int
    total_distance_km: float
    total_duration_min: float
    max_weight_capacity_kg: float
    utilized_weight_kg: float
    capacity_utilization_pct: float
    sla_breaches: int


class PlanMetrics(BaseModel):
    total_distance_km: float
    total_transit_duration_min: float
    total_dwell_duration_min: float
    total_duration_min: float
    sla_violations_count: int
    total_lateness_min: float
    unassigned_orders_count: int
    solve_time_seconds: float
    composite_objective_cost: float
    on_time_pct: float


class PlanResponse(BaseModel):
    workday_id: str
    solver_type: str
    risk_mode: str
    metrics: PlanMetrics
    vehicles: List[VehicleSummary]
    routes_geojson: Dict[str, Any]
    schedule: List[ScheduleStop]
