"""
backend/app/schemas/simulation.py
=================================
Pydantic schemas for the dynamic disruption simulation lab and MaskablePPO meta-actions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SimulationInitRequest(BaseModel):
    day_seed: int = Field(3000, description="Workday simulation seed")
    n_stops: int = Field(30, description="Base stops to distribute")


class SimulationStateSummary(BaseModel):
    sim_time_str: str = Field("09:00", description="Current simulation clock")
    sim_time_s: float = Field(32400.0, description="Current simulation time in seconds")
    active_vehicles_count: int
    completed_stops_count: int
    remaining_stops_count: int
    unassigned_queue_count: int
    mean_capacity_pct: float
    max_driver_delay_min: float
    route_churn_kendall_tau: float


class SimulationInitResponse(BaseModel):
    status: str
    message: str
    state: SimulationStateSummary
    routes_geojson: Dict[str, Any]
    vehicles: List[Dict[str, Any]]


class DisruptionInjectRequest(BaseModel):
    type: str = Field(
        ...,
        description="Shock type: 'TRAFFIC_SPIKE' | 'BREAKDOWN' | 'URGENT_ORDER' | 'FAILED_DELIVERY'",
    )
    target_h3_cell: Optional[str] = Field(None, description="H3 cell or zone name e.g. 'Alkapuri'")
    vehicle_id: Optional[str] = Field(None, description="Vehicle ID for breakdown event")
    severity: float = Field(2.5, description="Multiplier for traffic spike or severity")


class DisruptedOrder(BaseModel):
    stop_id: str
    vehicle_id: str
    lat: float
    lon: float
    original_eta_str: str
    new_eta_str: str
    delay_delta_min: float
    is_sla_breached: bool


class DisruptionInjectResponse(BaseModel):
    event_type: str
    description: str
    timestamp_str: str
    agent_action_id: int
    agent_action_name: str
    decision_latency_ms: float
    route_churn_score: float
    before_routes_geojson: Dict[str, Any]
    after_routes_geojson: Dict[str, Any]
    disrupted_orders: List[DisruptedOrder]
    state: SimulationStateSummary
