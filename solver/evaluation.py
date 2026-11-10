"""
solver/evaluation.py
====================
Unified, solver-agnostic evaluation harness for RouteMind.

Problem (Phase 1 metric trap):
  - Greedy baselines reported raw metres of haversine distance.
  - OR-Tools reported its internal composite objective value (scaled integer).
  These are INCOMPARABLE. This module fixes that by traversing every route
  step-by-step with actual cumulative timing and computing every metric from
  first principles using the same duration/distance matrices.

Usage:
    from solver.evaluation import evaluate_route_plan, SolutionMetrics
    metrics = evaluate_route_plan(routes, dur_mat, dist_mat, stops, vehicles, weights)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from solver.baselines.greedy import Stop, Vehicle

log = logging.getLogger(__name__)


@dataclass
class ObjectiveWeights:
    """Mirrors configs/settings.py weight fields — kept separate to avoid circular imports."""
    w1_distance: float = 1.0      # distance cost (per km)
    w2_lateness: float = 5.0      # per-minute SLA lateness penalty
    w3_overtime: float = 3.0      # per-minute driver overtime penalty
    w4_unassigned: float = 100.0  # per unassigned order penalty
    w5_stability: float = 2.0     # route stability penalty (Phase 4)
    w6_workload_var: float = 0.5  # workload variance penalty

    @classmethod
    def from_settings(cls, s: Any) -> "ObjectiveWeights":
        """Construct from a pydantic Settings object."""
        return cls(
            w1_distance=getattr(s, "w1_distance", 1.0),
            w2_lateness=getattr(s, "w2_lateness", 5.0),
            w3_overtime=getattr(s, "w3_overtime", 3.0),
            w4_unassigned=getattr(s, "w4_unassigned", 100.0),
            w5_stability=getattr(s, "w5_stability", 2.0),
            w6_workload_var=getattr(s, "w6_workload_var", 0.5),
        )


@dataclass
class SolutionMetrics:
    """
    Canonical, apples-to-apples solution quality metrics.
    Computed by traversing routes step-by-step — independent of solver internals.
    """
    # ── Distance & Time ──────────────────────────────────────────────────────
    total_distance_km: float = 0.0
    total_transit_duration_min: float = 0.0   # sum of arc travel times
    total_dwell_duration_min: float = 0.0     # sum of stop service times
    total_duration_min: float = 0.0           # transit + dwell + waiting

    # ── SLA Performance ──────────────────────────────────────────────────────
    sla_violations_count: int = 0             # stops where arrival > tw_close
    total_lateness_min: float = 0.0           # Σ max(0, arrival - tw_close) / 60

    # ── Fleet Utilization ────────────────────────────────────────────────────
    vehicles_used: int = 0
    total_overtime_min: float = 0.0           # Σ max(0, return_to_depot - shift_end)
    workload_variance: float = 0.0            # Var of per-vehicle duration_min

    # ── Feasibility ──────────────────────────────────────────────────────────
    unassigned_orders_count: int = 0
    capacity_violations: int = 0              # routes that exceed capacity

    # ── Solve Meta ───────────────────────────────────────────────────────────
    solve_time_seconds: float = 0.0
    algorithm: str = ""

    # ── Composite Objective (computed last) ──────────────────────────────────
    composite_objective_cost: float = 0.0

    def as_row(self) -> dict[str, Any]:
        """Flat dict for DataFrame / table display."""
        return {
            "Method": self.algorithm,
            "Distance (km)": f"{self.total_distance_km:.2f}",
            "Time (min)": f"{self.total_duration_min:.1f}",
            "SLA Breaches": self.sla_violations_count,
            "Lateness (min)": f"{self.total_lateness_min:.1f}",
            "Unassigned": self.unassigned_orders_count,
            "Solve Time (s)": f"{self.solve_time_seconds:.3f}",
            "Cost (J)": f"{self.composite_objective_cost:.1f}",
        }

    def markdown_row(self) -> str:
        """Single markdown table row."""
        d = self.as_row()
        return (
            f"| {d['Method']} | {d['Distance (km)']} | {d['Time (min)']} | "
            f"{d['SLA Breaches']} | {d['Lateness (min)']} | {d['Unassigned']} | "
            f"{d['Solve Time (s)']} | {d['Cost (J)']} |"
        )


def evaluate_route_plan(
    routes: dict[str, list[str]],       # vehicle_id → ordered list of stop_ids
    duration_matrix: np.ndarray,         # (N+1)×(N+1) seconds, row 0 = depot
    distance_matrix: np.ndarray,         # (N+1)×(N+1) metres, row 0 = depot
    stops: list[Stop],
    vehicles: list[Vehicle],
    weights: ObjectiveWeights | None = None,
    solve_time_seconds: float = 0.0,
    algorithm: str = "unknown",
    unassigned: list[str] | None = None,
) -> SolutionMetrics:
    """
    Traverse every route step-by-step using cumulative arrival times and
    compute all SolutionMetrics from first principles.

    Matrix convention:
        duration_matrix[i, j] = travel time in SECONDS from location i to j
        Row/col 0 = depot; row/col k = stops[k-1]

    Waiting is counted when a vehicle arrives before tw_open (must wait).
    Lateness is counted when a vehicle arrives after tw_close.
    """
    if weights is None:
        weights = ObjectiveWeights()

    # Build stop lookup
    stop_map: dict[str, Stop] = {s.stop_id: s for s in stops}
    stop_matrix_idx: dict[str, int] = {s.stop_id: i + 1 for i, s in enumerate(stops)}
    vehicle_map: dict[str, Vehicle] = {v.vehicle_id: v for v in vehicles}

    total_dist_m = 0.0
    total_transit_s = 0.0
    total_dwell_s = 0.0
    total_wait_s = 0.0
    sla_violations = 0
    total_late_s = 0.0
    total_overtime_s = 0.0
    capacity_violations = 0
    vehicle_durations: list[float] = []  # per-vehicle total active duration (s)
    vehicles_used = 0

    for vid, stop_ids in routes.items():
        if not stop_ids:
            continue
        vehicles_used += 1

        v = vehicle_map.get(vid)
        if v is None:
            log.warning("Vehicle %s in routes but not in fleet — skipping.", vid)
            continue

        cur_time_s = float(v.shift_start_s)
        cur_node = 0  # depot
        route_dist_m = 0.0
        route_transit_s = 0.0
        route_dwell_s = 0.0
        route_wait_s = 0.0
        load_weight = 0.0
        load_volume = 0.0

        for sid in stop_ids:
            if sid not in stop_map:
                log.warning("Stop %s not found in stop_map — skipping.", sid)
                continue

            s = stop_map[sid]
            j = stop_matrix_idx[sid]

            # Travel from current node to this stop
            travel_s = float(duration_matrix[cur_node, j])
            dist_m = float(distance_matrix[cur_node, j])
            arrival_s = cur_time_s + travel_s

            # Wait if we arrive before the time window opens
            wait_s = max(0.0, s.tw_open_s - arrival_s)
            start_service_s = arrival_s + wait_s

            # SLA: lateness is arrival (not service start) vs tw_close
            late_s = max(0.0, arrival_s - s.tw_close_s)
            if late_s > 0:
                sla_violations += 1
                total_late_s += late_s

            # Dwell at stop
            dwell_s = s.service_time_s

            # Accumulate for this stop
            route_dist_m += dist_m
            route_transit_s += travel_s
            route_dwell_s += dwell_s
            route_wait_s += wait_s
            load_weight += s.demand_weight
            load_volume += s.demand_volume

            # Advance clock
            cur_time_s = start_service_s + dwell_s
            cur_node = j

        # Return to depot
        return_travel_s = float(duration_matrix[cur_node, 0])
        return_dist_m = float(distance_matrix[cur_node, 0])
        return_time_s = cur_time_s + return_travel_s

        route_dist_m += return_dist_m
        route_transit_s += return_travel_s

        # Overtime: return after shift end
        overtime_s = max(0.0, return_time_s - v.shift_end_s)
        total_overtime_s += overtime_s

        # Capacity violations
        if load_weight > v.max_weight or load_volume > v.max_volume:
            capacity_violations += 1

        # Accumulate totals
        total_dist_m += route_dist_m
        total_transit_s += route_transit_s
        total_dwell_s += route_dwell_s
        total_wait_s += route_wait_s

        active_duration_s = return_time_s - v.shift_start_s
        vehicle_durations.append(active_duration_s)

    # Unassigned count
    assigned_ids: set[str] = {sid for route in routes.values() for sid in route}
    n_unassigned = len(unassigned) if unassigned is not None else len(
        [s for s in stops if s.stop_id not in assigned_ids]
    )

    # Workload variance (in minutes²)
    workload_min = [d / 60.0 for d in vehicle_durations]
    wl_var = float(np.var(workload_min)) if len(workload_min) > 1 else 0.0

    # Unit conversions
    dist_km = total_dist_m / 1000.0
    transit_min = total_transit_s / 60.0
    dwell_min = total_dwell_s / 60.0
    total_min = (total_transit_s + total_dwell_s + total_wait_s) / 60.0
    late_min = total_late_s / 60.0
    overtime_min = total_overtime_s / 60.0

    # ── Composite objective J ────────────────────────────────────────────────
    # J = w1*dist_km + w2*late_min + w3*overtime_min + w4*unassigned + w6*wl_var
    J = (
        weights.w1_distance * dist_km
        + weights.w2_lateness * late_min
        + weights.w3_overtime * overtime_min
        + weights.w4_unassigned * n_unassigned
        + weights.w6_workload_var * wl_var
    )

    return SolutionMetrics(
        total_distance_km=round(dist_km, 3),
        total_transit_duration_min=round(transit_min, 2),
        total_dwell_duration_min=round(dwell_min, 2),
        total_duration_min=round(total_min, 2),
        sla_violations_count=sla_violations,
        total_lateness_min=round(late_min, 2),
        vehicles_used=vehicles_used,
        total_overtime_min=round(overtime_min, 2),
        workload_variance=round(wl_var, 4),
        unassigned_orders_count=n_unassigned,
        capacity_violations=capacity_violations,
        solve_time_seconds=round(solve_time_seconds, 4),
        algorithm=algorithm,
        composite_objective_cost=round(J, 2),
    )


def print_comparison_table(metrics_list: list[SolutionMetrics]) -> str:
    """
    Render a clean Markdown comparison table for all evaluated solvers.
    Returns the table string and also prints it.
    """
    header = (
        "| Method | Distance (km) | Time (min) | SLA Breaches | "
        "Lateness (min) | Unassigned | Solve Time (s) | Cost (J) |"
    )
    separator = "|" + "|".join(["-" * (len(h) + 2) for h in header.split("|")[1:-1]]) + "|"
    rows = [m.markdown_row() for m in metrics_list]

    table = "\n".join([header, separator] + rows)
    print("\n" + table + "\n")
    return table
