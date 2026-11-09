"""
Phase 1 — PyVRP Benchmarking Wrapper
======================================
Wraps PyVRP (Hybrid Genetic Search) for CVRPTW to benchmark against
OR-Tools on CVRPLib Solomon/Homberger instances and Vadodara workdays.

PyVRP is used as the "Challenger 2 (Target)" in Experiment 1 of the
ablation table.

Usage:
    from solver.pyvrp_benchmark.pyvrp_solver import PyVRPSolver
    result = PyVRPSolver(stops, vehicles, duration_matrix, distance_matrix).solve()
"""
from __future__ import annotations

import logging
import time

import numpy as np
from pyvrp import Model, ProblemData, Client, Depot, VehicleType, Edge
from pyvrp.stop import MaxRuntime

from solver.baselines.greedy import Stop, Vehicle, SolverResult

log = logging.getLogger(__name__)

_SCALE = 100  # integer scaling (metres → centimetres)


class PyVRPSolver:
    """
    PyVRP (HGS) CVRPTW solver.

    Builds a pyvrp.Model from RouteMind Stop/Vehicle objects and
    runs Hybrid Genetic Search with a configurable time budget.
    """

    def __init__(
        self,
        stops: list[Stop],
        vehicles: list[Vehicle],
        duration_matrix: np.ndarray,  # seconds, shape (N+1, N+1), row 0 = depot
        distance_matrix: np.ndarray,  # metres
    ):
        self.stops = stops
        self.vehicles = vehicles
        self.dur_mat = duration_matrix
        self.dist_mat = distance_matrix

    def solve(self, time_limit_s: int = 60) -> SolverResult:
        t0 = time.perf_counter()
        m = Model()

        # ── Depot ──────────────────────────────────────────────────────────────
        depot_v = self.vehicles[0]
        m.add_depot(
            x=int(depot_v.depot_lon * 1e5),
            y=int(depot_v.depot_lat * 1e5),
        )

        # ── Clients (stops) ────────────────────────────────────────────────────
        for s in self.stops:
            m.add_client(
                x=int(s.lon * 1e5),
                y=int(s.lat * 1e5),
                tw_early=s.tw_open_s,
                tw_late=s.tw_close_s,
                service_duration=int(s.service_time_s),
                delivery=(int(s.demand_weight * _SCALE), int(s.demand_volume * _SCALE)),
                name=s.stop_id,
            )

        # ── Vehicle types ──────────────────────────────────────────────────────
        # Group vehicles by type for PyVRP VehicleType
        seen: dict[str, int] = {}
        for v in self.vehicles:
            key = f"{v.vtype}_{int(v.max_weight)}_{int(v.max_volume)}"
            if key not in seen:
                seen[key] = 0
                m.add_vehicle_type(
                    capacity=(int(v.max_weight * _SCALE), int(v.max_volume * _SCALE)),
                    num_available=0,
                    tw_early=v.shift_start_s,
                    tw_late=v.shift_end_s,
                    name=v.vtype,
                )
            seen[key] += 1

        # Update counts
        for idx, key in enumerate(seen):
            m.vehicle_types()[idx] = m.vehicle_types()[idx]._replace(
                num_available=seen[key]
            )

        # ── Distance/Duration matrix ───────────────────────────────────────────
        n = len(self.stops) + 1
        for i in range(n):
            for j in range(n):
                if i != j:
                    m.add_edge(
                        frm=i,
                        to=j,
                        distance=int(self.dist_mat[i, j]),
                        duration=int(self.dur_mat[i, j]),
                    )

        # ── Solve ──────────────────────────────────────────────────────────────
        result = m.solve(stop=MaxRuntime(time_limit_s), seed=42)
        solve_time = time.perf_counter() - t0

        # ── Extract routes ─────────────────────────────────────────────────────
        routes: dict[str, list[str]] = {}
        assigned: set[str] = set()
        stop_names = [s.stop_id for s in self.stops]

        for vi, route in enumerate(result.best.routes()):
            vid = self.vehicles[vi].vehicle_id if vi < len(self.vehicles) else f"V{vi:03d}"
            stop_ids = [stop_names[visit.client() - 1] for visit in route]
            routes[vid] = stop_ids
            assigned.update(stop_ids)

        unassigned = [s.stop_id for s in self.stops if s.stop_id not in assigned]
        cost = result.best.distance() if result.is_feasible() else float("inf")

        log.info(
            "[PyVRP HGS t=%ds] cost=%.0f | unassigned=%d | feasible=%s | solve=%.2f s",
            time_limit_s, cost, len(unassigned), result.is_feasible(), solve_time,
        )

        return SolverResult(
            routes=routes,
            total_distance_m=float(cost),
            total_lateness_s=0.0,  # PyVRP enforces TW as hard constraint
            unassigned=unassigned,
            solve_time_s=solve_time,
            algorithm=f"PyVRP-HGS-t{time_limit_s}s",
            extra={"feasible": result.is_feasible()},
        )
