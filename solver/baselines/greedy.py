"""
Phase 1 — Solver Baselines
===========================
Two deterministic construction heuristics used as the lowest bar
in every ablation table:

  1. NearestNeighborGreedy — builds a route greedily by always
     dispatching the nearest unvisited feasible stop.

  2. CheapestInsertionGreedy — inserts each unrouted stop into the
     cheapest feasible gap in an existing route.

Both run in < 1 ms per problem instance and serve as the "Baseline"
column in the ablation experiments.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ── Data contracts ─────────────────────────────────────────────────────────────

@dataclass
class Stop:
    stop_id: str
    lat: float
    lon: float
    demand_weight: float
    demand_volume: float
    tw_open_s: int           # earliest arrival (seconds from midnight)
    tw_close_s: int          # latest arrival (hard deadline)
    service_time_s: float    # dwell + handoff time
    is_priority: bool = False
    vehicle_compat: list[str] = field(default_factory=lambda: ["bike", "three_wheeler", "van"])


@dataclass
class Vehicle:
    vehicle_id: str
    vtype: str
    max_weight: float
    max_volume: float
    shift_start_s: int
    shift_end_s: int
    depot_lat: float
    depot_lon: float


@dataclass
class SolverResult:
    routes: dict[str, list[str]]          # vehicle_id → ordered list of stop_ids
    total_distance_m: float
    total_lateness_s: float               # sum of max(0, arrival - tw_close)
    unassigned: list[str]                 # stop_ids not served
    solve_time_s: float
    algorithm: str
    extra: dict[str, Any] = field(default_factory=dict)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Fast haversine distance in metres."""
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _travel_time_s(
    lat1: float, lon1: float, lat2: float, lon2: float,
    duration_matrix: np.ndarray | None,
    idx1: int, idx2: int,
    speed_factor: float = 1.0,
) -> float:
    """Return travel time in seconds. Uses matrix when available, else 30 km/h haversine."""
    if duration_matrix is not None:
        return float(duration_matrix[idx1, idx2]) / speed_factor
    dist = _haversine_m(lat1, lon1, lat2, lon2)
    return dist / (30_000 / 3600) / speed_factor   # 30 km/h default


# ── Nearest-Neighbor Greedy ────────────────────────────────────────────────────

class NearestNeighborGreedy:
    """
    For each vehicle (sorted by shift start), greedily extend the route
    with the closest feasible unvisited stop. Vehicles are refilled until
    all stops are served or infeasible.
    """

    def __init__(
        self,
        stops: list[Stop],
        vehicles: list[Vehicle],
        duration_matrix: np.ndarray | None = None,
        distance_matrix: np.ndarray | None = None,
        stop_index: dict[str, int] | None = None,   # stop_id → matrix row
    ):
        self.stops = stops
        self.vehicles = vehicles
        self.dur_mat = duration_matrix
        self.dist_mat = distance_matrix
        self.stop_idx = stop_index or {s.stop_id: i for i, s in enumerate(stops)}

    def solve(self) -> SolverResult:
        import time
        t0 = time.perf_counter()

        stops_map: dict[str, Stop] = {s.stop_id: s for s in self.stops}
        unvisited: set[str] = set(stops_map.keys())
        routes: dict[str, list[str]] = {v.vehicle_id: [] for v in self.vehicles}
        total_dist = 0.0
        total_late = 0.0

        for v in sorted(self.vehicles, key=lambda x: x.shift_start_s):
            cur_lat, cur_lon = v.depot_lat, v.depot_lon
            cur_time = float(v.shift_start_s)
            load_w = 0.0
            load_vol = 0.0
            route = routes[v.vehicle_id]
            cur_idx = -1  # depot has no matrix index

            while unvisited:
                best_id: str | None = None
                best_dist = float("inf")

                for sid in unvisited:
                    s = stops_map[sid]
                    if v.vtype not in s.vehicle_compat:
                        continue
                    if load_w + s.demand_weight > v.max_weight:
                        continue
                    if load_vol + s.demand_volume > v.max_volume:
                        continue

                    s_idx = self.stop_idx[sid]
                    travel = _travel_time_s(cur_lat, cur_lon, s.lat, s.lon, self.dur_mat, 0 if cur_idx < 0 else cur_idx, s_idx)
                    arr = cur_time + travel

                    # Hard shift check
                    depart = max(arr, s.tw_open_s) + s.service_time_s
                    if depart > v.shift_end_s:
                        continue

                    dist = _haversine_m(cur_lat, cur_lon, s.lat, s.lon)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = sid

                if best_id is None:
                    break

                s = stops_map[best_id]
                s_idx = self.stop_idx[best_id]
                travel = _travel_time_s(cur_lat, cur_lon, s.lat, s.lon, self.dur_mat, 0, s_idx)
                arr = cur_time + travel
                late = max(0.0, arr - s.tw_close_s)

                total_dist += best_dist
                total_late += late

                cur_time = max(arr, s.tw_open_s) + s.service_time_s
                cur_lat, cur_lon = s.lat, s.lon
                cur_idx = s_idx
                load_w += s.demand_weight
                load_vol += s.demand_volume

                route.append(best_id)
                unvisited.discard(best_id)

        solve_time = time.perf_counter() - t0
        log.info(
            "[NN-Greedy] dist=%.0f m | lateness=%.0f s | unassigned=%d | time=%.3f s",
            total_dist, total_late, len(unvisited), solve_time,
        )
        return SolverResult(
            routes=routes,
            total_distance_m=total_dist,
            total_lateness_s=total_late,
            unassigned=list(unvisited),
            solve_time_s=solve_time,
            algorithm="NearestNeighborGreedy",
        )


# ── Cheapest Insertion Greedy ──────────────────────────────────────────────────

class CheapestInsertionGreedy:
    """
    Builds routes by repeatedly inserting each unrouted stop into the
    cheapest feasible position across all vehicle routes.
    O(N² × R) — acceptable for ≤ 300 stops.
    """

    def __init__(
        self,
        stops: list[Stop],
        vehicles: list[Vehicle],
        duration_matrix: np.ndarray | None = None,
        distance_matrix: np.ndarray | None = None,
        stop_index: dict[str, int] | None = None,
    ):
        self.stops_map: dict[str, Stop] = {s.stop_id: s for s in stops}
        self.vehicles = vehicles
        self.dur_mat = duration_matrix
        self.dist_mat = distance_matrix
        self.stop_idx = stop_index or {s.stop_id: i for i, s in enumerate(stops)}

    def _insertion_cost(
        self, prev_lat: float, prev_lon: float, s: Stop, next_lat: float, next_lon: float
    ) -> float:
        """Additional distance when inserting stop s between prev and next."""
        return (
            _haversine_m(prev_lat, prev_lon, s.lat, s.lon)
            + _haversine_m(s.lat, s.lon, next_lat, next_lon)
            - _haversine_m(prev_lat, prev_lon, next_lat, next_lon)
        )

    def solve(self) -> SolverResult:
        import time
        t0 = time.perf_counter()

        unvisited = list(self.stops_map.keys())
        # Sort priority first
        unvisited.sort(key=lambda sid: (not self.stops_map[sid].is_priority, self.stops_map[sid].tw_close_s))

        # Each vehicle route: list of stop_ids (excluding depot)
        routes: dict[str, list[str]] = {v.vehicle_id: [] for v in self.vehicles}
        vehicle_map: dict[str, Vehicle] = {v.vehicle_id: v for v in self.vehicles}

        total_dist = 0.0
        total_late = 0.0
        still_unassigned: list[str] = []

        for sid in unvisited:
            s = self.stops_map[sid]
            best_cost = float("inf")
            best_vid: str | None = None
            best_pos: int = 0

            for v in self.vehicles:
                if v.vtype not in s.vehicle_compat:
                    continue

                route = routes[v.vehicle_id]
                # Check capacity
                total_w = sum(self.stops_map[r].demand_weight for r in route) + s.demand_weight
                total_vol = sum(self.stops_map[r].demand_volume for r in route) + s.demand_volume
                if total_w > v.max_weight or total_vol > v.max_volume:
                    continue

                # Try every insertion position
                for pos in range(len(route) + 1):
                    prev_lat = v.depot_lat if pos == 0 else self.stops_map[route[pos - 1]].lat
                    prev_lon = v.depot_lon if pos == 0 else self.stops_map[route[pos - 1]].lon
                    next_lat = v.depot_lat if pos == len(route) else self.stops_map[route[pos]].lat
                    next_lon = v.depot_lon if pos == len(route) else self.stops_map[route[pos]].lon

                    cost = self._insertion_cost(prev_lat, prev_lon, s, next_lat, next_lon)
                    if cost < best_cost:
                        best_cost = cost
                        best_vid = v.vehicle_id
                        best_pos = pos

            if best_vid is not None:
                routes[best_vid].insert(best_pos, sid)
                total_dist += best_cost
            else:
                still_unassigned.append(sid)

        solve_time = time.perf_counter() - t0
        log.info(
            "[Cheapest-Insertion] dist=%.0f m | unassigned=%d | time=%.3f s",
            total_dist, len(still_unassigned), solve_time,
        )
        return SolverResult(
            routes=routes,
            total_distance_m=total_dist,
            total_lateness_s=total_late,
            unassigned=still_unassigned,
            solve_time_s=solve_time,
            algorithm="CheapestInsertionGreedy",
        )
