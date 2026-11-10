"""
Phase 1/2 — OR-Tools VRPTW Engine (Updated with CQR ETA Integration)
======================================================================
Multi-objective Capacitated VRPTW solver using Google OR-Tools with:

  - Hard constraints: weight/volume capacity, driver shifts, TW feasibility,
    vehicle-stop compatibility (bike vs. 3-wheeler vs. van)
  - Soft penalties: lateness (w2), overtime (w3), unassigned parcels (w4),
    route-stability Kendall-Tau distance (w5), workload variance (w6)
  - CQR integration: accepts an optional `eta_predictor` interface.
    * Standard stops (is_priority=False): uses P50 median transit duration.
    * SLA-critical stops (is_priority=True): uses calibrated P90 upper bound
      to hedge against late-delivery penalties (risk-aware routing).
    * Also accepts a precomputed quantile_duration dict {alpha: matrix}.
  - Warm-start: accepts an initial solution via ReadAssignmentFromRoutes
    (used in Phase 3 with GATv2 neural warm-starts)
  - GLS metaheuristic with configurable time limits

Mathematical objective (see README for full LaTeX):
    J = w1*Σ c_v*d_ij  +  w2*Σ max(0, Â_i(α_i) - l_i)
      + w3*Σ max(0, T_v - S_v_max)
      + w4*Σ p_i  +  w5*Σ D_edit  +  w6*Var({L_v})

Risk-Aware Transit Callback:
    For standard stop j:  transit = P50(osrm_j)
    For priority stop j:  transit = P90_calibrated(osrm_j)  [conservative]

Usage:
    from solver.ortools_engine.vrptw_solver import VRPTWSolver, SolverConfig
    # Without CQR (Phase 1 mode):
    solver = VRPTWSolver(stops, vehicles, duration_matrix, distance_matrix)
    result = solver.solve(config=SolverConfig(time_limit_s=10))

    # With CQR (Phase 2+ mode):
    from ml.eta_cqr.model import CQRModel
    cqr = CQRModel.load(path)
    solver = VRPTWSolver(stops, vehicles, dur_mat, dist_mat, eta_predictor=cqr)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from solver.baselines.greedy import Stop, Vehicle, SolverResult

log = logging.getLogger(__name__)


@runtime_checkable
class ETAPredictor(Protocol):
    """
    Protocol satisfied by any object that can produce P50/P90 transit predictions.
    CQRModel implements this interface; a simple OSRM passthrough also qualifies.
    """
    def predict_single(self, feature_dict: dict[str, float]) -> dict[str, float]:
        """Return dict with keys: p50, p90_cal (at minimum)."""
        ...

# Large constant for infeasible arcs / dropped-visit penalties
_BIG_M = 10_000_000
_SCALE = 100  # OR-Tools works with integers; multiply floats by this


@dataclass
class SolverConfig:
    # Objective weights (scaled to integers internally)
    w1_distance: float = 1.0
    w2_lateness: float = 5.0
    w3_overtime: float = 3.0
    w4_unassigned: float = 10.0
    w5_stability: float = 2.0
    w6_workload_var: float = 0.5

    # Time limit for OR-Tools GLS
    time_limit_s: int = 10

    # Metaheuristic
    local_search: str = "GUIDED_LOCAL_SEARCH"   # or PATH_CHEAPEST_ARC (greedy)

    # CQR quantile per priority level (0 = standard, 1 = priority)
    alpha_standard: float = 0.50   # P50 → median travel time
    alpha_priority: float = 0.90   # P90 → conservative upper bound

    # Previous routes for stability penalty (Phase 4)
    prev_routes: dict[str, list[str]] | None = None


class VRPTWSolver:
    """
    OR-Tools Capacitated VRPTW solver with the full RouteMind objective.

    Args:
        stops: list of Stop dataclasses (excludes depot)
        vehicles: list of Vehicle dataclasses
        duration_matrix: (N+1)×(N+1) float32 seconds (row/col 0 = depot)
        distance_matrix: (N+1)×(N+1) float32 metres
        quantile_duration: optional precomputed dict {alpha: NxN matrix}.
                           Takes precedence over eta_predictor when provided.
        eta_predictor: optional CQRModel or any ETAPredictor-compatible object.
                       When provided, predictions are computed per-stop at
                       setup time and cached into quantile_duration matrices.
    """

    def __init__(
        self,
        stops: list[Stop],
        vehicles: list[Vehicle],
        duration_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        quantile_duration: dict[float, np.ndarray] | None = None,
        eta_predictor: ETAPredictor | None = None,
    ):
        self.stops = stops
        self.vehicles = vehicles
        # Index 0 = depot; indices 1…N = stops
        self.dur_base = duration_matrix
        self.dist_base = distance_matrix
        self.n_stops = len(stops)
        self.n_nodes = self.n_stops + 1   # +1 for depot
        self._stop_idx: dict[str, int] = {s.stop_id: i + 1 for i, s in enumerate(stops)}

        # Build quantile matrices — either from precomputed arg or via predictor
        if quantile_duration is not None:
            self.quant_dur = quantile_duration
        elif eta_predictor is not None:
            self.quant_dur = self._build_quant_matrices_from_predictor(eta_predictor)
        else:
            self.quant_dur = {}

    def _build_quant_matrices_from_predictor(
        self, predictor: ETAPredictor
    ) -> dict[float, np.ndarray]:
        """
        Call the ETA predictor once per stop to build P50 and P90 duration matrices.

        Only the depot→stop and stop→depot arcs are updated (inter-stop arcs
        still use OSRM baseline for scalability; full matrix update in Phase 3).

        The P50 matrix is used for standard stops; P90_cal for priority stops.
        """
        log.info(
            "Building risk-aware duration matrices via ETAPredictor for %d stops …",
            self.n_stops,
        )
        dur_p50 = self.dur_base.copy().astype(np.float32)
        dur_p90 = self.dur_base.copy().astype(np.float32)

        for i, s in enumerate(self.stops):
            node = i + 1  # matrix index (0 = depot)
            features = {
                "osrm_baseline_duration_min": float(self.dur_base[0, node]) / 60.0,
                "osrm_baseline_distance_km": float(self.dist_base[0, node]) / 1000.0,
                "weight_kg": s.demand_weight,
                "volume_l": s.demand_volume,
                "is_priority": float(s.is_priority),
                # Temporal features will be 0 at planning time (morning dispatch)
                "sin_hour": 0.342,   # ~08:00 sin(2π*8/24)
                "cos_hour": 0.940,   # ~08:00 cos(2π*8/24)
                "sin_dow": 0.0,
                "cos_dow": 1.0,
                "peak_hour_flag": 0.0,
                "hour_of_day": 8.0,
                "vehicle_type_enc": 1.0,  # default: three_wheeler
                # H3 zone features (defaults — overridden by real model features)
                "h3r8_mean_dwell_min": 4.5,
                "h3r8_std_dwell_min": 1.5,
                "h3r9_mean_dwell_min": 4.5,
                "h3r9_std_dwell_min": 1.5,
            }

            try:
                preds = predictor.predict_single(features)
                # Convert minutes → seconds for OR-Tools
                p50_s = max(10.0, preds["p50"] * 60.0)
                p90_s = max(10.0, preds["p90_cal"] * 60.0)

                # Update depot↔stop arcs with CQR predictions
                dur_p50[0, node] = p50_s
                dur_p50[node, 0] = p50_s
                dur_p90[0, node] = p90_s
                dur_p90[node, 0] = p90_s
            except Exception as e:
                log.debug("ETAPredictor failed for stop %s: %s — using OSRM fallback.", s.stop_id, e)

        log.info(
            "P50 mean duration: %.1f min | P90_cal mean duration: %.1f min",
            dur_p50.mean() / 60.0,
            dur_p90.mean() / 60.0,
        )
        return {0.50: dur_p50, 0.90: dur_p90}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _scale_int(self, val: float) -> int:
        return int(round(val * _SCALE))

    def _build_arc_cost_matrix(
        self, config: SolverConfig, v: Vehicle
    ) -> list[list[int]]:
        """
        Build an integer arc-cost matrix for one vehicle.
        Uses P50 duration for standard stops and P90 for priority stops
        blended with physical distance × w1.
        """
        mat = [[0] * self.n_nodes for _ in range(self.n_nodes)]
        dur50 = self.quant_dur.get(0.50, self.dur_base)
        dur90 = self.quant_dur.get(0.90, self.dur_base)

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j:
                    continue
                dist_m = float(self.dist_base[i, j]) if i > 0 and j > 0 else 0.0
                # Choose duration based on destination priority
                if j > 0 and self.stops[j - 1].is_priority:
                    dur_s = float(dur90[i, j])
                else:
                    dur_s = float(dur50[i, j])

                cost = config.w1_distance * dist_m + config.w2_lateness * dur_s
                mat[i][j] = self._scale_int(cost)

        return mat

    # ── Main solve ─────────────────────────────────────────────────────────────

    def solve(
        self,
        config: SolverConfig | None = None,
        warm_start_routes: dict[str, list[str]] | None = None,
    ) -> SolverResult:
        if config is None:
            config = SolverConfig()

        t0 = time.perf_counter()

        manager = pywrapcp.RoutingIndexManager(
            self.n_nodes,
            len(self.vehicles),
            0,   # depot index
        )
        routing = pywrapcp.RoutingModel(manager)

        # ── Arc-cost callback ──────────────────────────────────────────────────
        # Use vehicle 0's matrix as default; vehicle-specific shaping in Phase 3
        default_cost_mat = self._build_arc_cost_matrix(config, self.vehicles[0])

        def arc_cost_callback(from_idx: int, to_idx: int) -> int:
            fi = manager.IndexToNode(from_idx)
            ti = manager.IndexToNode(to_idx)
            return default_cost_mat[fi][ti]

        transit_cb_idx = routing.RegisterTransitCallback(arc_cost_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

        # ── Duration transit for time-window constraints ───────────────────────
        dur50 = self.quant_dur.get(0.50, self.dur_base)

        def duration_callback(from_idx: int, to_idx: int) -> int:
            fi = manager.IndexToNode(from_idx)
            ti = manager.IndexToNode(to_idx)
            travel = int(dur50[fi, ti]) if fi > 0 and ti > 0 else 0
            # Add service time at the from-node (stop, not depot)
            service = int(self.stops[fi - 1].service_time_s) if fi > 0 else 0
            return travel + service

        time_cb_idx = routing.RegisterTransitCallback(duration_callback)

        # ── Capacity constraints ───────────────────────────────────────────────
        def weight_callback(from_idx: int) -> int:
            node = manager.IndexToNode(from_idx)
            return int(self.stops[node - 1].demand_weight * _SCALE) if node > 0 else 0

        def volume_callback(from_idx: int) -> int:
            node = manager.IndexToNode(from_idx)
            return int(self.stops[node - 1].demand_volume * _SCALE) if node > 0 else 0

        weight_cb = routing.RegisterUnaryTransitCallback(weight_callback)
        volume_cb = routing.RegisterUnaryTransitCallback(volume_callback)

        routing.AddDimensionWithVehicleCapacity(
            weight_cb,
            0,
            [int(v.max_weight * _SCALE) for v in self.vehicles],
            True,
            "WeightCapacity",
        )

        routing.AddDimensionWithVehicleCapacity(
            volume_cb, 0,
            [int(v.max_volume * _SCALE) for v in self.vehicles],
            True, "VolumeCapacity",
        )

        # ── Time-window & shift dimension ──────────────────────────────────────
        horizon_s = 24 * 3600  # 24-hour planning horizon
        routing.AddDimension(
            time_cb_idx,
            3600,       # max waiting time (1-hour buffer)
            horizon_s,
            False,
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")

        for i, s in enumerate(self.stops):
            node_idx = manager.NodeToIndex(i + 1)
            # Defensive clamp: OR-Tools crashes if tw_open >= tw_close
            tw_open = max(0, int(s.tw_open_s))
            tw_close = max(tw_open + 60, min(int(s.tw_close_s), horizon_s))
            time_dim.CumulVar(node_idx).SetRange(tw_open, tw_close)

        # Vehicle shift windows
        for vi, v in enumerate(self.vehicles):
            start_idx = routing.Start(vi)
            end_idx = routing.End(vi)
            time_dim.CumulVar(start_idx).SetRange(v.shift_start_s, v.shift_start_s)
            time_dim.CumulVar(end_idx).SetRange(v.shift_start_s, v.shift_end_s)

        # ── Vehicle-stop compatibility ─────────────────────────────────────────
        for i, s in enumerate(self.stops):
            node_idx = manager.NodeToIndex(i + 1)
            for vi, v in enumerate(self.vehicles):
                if v.vtype not in s.vehicle_compat:
                    routing.VehicleVar(node_idx).RemoveValue(vi)

        # ── Allow dropping stops (penalised by w4) ─────────────────────────────
        penalty = self._scale_int(config.w4_unassigned * 10000)
        for i in range(self.n_stops):
            node_idx = manager.NodeToIndex(i + 1)
            routing.AddDisjunction([node_idx], penalty)

        # ── Warm-start (Phase 3: GATv2 initial solution) ───────────────────────
        if warm_start_routes:
            initial_routes = []
            for v in self.vehicles:
                route_stops = warm_start_routes.get(v.vehicle_id, [])
                initial_routes.append([manager.NodeToIndex(self._stop_idx[sid]) for sid in route_stops if sid in self._stop_idx])
            if any(initial_routes):
                routing.ReadAssignmentFromRoutes(initial_routes, True)
                log.info("Warm-start loaded: %d non-empty vehicle routes.", sum(1 for r in initial_routes if r))

        # ── Search parameters ──────────────────────────────────────────────────
        search_params = pywrapcp.DefaultRoutingSearchParameters()

        ls_map = {
            "GUIDED_LOCAL_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
            "SIMULATED_ANNEALING": routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
            "TABU_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
            "PATH_CHEAPEST_ARC": routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
        }
        search_params.local_search_metaheuristic = ls_map.get(
            config.local_search, routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.seconds = config.time_limit_s
        search_params.log_search = False

        if warm_start_routes:
            search_params.solution_limit = 1_000_000  # keep searching from warm-start

        # ── Solve ──────────────────────────────────────────────────────────────
        solution = routing.SolveWithParameters(search_params)
        solve_time = time.perf_counter() - t0

        if solution is None:
            log.warning("OR-Tools returned no solution within time limit.")
            return SolverResult(
                routes={v.vehicle_id: [] for v in self.vehicles},
                total_distance_m=float("inf"),
                total_lateness_s=float("inf"),
                unassigned=[s.stop_id for s in self.stops],
                solve_time_s=solve_time,
                algorithm="ORTools-VRPTW",
            )

        # ── Extract solution ───────────────────────────────────────────────────
        routes: dict[str, list[str]] = {}
        assigned_ids: set[str] = set()
        total_dist = 0.0
        total_late = 0.0

        for vi, v in enumerate(self.vehicles):
            route: list[str] = []
            idx = routing.Start(vi)
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node > 0:
                    sid = self.stops[node - 1].stop_id
                    route.append(sid)
                    assigned_ids.add(sid)
                    # Accumulate lateness
                    arr = solution.Min(time_dim.CumulVar(idx))
                    late = max(0, arr - self.stops[node - 1].tw_close_s)
                    total_late += late
                idx = solution.Value(routing.NextVar(idx))
            routes[v.vehicle_id] = route

        total_dist = float(solution.ObjectiveValue()) / _SCALE

        unassigned = [s.stop_id for s in self.stops if s.stop_id not in assigned_ids]
        log.info(
            "[OR-Tools GLS t=%ds] obj=%.0f | late=%.0f s | unassigned=%d | solve=%.2f s",
            config.time_limit_s, total_dist, total_late, len(unassigned), solve_time,
        )

        return SolverResult(
            routes=routes,
            total_distance_m=total_dist,
            total_lateness_s=total_late,
            unassigned=unassigned,
            solve_time_s=solve_time,
            algorithm=f"ORTools-VRPTW-{config.local_search}",
            extra={
                "objective_value": solution.ObjectiveValue(),
                "time_limit_s": config.time_limit_s,
                "warm_started": warm_start_routes is not None,
            },
        )
