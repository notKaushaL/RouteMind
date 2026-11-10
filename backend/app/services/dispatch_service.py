"""
backend/app/services/dispatch_service.py
========================================
Dispatch planning service orchestrating heuristics, OR-Tools, CQR, and GATv2 warm-start.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple
import numpy as np

from backend.app.schemas.dispatch import (
    PlanRequest,
    PlanResponse,
    PlanMetrics,
    VehicleSummary,
    ScheduleStop,
    ObjectiveWeightsSchema,
)
from backend.app.services.engine_registry import registry
from configs.settings import settings
from solver.baselines.greedy import Stop, Vehicle, NearestNeighborGreedy, CheapestInsertionGreedy
from solver.evaluation import evaluate_route_plan, ObjectiveWeights, SolutionMetrics
from solver.ortools_engine.vrptw_solver import VRPTWSolver, SolverConfig
from scripts.benchmark_phase3 import build_daily_instance, DEPOT_LAT, DEPOT_LON
from ml.gnn_sequencer.decoder import GNNSequenceDecoder
from ml.gnn_sequencer.schema import RouteGraphData, RouteGraphNode

log = logging.getLogger(__name__)

VEHICLE_COLORS = {
    "V_TW1": "#38bdf8",  # Sky Blue (Three-Wheeler 1)
    "V_TW2": "#34d399",  # Emerald Green (Three-Wheeler 2)
    "V_VAN": "#f472b6",  # Pink / Magenta (Van)
    "V_FALLBACK_0": "#38bdf8",
    "V_FALLBACK_1": "#34d399",
    "V_FALLBACK_2": "#f472b6",
}


def _sec_to_clock(seconds: float) -> str:
    """Format seconds from midnight to HH:MM."""
    s = int(seconds) % (24 * 3600)
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h:02d}:{m:02d}"


def generate_dispatch_plan(req: PlanRequest) -> PlanResponse:
    """Execute chosen dispatch solver and return full GeoJSON routes and metrics."""
    t0 = time.perf_counter()

    # 1. Build stop instances and matrices
    stops, fleet, dur_matrix, dist_matrix, stop_clusters = build_daily_instance(
        n_stops=req.n_stops, seed=req.day_seed
    )
    stops_dict = {s.stop_id: s for s in stops}
    stop_to_idx = {s.stop_id: i + 1 for i, s in enumerate(stops)}
    stop_to_idx["DEPOT"] = 0

    # 2. Objective Weights
    w_schema = req.weights or ObjectiveWeightsSchema()
    weights = ObjectiveWeights(
        w1_distance=w_schema.w1_distance,
        w2_lateness=w_schema.w2_lateness,
        w3_overtime=w_schema.w3_overtime,
        w4_unassigned=w_schema.w4_unassigned,
        w5_stability=w_schema.w5_stability,
        w6_workload_var=w_schema.w6_workload_var,
    )

    # 3. Solver Execution
    routes: Dict[str, List[str]] = {v.vehicle_id: [] for v in fleet}
    unassigned: List[str] = []

    if req.solver_type == "greedy":
        # Heuristic insertion
        solver = CheapestInsertionGreedy(stops, fleet, dur_matrix, dist_matrix)
        res = solver.solve()
        routes = res.routes
        unassigned = res.unassigned

    elif req.solver_type == "ortools_cold":
        # Cold-start OR-Tools
        cqr_predictor = None
        if req.risk_mode == "p90_robust" and registry.cqr_model:
            cqr_predictor = registry.cqr_model

        solver = VRPTWSolver(
            stops,
            fleet,
            dur_matrix,
            dist_matrix,
            eta_predictor=cqr_predictor,
        )
        cfg = SolverConfig(
            w1_distance=weights.w1_distance,
            w2_lateness=weights.w2_lateness,
            w3_overtime=weights.w3_overtime,
            w4_unassigned=weights.w4_unassigned,
            w5_stability=weights.w5_stability,
            w6_workload_var=weights.w6_workload_var,
            time_limit_s=2,
            local_search="PATH_CHEAPEST_ARC",
        )
        res = solver.solve(config=cfg)
        routes = res.routes
        unassigned = res.unassigned

    else:
        # Proposed: GATv2 Neural Warm-Start + OR-Tools
        cqr_predictor = None
        if req.risk_mode == "p90_robust" and registry.cqr_model:
            cqr_predictor = registry.cqr_model

        warm_routes = None
        if registry.gatv2_model:
            try:
                graph_nodes = [
                    RouteGraphNode(
                        stop_id="DEPOT",
                        lat=DEPOT_LAT,
                        lon=DEPOT_LON,
                        tw_open_s=32400.0,
                        tw_close_s=61200.0,
                        demand_weight=0.0,
                        demand_volume=0.0,
                        service_time_s=0.0,
                        cluster_id=0,
                    )
                ] + [
                    RouteGraphNode(
                        stop_id=s.stop_id,
                        lat=s.lat,
                        lon=s.lon,
                        tw_open_s=s.tw_open_s,
                        tw_close_s=s.tw_close_s,
                        demand_weight=s.demand_weight,
                        demand_volume=s.demand_volume,
                        service_time_s=s.service_time_s,
                        cluster_id=stop_clusters.get(s.stop_id, 0),
                    )
                    for s in stops
                ]
                graph_data = RouteGraphData(nodes=graph_nodes, dur_matrix=dur_matrix, dist_matrix=dist_matrix)
                decoder = GNNSequenceDecoder(model=registry.gatv2_model)
                warm_routes = decoder.decode_all_vehicles(graph_data, fleet)
            except Exception as e:
                log.warning("GATv2 warm start decoding failed, falling back: %s", e)
                warm_routes = None

        solver = VRPTWSolver(
            stops,
            fleet,
            dur_matrix,
            dist_matrix,
            eta_predictor=cqr_predictor,
        )
        cfg = SolverConfig(
            w1_distance=weights.w1_distance,
            w2_lateness=weights.w2_lateness,
            w3_overtime=weights.w3_overtime,
            w4_unassigned=weights.w4_unassigned,
            w5_stability=weights.w5_stability,
            w6_workload_var=weights.w6_workload_var,
            time_limit_s=2,
            local_search="PATH_CHEAPEST_ARC",
        )
        res = solver.solve(config=cfg, initial_routes=warm_routes)
        routes = res.routes
        unassigned = res.unassigned

    solve_time_sec = time.perf_counter() - t0

    # 4. Standardized Evaluation Harness
    eval_metrics = evaluate_route_plan(
        routes=routes,
        duration_matrix=dur_matrix,
        distance_matrix=dist_matrix,
        stops=stops,
        vehicles=fleet,
        weights=weights,
        solve_time_seconds=solve_time_sec,
        algorithm=req.solver_type,
        unassigned=unassigned,
    )

    total_stops_eval = req.n_stops
    delivered_stops = total_stops_eval - len(unassigned) - eval_metrics.sla_violations_count
    on_time_pct = max(0.0, min(100.0, (delivered_stops / max(total_stops_eval, 1)) * 100.0))

    metrics_resp = PlanMetrics(
        total_distance_km=round(eval_metrics.total_distance_km, 2),
        total_transit_duration_min=round(eval_metrics.total_transit_duration_min, 1),
        total_dwell_duration_min=round(eval_metrics.total_dwell_duration_min, 1),
        total_duration_min=round(eval_metrics.total_duration_min, 1),
        sla_violations_count=eval_metrics.sla_violations_count,
        total_lateness_min=round(eval_metrics.total_lateness_min, 1),
        unassigned_orders_count=len(unassigned),
        solve_time_seconds=round(solve_time_sec, 3),
        composite_objective_cost=round(eval_metrics.composite_objective_cost, 2),
        on_time_pct=round(on_time_pct, 1),
    )

    # 5. Build GeoJSON Features
    geojson_features: List[Dict[str, Any]] = []

    # Depot Point Feature
    geojson_features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [DEPOT_LON, DEPOT_LAT],
        },
        "properties": {
            "type": "depot",
            "stop_id": "DEPOT",
            "name": "Vadodara Central Depot",
            "address": "Alkapuri Hub, Vadodara",
        },
    })

    # Schedule Table & Vehicle Summaries
    schedule_rows: List[ScheduleStop] = []
    vehicle_summaries: List[VehicleSummary] = []

    vehicle_map = {v.vehicle_id: v for v in fleet}

    for vi, (vid, stop_ids) in enumerate(routes.items()):
        v = vehicle_map.get(vid)
        color = VEHICLE_COLORS.get(vid, "#38bdf8")

        v_dist_km = 0.0
        v_dur_min = 0.0
        v_weight_kg = 0.0
        v_breaches = 0

        coords_line = [[DEPOT_LON, DEPOT_LAT]]
        cur_node = 0
        cur_t_s = float(v.shift_start_s) if v else 32400.0

        for seq, sid in enumerate(stop_ids):
            s = stops_dict.get(sid)
            if not s:
                continue

            j = stop_to_idx.get(sid, 0)
            transit_s = float(dur_matrix[cur_node, j])
            dist_m = float(dist_matrix[cur_node, j])
            arr_s = cur_t_s + transit_s

            wait_s = max(0.0, s.tw_open_s - arr_s)
            service_start_s = arr_s + wait_s
            dep_s = service_start_s + s.service_time_s

            late_min = max(0.0, (arr_s - s.tw_close_s) / 60.0)
            is_breached = arr_s > s.tw_close_s
            if is_breached:
                v_breaches += 1

            v_dist_km += dist_m / 1000.0
            v_dur_min += (dep_s - cur_t_s) / 60.0
            v_weight_kg += s.demand_weight

            coords_line.append([s.lon, s.lat])
            h3_cell_val = getattr(s, "h3_cell", None) or f"88619b456{seq % 16:01x}fff"

            # Stop Point Feature
            geojson_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [s.lon, s.lat],
                },
                "properties": {
                    "type": "stop",
                    "stop_id": s.stop_id,
                    "sequence_idx": seq + 1,
                    "vehicle_id": vid,
                    "color": color,
                    "h3_cell": h3_cell_val,
                    "cluster_id": stop_clusters.get(s.stop_id, 0),
                    "demand_weight_kg": s.demand_weight,
                    "demand_volume_l": s.demand_volume,
                    "tw_open_time": _sec_to_clock(s.tw_open_s),
                    "tw_close_time": _sec_to_clock(s.tw_close_s),
                    "arrival_time": _sec_to_clock(arr_s),
                    "departure_time": _sec_to_clock(dep_s),
                    "lateness_min": round(late_min, 1),
                    "is_sla_breached": is_breached,
                    "is_priority": s.is_priority,
                },
            })

            schedule_rows.append(ScheduleStop(
                stop_id=s.stop_id,
                vehicle_id=vid,
                sequence_idx=seq + 1,
                lat=s.lat,
                lon=s.lon,
                h3_cell=h3_cell_val,
                cluster_id=stop_clusters.get(s.stop_id, 0),
                demand_weight_kg=s.demand_weight,
                demand_volume_l=s.demand_volume,
                tw_open_time=_sec_to_clock(s.tw_open_s),
                tw_close_time=_sec_to_clock(s.tw_close_s),
                arrival_time=_sec_to_clock(arr_s),
                departure_time=_sec_to_clock(dep_s),
                lateness_min=round(late_min, 1),
                is_sla_breached=is_breached,
                is_priority=s.is_priority,
            ))

            cur_node = j
            cur_t_s = dep_s

        # Return to depot
        if stop_ids:
            coords_line.append([DEPOT_LON, DEPOT_LAT])

        # Route LineString Feature
        geojson_features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords_line,
            },
            "properties": {
                "type": "route",
                "vehicle_id": vid,
                "color": color,
                "stop_count": len(stop_ids),
                "total_distance_km": round(v_dist_km, 2),
                "total_duration_min": round(v_dur_min, 1),
            },
        })

        max_cap = v.max_weight if v else 200.0
        util_pct = round((v_weight_kg / max(max_cap, 1e-3)) * 100.0, 1)

        vehicle_summaries.append(VehicleSummary(
            vehicle_id=vid,
            vtype=v.vtype if v else "three_wheeler",
            color=color,
            total_stops=len(stop_ids),
            total_distance_km=round(v_dist_km, 2),
            total_duration_min=round(v_dur_min, 1),
            max_weight_capacity_kg=max_cap,
            utilized_weight_kg=round(v_weight_kg, 1),
            capacity_utilization_pct=util_pct,
            sla_breaches=v_breaches,
        ))

    geojson_collection = {
        "type": "FeatureCollection",
        "features": geojson_features,
    }

    return PlanResponse(
        workday_id=req.workday_id,
        solver_type=req.solver_type,
        risk_mode=req.risk_mode,
        metrics=metrics_resp,
        vehicles=vehicle_summaries,
        routes_geojson=geojson_collection,
        schedule=schedule_rows,
    )
