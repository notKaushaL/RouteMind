"""
scripts/benchmark_phase3.py
============================
Phase 3 Ablation Benchmark — Neural-Guided Routing (GATv2) & OR-Tools Hybridization

Runs a 10-day evaluation benchmark comparing 3 dispatch configurations under identical
compute budgets:
  1. Config 1: Cold-Start OR-Tools GLS (10s budget, Path Cheapest Arc initial heuristic)
  2. Config 2: Arc-Cost Shaped OR-Tools GLS (10s budget, lambda = 0.5)
  3. Config 3 (Proposed): GATv2 Warm-Start + OR-Tools GLS (GATv2 decoded initial routes + 8s GLS)

Outputs a comprehensive Markdown comparison table:
  - Time-to-convergence (Composite Cost J at 2s, 5s, 10s)
  - Final Composite Cost J
  - Total Distance (km)
  - SLA Breach Rate (%)
  - Zone Coherence Score (% intra-cluster transitions)

Usage:
    python scripts/benchmark_phase3.py --n-stops 30 --n-days 10 --train-epochs 10
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

from configs.settings import settings
from solver.baselines.greedy import Stop, Vehicle
from ml.gnn_sequencer.dataset import (
    HighFidelityExpertRouteGenerator,
    VADODARA_CLUSTERS,
    DEPOT_LAT,
    DEPOT_LON,
    haversine_matrix,
)
from ml.gnn_sequencer.decoder import GNNSequenceDecoder
from ml.gnn_sequencer.model import RouteGATv2
from ml.gnn_sequencer.schema import RouteGraphData, RouteGraphNode
from ml.gnn_sequencer.train import train_gatv2, MODELS_DIR
from solver.evaluation import evaluate_route_plan, ObjectiveWeights, SolutionMetrics
from solver.ortools_engine.arc_shaping import ArcCostShaper
from solver.ortools_engine.vrptw_solver import VRPTWSolver, SolverConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "benchmark_results"


def compute_zone_coherence(
    routes: Dict[str, List[str]],
    stop_cluster_map: Dict[str, int],
) -> float:
    """
    Compute Zone Coherence Score:
    Ratio of consecutive stop-to-stop transitions that stay within the same neighborhood cluster.
    """
    intra_count = 0
    total_transitions = 0

    for vid, stop_ids in routes.items():
        if len(stop_ids) < 2:
            continue
        for k in range(len(stop_ids) - 1):
            c1 = stop_cluster_map.get(stop_ids[k], -1)
            c2 = stop_cluster_map.get(stop_ids[k + 1], -2)
            if c1 == c2 and c1 >= 0:
                intra_count += 1
            total_transitions += 1

    if total_transitions == 0:
        return 1.0
    return float(intra_count / total_transitions)


def build_daily_instance(
    n_stops: int,
    seed: int,
) -> Tuple[List[Stop], List[Vehicle], np.ndarray, np.ndarray, Dict[str, int]]:
    """Build a deterministic workday instance with cluster assignments."""
    rng = np.random.default_rng(seed)

    # Vehicles: 2 three-wheelers, 1 van
    fleet = [
        Vehicle(vehicle_id="V_TW1", vtype="three_wheeler", max_weight=200.0, max_volume=600.0, shift_start_s=6 * 3600, shift_end_s=18 * 3600, depot_lat=DEPOT_LAT, depot_lon=DEPOT_LON),
        Vehicle(vehicle_id="V_TW2", vtype="three_wheeler", max_weight=200.0, max_volume=600.0, shift_start_s=6 * 3600, shift_end_s=18 * 3600, depot_lat=DEPOT_LAT, depot_lon=DEPOT_LON),
        Vehicle(vehicle_id="V_VAN", vtype="van", max_weight=500.0, max_volume=1500.0, shift_start_s=6 * 3600, shift_end_s=18 * 3600, depot_lat=DEPOT_LAT, depot_lon=DEPOT_LON),
    ]

    active_cids = rng.choice(len(VADODARA_CLUSTERS), size=4, replace=False)
    clusters = [VADODARA_CLUSTERS[i] for i in active_cids]

    stops: List[Stop] = []
    stop_cluster_map: Dict[str, int] = {}

    for i in range(1, n_stops + 1):
        c_lat, c_lon, _, cid = clusters[i % len(clusters)]
        lat = float(np.clip(c_lat + rng.normal(0.0, 0.007), 22.20, 22.42))
        lon = float(np.clip(c_lon + rng.normal(0.0, 0.007), 73.10, 73.30))

        is_p = bool(rng.random() < 0.15)
        open_h = 7 if is_p else int(rng.integers(8, 14))
        tw_open_s = open_h * 3600
        tw_close_s = tw_open_s + 4 * 3600

        sid = f"S{i:03d}"
        stops.append(
            Stop(
                stop_id=sid,
                lat=lat,
                lon=lon,
                demand_weight=float(rng.uniform(1.0, 15.0)),
                demand_volume=float(rng.uniform(2.0, 25.0)),
                tw_open_s=tw_open_s,
                tw_close_s=tw_close_s,
                service_time_s=float(rng.uniform(120.0, 300.0)),
                is_priority=is_p,
                vehicle_compat=["bike", "three_wheeler", "van"],
            )
        )
        stop_cluster_map[sid] = cid

    # Build distance & duration matrices
    coords = np.array([[DEPOT_LAT, DEPOT_LON]] + [[s.lat, s.lon] for s in stops])
    dist_matrix = haversine_matrix(coords)
    speed_mps = 22.0 * 1000.0 / 3600.0
    dur_matrix = dist_matrix / speed_mps

    return stops, fleet, dur_matrix, dist_matrix, stop_cluster_map


def run_phase3_ablation(
    n_stops: int = 30,
    n_days: int = 10,
    epochs: int = 10,
    force_retrain: bool = False,
) -> pd.DataFrame:
    """
    Run Phase 3 benchmark across n_days.
    """
    model_path = MODELS_DIR / "gatv2_sequencer.pt"
    if force_retrain or not model_path.exists():
        log.info("Training GATv2 model (%d epochs)...", epochs)
        train_gatv2(epochs=epochs, save_path=model_path)
    else:
        log.info("Found trained GATv2 model at %s", model_path)

    decoder = GNNSequenceDecoder(model_path=model_path)
    weights = ObjectiveWeights.from_settings(settings)
    shaper = ArcCostShaper(lambda_shaping=0.50, scale_factor=1000.0)

    rows: List[Dict[str, Any]] = []

    for day_idx in range(n_days):
        seed = 2000 + day_idx
        log.info("--- Day %d/%d (seed=%d, n_stops=%d) ---", day_idx + 1, n_days, seed, n_stops)
        stops, fleet, dur_matrix, dist_matrix, stop_clusters = build_daily_instance(n_stops, seed)

        # ── Pre-compute GATv2 transition probabilities ────────────────────────
        # Build graph for this instance to extract P_theta
        depot_node = RouteGraphNode(
            node_id="DEPOT", lat=DEPOT_LAT, lon=DEPOT_LON,
            tw_open_s=6.0 * 3600.0, tw_close_s=22.0 * 3600.0,
            demand_weight=0.0, demand_volume=0.0, service_time_s=0.0,
            is_depot=True, is_priority=False, h3_cluster_id=0,
        )
        stop_nodes = [
            RouteGraphNode(
                node_id=s.stop_id, lat=s.lat, lon=s.lon,
                tw_open_s=float(s.tw_open_s), tw_close_s=float(s.tw_close_s),
                demand_weight=float(s.demand_weight), demand_volume=float(s.demand_volume),
                service_time_s=float(s.service_time_s), is_depot=False, is_priority=s.is_priority,
                h3_cluster_id=stop_clusters.get(s.stop_id, 0),
            )
            for s in stops
        ]
        g_data = RouteGraphData.build_from_stops_and_routes(
            [depot_node] + stop_nodes, dur_matrix, dist_matrix, []
        )
        data_t = g_data.to_torch("cpu")
        x = data_t.x if hasattr(data_t, "x") else data_t["x"]
        e_idx = data_t.edge_index if hasattr(data_t, "edge_index") else data_t["edge_index"]
        e_attr = data_t.edge_attr if hasattr(data_t, "edge_attr") else data_t["edge_attr"]

        with torch.no_grad():
            prob_mat = decoder.model.predict_transition_matrix(x, e_idx, e_attr, len(stops) + 1).numpy()

        shaped_dist_matrix = shaper.shape_cost_matrix(dist_matrix, prob_mat)

        # ── Config 1: Cold-Start OR-Tools GLS (10s) ───────────────────────────
        # Measure intermediate convergence at 2s, 5s, 10s
        solver_c1 = VRPTWSolver(stops, fleet, dur_matrix, dist_matrix)
        res_c1_2s = solver_c1.solve(config=SolverConfig(time_limit_s=2, local_search="GUIDED_LOCAL_SEARCH"))
        m_c1_2s = evaluate_route_plan(res_c1_2s.routes, dur_matrix, dist_matrix, stops, fleet, weights)

        res_c1_5s = solver_c1.solve(config=SolverConfig(time_limit_s=5, local_search="GUIDED_LOCAL_SEARCH"))
        m_c1_5s = evaluate_route_plan(res_c1_5s.routes, dur_matrix, dist_matrix, stops, fleet, weights)

        res_c1_10s = solver_c1.solve(config=SolverConfig(time_limit_s=10, local_search="GUIDED_LOCAL_SEARCH"))
        m_c1_10s = evaluate_route_plan(res_c1_10s.routes, dur_matrix, dist_matrix, stops, fleet, weights)
        coherence_c1 = compute_zone_coherence(res_c1_10s.routes, stop_clusters)

        rows.append({
            "day": day_idx + 1,
            "config": "Cold-Start OR-Tools GLS (Baseline)",
            "j_2s": m_c1_2s.composite_objective_cost,
            "j_5s": m_c1_5s.composite_objective_cost,
            "j_final": m_c1_10s.composite_objective_cost,
            "distance_km": m_c1_10s.total_distance_km,
            "sla_breach_rate": (m_c1_10s.sla_violations_count / n_stops) * 100.0,
            "zone_coherence": coherence_c1 * 100.0,
        })

        # ── Config 2: Arc-Cost Shaped OR-Tools (10s) ───────────────────────────
        solver_c2 = VRPTWSolver(stops, fleet, dur_matrix, dist_matrix, shaped_distance_matrix=shaped_dist_matrix)
        res_c2_2s = solver_c2.solve(config=SolverConfig(time_limit_s=2, local_search="GUIDED_LOCAL_SEARCH"))
        m_c2_2s = evaluate_route_plan(res_c2_2s.routes, dur_matrix, dist_matrix, stops, fleet, weights)

        res_c2_5s = solver_c2.solve(config=SolverConfig(time_limit_s=5, local_search="GUIDED_LOCAL_SEARCH"))
        m_c2_5s = evaluate_route_plan(res_c2_5s.routes, dur_matrix, dist_matrix, stops, fleet, weights)

        res_c2_10s = solver_c2.solve(config=SolverConfig(time_limit_s=10, local_search="GUIDED_LOCAL_SEARCH"))
        m_c2_10s = evaluate_route_plan(res_c2_10s.routes, dur_matrix, dist_matrix, stops, fleet, weights)
        coherence_c2 = compute_zone_coherence(res_c2_10s.routes, stop_clusters)

        rows.append({
            "day": day_idx + 1,
            "config": "Arc-Cost Shaped OR-Tools",
            "j_2s": m_c2_2s.composite_objective_cost,
            "j_5s": m_c2_5s.composite_objective_cost,
            "j_final": m_c2_10s.composite_objective_cost,
            "distance_km": m_c2_10s.total_distance_km,
            "sla_breach_rate": (m_c2_10s.sla_violations_count / n_stops) * 100.0,
            "zone_coherence": coherence_c2 * 100.0,
        })

        # ── Config 3 (Proposed): GATv2 Warm-Start + OR-Tools (8s) ──────────────
        initial_routes = decoder.decode_initial_routes(
            depot=(DEPOT_LAT, DEPOT_LON),
            stops=stops,
            vehicles=fleet,
            duration_matrix=dur_matrix,
            distance_matrix=dist_matrix,
        )

        solver_c3 = VRPTWSolver(stops, fleet, dur_matrix, dist_matrix)
        # Evaluate initial decoded solution (at t=0s)
        # Map decoded routes back to stop_ids for evaluation
        init_routes_dict = {}
        for vi, r in enumerate(initial_routes):
            vid = fleet[vi].vehicle_id
            init_routes_dict[vid] = [stops[idx - 1].stop_id for idx in r if idx > 0]
        m_init = evaluate_route_plan(init_routes_dict, dur_matrix, dist_matrix, stops, fleet, weights)

        res_c3_5s = solver_c3.solve(
            config=SolverConfig(time_limit_s=3, local_search="GUIDED_LOCAL_SEARCH"),
            initial_routes=initial_routes,
        )
        m_c3_5s = evaluate_route_plan(res_c3_5s.routes, dur_matrix, dist_matrix, stops, fleet, weights)

        res_c3_final = solver_c3.solve(
            config=SolverConfig(time_limit_s=8, local_search="GUIDED_LOCAL_SEARCH"),
            initial_routes=initial_routes,
        )
        m_c3_final = evaluate_route_plan(res_c3_final.routes, dur_matrix, dist_matrix, stops, fleet, weights)
        coherence_c3 = compute_zone_coherence(res_c3_final.routes, stop_clusters)

        rows.append({
            "day": day_idx + 1,
            "config": "GATv2 Warm-Start + OR-Tools (Proposed)",
            "j_2s": m_init.composite_objective_cost,  # Initial GNN decode cost
            "j_5s": m_c3_5s.composite_objective_cost,
            "j_final": m_c3_final.composite_objective_cost,
            "distance_km": m_c3_final.total_distance_km,
            "sla_breach_rate": (m_c3_final.sla_violations_count / n_stops) * 100.0,
            "zone_coherence": coherence_c3 * 100.0,
        })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "phase3_benchmark.csv", index=False)

    # Compute Summary
    summary = (
        df.groupby("config")
        .agg(
            j_2s=("j_2s", "mean"),
            j_5s=("j_5s", "mean"),
            j_final=("j_final", "mean"),
            distance_km=("distance_km", "mean"),
            sla_breach_rate=("sla_breach_rate", "mean"),
            zone_coherence=("zone_coherence", "mean"),
        )
        .round(2)
        .reset_index()
    )
    summary.to_csv(RESULTS_DIR / "phase3_summary.csv", index=False)

    print("\n" + "=" * 95)
    print(" PHASE 3 ABLATION — Neural-Guided Routing (GATv2 + OR-Tools)")
    print("=" * 95)
    print(f"n_stops={n_stops} | n_days={n_days} | identical compute budget\n")

    markdown_table = summary.to_markdown(index=False)
    print(markdown_table)
    print("\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-stops", type=int, default=30)
    parser.add_argument("--n-days", type=int, default=10)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    run_phase3_ablation(
        n_stops=args.n_stops,
        n_days=args.n_days,
        epochs=args.train_epochs,
        force_retrain=args.force_retrain,
    )
