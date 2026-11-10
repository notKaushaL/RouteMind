"""
scripts/benchmark_phase4.py
============================
Phase 4 Ablation Benchmark — Dynamic Disruption Simulation & Action-Masked RL Meta-Controller

Evaluates 3 real-time dispatch strategies across 20 simulated test days under identical
disruption shocks (2 traffic spikes, 1 vehicle breakdown, 5 urgent orders):

  1. Strategy 1: Naive Full Re-solve (OR-Tools from scratch with w5 = 0)
  2. Strategy 2: Greedy Local Insertion (cheapest insertion, no resequencing)
  3. Strategy 3 (Proposed): MaskablePPO Meta-Controller (dynamic buffering / insertion / stability-tiered partial re-solve)

Metrics:
  - Final On-Time SLA %
  - Average Lateness (min)
  - Unassigned Orders Count
  - Route Churn (Normalized Kendall-Tau distance on published stops: 0 = stable, 1 = complete reshuffle)
  - Total Operational Cost J
  - Average Decision Latency (ms)

Usage:
    python scripts/benchmark_phase4.py --n-days 20
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Silent protobuf / tensorboard compatibility
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
sys.modules["tensorboard"] = None

# Ensure RouteMind is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.gnn_sequencer.dataset import haversine_matrix, DEPOT_LAT, DEPOT_LON
from configs.settings import settings
from solver.baselines.greedy import Stop, Vehicle
from solver.evaluation import evaluate_route_plan, ObjectiveWeights, SolutionMetrics
from solver.ortools_engine.vrptw_solver import VRPTWSolver, SolverConfig
from solver.stability.kendall_tau import route_stability_penalty, kendall_tau_normalized
from simulator.disruptions.engine import (
    DisruptionEngine,
    DisruptionEvent,
    DisruptionType,
    TrafficSpikeEvent,
    VehicleBreakdownEvent,
    FailedDeliveryEvent,
    UrgentOrderEvent,
    WORKDAY_START_S,
    WORKDAY_END_S,
)
from simulator.fleet.state_tracker import FleetStateTracker
from ml.rl_dispatcher.agent import DispatchMetaAgent
from simulator.environments.dispatch_env import W5_STABILITY_TIERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "benchmark_results"
MODEL_PATH = Path(__file__).resolve().parents[1] / "ml" / "rl_dispatcher" / "models" / "maskable_ppo_dispatcher.zip"


def simulate_day(
    strategy: str,
    day_seed: int,
    n_stops: int = 30,
    agent: Optional[DispatchMetaAgent] = None,
) -> Dict[str, Any]:
    """
    Run an 8-hour dynamic disruption simulation for a single workday under a specific strategy.
    """
    from scripts.benchmark_phase3 import build_daily_instance
    stops, fleet, dur_matrix, dist_matrix, stop_clusters = build_daily_instance(n_stops, day_seed)

    stops_dict = {s.stop_id: s for s in stops}
    stop_to_idx = {s.stop_id: i + 1 for i, s in enumerate(stops)}
    stop_to_idx["DEPOT"] = 0

    # 1. Initial Plan (Cold-Start OR-Tools)
    init_solver = VRPTWSolver(stops, fleet, dur_matrix, dist_matrix)
    init_res = init_solver.solve(config=SolverConfig(time_limit_s=3))
    initial_routes = init_res.routes

    # 2. Initialize Tracker & Disruption Timeline
    tracker = FleetStateTracker(
        vehicles=fleet,
        initial_routes=initial_routes,
        stops_dict=stops_dict,
        start_time_s=WORKDAY_START_S,
    )

    disruptions = DisruptionEngine.generate_benchmark_scenario(
        seed=day_seed,
        num_urgent=5,
        num_traffic_spikes=2,
        include_breakdown=True,
        vehicle_ids=[v.vehicle_id for v in fleet],
    )

    unassigned_orders: List[Stop] = []
    latencies_ms: List[float] = []

    # 3. Step Through Operational Events
    while disruptions.events:
        next_event_time = disruptions.events[0].timestamp_s
        tracker.advance_time(next_event_time, dur_matrix, stop_to_idx, disruptions)
        due_events = disruptions.get_due_events(next_event_time)

        for event in due_events:
            t_start = time.perf_counter()

            if strategy == "Naive Full Re-solve":
                # Naive Full Re-solve: collects all unvisited stops across all active vehicles and solves from scratch (w5 = 0)
                if isinstance(event, VehicleBreakdownEvent):
                    salvaged = tracker.handle_breakdown(event.vehicle_id)
                elif isinstance(event, UrgentOrderEvent):
                    tracker.stops_dict[event.stop.stop_id] = event.stop
                    stop_to_idx[event.stop.stop_id] = len(stop_to_idx)
                    unassigned_orders.append(event.stop)

                # Re-solve remaining stops
                rem = tracker.get_remaining_routes()
                all_unvisited = [
                    tracker.stops_dict[sid]
                    for r in rem.values()
                    for sid in r
                    if sid in tracker.stops_dict
                ] + unassigned_orders

                active_v = [v for v in fleet if tracker.vehicles[v.vehicle_id].is_active]
                if all_unvisited and active_v:
                    re_coords = np.array([[DEPOT_LAT, DEPOT_LON]] + [[s.lat, s.lon] for s in all_unvisited], dtype=np.float64)
                    re_dist = haversine_matrix(re_coords).astype(np.float32)
                    speed_mps = 22.0 * 1000.0 / 3600.0
                    re_dur = (re_dist / speed_mps).astype(np.float32)
                    re_s = VRPTWSolver(all_unvisited, active_v, re_dur, re_dist)
                    cfg = SolverConfig(w5_stability=0.0, time_limit_s=1, local_search="PATH_CHEAPEST_ARC")  # Zero stability penalty
                    res = re_s.solve(config=cfg)
                    if res and res.routes:
                        tracker.apply_replan(res.routes)
                        unassigned_orders.clear()

            elif strategy == "Greedy Local Insertion":
                # Greedy Local Insertion: inserts into cheapest feasible vehicle without re-sequencing
                if isinstance(event, VehicleBreakdownEvent):
                    salvaged = tracker.handle_breakdown(event.vehicle_id)
                    for sid in salvaged:
                        s_obj = tracker.stops_dict.get(sid)
                        if s_obj:
                            inserted = False
                            for v in fleet:
                                if tracker.vehicles[v.vehicle_id].is_active:
                                    if tracker.insert_cheapest(v.vehicle_id, s_obj, dur_matrix, stop_to_idx):
                                        inserted = True
                                        break
                            if not inserted:
                                unassigned_orders.append(s_obj)

                elif isinstance(event, UrgentOrderEvent):
                    stop_to_idx[event.stop.stop_id] = len(stop_to_idx)
                    inserted = False
                    for v in fleet:
                        if tracker.vehicles[v.vehicle_id].is_active:
                            if tracker.insert_cheapest(v.vehicle_id, event.stop, dur_matrix, stop_to_idx):
                                inserted = True
                                break
                    if not inserted:
                        unassigned_orders.append(event.stop)

            elif strategy == "MaskablePPO Meta-Controller":
                # MaskablePPO: selects meta-action using trained RL policy with boolean action masks
                if isinstance(event, VehicleBreakdownEvent):
                    salvaged = tracker.handle_breakdown(event.vehicle_id)
                    for sid in salvaged:
                        s_obj = tracker.stops_dict.get(sid)
                        if s_obj:
                            unassigned_orders.append(s_obj)

                elif isinstance(event, UrgentOrderEvent):
                    stop_to_idx[event.stop.stop_id] = len(stop_to_idx)
                    unassigned_orders.append(event.stop)

                # Query agent for each unassigned item
                if unassigned_orders and agent:
                    target_stop = unassigned_orders.pop(0)

                    # Build mock observation vector
                    metrics = tracker.summary_metrics()
                    obs = np.zeros(24, dtype=np.float32)
                    obs[3] = 1.0  # urgent order flag
                    obs[4] = float(target_stop.demand_weight) / 20.0
                    obs[5] = float(target_stop.demand_volume) / 40.0
                    obs[6] = 1.0
                    obs[9] = metrics["active_vehicles_count"] / float(len(fleet))
                    obs[10] = metrics["mean_capacity_pct"]
                    obs[11] = metrics["mean_shift_remaining_min"] / 480.0
                    obs[12] = min(metrics["max_driver_delay_min"] / 60.0, 2.0)

                    # Build action mask
                    mask = np.ones(1 + len(fleet) * 4, dtype=bool)
                    for vi, v in enumerate(fleet):
                        v_st = tracker.vehicles.get(v.vehicle_id)
                        ok = v_st and v_st.is_active and not v_st.is_broken_down
                        mask[1 + vi] = ok
                        for t in range(3):
                            mask[1 + len(fleet) + vi * 3 + t] = ok

                    action = agent.select_action(obs, mask, deterministic=True)

                    if action == 0:
                        # Buffer
                        unassigned_orders.append(target_stop)
                    elif 1 <= action <= len(fleet):
                        # Insertion
                        t_vid = fleet[action - 1].vehicle_id
                        if not tracker.insert_cheapest(t_vid, target_stop, dur_matrix, stop_to_idx):
                            unassigned_orders.append(target_stop)
                    else:
                        # Partial resolve with selected w5 tier
                        res_idx = action - 1 - len(fleet)
                        t_vid = fleet[res_idx // 3].vehicle_id
                        w5 = W5_STABILITY_TIERS[res_idx % 3]

                        tracker.insert_cheapest(t_vid, target_stop, dur_matrix, stop_to_idx)
                        rem = tracker.get_remaining_routes()
                        active_v = [v for v in fleet if tracker.vehicles[v.vehicle_id].is_active]
                        unvis = [
                            tracker.stops_dict[sid]
                            for r in rem.values()
                            for sid in r
                            if sid in tracker.stops_dict
                        ]
                        if unvis and active_v:
                            re_coords = np.array([[DEPOT_LAT, DEPOT_LON]] + [[s.lat, s.lon] for s in unvis], dtype=np.float64)
                            re_dist = haversine_matrix(re_coords).astype(np.float32)
                            speed_mps = 22.0 * 1000.0 / 3600.0
                            re_dur = (re_dist / speed_mps).astype(np.float32)
                            re_s = VRPTWSolver(unvis, active_v, re_dur, re_dist)
                            cfg = SolverConfig(w5_stability=w5, time_limit_s=1, local_search="PATH_CHEAPEST_ARC")
                            res = re_s.solve(config=cfg, initial_routes=list(rem.values()))
                            if res and res.routes:
                                tracker.apply_replan(res.routes)

            t_elapsed = (time.perf_counter() - t_start) * 1000.0  # ms
            latencies_ms.append(t_elapsed)

    # Advance to end of workday
    tracker.advance_time(WORKDAY_END_S, dur_matrix, stop_to_idx, disruptions)

    # 4. Evaluate Final Completed Day
    final_routes = tracker.get_current_routes()
    published_routes = tracker.get_published_routes()

    completed_stops_set = set()
    for v in tracker.vehicles.values():
        completed_stops_set.update(v.completed_stops)

    # Compute Route Churn (Normalized Kendall-Tau on published stops)
    churn = route_stability_penalty(final_routes, published_routes, completed_stops=completed_stops_set)

    # Evaluate SLA & lateness
    sla_violations = 0
    total_lateness_min = 0.0

    for sid, arr_t in tracker.arrival_times.items():
        s_obj = tracker.stops_dict.get(sid)
        if s_obj and arr_t > s_obj.tw_close_s:
            sla_violations += 1
            total_lateness_min += (arr_t - s_obj.tw_close_s) / 60.0

    total_evaluated_stops = len(tracker.arrival_times) + len(unassigned_orders)
    on_time_pct = (
        ((total_evaluated_stops - sla_violations - len(unassigned_orders)) / max(total_evaluated_stops, 1))
        * 100.0
    )

    # Composite cost J
    weights = ObjectiveWeights.from_settings(settings)
    all_stops_list = list(tracker.stops_dict.values())
    eval_coords = np.array([[DEPOT_LAT, DEPOT_LON]] + [[s.lat, s.lon] for s in all_stops_list], dtype=np.float64)
    eval_dist = haversine_matrix(eval_coords).astype(np.float32)
    speed_mps = 22.0 * 1000.0 / 3600.0
    eval_dur = (eval_dist / speed_mps).astype(np.float32)

    metrics_eval = evaluate_route_plan(
        routes=final_routes,
        duration_matrix=eval_dur,
        distance_matrix=eval_dist,
        stops=all_stops_list,
        vehicles=fleet,
        weights=weights,
    )
    cost_j = metrics_eval.composite_objective_cost + (10.0 * len(unassigned_orders)) + (1.5 * churn)

    return {
        "strategy": strategy,
        "day": day_seed,
        "on_time_pct": max(0.0, min(100.0, on_time_pct)),
        "lateness_min": total_lateness_min / max(len(tracker.arrival_times), 1),
        "unassigned_count": len(unassigned_orders),
        "route_churn": churn,
        "cost_j": cost_j,
        "latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 5.0,
    }


def run_phase4_ablation(
    n_days: int = 20,
    model_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run Phase 4 ablation benchmark across n_days.
    """
    if model_path is None:
        model_path = MODEL_PATH

    agent = DispatchMetaAgent(model_path=model_path)
    strategies = [
        "Naive Full Re-solve",
        "Greedy Local Insertion",
        "MaskablePPO Meta-Controller",
    ]

    all_results: List[Dict[str, Any]] = []

    log.info("Starting Phase 4 Ablation Benchmark across %d simulated workdays...", n_days)

    for day_idx in range(n_days):
        seed = 3000 + day_idx
        log.info("--- Simulating Day %02d/%02d (seed=%d) ---", day_idx + 1, n_days, seed)

        for strat in strategies:
            res = simulate_day(strategy=strat, day_seed=seed, agent=agent)
            all_results.append(res)

    df = pd.DataFrame(all_results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "phase4_benchmark.csv", index=False)

    summary = (
        df.groupby("strategy")
        .agg(
            on_time_pct=("on_time_pct", "mean"),
            lateness_min=("lateness_min", "mean"),
            unassigned_count=("unassigned_count", "mean"),
            route_churn=("route_churn", "mean"),
            cost_j=("cost_j", "mean"),
            latency_ms=("latency_ms", "mean"),
        )
        .round(2)
        .reset_index()
    )
    summary.to_csv(RESULTS_DIR / "phase4_summary.csv", index=False)

    print("\n" + "=" * 105)
    print(" PHASE 4 ABLATION — Dynamic Disruption Simulation & MaskablePPO Meta-Controller")
    print("=" * 105)
    print(f"n_days={n_days} | Live shocks: 2 traffic spikes, 1 breakdown, 5 urgent orders\n")

    markdown_table = summary.to_markdown(index=False)
    print(markdown_table)
    print("\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-days", type=int, default=20)
    args = parser.parse_args()

    run_phase4_ablation(n_days=args.n_days)
