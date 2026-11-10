"""
scripts/benchmark_phase1.py  (UPDATED — Phase 1 Metric Fix)
=============================================================
Runs Experiment 1: Static Dispatch Benchmark using the UNIFIED
apples-to-apples evaluation harness (solver/evaluation.py).

All four solvers (NN Greedy, Cheapest Insertion, OR-Tools GLS, PyVRP HGS)
are evaluated through evaluate_route_plan() so every metric
(distance_km, time_min, SLA_breaches, lateness_min, Cost_J) is computed
from first principles using the same duration/distance matrices.

Output:
  - Markdown table printed to stdout
  - CSV saved to data/processed/benchmark_results/phase1_benchmark.csv

Usage:
    python scripts/benchmark_phase1.py --n-stops 50
    python scripts/benchmark_phase1.py --n-stops 100 --ortools-time 30 --seed 42
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "benchmark_results"


# ── Instance builder ──────────────────────────────────────────────────────────

def build_instance(
    n_stops: int = 50,
    seed: int = 42,
) -> tuple:
    """
    Build a reproducible Vadodara-like VRPTW instance entirely in-memory.
    Returns: stops, vehicles, dur_mat, dist_mat
    """
    import math
    from configs.settings import settings
    from simulator.demand_generator.generate_workday import VEHICLE_TYPES
    from solver.baselines.greedy import Stop, Vehicle

    rng = np.random.default_rng(seed)
    depot_lat, depot_lon = settings.depot_lat, settings.depot_lon

    stops = []
    for i in range(n_stops):
        lat = depot_lat + rng.uniform(-0.08, 0.08)
        lon = depot_lon + rng.uniform(-0.08, 0.08)
        is_p = rng.random() < 0.15
        # Generate TW consistently: open first, then close = open + 4h
        if is_p:
            tw_open_s = 6 * 3600
            tw_close_s = 12 * 3600
        else:
            open_h = int(rng.integers(8, 15))  # 8..14 AM open
            tw_open_s = open_h * 3600
            tw_close_s = tw_open_s + 4 * 3600  # always +4h window
        stops.append(Stop(
            stop_id=f"S{i:04d}",
            lat=lat, lon=lon,
            demand_weight=float(rng.uniform(0.5, 20.0)),
            demand_volume=float(rng.uniform(1.0, 40.0)),
            tw_open_s=tw_open_s,
            tw_close_s=tw_close_s,
            service_time_s=float(rng.uniform(120, 360)),
            is_priority=is_p,
            vehicle_compat=["bike", "three_wheeler", "van"],
        ))

    vehicles = []
    vid = 0
    for vtype, cfg in VEHICLE_TYPES.items():
        for _ in range(cfg["count"]):
            vehicles.append(Vehicle(
                vehicle_id=f"V{vid:03d}",
                vtype=vtype,
                max_weight=cfg["max_weight_kg"],
                max_volume=cfg["max_volume_l"],
                shift_start_s=cfg["shift_start_h"] * 3600,
                shift_end_s=cfg["shift_end_h"] * 3600,
                depot_lat=depot_lat,
                depot_lon=depot_lon,
            ))
            vid += 1

    # Build haversine distance + 30 km/h duration matrices (N+1 × N+1)
    all_lats = [depot_lat] + [s.lat for s in stops]
    all_lons = [depot_lon] + [s.lon for s in stops]
    n = len(all_lats)
    dist_mat = np.zeros((n, n), dtype=np.float32)
    dur_mat = np.zeros((n, n), dtype=np.float32)
    AVG_SPEED = 30_000 / 3600  # 30 km/h in m/s

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            R = 6_371_000.0
            phi1, phi2 = math.radians(all_lats[i]), math.radians(all_lats[j])
            dphi = math.radians(all_lats[j] - all_lats[i])
            dlam = math.radians(all_lons[j] - all_lons[i])
            a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
            d = 2 * R * math.asin(math.sqrt(a))
            dist_mat[i, j] = d
            dur_mat[i, j] = d / AVG_SPEED

    return stops, vehicles, dur_mat, dist_mat


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(
    n_stops: int = 50,
    seed: int = 42,
    ortools_time: int = 10,
    pyvrp_time: int = 60,
    skip_pyvrp: bool = False,
) -> list[dict]:
    from solver.baselines.greedy import NearestNeighborGreedy, CheapestInsertionGreedy
    from solver.ortools_engine.vrptw_solver import VRPTWSolver, SolverConfig
    from solver.evaluation import evaluate_route_plan, ObjectiveWeights, print_comparison_table

    stops, vehicles, dur_mat, dist_mat = build_instance(n_stops=n_stops, seed=seed)
    weights = ObjectiveWeights()
    all_metrics = []

    # 1. NN Greedy
    log.info("Running NearestNeighborGreedy …")
    r = NearestNeighborGreedy(stops, vehicles, dur_mat, dist_mat).solve()
    m = evaluate_route_plan(r.routes, dur_mat, dist_mat, stops, vehicles, weights,
                            r.solve_time_s, "NN Greedy", r.unassigned)
    all_metrics.append(m)

    # 2. Cheapest Insertion
    log.info("Running CheapestInsertionGreedy …")
    r = CheapestInsertionGreedy(stops, vehicles, dur_mat, dist_mat).solve()
    m = evaluate_route_plan(r.routes, dur_mat, dist_mat, stops, vehicles, weights,
                            r.solve_time_s, "Cheapest Insertion", r.unassigned)
    all_metrics.append(m)

    # 3. OR-Tools GLS (cold start)
    log.info("Running OR-Tools GLS (t=%ds) …", ortools_time)
    r = VRPTWSolver(stops, vehicles, dur_mat, dist_mat).solve(
        config=SolverConfig(time_limit_s=ortools_time, local_search="GUIDED_LOCAL_SEARCH")
    )
    m = evaluate_route_plan(r.routes, dur_mat, dist_mat, stops, vehicles, weights,
                            r.solve_time_s, f"OR-Tools GLS (t={ortools_time}s)", r.unassigned)
    all_metrics.append(m)

    # 4. PyVRP HGS (optional)
    if not skip_pyvrp:
        log.info("Running PyVRP HGS (t=%ds) …", pyvrp_time)
        try:
            from solver.pyvrp_benchmark.pyvrp_solver import PyVRPSolver
            r = PyVRPSolver(stops, vehicles, dur_mat, dist_mat).solve(time_limit_s=pyvrp_time)
            m = evaluate_route_plan(r.routes, dur_mat, dist_mat, stops, vehicles, weights,
                                    r.solve_time_s, f"PyVRP HGS (t={pyvrp_time}s)", r.unassigned)
            all_metrics.append(m)
        except Exception as e:
            log.warning("PyVRP failed: %s", e)

    # Print the unified comparison table
    print(f"\n{'='*80}")
    print(f" PHASE 1 ABLATION — Experiment 1: Static Dispatch  (n_stops={n_stops}, seed={seed})")
    print(f"{'='*80}")
    print_comparison_table(all_metrics)

    return [m.as_row() for m in all_metrics]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 static dispatch benchmark.")
    parser.add_argument("--n-stops", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ortools-time", type=int, default=10)
    parser.add_argument("--pyvrp-time", type=int, default=60)
    parser.add_argument("--skip-pyvrp", action="store_true")
    args = parser.parse_args()

    rows = run_benchmark(
        n_stops=args.n_stops,
        seed=args.seed,
        ortools_time=args.ortools_time,
        pyvrp_time=args.pyvrp_time,
        skip_pyvrp=args.skip_pyvrp,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "phase1_benchmark.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("Results → %s", out)


if __name__ == "__main__":
    main()
