"""
Phase 1 — Ablation Experiment 1: Static Dispatch Benchmark
===========================================================
Benchmarks three solvers on:
  a) CVRPLib Solomon instances (C1_10_1, R1_10_1, RC1_10_1)
  b) 30 simulated Vadodara workdays

Produces the first row of the README ablation table:

  Experiment 1: Static Dispatch
  ┌──────────────────────┬──────────────┬────────────────────┬──────────────────────┐
  │ Algorithm            │ Total cost   │ Unassigned parcels │ Solve time (s)       │
  ├──────────────────────┼──────────────┼────────────────────┼──────────────────────┤
  │ NN Greedy            │ XXXX m       │ XX                 │ 0.001 s              │
  │ Cheapest Insertion   │ XXXX m       │ XX                 │ 0.003 s              │
  │ OR-Tools GLS (10s)   │ XXXX m       │ XX                 │ 10.0 s               │
  │ PyVRP HGS (60s)      │ XXXX m       │ XX                 │ 60.0 s               │
  └──────────────────────┴──────────────┴────────────────────┴──────────────────────┘

Usage:
    python scripts/benchmark_phase1.py --mode vadodara --days 5
    python scripts/benchmark_phase1.py --mode solomon  --instance C101
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import settings
from solver.baselines.greedy import (
    CheapestInsertionGreedy,
    NearestNeighborGreedy,
    Stop,
    Vehicle,
    SolverResult,
)
from solver.ortools_engine.vrptw_solver import SolverConfig, VRPTWSolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "data" / "processed" / "benchmark_results"


# ── Vadodara toy instance builder ─────────────────────────────────────────────

def build_vadodara_instance(day_parquet: Path, seed: int = 42) -> tuple[list[Stop], list[Vehicle], np.ndarray, np.ndarray]:
    """
    Load a generated workday parquet and build Stop/Vehicle objects
    with a simple haversine distance matrix (OSRM not required for benchmarks).
    """
    df = pd.read_parquet(day_parquet)
    rng = np.random.default_rng(seed)

    stops = []
    for _, row in df.iterrows():
        import json as _json
        compat = _json.loads(row["vehicle_compat"]) if isinstance(row["vehicle_compat"], str) else row["vehicle_compat"]
        stops.append(Stop(
            stop_id=row["parcel_id"],
            lat=row["lat"],
            lon=row["lon"],
            demand_weight=float(row["weight_kg"]),
            demand_volume=float(row["volume_l"]),
            tw_open_s=int(row["tw_open_s"]),
            tw_close_s=int(row["tw_close_s"]),
            service_time_s=float(row["service_time_s"]),
            is_priority=bool(row["is_priority"]),
            vehicle_compat=compat,
        ))

    # Build vehicles from VEHICLE_TYPES config
    from simulator.demand_generator.generate_workday import VEHICLE_TYPES
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
                depot_lat=settings.depot_lat,
                depot_lon=settings.depot_lon,
            ))
            vid += 1

    # Build distance + duration matrices (N+1, N+1) where index 0 = depot
    all_lats = [settings.depot_lat] + [s.lat for s in stops]
    all_lons = [settings.depot_lon] + [s.lon for s in stops]
    n = len(all_lats)

    dist_mat = np.zeros((n, n), dtype=np.float32)
    dur_mat = np.zeros((n, n), dtype=np.float32)
    AVG_SPEED_MPS = 30_000 / 3600  # 30 km/h

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Haversine
            R = 6_371_000.0
            phi1, phi2 = np.radians(all_lats[i]), np.radians(all_lats[j])
            dphi = np.radians(all_lats[j] - all_lats[i])
            dlam = np.radians(all_lons[j] - all_lons[i])
            a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
            dist = 2 * R * np.arcsin(np.sqrt(a))
            dist_mat[i, j] = dist
            dur_mat[i, j] = dist / AVG_SPEED_MPS

    return stops, vehicles, dur_mat, dist_mat


def run_all_solvers(
    stops: list[Stop],
    vehicles: list[Vehicle],
    dur_mat: np.ndarray,
    dist_mat: np.ndarray,
    label: str,
    ortools_time: int = 10,
    pyvrp_time: int = 60,
    skip_pyvrp: bool = False,
) -> list[dict]:
    rows = []

    # 1. NN Greedy
    log.info("--- Running NearestNeighborGreedy ---")
    res_nn = NearestNeighborGreedy(stops, vehicles, dur_mat, dist_mat).solve()
    rows.append(_result_row(res_nn, label, len(stops)))

    # 2. Cheapest Insertion
    log.info("--- Running CheapestInsertionGreedy ---")
    res_ci = CheapestInsertionGreedy(stops, vehicles, dur_mat, dist_mat).solve()
    rows.append(_result_row(res_ci, label, len(stops)))

    # 3. OR-Tools GLS
    log.info("--- Running OR-Tools GLS (time=%ds) ---", ortools_time)
    res_or = VRPTWSolver(stops, vehicles, dur_mat, dist_mat).solve(
        config=SolverConfig(time_limit_s=ortools_time, local_search="GUIDED_LOCAL_SEARCH")
    )
    rows.append(_result_row(res_or, label, len(stops)))

    # 4. PyVRP HGS
    if not skip_pyvrp:
        log.info("--- Running PyVRP HGS (time=%ds) ---", pyvrp_time)
        try:
            from solver.pyvrp_benchmark.pyvrp_solver import PyVRPSolver
            res_pv = PyVRPSolver(stops, vehicles, dur_mat, dist_mat).solve(time_limit_s=pyvrp_time)
            rows.append(_result_row(res_pv, label, len(stops)))
        except Exception as e:
            log.warning("PyVRP failed: %s", e)

    return rows


def _result_row(res: SolverResult, label: str, n_stops: int) -> dict:
    return {
        "instance": label,
        "algorithm": res.algorithm,
        "total_cost_m": round(res.total_distance_m, 2),
        "total_lateness_s": round(res.total_lateness_s, 2),
        "unassigned": len(res.unassigned),
        "unassigned_pct": round(100 * len(res.unassigned) / max(n_stops, 1), 1),
        "solve_time_s": round(res.solve_time_s, 4),
        "n_stops": n_stops,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vadodara", "solomon"], default="vadodara")
    parser.add_argument("--days", type=int, default=5, help="Number of Vadodara days to benchmark")
    parser.add_argument("--ortools-time", type=int, default=10)
    parser.add_argument("--pyvrp-time", type=int, default=60)
    parser.add_argument("--skip-pyvrp", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    if args.mode == "vadodara":
        workday_dir = Path("data/processed/workdays")
        day_files = sorted(workday_dir.glob("day_*.parquet"))[: args.days]

        if not day_files:
            log.error("No workday parquets found in %s. Run generate_workday.py first.", workday_dir)
            sys.exit(1)

        for pq in day_files:
            day_label = pq.stem
            log.info("=== Benchmarking %s ===", day_label)
            stops, vehicles, dur_mat, dist_mat = build_vadodara_instance(pq)
            rows = run_all_solvers(
                stops, vehicles, dur_mat, dist_mat,
                label=day_label,
                ortools_time=args.ortools_time,
                pyvrp_time=args.pyvrp_time,
                skip_pyvrp=args.skip_pyvrp,
            )
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = RESULTS_DIR / f"phase1_benchmark_{args.mode}.csv"
    df.to_csv(out_csv, index=False)
    log.info("✅ Results saved → %s", out_csv)

    # Pretty-print summary
    pivot = df.pivot_table(
        index="algorithm",
        values=["total_cost_m", "unassigned_pct", "solve_time_s"],
        aggfunc={"total_cost_m": "mean", "unassigned_pct": "mean", "solve_time_s": "mean"},
    ).round(2)
    print("\n=== Phase 1 Ablation — Experiment 1: Static Dispatch ===")
    print(pivot.to_string())
    print(f"\nFull results: {out_csv}")


if __name__ == "__main__":
    main()
