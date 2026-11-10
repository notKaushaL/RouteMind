"""
scripts/benchmark_phase2.py
============================
Phase 2 Ablation — Experiment 2: Uncertainty-Aware Routing

Compares three ETA configurations under identical order distributions
across 10 simulated test days:

  Baseline:    OR-Tools + OSRM static deterministic durations (no ML)
  Challenger 1: OR-Tools + LightGBM P50 point estimates (median)
  Challenger 2: OR-Tools + Conformalized P90 SLA bounds (risk-aware)

Primary metrics:
  On-Time SLA %  |  Late Deliveries  |  90th-Pctile Lateness (min)
  Total Distance (km)  |  Total Cost (J)

Usage:
    python scripts/benchmark_phase2.py
    python scripts/benchmark_phase2.py --n-stops 40 --n-days 5 --skip-train
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[1] / "ml" / "eta_cqr" / "models" / "eta_cqr_pipeline.joblib"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "benchmark_results"


# ── Instance builder (identical to Phase 1 for comparability) ─────────────────

def build_instance(n_stops: int, seed: int):
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
            open_h = int(rng.integers(8, 15))  # 8..14 h
            tw_open_s = open_h * 3600
            tw_close_s = tw_open_s + 4 * 3600
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
                vehicle_id=f"V{vid:03d}", vtype=vtype,
                max_weight=cfg["max_weight_kg"], max_volume=cfg["max_volume_l"],
                shift_start_s=cfg["shift_start_h"] * 3600,
                shift_end_s=cfg["shift_end_h"] * 3600,
                depot_lat=depot_lat, depot_lon=depot_lon,
            ))
            vid += 1

    # Build distance + duration matrices
    all_lats = [depot_lat] + [s.lat for s in stops]
    all_lons = [depot_lon] + [s.lon for s in stops]
    n = len(all_lats)
    dist_mat = np.zeros((n, n), dtype=np.float32)
    dur_mat = np.zeros((n, n), dtype=np.float32)
    AVG_SPEED = 30_000 / 3600

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


# ── Realistic congestion simulation for test evaluation ───────────────────────

def simulate_actual_durations(
    dur_mat: np.ndarray,
    stops,
    seed: int,
    cong_mean: float = 1.25,   # 25% average congestion
    cong_std: float = 0.15,
) -> np.ndarray:
    """
    Simulate what ACTUALLY happens on delivery day — apply random congestion
    multipliers to the OSRM baseline to get ground-truth arrival times.
    This is what we measure SLA performance against.
    """
    rng = np.random.default_rng(seed + 99999)
    n = dur_mat.shape[0]
    actual = dur_mat.copy()
    for i in range(n):
        for j in range(n):
            if i != j:
                cong = rng.normal(cong_mean, cong_std)
                actual[i, j] = dur_mat[i, j] * max(0.7, cong)
    return actual.astype(np.float32)


# ── Build quantile matrices from CQR model ────────────────────────────────────

def build_cqr_duration_matrices(
    model,
    stops,
    dur_mat: np.ndarray,
    dist_mat: np.ndarray,
    hour_of_day: float = 8.0,
    is_peak: bool = False,
) -> dict[float, np.ndarray]:
    """
    Use the trained CQR model to predict P50/P90 durations for every stop.
    Returns {0.50: p50_matrix, 0.90: p90_cal_matrix}.
    """
    n = dur_mat.shape[0]
    dur_p50 = dur_mat.copy().astype(np.float32)
    dur_p90 = dur_mat.copy().astype(np.float32)

    sin_h = math.sin(2 * math.pi * hour_of_day / 24.0)
    cos_h = math.cos(2 * math.pi * hour_of_day / 24.0)

    for i, s in enumerate(stops):
        node = i + 1
        features = {
            "sin_hour": sin_h,
            "cos_hour": cos_h,
            "sin_dow": 0.0,
            "cos_dow": 1.0,
            "peak_hour_flag": float(is_peak),
            "hour_of_day": hour_of_day,
            "osrm_baseline_duration_min": float(dur_mat[0, node]) / 60.0,
            "osrm_baseline_distance_km": float(dist_mat[0, node]) / 1000.0,
            "weight_kg": s.demand_weight,
            "volume_l": s.demand_volume,
            "is_priority": float(s.is_priority),
            "vehicle_type_enc": 1.0,
            "h3r8_mean_dwell_min": 4.5,
            "h3r8_std_dwell_min": 1.5,
            "h3r9_mean_dwell_min": 4.5,
            "h3r9_std_dwell_min": 1.5,
        }
        try:
            preds = model.predict_single(features)
            p50_s = max(10.0, preds["p50"] * 60.0)
            p90_s = max(10.0, preds["p90_cal"] * 60.0)
            dur_p50[0, node] = p50_s
            dur_p50[node, 0] = p50_s
            dur_p90[0, node] = p90_s
            dur_p90[node, 0] = p90_s
        except Exception:
            pass  # fallback to OSRM

    return {0.50: dur_p50, 0.90: dur_p90}


# ── SLA evaluation (against actual durations, not solver's view) ──────────────

def compute_sla_metrics(
    routes: dict,
    actual_dur_mat: np.ndarray,
    stops,
    vehicles,
    stop_index: dict,
) -> dict:
    """Evaluate SLA performance using actual (ground-truth congested) durations."""
    from solver.baselines.greedy import Stop, Vehicle

    stop_map = {s.stop_id: s for s in stops}
    veh_map = {v.vehicle_id: v for v in vehicles}

    total_stops = sum(len(r) for r in routes.values())
    on_time = 0
    total_lateness = []

    for vid, stop_ids in routes.items():
        if not stop_ids:
            continue
        v = veh_map.get(vid)
        if v is None:
            continue

        cur_time = float(v.shift_start_s)
        cur_node = 0

        for sid in stop_ids:
            if sid not in stop_map:
                continue
            s = stop_map[sid]
            j = stop_index[sid]
            travel_s = float(actual_dur_mat[cur_node, j])
            arrival_s = cur_time + travel_s
            late_s = max(0.0, arrival_s - s.tw_close_s)

            if late_s == 0:
                on_time += 1
            total_lateness.append(late_s / 60.0)

            cur_time = max(arrival_s, s.tw_open_s) + s.service_time_s
            cur_node = j

    n_served = len(total_lateness)
    late_deliveries = sum(1 for l in total_lateness if l > 0)
    pct_ontime = 100.0 * on_time / max(n_served, 1)
    p90_lateness = float(np.percentile(total_lateness, 90)) if total_lateness else 0.0

    return {
        "on_time_pct": round(pct_ontime, 2),
        "late_deliveries": late_deliveries,
        "p90_lateness_min": round(p90_lateness, 2),
        "n_served": n_served,
    }


# ── Main benchmark ─────────────────────────────────────────────────────────────

def run_phase2_benchmark(
    n_stops: int = 40,
    n_days: int = 10,
    ortools_time: int = 10,
    skip_train: bool = False,
) -> pd.DataFrame:
    from solver.ortools_engine.vrptw_solver import VRPTWSolver, SolverConfig
    from solver.evaluation import evaluate_route_plan, ObjectiveWeights

    # ── Train / load model ────────────────────────────────────────────────────
    model = None
    if not skip_train:
        log.info("=" * 60)
        log.info("Step 1: Generating synthetic telemetry …")
        log.info("=" * 60)

        from simulator.demand_generator.generate_telemetry import main as gen_telem
        gen_telem()

        log.info("Step 2: Building feature store …")
        from ml.feature_store.store import FeatureStore
        store = FeatureStore()
        store.build()

        log.info("Step 3: Training CQR model …")
        from ml.eta_cqr.train import train as train_cqr
        train_cqr(n_estimators=400, mlflow_uri="file:./mlruns")

    if MODEL_PATH.exists():
        from ml.eta_cqr.model import CQRModel
        model = CQRModel.load(MODEL_PATH)
        log.info("CQR model loaded: coverage=%.1f%%", model.coverage_ * 100)
    else:
        log.warning("No trained model found — Challengers 1 & 2 will use OSRM only.")

    # ── Run ablation across test days ─────────────────────────────────────────
    weights = ObjectiveWeights()
    all_rows = []

    for day in range(n_days):
        seed = 1000 + day
        stops, vehicles, dur_mat, dist_mat = build_instance(n_stops=n_stops, seed=seed)
        actual_dur_mat = simulate_actual_durations(dur_mat, stops, seed=seed)
        stop_index = {s.stop_id: i + 1 for i, s in enumerate(stops)}

        log.info("--- Day %d/%d (seed=%d, n_stops=%d) ---", day + 1, n_days, seed, n_stops)

        # Configuration 1: OSRM Static (Baseline)
        r = VRPTWSolver(stops, vehicles, dur_mat, dist_mat).solve(
            SolverConfig(time_limit_s=ortools_time)
        )
        sla = compute_sla_metrics(r.routes, actual_dur_mat, stops, vehicles, stop_index)
        m = evaluate_route_plan(r.routes, actual_dur_mat, dist_mat, stops, vehicles, weights,
                                r.solve_time_s, "OSRM Static", r.unassigned)
        all_rows.append({
            "day": day, "config": "OSRM Static (Baseline)",
            **sla,
            "total_distance_km": m.total_distance_km,
            "total_cost_J": m.composite_objective_cost,
            "unassigned": m.unassigned_orders_count,
        })

        if model is not None:
            # Configuration 2: OR-Tools + LightGBM P50
            quant_mats_p50_only = build_cqr_duration_matrices(model, stops, dur_mat, dist_mat)
            dur_p50_only = {0.50: quant_mats_p50_only[0.50], 0.90: quant_mats_p50_only[0.50]}  # both P50
            r = VRPTWSolver(stops, vehicles, dur_mat, dist_mat, quantile_duration=dur_p50_only).solve(
                SolverConfig(time_limit_s=ortools_time)
            )
            sla = compute_sla_metrics(r.routes, actual_dur_mat, stops, vehicles, stop_index)
            m = evaluate_route_plan(r.routes, actual_dur_mat, dist_mat, stops, vehicles, weights,
                                    r.solve_time_s, "LightGBM P50", r.unassigned)
            all_rows.append({
                "day": day, "config": "LightGBM P50 (Challenger 1)",
                **sla,
                "total_distance_km": m.total_distance_km,
                "total_cost_J": m.composite_objective_cost,
                "unassigned": m.unassigned_orders_count,
            })

            # Configuration 3: OR-Tools + CQR P90 (risk-aware)
            quant_mats = build_cqr_duration_matrices(model, stops, dur_mat, dist_mat)
            r = VRPTWSolver(stops, vehicles, dur_mat, dist_mat, quantile_duration=quant_mats).solve(
                SolverConfig(time_limit_s=ortools_time)
            )
            sla = compute_sla_metrics(r.routes, actual_dur_mat, stops, vehicles, stop_index)
            m = evaluate_route_plan(r.routes, actual_dur_mat, dist_mat, stops, vehicles, weights,
                                    r.solve_time_s, "CQR P90 Risk-Aware", r.unassigned)
            all_rows.append({
                "day": day, "config": "CQR P90 Risk-Aware (Proposed)",
                **sla,
                "total_distance_km": m.total_distance_km,
                "total_cost_J": m.composite_objective_cost,
                "unassigned": m.unassigned_orders_count,
            })

    df = pd.DataFrame(all_rows)

    # ── Aggregate and print results ────────────────────────────────────────────
    summary = df.groupby("config").agg({
        "on_time_pct": "mean",
        "late_deliveries": "mean",
        "p90_lateness_min": "mean",
        "total_distance_km": "mean",
        "total_cost_J": "mean",
    }).round(2).reset_index()
    summary.columns = [
        "Configuration", "On-Time SLA %", "Late Deliveries",
        "90th-Pctile Lateness (min)", "Total Distance (km)", "Total Cost (J)"
    ]

    print(f"\n{'='*90}")
    print(" PHASE 2 ABLATION — Experiment 2: Uncertainty-Aware Routing")
    print(f"{'='*90}")
    print(f"n_stops={n_stops} | n_days={n_days} | OR-Tools t={ortools_time}s")
    print()

    # Markdown table
    cols = summary.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["-" * (len(c) + 2) for c in cols]) + "|"
    print(header)
    print(sep)
    for _, row in summary.iterrows():
        print("| " + " | ".join(str(v) for v in row.values) + " |")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "phase2_benchmark.csv"
    df.to_csv(out, index=False)
    summary.to_csv(RESULTS_DIR / "phase2_summary.csv", index=False)
    log.info("Results → %s", out)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 CQR ablation benchmark.")
    parser.add_argument("--n-stops", type=int, default=40)
    parser.add_argument("--n-days", type=int, default=5)
    parser.add_argument("--ortools-time", type=int, default=10)
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip telemetry gen & training; use pre-trained model.")
    args = parser.parse_args()

    run_phase2_benchmark(
        n_stops=args.n_stops,
        n_days=args.n_days,
        ortools_time=args.ortools_time,
        skip_train=args.skip_train,
    )


if __name__ == "__main__":
    main()
