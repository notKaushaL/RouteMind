"""
Phase 1 — Step 3: Synthetic Workday Demand Generator
=====================================================
Generates realistic daily parcel delivery workdays for Vadodara:
  - Randomly snaps parcels to real OSMnx road-network nodes
  - Assigns time windows (early-morning priority, daytime standard)
  - Assigns vehicle-compatibility flags (two-wheeler / three-wheeler)
  - Applies zone-aware dwell-time noise using H3 friction maps
  - Writes each simulated day as a Parquet file + a JSON summary

Dwell-time zones (Vadodara context):
  - Alkapuri / Fatehgunj (commercial high-rise): +8–15 min parking friction
  - GIDC Makarpura / Gorwa (industrial): +3–6 min dock queues
  - Gotri / Manjalpur (residential): +1–3 min standard

Usage:
    python simulator/demand_generator/generate_workday.py --days 30 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import h3
import numpy as np
import osmnx as ox
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Output path ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "workdays"

# ── Vehicle fleet definition ──────────────────────────────────────────────────
VEHICLE_TYPES = {
    "bike": {
        "count": 15,
        "max_weight_kg": 20,
        "max_volume_l": 40,
        "speed_factor": 1.1,   # faster on narrow lanes
        "shift_start_h": 8,
        "shift_end_h": 20,
    },
    "three_wheeler": {
        "count": 8,
        "max_weight_kg": 300,
        "max_volume_l": 600,
        "speed_factor": 0.85,
        "shift_start_h": 7,
        "shift_end_h": 19,
    },
    "van": {
        "count": 2,
        "max_weight_kg": 1200,
        "max_volume_l": 3000,
        "speed_factor": 0.9,
        "shift_start_h": 6,
        "shift_end_h": 18,
    },
}

# ── H3 zone dwell friction lookup (minutes added to base service time) ────────
# Coarse-grained; refined per-hex in Phase 2 with real historical telemetry
ZONE_DWELL_CONFIG = {
    "alkapuri":       {"h3_center": (22.3100, 73.1850), "extra_min_lo": 8,  "extra_min_hi": 15},
    "fatehgunj":      {"h3_center": (22.3215, 73.1932), "extra_min_lo": 6,  "extra_min_hi": 12},
    "makarpura_gidc": {"h3_center": (22.2680, 73.1920), "extra_min_lo": 3,  "extra_min_hi": 6},
    "gorwa":          {"h3_center": (22.3380, 73.1700), "extra_min_lo": 2,  "extra_min_hi": 5},
    "gotri":          {"h3_center": (22.3290, 73.1480), "extra_min_lo": 1,  "extra_min_hi": 3},
    "manjalpur":      {"h3_center": (22.2800, 73.1750), "extra_min_lo": 1,  "extra_min_hi": 3},
}

# Pre-compute zone H3 cells at resolution 8
_ZONE_CELLS: dict[str, str] = {
    name: h3.latlng_to_cell(cfg["h3_center"][0], cfg["h3_center"][1], settings.h3_res_coarse)
    for name, cfg in ZONE_DWELL_CONFIG.items()
}
# Flatten to cell→config mapping
_CELL_TO_DWELL: dict[str, dict] = {cell: ZONE_DWELL_CONFIG[name] for name, cell in _ZONE_CELLS.items()}


def _build_fleet(rng: np.random.Generator) -> list[dict[str, Any]]:
    vehicles = []
    vid = 0
    for vtype, cfg in VEHICLE_TYPES.items():
        for _ in range(cfg["count"]):
            vehicles.append({
                "vehicle_id": f"V{vid:03d}",
                "type": vtype,
                "max_weight_kg": cfg["max_weight_kg"],
                "max_volume_l": cfg["max_volume_l"],
                "speed_factor": cfg["speed_factor"],
                "shift_start_s": cfg["shift_start_h"] * 3600,
                "shift_end_s": cfg["shift_end_h"] * 3600,
                "depot_lat": settings.depot_lat,
                "depot_lon": settings.depot_lon,
            })
            vid += 1
    return vehicles


def _dwell_extra_min(h3_cell_r8: str, rng: np.random.Generator) -> float:
    """Return extra dwell minutes for a stop given its H3 cell."""
    if h3_cell_r8 in _CELL_TO_DWELL:
        cfg = _CELL_TO_DWELL[h3_cell_r8]
        return rng.uniform(cfg["extra_min_lo"], cfg["extra_min_hi"])
    # Default residential: 1–4 min
    return rng.uniform(1.0, 4.0)


def generate_workday(
    day_index: int,
    seed: int,
    graph_path: Path,
) -> pd.DataFrame:
    """
    Generate one synthetic workday of parcel deliveries.

    Returns a DataFrame with one row per parcel/stop.
    """
    rng = np.random.default_rng(seed + day_index * 7919)

    # Load OSMnx graph (cached after first call)
    log.info("Day %03d — Loading OSMnx graph …", day_index)
    G = ox.load_graphml(str(graph_path))
    nodes, _ = ox.graph_to_gdfs(G)
    nodes_wgs = nodes.to_crs("EPSG:4326")

    n_parcels = int(rng.integers(settings.parcels_per_day_min, settings.parcels_per_day_max + 1))
    log.info("Day %03d — Generating %d parcels …", day_index, n_parcels)

    # Sample random OSMnx node IDs as delivery locations (excluding depot)
    sampled_nodes = nodes_wgs.sample(n=n_parcels, random_state=int(seed + day_index), replace=False)

    parcels = []
    for idx, (osmid, row) in enumerate(sampled_nodes.iterrows()):
        lat = row.geometry.y
        lon = row.geometry.x

        # H3 cell for dwell noise
        cell_r8 = h3.latlng_to_cell(lat, lon, settings.h3_res_coarse)
        cell_r9 = h3.latlng_to_cell(lat, lon, settings.h3_res_fine)

        # Parcel characteristics
        is_priority = rng.random() < 0.15   # 15% SLA-critical
        weight_kg = float(rng.uniform(0.1, 25.0))
        volume_l = float(rng.uniform(0.5, 50.0))

        # Time window
        if is_priority:
            # Priority: must arrive before 12:00
            tw_open_s = 6 * 3600
            tw_close_s = 12 * 3600
        else:
            # Standard: flexible 4-hour window drawn from operational hours
            open_h = rng.integers(8, 16)
            tw_open_s = int(open_h) * 3600
            tw_close_s = tw_open_s + 4 * 3600

        # Base service time: 2–5 min + zone dwell noise
        base_service_s = float(rng.uniform(120, 300))
        dwell_extra_s = _dwell_extra_min(cell_r8, rng) * 60.0
        service_time_s = base_service_s + dwell_extra_s

        # Vehicle compatibility
        if weight_kg <= 20 and volume_l <= 40:
            vehicle_compat = ["bike", "three_wheeler", "van"]
        elif weight_kg <= 300 and volume_l <= 600:
            vehicle_compat = ["three_wheeler", "van"]
        else:
            vehicle_compat = ["van"]

        parcels.append({
            "day": day_index,
            "parcel_id": f"D{day_index:03d}P{idx:04d}",
            "node_osmid": osmid,
            "lat": lat,
            "lon": lon,
            "h3_r8": cell_r8,
            "h3_r9": cell_r9,
            "weight_kg": weight_kg,
            "volume_l": volume_l,
            "is_priority": is_priority,
            "tw_open_s": tw_open_s,
            "tw_close_s": tw_close_s,
            "service_time_s": service_time_s,
            "dwell_extra_s": dwell_extra_s,
            "vehicle_compat": json.dumps(vehicle_compat),
        })

    return pd.DataFrame(parcels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Vadodara workdays.")
    parser.add_argument("--days", type=int, default=30, help="Number of workdays to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    args = parser.parse_args()

    graph_path = (
        Path(__file__).parent.parent / "vadodara_network" / "vadodara_drive.graphml"
    )
    if not graph_path.exists():
        log.error(
            "Graph not found at %s — run build_network.py first.", graph_path
        )
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_days = []
    for day in range(args.days):
        df = generate_workday(day_index=day, seed=args.seed, graph_path=graph_path)
        day_path = OUT_DIR / f"day_{day:03d}.parquet"
        df.to_parquet(day_path, index=False)
        all_days.append(df)
        log.info("Day %03d → %s (%d parcels)", day, day_path, len(df))

    combined = pd.concat(all_days, ignore_index=True)
    combined_path = OUT_DIR / "all_days.parquet"
    combined.to_parquet(combined_path, index=False)
    log.info("✅ All %d days → %s (total %d parcels)", args.days, combined_path, len(combined))

    # Summary stats
    log.info("Priority parcels: %.1f%%", combined["is_priority"].mean() * 100)
    log.info("H3 R8 unique cells: %d", combined["h3_r8"].nunique())
    log.info("Avg service time: %.0f s", combined["service_time_s"].mean())


if __name__ == "__main__":
    main()
