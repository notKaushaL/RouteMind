"""
simulator/demand_generator/generate_telemetry.py
=================================================
Phase 2 — Synthetic Telemetry Generator

Generates 90 simulated historical delivery days (~20,000 completed legs)
across Vadodara with realistic operational dynamics:

  * Diurnal congestion waves (sinusoidal, peaking 09:00–11:00 & 17:30–20:30)
  * Localized H3 dwell friction by commercial/residential zone type
  * Vehicle-type speed modifiers (bikes faster in traffic, vans slower)
  * Random monsoon/weather delay shocks (June–September months)

Output: data/synthetic/telemetry_90d.parquet (~20k rows)

Schema per row (one completed delivery leg):
  day, leg_id, vehicle_id, vehicle_type,
  stop_id, h3_r8, h3_r9, lat, lon,
  departure_time_s,          -- actual departure from previous stop / depot
  osrm_baseline_duration_s,  -- deterministic matrix value (ground truth free-flow)
  actual_transit_time_s,     -- osrm * congestion_factor + noise
  actual_dwell_time_s,       -- zone_dwell_base + friction_noise
  actual_total_leg_time_s,   -- transit + dwell
  tw_open_s, tw_close_s,
  arrival_time_s,            -- departure + actual_transit_time_s
  is_late,                   -- arrival > tw_close
  lateness_s,                -- max(0, arrival - tw_close)
  weight_kg, volume_l,
  is_priority,
  peak_hour_flag,
  congestion_factor,
  weather_shock_factor,
  hour_of_day, day_of_week,
  sin_hour, cos_hour, sin_dow, cos_dow
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
OUT_PATH = OUT_DIR / "telemetry_90d.parquet"

# ── Zone dwell friction config (H3 resolution 8) ─────────────────────────────
# (zone_name, center_lat, center_lon, mean_dwell_min, std_dwell_min)
ZONE_CONFIGS = [
    # Commercial / high-friction zones
    ("alkapuri",        22.3100, 73.1850, 11.0, 3.5),
    ("fatehgunj",       22.3215, 73.1932,  9.5, 3.0),
    ("makarpura_gidc",  22.2680, 73.1920,  7.5, 2.5),
    ("gorwa",           22.3380, 73.1700,  6.5, 2.0),
    ("sayajigunj",      22.3147, 73.1844,  8.0, 2.8),
    # Residential / low-friction zones
    ("gotri",           22.3290, 73.1480,  4.0, 1.5),
    ("vasna",           22.2950, 73.1700,  4.5, 1.8),
    ("manjalpur",       22.2800, 73.1750,  3.5, 1.2),
    ("waghodia_road",   22.3400, 73.2100,  3.8, 1.3),
    ("sama",            22.3200, 73.2200,  5.0, 2.0),
]

# Build cell → (mean, std) lookup at resolution 8
_CELL_DWELL: dict[str, tuple[float, float]] = {}
for _name, _lat, _lon, _mean, _std in ZONE_CONFIGS:
    _cell = h3.latlng_to_cell(_lat, _lon, settings.h3_res_coarse)
    _CELL_DWELL[_cell] = (_mean, _std)

# Default for cells not in lookup
_DEFAULT_DWELL = (4.5, 1.5)

# ── Vehicle speed modifiers ───────────────────────────────────────────────────
# Applied as a multiplier on OSRM duration (>1 = slower)
VEHICLE_PEAK_FACTOR: dict[str, float] = {
    "bike":           0.85,   # bikes filter through traffic — faster than free-flow
    "three_wheeler":  1.15,   # moderate congestion impact
    "van":            1.40,   # vans suffer most in congested lanes
}
VEHICLE_OFFPEAK_FACTOR: dict[str, float] = {
    "bike":           1.00,
    "three_wheeler":  1.05,
    "van":            1.10,
}

# Monsoon months (June=6 … September=9) — adds stochastic shock
MONSOON_MONTHS = {6, 7, 8, 9}


def _congestion_factor(hour: float, vehicle_type: str, day_num: int) -> tuple[float, bool]:
    """
    Diurnal sinusoidal congestion model.

    Morning peak:   09:00–11:00 → factor peaks at 1.45
    Evening peak:   17:30–20:30 → factor peaks at 1.60
    Off-peak:       factor ~1.0 with mild noise

    Returns (congestion_factor, is_peak_hour).
    """
    # Morning wave: centred at 10h, half-width 1.5h
    am = 0.45 * max(0.0, math.cos(math.pi * (hour - 10.0) / 1.5)) if 8.5 <= hour <= 11.5 else 0.0
    # Evening wave: centred at 19h, half-width 1.5h
    pm = 0.60 * max(0.0, math.cos(math.pi * (hour - 19.0) / 1.5)) if 17.5 <= hour <= 20.5 else 0.0
    base_factor = 1.0 + max(am, pm)

    is_peak = (8.5 <= hour <= 11.5) or (17.5 <= hour <= 20.5)

    # Vehicle modifier
    veh_mod = VEHICLE_PEAK_FACTOR[vehicle_type] if is_peak else VEHICLE_OFFPEAK_FACTOR[vehicle_type]
    return base_factor * veh_mod, is_peak


def _weather_shock(day_num: int, rng: np.random.Generator) -> float:
    """
    Monsoon-season shock: 15% probability of a 1.1–1.5× delay event
    during June–September; smaller random noise otherwise.
    """
    # day_num 0-based, assume day 0 = March 1 (91 days covers all of monsoon)
    month = ((day_num // 30) + 2) % 12 + 1  # rough month estimate
    if month in MONSOON_MONTHS and rng.random() < 0.15:
        return rng.uniform(1.10, 1.50)
    return rng.uniform(0.95, 1.05)  # small daily noise


def generate_telemetry(n_days: int = 90, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic telemetry for `n_days` delivery days."""
    import math  # noqa: F811 — needed inside function scope for _congestion_factor

    rng = np.random.default_rng(seed)
    depot_lat, depot_lon = settings.depot_lat, settings.depot_lon

    # Sample stop locations across Vadodara bounding box
    all_records: list[dict] = []
    leg_counter = 0

    for day in range(n_days):
        day_rng = np.random.default_rng(seed + day * 31337)
        n_parcels = int(day_rng.integers(settings.parcels_per_day_min, settings.parcels_per_day_max + 1))
        weather_shock = _weather_shock(day, day_rng)
        day_of_week = day % 7

        # Generate n_parcels random stops inside Vadodara bbox
        lats = day_rng.uniform(settings.vadodara_south, settings.vadodara_north, n_parcels)
        lons = day_rng.uniform(settings.vadodara_west, settings.vadodara_east, n_parcels)

        # Assign vehicles round-robin across fleet
        from simulator.demand_generator.generate_workday import VEHICLE_TYPES
        vehicle_pool: list[tuple[str, str]] = []  # (vehicle_id, vehicle_type)
        vid = 0
        for vtype, cfg in VEHICLE_TYPES.items():
            for _ in range(cfg["count"]):
                vehicle_pool.append((f"V{vid:03d}", vtype))
                vid += 1

        # Shuffle stops into vehicle assignments
        stop_assignments = day_rng.choice(len(vehicle_pool), size=n_parcels)

        for idx in range(n_parcels):
            lat = lats[idx]
            lon = lons[idx]
            h3_r8 = h3.latlng_to_cell(lat, lon, settings.h3_res_coarse)
            h3_r9 = h3.latlng_to_cell(lat, lon, settings.h3_res_fine)

            veh_id, veh_type = vehicle_pool[stop_assignments[idx]]

            # Simulate actual departure time (8–17h spread)
            departure_s = int(day_rng.uniform(8 * 3600, 17 * 3600))
            hour_of_day = departure_s / 3600.0

            # Haversine distance from depot (metres) as OSRM baseline proxy
            R = 6_371_000.0
            phi1, phi2 = math.radians(depot_lat), math.radians(lat)
            dphi = math.radians(lat - depot_lat)
            dlam = math.radians(lon - depot_lon)
            a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
            dist_m = 2 * R * math.asin(math.sqrt(a))

            # OSRM baseline: free-flow at 30 km/h
            osrm_baseline_s = dist_m / (30_000 / 3600)

            # Actual transit: baseline × congestion × weather + Gaussian noise
            cong_factor, is_peak = _congestion_factor(hour_of_day, veh_type, day)
            noise = day_rng.normal(1.0, 0.08)  # ±8% random noise
            actual_transit_s = osrm_baseline_s * cong_factor * weather_shock * max(0.6, noise)

            # Zone dwell
            dwell_mean, dwell_std = _CELL_DWELL.get(h3_r8, _DEFAULT_DWELL)
            actual_dwell_s = max(60.0, day_rng.normal(dwell_mean * 60, dwell_std * 60))

            # Parcel attributes
            is_priority = day_rng.random() < 0.15
            weight_kg = float(day_rng.uniform(0.1, 25.0))
            volume_l = float(day_rng.uniform(0.5, 50.0))

            # Time window
            if is_priority:
                tw_open_s = 6 * 3600
                tw_close_s = 12 * 3600
            else:
                open_h = int(day_rng.integers(8, 16))
                tw_open_s = open_h * 3600
                tw_close_s = tw_open_s + 4 * 3600

            arrival_s = departure_s + actual_transit_s
            is_late = arrival_s > tw_close_s
            lateness_s = max(0.0, arrival_s - tw_close_s)

            # Cyclic time encodings
            sin_hour = math.sin(2 * math.pi * hour_of_day / 24.0)
            cos_hour = math.cos(2 * math.pi * hour_of_day / 24.0)
            sin_dow = math.sin(2 * math.pi * day_of_week / 7.0)
            cos_dow = math.cos(2 * math.pi * day_of_week / 7.0)

            all_records.append({
                "day": day,
                "leg_id": f"D{day:03d}L{leg_counter:06d}",
                "vehicle_id": veh_id,
                "vehicle_type": veh_type,
                "stop_id": f"D{day:03d}P{idx:04d}",
                "h3_r8": h3_r8,
                "h3_r9": h3_r9,
                "lat": lat,
                "lon": lon,
                "distance_m": dist_m,
                "departure_time_s": departure_s,
                "osrm_baseline_duration_s": osrm_baseline_s,
                "actual_transit_time_s": actual_transit_s,
                "actual_dwell_time_s": actual_dwell_s,
                "actual_total_leg_time_s": actual_transit_s + actual_dwell_s,
                "tw_open_s": tw_open_s,
                "tw_close_s": tw_close_s,
                "arrival_time_s": arrival_s,
                "is_late": bool(is_late),
                "lateness_s": lateness_s,
                "weight_kg": weight_kg,
                "volume_l": volume_l,
                "is_priority": bool(is_priority),
                "peak_hour_flag": bool(is_peak),
                "congestion_factor": cong_factor,
                "weather_shock_factor": weather_shock,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "sin_hour": sin_hour,
                "cos_hour": cos_hour,
                "sin_dow": sin_dow,
                "cos_dow": cos_dow,
            })
            leg_counter += 1

        if (day + 1) % 10 == 0:
            log.info("Generated day %d / %d (%d total legs)", day + 1, n_days, leg_counter)

    df = pd.DataFrame(all_records)
    log.info("Total legs generated: %d across %d days", len(df), n_days)
    return df


def main() -> None:
    import math  # noqa
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Generating 90-day telemetry dataset …")
    df = generate_telemetry(n_days=90, seed=42)

    df.to_parquet(OUT_PATH, index=False)
    log.info("Saved → %s  (%d rows, %.1f MB)", OUT_PATH, len(df), OUT_PATH.stat().st_size / 1e6)

    # Quick stats
    log.info("On-time rate: %.1f%%", 100 * (1 - df["is_late"].mean()))
    log.info("Mean transit time: %.1f min", df["actual_transit_time_s"].mean() / 60)
    log.info("Mean dwell time: %.1f min", df["actual_dwell_time_s"].mean() / 60)
    log.info("Priority share: %.1f%%", df["is_priority"].mean() * 100)


# ── fix missing import in module body ─────────────────────────────────────────
import math  # noqa: E402

if __name__ == "__main__":
    main()
