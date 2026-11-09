"""
RouteMind Global Configuration
Loaded from .env at repo root via python-dotenv.
"""
from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Vadodara bounding box ─────────────────────────────────────────────────
    vadodara_north: float = 22.38
    vadodara_south: float = 22.24
    vadodara_east: float = 73.26
    vadodara_west: float = 73.10

    # ── H3 resolution ─────────────────────────────────────────────────────────
    h3_res_coarse: int = 8   # ~460 m hex — traffic zone clustering
    h3_res_fine: int = 9     # ~174 m hex — per-stop dwell noise

    # ── OSRM ─────────────────────────────────────────────────────────────────
    osrm_url: str = "http://localhost:5000"

    # ── Fleet defaults ────────────────────────────────────────────────────────
    depot_lat: float = 22.3072
    depot_lon: float = 73.1812
    num_vehicles: int = 25
    parcels_per_day_min: int = 150
    parcels_per_day_max: int = 300

    # ── Solver ────────────────────────────────────────────────────────────────
    solver_time_limit_s: int = 10          # dispatch budget (hard)
    solver_benchmark_time_s: int = 60      # extended runs for benchmarks

    # ── Objective weights (default) ───────────────────────────────────────────
    w1_distance: float = 1.0
    w2_lateness: float = 5.0
    w3_overtime: float = 3.0
    w4_unassigned: float = 10.0
    w5_stability: float = 2.0
    w6_workload_var: float = 0.5

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5001"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: Path = ROOT_DIR / "data"
    osmnx_cache_dir: Path = ROOT_DIR / "data" / "osmnx_cache"
    osrm_data_dir: Path = ROOT_DIR / "data" / "osrm"
    features_dir: Path = ROOT_DIR / "data" / "features"


settings = Settings()
