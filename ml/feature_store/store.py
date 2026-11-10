"""
ml/feature_store/store.py
==========================
Phase 2 — Offline Feature Store using DuckDB + Parquet

Ingests raw telemetry, computes the full ML feature tensor, and
provides clean train / calibration / test split methods.

Feature groups:
  SPATIAL:    H3 R8/R9 historical mean/std dwell, spatial bearing
  TEMPORAL:   sin/cos hour-of-day & day-of-week, peak_hour_flag
  ROUTE/CARGO: osrm_baseline_duration, distance_m, weight_kg,
               volume_l, vehicle_type (encoded), is_priority
  TARGETS:    actual_transit_time_min, actual_dwell_time_min,
              actual_total_leg_time_min

Split strategy (temporal — no data leakage):
  Train:       Days 0–62  (70%)
  Calibration: Days 63–74 (13.3%) — for conformal coverage adjustment
  Test:        Days 75–89 (16.7%)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import duckdb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TELEMETRY_PATH = _REPO_ROOT / "data" / "synthetic" / "telemetry_90d.parquet"
_FEATURE_DIR = _REPO_ROOT / "data" / "features"
_FEATURE_PATH = _FEATURE_DIR / "eta_features.parquet"
_DB_PATH = _FEATURE_DIR / "feature_store.duckdb"

# Split boundaries (day index)
TRAIN_END_DAY = 62
CALIB_END_DAY = 74
# test: days 75–89

# Vehicle type → integer encoding
VEHICLE_TYPE_ENC: dict[str, int] = {
    "bike": 0,
    "three_wheeler": 1,
    "van": 2,
}

# ── Feature column sets ────────────────────────────────────────────────────────
FEATURE_COLS: list[str] = [
    # Temporal
    "sin_hour", "cos_hour", "sin_dow", "cos_dow",
    "peak_hour_flag",
    "hour_of_day",
    # Route / Cargo
    "osrm_baseline_duration_min",
    "osrm_baseline_distance_km",
    "weight_kg", "volume_l", "is_priority",
    "vehicle_type_enc",
    # Spatial — H3 zone aggregates
    "h3r8_mean_dwell_min", "h3r8_std_dwell_min",
    "h3r9_mean_dwell_min", "h3r9_std_dwell_min",
]

TARGET_TRANSIT = "actual_transit_time_min"
TARGET_DWELL = "actual_dwell_time_min"
TARGET_TOTAL = "actual_total_leg_time_min"
ALL_TARGETS = [TARGET_TRANSIT, TARGET_DWELL, TARGET_TOTAL]


class FeatureStore:
    """
    DuckDB-backed offline feature store.

    Usage:
        store = FeatureStore()
        store.build()   # only need to run once
        X_train, y_train = store.get_training_data()
        X_cal, y_cal   = store.get_calibration_data()
        X_test, y_test = store.get_test_data()
    """

    def __init__(
        self,
        telemetry_path: Path = _TELEMETRY_PATH,
        feature_path: Path = _FEATURE_PATH,
        db_path: Path = _DB_PATH,
    ):
        self.telemetry_path = telemetry_path
        self.feature_path = feature_path
        self.db_path = db_path

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, force: bool = False) -> None:
        """
        Ingest telemetry, engineer features, write `eta_features.parquet`.
        Skips if the output already exists unless `force=True`.
        """
        if self.feature_path.exists() and not force:
            log.info("Feature store already built at %s. Use force=True to rebuild.", self.feature_path)
            return

        if not self.telemetry_path.exists():
            raise FileNotFoundError(
                f"Telemetry not found at {self.telemetry_path}. "
                "Run simulator/demand_generator/generate_telemetry.py first."
            )

        log.info("Building feature store from %s …", self.telemetry_path)
        _FEATURE_DIR.mkdir(parents=True, exist_ok=True)

        con = duckdb.connect(str(self.db_path))
        con.execute(f"""
            CREATE OR REPLACE TABLE raw_telemetry AS
            SELECT * FROM read_parquet('{self.telemetry_path.as_posix()}')
        """)

        # ── H3 zone aggregates (using training days only to avoid leakage) ───
        con.execute(f"""
            CREATE OR REPLACE TABLE h3r8_stats AS
            SELECT
                h3_r8,
                AVG(actual_dwell_time_s / 60.0) AS h3r8_mean_dwell_min,
                STDDEV(actual_dwell_time_s / 60.0) AS h3r8_std_dwell_min
            FROM raw_telemetry
            WHERE day <= {TRAIN_END_DAY}
            GROUP BY h3_r8
        """)

        con.execute(f"""
            CREATE OR REPLACE TABLE h3r9_stats AS
            SELECT
                h3_r9,
                AVG(actual_dwell_time_s / 60.0) AS h3r9_mean_dwell_min,
                STDDEV(actual_dwell_time_s / 60.0) AS h3r9_std_dwell_min
            FROM raw_telemetry
            WHERE day <= {TRAIN_END_DAY}
            GROUP BY h3_r9
        """)

        # ── Feature join ──────────────────────────────────────────────────────
        con.execute("""
            CREATE OR REPLACE TABLE features AS
            SELECT
                t.day,
                t.leg_id,
                t.vehicle_type,
                t.stop_id,
                t.h3_r8, t.h3_r9,

                -- Temporal
                t.sin_hour, t.cos_hour,
                t.sin_dow,  t.cos_dow,
                CAST(t.peak_hour_flag AS INTEGER)   AS peak_hour_flag,
                t.hour_of_day,

                -- Route / cargo
                t.osrm_baseline_duration_s / 60.0  AS osrm_baseline_duration_min,
                t.distance_m / 1000.0              AS osrm_baseline_distance_km,
                t.weight_kg,
                t.volume_l,
                CAST(t.is_priority AS INTEGER)     AS is_priority,

                -- H3 spatial aggregates (left join → NULL for unseen cells)
                COALESCE(r8.h3r8_mean_dwell_min, 4.5)  AS h3r8_mean_dwell_min,
                COALESCE(r8.h3r8_std_dwell_min,  1.5)  AS h3r8_std_dwell_min,
                COALESCE(r9.h3r9_mean_dwell_min, 4.5)  AS h3r9_mean_dwell_min,
                COALESCE(r9.h3r9_std_dwell_min,  1.5)  AS h3r9_std_dwell_min,

                -- Targets (minutes)
                t.actual_transit_time_s / 60.0  AS actual_transit_time_min,
                t.actual_dwell_time_s   / 60.0  AS actual_dwell_time_min,
                t.actual_total_leg_time_s / 60.0 AS actual_total_leg_time_min

            FROM raw_telemetry t
            LEFT JOIN h3r8_stats r8 ON t.h3_r8 = r8.h3_r8
            LEFT JOIN h3r9_stats r9 ON t.h3_r9 = r9.h3_r9
        """)

        # Persist as Parquet
        con.execute(f"""
            COPY features TO '{self.feature_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        n = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        con.close()

        log.info("Feature store built: %d rows → %s", n, self.feature_path)

    # ── Split accessors ────────────────────────────────────────────────────────

    def _load_split(self, day_min: int, day_max: int) -> pd.DataFrame:
        """Load a day-range slice from the feature parquet."""
        if not self.feature_path.exists():
            raise FileNotFoundError(
                f"Feature store not found at {self.feature_path}. Run build() first."
            )
        con = duckdb.connect()
        df = con.execute(
            f"SELECT * FROM read_parquet('{self.feature_path.as_posix()}')"
            f" WHERE day >= {day_min} AND day <= {day_max}"
        ).df()
        con.close()
        return df

    @staticmethod
    def _encode(df: pd.DataFrame) -> pd.DataFrame:
        """Add vehicle_type_enc column."""
        df = df.copy()
        df["vehicle_type_enc"] = df["vehicle_type"].map(VEHICLE_TYPE_ENC).fillna(0).astype(int)
        return df

    def _split_xy(
        self, df: pd.DataFrame, target: str = TARGET_TOTAL
    ) -> tuple[pd.DataFrame, pd.Series]:
        df = self._encode(df)
        X = df[FEATURE_COLS].copy()
        y = df[target].copy()
        return X, y

    def get_training_data(
        self, target: str = TARGET_TOTAL
    ) -> tuple[pd.DataFrame, pd.Series]:
        df = self._load_split(0, TRAIN_END_DAY)
        log.info("Training split: %d rows (days 0–%d)", len(df), TRAIN_END_DAY)
        return self._split_xy(df, target)

    def get_calibration_data(
        self, target: str = TARGET_TOTAL
    ) -> tuple[pd.DataFrame, pd.Series]:
        df = self._load_split(TRAIN_END_DAY + 1, CALIB_END_DAY)
        log.info("Calibration split: %d rows (days %d–%d)", len(df), TRAIN_END_DAY + 1, CALIB_END_DAY)
        return self._split_xy(df, target)

    def get_test_data(
        self, target: str = TARGET_TOTAL
    ) -> tuple[pd.DataFrame, pd.Series]:
        df = self._load_split(CALIB_END_DAY + 1, 999)
        log.info("Test split: %d rows (days %d+)", len(df), CALIB_END_DAY + 1)
        return self._split_xy(df, target)

    def get_full_dataframe(self) -> pd.DataFrame:
        """Return the complete encoded feature dataframe (all days)."""
        con = duckdb.connect()
        df = con.execute(
            f"SELECT * FROM read_parquet('{self.feature_path.as_posix()}')"
        ).df()
        con.close()
        return self._encode(df)
