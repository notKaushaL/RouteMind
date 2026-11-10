"""
backend/tests/test_api.py
=========================
End-to-end integration tests for RouteMind FastAPI dispatch, simulation, and MLOps endpoints.
"""
from __future__ import annotations

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure RouteMind is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models" in data


def test_dispatch_plan_greedy():
    payload = {
        "workday_id": "TEST_DAY",
        "day_seed": 42,
        "n_stops": 15,
        "solver_type": "greedy",
        "risk_mode": "p50_median",
    }
    response = client.post("/api/v1/dispatch/plan", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "routes_geojson" in data
    assert data["routes_geojson"]["type"] == "FeatureCollection"
    assert len(data["vehicles"]) == 3
    assert len(data["schedule"]) > 0


def test_dispatch_plan_ortools():
    payload = {
        "workday_id": "TEST_DAY",
        "day_seed": 42,
        "n_stops": 10,
        "solver_type": "ortools_cold",
        "risk_mode": "p90_robust",
    }
    response = client.post("/api/v1/dispatch/plan", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["metrics"]["total_distance_km"] > 0
    assert data["metrics"]["on_time_pct"] >= 90.0


def test_simulation_lifecycle_and_disruption():
    # 1. Initialize Simulation
    init_res = client.post("/api/v1/simulation/init", json={"day_seed": 3000, "n_stops": 15})
    assert init_res.status_code == 200
    init_data = init_res.json()
    assert init_data["status"] == "success"
    assert init_data["state"]["active_vehicles_count"] == 3

    # 2. Inject Disruption (Traffic Spike)
    spike_payload = {
        "type": "TRAFFIC_SPIKE",
        "target_h3_cell": "88619b4563fffff",
        "severity": 2.5,
    }
    disrupt_res = client.post("/api/v1/simulation/inject-disruption", json=spike_payload)
    assert disrupt_res.status_code == 200
    disrupt_data = disrupt_res.json()
    assert disrupt_data["event_type"] == "TRAFFIC_SPIKE"
    assert "before_routes_geojson" in disrupt_data
    assert "after_routes_geojson" in disrupt_data
    assert disrupt_data["decision_latency_ms"] >= 0.0


def test_explain_delay():
    payload = {
        "h3_cell": "88619b4563fffff",
        "cluster_id": 0,
        "osrm_base_time_s": 500.0,
        "hour": 18,
        "vehicle_class": "three_wheeler",
        "package_weight_kg": 14.5,
        "is_weekend": 0,
    }
    response = client.post("/api/v1/explain/delay", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["cqr_p90_robust_min"] >= data["cqr_p50_predicted_min"]
    assert len(data["top_factors"]) > 0
    assert any(f["direction"] == "increase" for f in data["top_factors"])


def test_mlops_metrics():
    response = client.get("/api/v1/mlops/metrics")
    assert response.status_code == 200
    data = response.json()
    assert len(data["phase1_deterministic_baselines"]) == 3
    assert len(data["phase2_cqr_ablation"]) == 3
    assert len(data["phase3_gnn_warmstart_ablation"]) == 3
    assert len(data["phase4_rl_meta_ablation"]) == 3
    assert data["cqr_calibration"]["empirical_coverage_pct"] >= 90.0
    assert data["drift_health"]["status"] == "HEALTHY"
