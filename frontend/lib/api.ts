/**
 * frontend/lib/api.ts
 * ===================
 * Typed API client for RouteMind FastAPI backend.
 */

export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error("Backend offline");
  return res.json();
}

export async function planDispatch(params: {
  workday_id?: string;
  day_seed?: number;
  n_stops?: number;
  solver_type: string;
  risk_mode: string;
  weights?: {
    w1_distance: number;
    w2_lateness: number;
    w3_overtime: number;
    w4_unassigned: number;
    w5_stability: number;
    w6_workload_var: number;
  };
}) {
  const res = await fetch(`${API_BASE}/dispatch/plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Planning error" }));
    throw new Error(err.detail || "Route planning failed");
  }
  return res.json();
}

export async function initSimulation(day_seed = 3000, n_stops = 30) {
  const res = await fetch(`${API_BASE}/simulation/init`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ day_seed, n_stops }),
  });
  if (!res.ok) throw new Error("Simulation init failed");
  return res.json();
}

export async function injectDisruption(params: {
  type: string;
  target_h3_cell?: string;
  vehicle_id?: string;
  severity?: number;
}) {
  const res = await fetch(`${API_BASE}/simulation/inject-disruption`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error("Disruption injection failed");
  return res.json();
}

export async function explainDelay(params: {
  h3_cell?: string;
  cluster_id?: number;
  osrm_base_time_s: number;
  hour: number;
  vehicle_class?: string;
  package_weight_kg?: number;
  is_weekend?: number;
}) {
  const res = await fetch(`${API_BASE}/explain/delay`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error("Explainability attribution failed");
  return res.json();
}

export async function getMLOpsMetrics() {
  const res = await fetch(`${API_BASE}/mlops/metrics`, { cache: "no-store" });
  if (!res.ok) throw new Error("MLOps metrics fetch failed");
  return res.json();
}
