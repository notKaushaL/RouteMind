"""
backend/app/services/simulation_service.py
==========================================
Interactive simulation lab managing live fleet tracking, disruption events, and MaskablePPO meta-actions.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.app.schemas.simulation import (
    SimulationInitResponse,
    SimulationStateSummary,
    DisruptionInjectRequest,
    DisruptionInjectResponse,
    DisruptedOrder,
)
from backend.app.services.engine_registry import registry
from configs.settings import settings
from solver.baselines.greedy import Stop, Vehicle
from solver.stability.kendall_tau import route_stability_penalty
from solver.ortools_engine.vrptw_solver import VRPTWSolver, SolverConfig
from scripts.benchmark_phase3 import build_daily_instance, DEPOT_LAT, DEPOT_LON
from ml.gnn_sequencer.dataset import haversine_matrix
from simulator.fleet.state_tracker import FleetStateTracker
from simulator.disruptions.engine import (
    DisruptionEngine,
    TrafficSpikeEvent,
    VehicleBreakdownEvent,
    FailedDeliveryEvent,
    UrgentOrderEvent,
    WORKDAY_START_S,
    WORKDAY_END_S,
)
from simulator.environments.dispatch_env import W5_STABILITY_TIERS

log = logging.getLogger(__name__)

VEHICLE_COLORS = {
    "V_TW1": "#38bdf8",
    "V_TW2": "#34d399",
    "V_VAN": "#f472b6",
}


def _sec_to_clock(seconds: float) -> str:
    s = int(seconds) % (24 * 3600)
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h:02d}:{m:02d}"


class SimulationLab:
    """Maintains active state of the live dispatch simulation."""

    def __init__(self):
        self.day_seed: int = 3000
        self.fleet: List[Vehicle] = []
        self.stops: List[Stop] = []
        self.stops_dict: Dict[str, Stop] = {}
        self.stop_to_idx: Dict[str, int] = {}
        self.dur_matrix: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self.dist_matrix: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self.tracker: Optional[FleetStateTracker] = None
        self.disruption_engine: Optional[DisruptionEngine] = None
        self.unassigned_queue: List[Stop] = []
        self.published_plan: Dict[str, List[str]] = {}
        self.current_time_s: float = WORKDAY_START_S
        self.is_initialized: bool = False

    def init_simulation(self, day_seed: int = 3000, n_stops: int = 30) -> SimulationInitResponse:
        self.day_seed = day_seed
        self.stops, self.fleet, self.dur_matrix, self.dist_matrix, _ = build_daily_instance(
            n_stops=n_stops, seed=day_seed
        )
        self.stops_dict = {s.stop_id: s for s in self.stops}
        self.stop_to_idx = {s.stop_id: i + 1 for i, s in enumerate(self.stops)}
        self.stop_to_idx["DEPOT"] = 0
        self.unassigned_queue = []
        self.current_time_s = WORKDAY_START_S

        # Solve initial cold-start routes with OR-Tools
        solver = VRPTWSolver(self.stops, self.fleet, self.dur_matrix, self.dist_matrix)
        res = solver.solve(config=SolverConfig(time_limit_s=2, local_search="PATH_CHEAPEST_ARC"))
        initial_routes = res.routes if res and res.routes else {v.vehicle_id: [] for v in self.fleet}
        self.published_plan = {v: list(r) for v, r in initial_routes.items()}

        # Initialize fleet tracker & disruptions
        self.tracker = FleetStateTracker(
            vehicles=self.fleet,
            initial_routes=initial_routes,
            stops_dict=self.stops_dict,
            start_time_s=WORKDAY_START_S,
        )
        self.disruption_engine = DisruptionEngine()
        self.is_initialized = True

        state_sum = self._get_state_summary()
        routes_geo = self._build_routes_geojson(self.tracker.get_current_routes())

        v_list = [
            {
                "vehicle_id": v.vehicle_id,
                "vtype": v.vtype,
                "color": VEHICLE_COLORS.get(v.vehicle_id, "#38bdf8"),
                "max_weight": v.max_weight,
                "max_volume": v.max_volume,
                "is_active": True,
                "stops_assigned": len(initial_routes.get(v.vehicle_id, [])),
            }
            for v in self.fleet
        ]

        return SimulationInitResponse(
            status="success",
            message=f"Initialized dispatch simulation with {n_stops} stops across 3 vehicles.",
            state=state_sum,
            routes_geojson=routes_geo,
            vehicles=v_list,
        )

    def _get_state_summary(self) -> SimulationStateSummary:
        if not self.tracker:
            return SimulationStateSummary(
                sim_time_str="09:00",
                sim_time_s=32400.0,
                active_vehicles_count=3,
                completed_stops_count=0,
                remaining_stops_count=30,
                unassigned_queue_count=0,
                mean_capacity_pct=0.0,
                max_driver_delay_min=0.0,
                route_churn_kendall_tau=0.0,
            )

        metrics = self.tracker.summary_metrics()
        current_routes = self.tracker.get_current_routes()
        completed_set = {
            s for v in self.tracker.vehicles.values() for s in v.completed_stops
        }
        churn = route_stability_penalty(current_routes, self.published_plan, completed_stops=completed_set)

        return SimulationStateSummary(
            sim_time_str=_sec_to_clock(self.tracker.current_time_s),
            sim_time_s=self.tracker.current_time_s,
            active_vehicles_count=int(metrics["active_vehicles_count"]),
            completed_stops_count=int(metrics["completed_stops_count"]),
            remaining_stops_count=int(metrics["remaining_stops_count"]),
            unassigned_queue_count=len(self.unassigned_queue),
            mean_capacity_pct=round(metrics["mean_capacity_pct"] * 100.0, 1),
            max_driver_delay_min=round(metrics["max_driver_delay_min"], 1),
            route_churn_kendall_tau=round(churn, 3),
        )

    def _build_routes_geojson(self, routes_dict: Dict[str, List[str]]) -> Dict[str, Any]:
        features: List[Dict[str, Any]] = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [DEPOT_LON, DEPOT_LAT]},
                "properties": {"type": "depot", "stop_id": "DEPOT", "name": "Vadodara Central Depot"},
            }
        ]

        for vid, stop_ids in routes_dict.items():
            color = VEHICLE_COLORS.get(vid, "#38bdf8")
            coords = [[DEPOT_LON, DEPOT_LAT]]

            for seq, sid in enumerate(stop_ids):
                s = self.stops_dict.get(sid)
                if not s:
                    continue
                coords.append([s.lon, s.lat])

                arr_t = self.tracker.arrival_times.get(sid, s.tw_open_s) if self.tracker else s.tw_open_s
                is_breached = arr_t > s.tw_close_s

                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [s.lon, s.lat]},
                    "properties": {
                        "type": "stop",
                        "stop_id": s.stop_id,
                        "sequence_idx": seq + 1,
                        "vehicle_id": vid,
                        "color": color,
                        "is_priority": s.is_priority,
                        "is_sla_breached": is_breached,
                        "arrival_time_str": _sec_to_clock(arr_t),
                        "tw_close_str": _sec_to_clock(s.tw_close_s),
                    },
                })

            if stop_ids:
                coords.append([DEPOT_LON, DEPOT_LAT])

            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "type": "route",
                    "vehicle_id": vid,
                    "color": color,
                    "stop_count": len(stop_ids),
                },
            })

        return {"type": "FeatureCollection", "features": features}

    def inject_disruption(self, req: DisruptionInjectRequest) -> DisruptionInjectResponse:
        if not self.is_initialized or not self.tracker:
            self.init_simulation(day_seed=3000)

        # 1. Capture Before State
        before_routes = self.tracker.get_current_routes()
        before_geojson = self._build_routes_geojson(before_routes)
        before_arrivals = dict(self.tracker.arrival_times)

        # Advance simulation clock by 45 minutes to simulate active midday shock
        target_t = min(self.tracker.current_time_s + 2700.0, WORKDAY_END_S - 3600.0)
        self.tracker.advance_time(target_t, self.dur_matrix, self.stop_to_idx, self.disruption_engine)

        event_name = req.type.upper()
        t_start = time.perf_counter()
        target_stop: Optional[Stop] = None

        if event_name == "TRAFFIC_SPIKE":
            cid = 0  # Alkapuri zone
            spike = TrafficSpikeEvent(
                event_id=f"SPIKE_{int(time.time()) % 1000}",
                timestamp_s=self.tracker.current_time_s,
                end_time_s=self.tracker.current_time_s + 3600.0,
                h3_cluster_ids=[cid],
                duration_multiplier=req.severity,
                description=f"Injected {req.severity:.1f}x traffic delay in Alkapuri corridor.",
            )
            self.disruption_engine.add_event(spike)
            desc = f"Injected {req.severity:.1f}x traffic delay wave over Alkapuri arterial corridor."

        elif event_name == "BREAKDOWN":
            vid = req.vehicle_id or "V_TW1"
            bd = VehicleBreakdownEvent(
                event_id=f"BD_{int(time.time()) % 1000}",
                timestamp_s=self.tracker.current_time_s,
                vehicle_id=vid,
                description=f"Breakdown of vehicle {vid}",
            )
            self.disruption_engine.add_event(bd)
            salvaged = self.tracker.handle_breakdown(vid)
            desc = f"Vehicle {vid} suffered engine breakdown. {len(salvaged)} unvisited stops salvaged for recovery."
            for sid in salvaged:
                s_obj = self.stops_dict.get(sid)
                if s_obj:
                    self.unassigned_queue.append(s_obj)
            if self.unassigned_queue:
                target_stop = self.unassigned_queue.pop(0)

        elif event_name == "URGENT_ORDER":
            urgent_id = f"URGENT_{int(time.time()) % 1000}"
            # Spawn near Alkapuri / Sayajigunj
            lat = 22.3120 + float(np.random.uniform(-0.015, 0.015))
            lon = 73.1850 + float(np.random.uniform(-0.015, 0.015))
            open_s = self.tracker.current_time_s + 600.0
            close_s = open_s + 3600.0  # Tight 60-min SLA
            urgent_stop = Stop(
                stop_id=urgent_id,
                lat=lat,
                lon=lon,
                demand_weight=12.0,
                demand_volume=25.0,
                tw_open_s=int(open_s),
                tw_close_s=int(close_s),
                service_time_s=180.0,
                is_priority=True,
            )
            self.stops_dict[urgent_id] = urgent_stop
            self.stop_to_idx[urgent_id] = len(self.stop_to_idx)
            self.unassigned_queue.append(urgent_stop)
            target_stop = self.unassigned_queue.pop(0)

            urgent_ev = UrgentOrderEvent(
                event_id=f"URG_{int(time.time()) % 1000}",
                timestamp_s=self.tracker.current_time_s,
                stop=urgent_stop,
                description=f"Urgent parcel arrival {urgent_id}",
            )
            self.disruption_engine.add_event(urgent_ev)
            desc = f"Urgent parcel arrival {urgent_id} with 60-min priority SLA window."

        else:
            desc = f"Triggered event {event_name}."

        # 2. Query MaskablePPO Agent
        agent = registry.rl_agent
        action = 0
        action_name = "BUFFER"

        if target_stop and agent:
            # Build observation vector
            metrics = self.tracker.summary_metrics()
            obs = np.zeros(24, dtype=np.float32)
            obs[3] = 1.0  # urgent stop flag
            obs[4] = float(target_stop.demand_weight) / 20.0
            obs[5] = float(target_stop.demand_volume) / 40.0
            obs[6] = 1.0
            obs[9] = metrics["active_vehicles_count"] / float(len(self.fleet))
            obs[10] = metrics["mean_capacity_pct"]
            obs[11] = metrics["mean_shift_remaining_min"] / 480.0
            obs[12] = min(metrics["max_driver_delay_min"] / 60.0, 2.0)

            # Build action mask
            mask = np.ones(1 + len(self.fleet) * 4, dtype=bool)
            for vi, v in enumerate(self.fleet):
                v_st = self.tracker.vehicles.get(v.vehicle_id)
                ok = v_st and v_st.is_active and not v_st.is_broken_down
                mask[1 + vi] = ok
                for t in range(3):
                    mask[1 + len(self.fleet) + vi * 3 + t] = ok

            action = agent.select_action(obs, mask, deterministic=True)

            if action == 0:
                action_name = "BUFFER_PACKAGE"
                self.unassigned_queue.append(target_stop)
            elif 1 <= action <= len(self.fleet):
                t_vid = self.fleet[action - 1].vehicle_id
                action_name = f"CHEAPEST_INSERTION({t_vid})"
                if not self.tracker.insert_cheapest(t_vid, target_stop, self.dur_matrix, self.stop_to_idx):
                    self.unassigned_queue.append(target_stop)
            else:
                res_idx = action - 1 - len(self.fleet)
                t_vid = self.fleet[res_idx // 3].vehicle_id
                w5 = W5_STABILITY_TIERS[res_idx % 3]
                action_name = f"PARTIAL_RESOLVE({t_vid}, w5={w5})"

                self.tracker.insert_cheapest(t_vid, target_stop, self.dur_matrix, self.stop_to_idx)
                rem = self.tracker.get_remaining_routes()
                active_v = [v for v in self.fleet if self.tracker.vehicles[v.vehicle_id].is_active]
                unvis = [
                    self.tracker.stops_dict[sid]
                    for r in rem.values()
                    for sid in r
                    if sid in self.tracker.stops_dict
                ]
                if unvis and active_v:
                    re_coords = np.array([[DEPOT_LAT, DEPOT_LON]] + [[s.lat, s.lon] for s in unvis], dtype=np.float64)
                    re_dist = haversine_matrix(re_coords).astype(np.float32)
                    re_dur = (re_dist / (22.0 * 1000.0 / 3600.0)).astype(np.float32)
                    re_s = VRPTWSolver(unvis, active_v, re_dur, re_dist)
                    cfg = SolverConfig(w5_stability=w5, time_limit_s=1, local_search="PATH_CHEAPEST_ARC")
                    res = re_s.solve(config=cfg, initial_routes=list(rem.values()))
                    if res and res.routes:
                        self.tracker.apply_replan(res.routes)

        t_latency_ms = (time.perf_counter() - t_start) * 1000.0

        # 3. Capture After State
        after_routes = self.tracker.get_current_routes()
        after_geojson = self._build_routes_geojson(after_routes)
        after_arrivals = dict(self.tracker.arrival_times)

        # Compute Churn
        completed_set = {
            s for v in self.tracker.vehicles.values() for s in v.completed_stops
        }
        churn = route_stability_penalty(after_routes, self.published_plan, completed_stops=completed_set)

        # 4. Compute Disrupted Orders
        disrupted_orders: List[DisruptedOrder] = []
        for sid, after_t in after_arrivals.items():
            before_t = before_arrivals.get(sid, after_t)
            delta_min = (after_t - before_t) / 60.0
            s_obj = self.stops_dict.get(sid)
            if s_obj and (abs(delta_min) > 0.5 or after_t > s_obj.tw_close_s):
                # Locate vehicle
                v_found = "UNKNOWN"
                for vid, r in after_routes.items():
                    if sid in r:
                        v_found = vid
                        break
                disrupted_orders.append(DisruptedOrder(
                    stop_id=sid,
                    vehicle_id=v_found,
                    lat=s_obj.lat,
                    lon=s_obj.lon,
                    original_eta_str=_sec_to_clock(before_t),
                    new_eta_str=_sec_to_clock(after_t),
                    delay_delta_min=round(delta_min, 1),
                    is_sla_breached=after_t > s_obj.tw_close_s,
                ))

        state_sum = self._get_state_summary()

        return DisruptionInjectResponse(
            event_type=event_name,
            description=desc,
            timestamp_str=_sec_to_clock(self.tracker.current_time_s),
            agent_action_id=int(action),
            agent_action_name=action_name,
            decision_latency_ms=round(t_latency_ms, 2),
            route_churn_score=round(churn, 3),
            before_routes_geojson=before_geojson,
            after_routes_geojson=after_geojson,
            disrupted_orders=disrupted_orders,
            state=state_sum,
        )


simulation_lab = SimulationLab()
