"""
simulator/fleet/state_tracker.py
=================================
Phase 4 — Route State & Driver Progress Tracker

Tracks live fleet status minute-by-minute across an operational workday:
  - Completed stops (LOCKED: customer delivered, cannot be modified)
  - Currently in-transit stop (LOCKED: driver currently en-route)
  - Remaining unvisited stops (ELIGIBLE: can be reassigned or re-sequenced)
  - Driver shift progress, capacity usage, and cumulative delay against planned TWs
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from solver.baselines.greedy import Stop, Vehicle
from simulator.disruptions.engine import DisruptionEngine

log = logging.getLogger(__name__)


@dataclass
class VehicleState:
    """Live state of a single delivery vehicle."""
    vehicle_id: str
    vtype: str
    max_weight: float
    max_volume: float
    current_weight: float = 0.0
    current_volume: float = 0.0
    shift_start_s: float = 9.0 * 3600.0
    shift_end_s: float = 18.0 * 3600.0
    is_active: bool = True
    is_broken_down: bool = False

    # Current geographic coordinates (lat, lon)
    current_loc: Tuple[float, float] = (22.3072, 73.1812)
    current_time_s: float = 9.0 * 3600.0

    # Route partitions:
    completed_stops: List[str] = field(default_factory=list)      # LOCKED
    in_transit_stop: Optional[str] = None                         # LOCKED
    remaining_stops: List[str] = field(default_factory=list)      # ELIGIBLE for re-optimization

    # Timeline details
    in_transit_eta_s: float = 0.0
    service_end_s: float = 0.0
    current_delay_min: float = 0.0


class FleetStateTracker:
    """
    Manages and advances the operational state of the entire fleet.
    """

    def __init__(
        self,
        vehicles: List[Vehicle],
        initial_routes: Dict[str, List[str]],
        stops_dict: Dict[str, Stop],
        depot_loc: Tuple[float, float] = (22.3072, 73.1812),
        start_time_s: float = 9.0 * 3600.0,
    ):
        self.depot_loc = depot_loc
        self.current_time_s = start_time_s
        self.stops_dict = stops_dict
        self.published_plan: Dict[str, List[str]] = {v: list(r) for v, r in initial_routes.items()}
        self.arrival_times: Dict[str, float] = {}

        self.vehicles: Dict[str, VehicleState] = {}
        for v in vehicles:
            route = list(initial_routes.get(v.vehicle_id, []))
            # Calculate initial onboard cargo weight & volume
            init_w = sum(stops_dict[sid].demand_weight for sid in route if sid in stops_dict)
            init_v = sum(stops_dict[sid].demand_volume for sid in route if sid in stops_dict)

            v_state = VehicleState(
                vehicle_id=v.vehicle_id,
                vtype=v.vtype,
                max_weight=v.max_weight,
                max_volume=v.max_volume,
                current_weight=init_w,
                current_volume=init_v,
                shift_start_s=float(v.shift_start_s),
                shift_end_s=float(v.shift_end_s),
                current_loc=depot_loc,
                current_time_s=start_time_s,
                remaining_stops=route,
            )
            self.vehicles[v.vehicle_id] = v_state

    def _get_leg_duration(
        self,
        prev_stop_id: str,
        next_stop_id: str,
        duration_matrix: np.ndarray,
        stop_to_idx: Dict[str, int],
    ) -> float:
        """Return travel time in seconds between two stops, with dynamic haversine fallback."""
        pi = stop_to_idx.get(prev_stop_id, 0)
        ni = stop_to_idx.get(next_stop_id, 0)
        if pi < len(duration_matrix) and ni < len(duration_matrix):
            return float(duration_matrix[pi, ni])

        p_obj = self.stops_dict.get(prev_stop_id)
        n_obj = self.stops_dict.get(next_stop_id)
        p_lat, p_lon = (p_obj.lat, p_obj.lon) if p_obj else self.depot_loc
        n_lat, n_lon = (n_obj.lat, n_obj.lon) if n_obj else self.depot_loc

        dlat = np.radians(n_lat - p_lat)
        dlon = np.radians(n_lon - p_lon)
        a = np.sin(dlat / 2.0)**2 + np.cos(np.radians(p_lat)) * np.cos(np.radians(n_lat)) * np.sin(dlon / 2.0)**2
        dist_m = 6371000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        return float(dist_m / (22.0 * 1000.0 / 3600.0))

    def advance_time(
        self,
        target_time_s: float,
        duration_matrix: np.ndarray,
        stop_to_idx: Dict[str, int],
        disruption_engine: Optional[DisruptionEngine] = None,
    ):
        """
        Advance the discrete-event simulation time to target_time_s,
        progressing vehicle journeys, deliveries, and delays.
        """
        step_dt = 60.0  # 1-minute discrete simulation granularity
        curr_t = self.current_time_s

        while curr_t < target_time_s:
            curr_t = min(curr_t + step_dt, target_time_s)
            self.current_time_s = curr_t

            for v in self.vehicles.values():
                if not v.is_active or v.is_broken_down:
                    continue

                # Case A: Driver is currently performing service at stop
                if v.in_transit_stop and curr_t < v.service_end_s:
                    continue  # Still actively delivering

                # Case B: Driver just finished service at in_transit_stop
                if v.in_transit_stop and curr_t >= v.service_end_s:
                    v.completed_stops.append(v.in_transit_stop)
                    s_obj = self.stops_dict.get(v.in_transit_stop)
                    if s_obj:
                        v.current_weight = max(0.0, v.current_weight - s_obj.demand_weight)
                        v.current_volume = max(0.0, v.current_volume - s_obj.demand_volume)
                    v.in_transit_stop = None

                # Case C: Driver is idle and has remaining stops
                if v.in_transit_stop is None and v.remaining_stops:
                    next_stop_id = v.remaining_stops.pop(0)
                    v.in_transit_stop = next_stop_id

                    prev_stop_id = v.completed_stops[-1] if v.completed_stops else "DEPOT"
                    base_dur = self._get_leg_duration(prev_stop_id, next_stop_id, duration_matrix, stop_to_idx)

                    # Apply active traffic multiplier if applicable
                    mult = 1.0
                    if disruption_engine is not None and next_stop_id in self.stops_dict:
                        s_obj = self.stops_dict[next_stop_id]
                        cid = int(hash(f"{round(s_obj.lat, 2)}_{round(s_obj.lon, 2)}") % 8)
                        mult = disruption_engine.get_active_traffic_multiplier(cid, curr_t)

                    transit_time = base_dur * mult
                    v.in_transit_eta_s = curr_t + transit_time

                    s_obj = self.stops_dict.get(next_stop_id)
                    service_s = s_obj.service_time_s if s_obj else 180.0
                    v.service_end_s = v.in_transit_eta_s + service_s
                    self.arrival_times[next_stop_id] = v.in_transit_eta_s

                    # Update delay against planned window
                    if s_obj and v.in_transit_eta_s > s_obj.tw_close_s:
                        v.current_delay_min = (v.in_transit_eta_s - s_obj.tw_close_s) / 60.0
                    else:
                        v.current_delay_min = 0.0

                    if s_obj:
                        v.current_loc = (s_obj.lat, s_obj.lon)

    def handle_breakdown(self, vehicle_id: str) -> List[str]:
        """
        Handle a vehicle breakdown: disables vehicle and releases unvisited stops for salvage.
        """
        v = self.vehicles.get(vehicle_id)
        if not v or not v.is_active:
            return []

        v.is_active = False
        v.is_broken_down = True
        salvaged_stops = list(v.remaining_stops)
        v.remaining_stops = []
        log.warning("Vehicle %s disabled due to breakdown — salvaged %d stops", vehicle_id, len(salvaged_stops))
        return salvaged_stops

    def insert_cheapest(
        self,
        vehicle_id: str,
        stop: Stop,
        duration_matrix: np.ndarray,
        stop_to_idx: Dict[str, int],
    ) -> bool:
        """
        Greedily insert a stop into vehicle's remaining route at the minimum incremental duration point.
        """
        v = self.vehicles.get(vehicle_id)
        if not v or not v.is_active or v.is_broken_down:
            return False

        # Capacity check
        if v.current_weight + stop.demand_weight > v.max_weight:
            return False
        if v.current_volume + stop.demand_volume > v.max_volume:
            return False

        self.stops_dict[stop.stop_id] = stop
        rem = v.remaining_stops
        new_idx = stop_to_idx.get(stop.stop_id, 0)

        # Find best insertion index k in remaining_stops
        best_cost = float("inf")
        best_pos = len(rem)  # default append

        # Starting point is currently in-transit stop or depot
        start_stop = v.in_transit_stop or (v.completed_stops[-1] if v.completed_stops else "DEPOT")
        start_idx = stop_to_idx.get(start_stop, 0)

        for pos in range(len(rem) + 1):
            prev_sid = start_stop if pos == 0 else rem[pos - 1]
            next_sid = "DEPOT" if pos == len(rem) else rem[pos]

            d_prev_new = self._get_leg_duration(prev_sid, stop.stop_id, duration_matrix, stop_to_idx)
            d_new_next = self._get_leg_duration(stop.stop_id, next_sid, duration_matrix, stop_to_idx)
            d_prev_next = self._get_leg_duration(prev_sid, next_sid, duration_matrix, stop_to_idx)

            delta = d_prev_new + d_new_next - d_prev_next
            if delta < best_cost:
                best_cost = delta
                best_pos = pos

        rem.insert(best_pos, stop.stop_id)
        v.current_weight += stop.demand_weight
        v.current_volume += stop.demand_volume
        return True

    def get_published_routes(self) -> Dict[str, List[str]]:
        """Return the initial published routes (for Kendall-Tau churn calculation)."""
        return {v: list(r) for v, r in self.published_plan.items()}

    def get_current_routes(self) -> Dict[str, List[str]]:
        """Return current full route per vehicle (completed + in-transit + remaining)."""
        full: Dict[str, List[str]] = {}
        for vid, v in self.vehicles.items():
            r = list(v.completed_stops)
            if v.in_transit_stop:
                r.append(v.in_transit_stop)
            r.extend(v.remaining_stops)
            full[vid] = r
        return full

    def get_remaining_routes(self) -> Dict[str, List[str]]:
        """Return only the unvisited, re-plannable stops per vehicle."""
        return {vid: list(v.remaining_stops) for vid, v in self.vehicles.items() if v.is_active}

    def apply_replan(self, new_remaining: Dict[str, List[str]]):
        """Update the remaining route sequences without modifying locked stops."""
        for vid, stops in new_remaining.items():
            if vid in self.vehicles and self.vehicles[vid].is_active:
                self.vehicles[vid].remaining_stops = list(stops)

    def summary_metrics(self) -> Dict[str, float]:
        """Aggregate live operational metrics for state observation vector."""
        active_vs = [v for v in self.vehicles.values() if v.is_active and not v.is_broken_down]
        n_active = len(active_vs)

        if n_active > 0:
            mean_cap_pct = float(np.mean([v.current_weight / max(v.max_weight, 1e-4) for v in active_vs]))
            mean_shift_rem_min = float(np.mean([max(0.0, (v.shift_end_s - self.current_time_s) / 60.0) for v in active_vs]))
            max_delay_min = float(max([v.current_delay_min for v in active_vs]))
        else:
            mean_cap_pct = 1.0
            mean_shift_rem_min = 0.0
            max_delay_min = 60.0

        n_completed = sum(len(v.completed_stops) for v in self.vehicles.values())
        n_remaining = sum(len(v.remaining_stops) for v in self.vehicles.values())

        return {
            "active_vehicles_count": float(n_active),
            "mean_capacity_pct": mean_cap_pct,
            "mean_shift_remaining_min": mean_shift_rem_min,
            "max_driver_delay_min": max_delay_min,
            "completed_stops_count": float(n_completed),
            "remaining_stops_count": float(n_remaining),
        }
