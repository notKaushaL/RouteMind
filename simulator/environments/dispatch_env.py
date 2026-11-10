"""
simulator/environments/dispatch_env.py
======================================
Phase 4 — Gymnasium Discrete-Event Dispatch Environment with Action Masking

Hierarchy:
  - High-level: MaskablePPO decides meta-action (BUFFER vs LOCAL_INSERTION vs STABILITY-TIERED RESOLVE).
  - Low-level: Executes heuristic insertion or rolling-horizon OR-Tools re-solve.
  - Hard Constraints: Boolean action masking prunes illegal/infeasible vehicles.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Silent protobuf / tensorboard compatibility
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
sys.modules["tensorboard"] = None

from configs.settings import settings
from solver.baselines.greedy import Stop, Vehicle
from ml.gnn_sequencer.dataset import haversine_matrix, DEPOT_LAT, DEPOT_LON
from solver.stability.kendall_tau import route_stability_penalty, kendall_tau_normalized
from simulator.disruptions.engine import (
    DisruptionEngine,
    DisruptionEvent,
    DisruptionType,
    TrafficSpikeEvent,
    VehicleBreakdownEvent,
    FailedDeliveryEvent,
    UrgentOrderEvent,
    WORKDAY_START_S,
    WORKDAY_END_S,
)
from simulator.fleet.state_tracker import FleetStateTracker

log = logging.getLogger(__name__)

# Constants
W5_STABILITY_TIERS = [0.20, 1.00, 5.00]  # Low, Medium, High


class DispatchMetaEnv(gym.Env):
    """
    Gymnasium discrete-event dispatch environment with action masking for MaskablePPO.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_vehicles: int = 3,
        seed: int = 42,
        max_events_per_day: int = 8,
    ):
        super().__init__()
        self.num_vehicles = num_vehicles
        self.max_events = max_events_per_day
        self.rng = np.random.default_rng(seed)

        # 24-dimensional continuous state vector
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24,),
            dtype=np.float32,
        )

        # Action space: 0 = BUFFER, 1..V = INSERT(v), V+1..V*4 = PARTIAL_RESOLVE(v, tier)
        # Total actions = 1 + V + (V * 3) = 1 + V * 4
        self.num_actions = 1 + (num_vehicles * 4)
        self.action_space = spaces.Discrete(self.num_actions)

        # Runtime state
        self.fleet: List[Vehicle] = []
        self.tracker: Optional[FleetStateTracker] = None
        self.disruption_engine: Optional[DisruptionEngine] = None
        self.dur_matrix: Optional[np.ndarray] = None
        self.dist_matrix: Optional[np.ndarray] = None
        self.stop_to_idx: Dict[str, int] = {}
        self.current_event: Optional[DisruptionEvent] = None
        self.unassigned_queue: List[Stop] = []
        self.event_count = 0
        self.total_reward = 0.0

    def _build_default_fleet(self) -> List[Vehicle]:
        """Construct standard 3-vehicle fleet for Vadodara dispatch."""
        depot_lat, depot_lon = 22.3072, 73.1812
        return [
            Vehicle(vehicle_id="V_TW1", vtype="three_wheeler", max_weight=200.0, max_volume=600.0, shift_start_s=int(WORKDAY_START_S), shift_end_s=int(WORKDAY_END_S), depot_lat=depot_lat, depot_lon=depot_lon),
            Vehicle(vehicle_id="V_TW2", vtype="three_wheeler", max_weight=200.0, max_volume=600.0, shift_start_s=int(WORKDAY_START_S), shift_end_s=int(WORKDAY_END_S), depot_lat=depot_lat, depot_lon=depot_lon),
            Vehicle(vehicle_id="V_VAN", vtype="van", max_weight=500.0, max_volume=1500.0, shift_start_s=int(WORKDAY_START_S), shift_end_s=int(WORKDAY_END_S), depot_lat=depot_lat, depot_lon=depot_lon),
        ]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new simulated workday."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.fleet = self._build_default_fleet()
        self.unassigned_queue = []
        self.event_count = 0
        self.total_reward = 0.0

        # Build daily base instance with 30 initial stops
        from scripts.benchmark_phase3 import build_daily_instance
        day_seed = int(self.rng.integers(1000, 99999))
        stops, _, self.dur_matrix, self.dist_matrix, _ = build_daily_instance(30, day_seed)

        stops_dict = {s.stop_id: s for s in stops}
        self.stop_to_idx = {s.stop_id: i + 1 for i, s in enumerate(stops)}
        self.stop_to_idx["DEPOT"] = 0

        # Build initial published plan using cold-start heuristic
        initial_routes: Dict[str, List[str]] = {v.vehicle_id: [] for v in self.fleet}
        stops_per_v = len(stops) // len(self.fleet)
        for i, s in enumerate(stops):
            v_idx = min(i // stops_per_v, len(self.fleet) - 1)
            initial_routes[self.fleet[v_idx].vehicle_id].append(s.stop_id)

        # Initialize Fleet Tracker
        self.tracker = FleetStateTracker(
            vehicles=self.fleet,
            initial_routes=initial_routes,
            stops_dict=stops_dict,
            start_time_s=WORKDAY_START_S,
        )

        # Setup Disruption Engine
        self.disruption_engine = DisruptionEngine.generate_benchmark_scenario(
            seed=day_seed,
            num_urgent=4,
            num_traffic_spikes=2,
            include_breakdown=True,
            vehicle_ids=[v.vehicle_id for v in self.fleet],
        )

        # Advance to first disruption event
        self.current_event = self._pop_next_event()
        obs = self._build_observation()
        info = {"event": self.current_event.description if self.current_event else "NONE"}

        return obs, info

    def _pop_next_event(self) -> Optional[DisruptionEvent]:
        """Advance time until a disruption event fires or workday ends."""
        while self.disruption_engine.events:
            next_t = self.disruption_engine.events[0].timestamp_s
            self.tracker.advance_time(
                target_time_s=next_t,
                duration_matrix=self.dur_matrix,
                stop_to_idx=self.stop_to_idx,
                disruption_engine=self.disruption_engine,
            )
            due = self.disruption_engine.get_due_events(next_t)
            if due:
                return due[0]

        # Advance to end of workday
        self.tracker.advance_time(
            target_time_s=WORKDAY_END_S,
            duration_matrix=self.dur_matrix,
            stop_to_idx=self.stop_to_idx,
            disruption_engine=self.disruption_engine,
        )
        return None

    def _build_observation(self) -> np.ndarray:
        """
        Construct the 24-dimensional state representation.
        """
        obs = np.zeros(24, dtype=np.float32)
        if self.tracker is None:
            return obs

        # 1. Event Type One-Hot [0..3]
        e_type_idx = 0
        if self.current_event:
            type_map = {
                DisruptionType.TRAFFIC_SPIKE: 0,
                DisruptionType.VEHICLE_BREAKDOWN: 1,
                DisruptionType.FAILED_DELIVERY: 2,
                DisruptionType.URGENT_ORDER: 3,
            }
            e_type_idx = type_map.get(self.current_event.event_type, 0)
            obs[e_type_idx] = 1.0

        # 2. Event Parcel Features [4..8]
        if self.current_event and isinstance(self.current_event, UrgentOrderEvent):
            s = self.current_event.stop
            obs[4] = float(s.demand_weight) / 20.0
            obs[5] = float(s.demand_volume) / 40.0
            obs[6] = float(self.current_event.sla_window_minutes) / 90.0
            obs[7] = (s.lat - 22.30) / 0.10
            obs[8] = (s.lon - 73.18) / 0.10

        # 3. Fleet State Summary [9..14]
        metrics = self.tracker.summary_metrics()
        obs[9] = metrics["active_vehicles_count"] / float(self.num_vehicles)
        obs[10] = metrics["mean_capacity_pct"]
        obs[11] = metrics["mean_shift_remaining_min"] / 480.0
        obs[12] = min(metrics["max_driver_delay_min"] / 60.0, 2.0)
        obs[13] = metrics["completed_stops_count"] / 40.0
        obs[14] = metrics["remaining_stops_count"] / 40.0

        # 4. Operational Context [15..18]
        obs[15] = float(len(self.unassigned_queue)) / 10.0
        obs[16] = min(sum(v.current_delay_min for v in self.tracker.vehicles.values()) / 60.0, 3.0)
        obs[17] = 2.0 if self.disruption_engine and self.disruption_engine._active_traffic_spikes else 1.0
        obs[18] = (self.tracker.current_time_s - WORKDAY_START_S) / (WORKDAY_END_S - WORKDAY_START_S)

        # 5. Per-Vehicle Capacity & Status [19..23]
        for idx, (vid, v) in enumerate(list(self.tracker.vehicles.items())[:5]):
            slot = 19 + idx
            if slot < 24:
                obs[slot] = 0.0 if (v.is_broken_down or not v.is_active) else (v.current_weight / max(v.max_weight, 1.0))

        return obs

    def action_masks(self) -> np.ndarray:
        """
        Hard Boolean Action Masking:
          - Masks out vehicle v if capacity exceeded or vehicle broken down
          - Masks out BUFFER if urgent parcel requires immediate dispatch
        """
        mask = np.ones(self.num_actions, dtype=bool)

        # If current event is high-priority urgent, BUFFER (action 0) is forbidden
        is_urgent = (
            self.current_event is not None
            and isinstance(self.current_event, UrgentOrderEvent)
            and self.current_event.sla_window_minutes <= 75.0
        )
        if is_urgent:
            mask[0] = False

        # Check vehicle feasibility
        for vi, v in enumerate(self.fleet):
            v_state = self.tracker.vehicles.get(v.vehicle_id)
            is_valid_vehicle = (
                v_state is not None
                and v_state.is_active
                and not v_state.is_broken_down
                and (v_state.shift_end_s - self.tracker.current_time_s) > 900.0  # >15m shift left
            )

            # Check capacity if parcel is present
            if self.current_event and isinstance(self.current_event, UrgentOrderEvent):
                p_w = self.current_event.stop.demand_weight
                p_v = self.current_event.stop.demand_volume
                if v_state and (v_state.current_weight + p_w > v_state.max_weight or v_state.current_volume + p_v > v_state.max_volume):
                    is_valid_vehicle = False

            # Action for insertion: 1 + vi
            act_insert = 1 + vi
            mask[act_insert] = is_valid_vehicle

            # Actions for partial resolve (3 tiers): (V + 1) + (vi * 3) .. + 2
            base_resolve_act = 1 + self.num_vehicles + (vi * 3)
            for tier in range(3):
                mask[base_resolve_act + tier] = is_valid_vehicle and len(v_state.remaining_stops) > 0 if v_state else False

        # Defensive fallback: If all actions masked out, allow BUFFER
        if not np.any(mask):
            mask[0] = True

        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute the meta-action, update fleet state, compute reward, and advance.
        """
        self.event_count += 1
        reward = 0.0
        action_name = "BUFFER"

        # Record pre-action state for Kendall-Tau stability comparison
        prev_published = self.tracker.get_published_routes()
        completed_set = set()
        for v in self.tracker.vehicles.values():
            completed_set.update(v.completed_stops)

        # ── 1. Action 0: BUFFER_ORDER ──────────────────────────────────────────
        if action == 0:
            action_name = "BUFFER_ORDER"
            if self.current_event and isinstance(self.current_event, UrgentOrderEvent):
                self.unassigned_queue.append(self.current_event.stop)
                reward -= 5.0  # Holding urgent order incurs penalty

        # ── 2. Action 1..V: CHEAPEST_INSERTION(v) ──────────────────────────────
        elif 1 <= action <= self.num_vehicles:
            v_idx = action - 1
            target_vehicle = self.fleet[v_idx].vehicle_id
            action_name = f"INSERT_{target_vehicle}"

            if self.current_event and isinstance(self.current_event, UrgentOrderEvent):
                inserted = self.tracker.insert_cheapest(
                    target_vehicle,
                    self.current_event.stop,
                    self.dur_matrix,
                    self.stop_to_idx,
                )
                if inserted:
                    reward += 15.0  # Successful insertion
                else:
                    self.unassigned_queue.append(self.current_event.stop)
                    reward -= 10.0

            elif self.current_event and isinstance(self.current_event, VehicleBreakdownEvent):
                # Salvage unvisited stops from broken vehicle into target_vehicle
                salvaged = self.tracker.handle_breakdown(self.current_event.vehicle_id)
                for sid in salvaged:
                    s_obj = self.tracker.stops_dict.get(sid)
                    if s_obj:
                        self.tracker.insert_cheapest(target_vehicle, s_obj, self.dur_matrix, self.stop_to_idx)
                reward += 10.0

        # ── 3. Action V+1..V*4: PARTIAL_RESOLVE(v, tier) ───────────────────────
        else:
            resolve_idx = action - 1 - self.num_vehicles
            v_idx = resolve_idx // 3
            tier_idx = resolve_idx % 3
            target_vehicle = self.fleet[v_idx].vehicle_id
            w5_weight = W5_STABILITY_TIERS[tier_idx]
            action_name = f"RESOLVE_{target_vehicle}_w5_{w5_weight}"

            # If an urgent order is present, insert it first into the pool
            if self.current_event and isinstance(self.current_event, UrgentOrderEvent):
                self.tracker.insert_cheapest(target_vehicle, self.current_event.stop, self.dur_matrix, self.stop_to_idx)

            # Perform localized re-solve on remaining unvisited stops with w5 penalty
            rem_routes = self.tracker.get_remaining_routes()
            from solver.ortools_engine.vrptw_solver import VRPTWSolver, SolverConfig

            # Collect unvisited stops to re-solve
            unvisited_stops = [
                self.tracker.stops_dict[sid]
                for stops in rem_routes.values()
                for sid in stops
                if sid in self.tracker.stops_dict
            ]

            active_fleet = [v for v in self.fleet if self.tracker.vehicles[v.vehicle_id].is_active]

            if unvisited_stops and len(active_fleet) > 0:
                re_coords = np.array([[DEPOT_LAT, DEPOT_LON]] + [[s.lat, s.lon] for s in unvisited_stops], dtype=np.float64)
                re_dist = haversine_matrix(re_coords).astype(np.float32)
                speed_mps = 22.0 * 1000.0 / 3600.0
                re_dur = (re_dist / speed_mps).astype(np.float32)
                re_solver = VRPTWSolver(unvisited_stops, active_fleet, re_dur, re_dist)
                cfg = SolverConfig(w5_stability=w5_weight, time_limit_s=1, local_search="PATH_CHEAPEST_ARC")
                res = re_solver.solve(config=cfg, initial_routes=list(rem_routes.values()))
                if res and res.routes:
                    self.tracker.apply_replan(res.routes)
                    reward += 20.0

        # ── Reward Shaping: Lateness, Route Churn, and Unassigned Penalties ────
        current_routes = self.tracker.get_current_routes()
        churn = route_stability_penalty(current_routes, prev_published, completed_stops=completed_set)
        reward -= 1.5 * min(churn, 5.0)

        # Lateness penalty
        metrics = self.tracker.summary_metrics()
        total_delay = metrics["max_driver_delay_min"]
        reward -= 2.0 * min(total_delay, 10.0)

        # Advance to next event
        self.current_event = self._pop_next_event()
        terminated = (self.current_event is None) or (self.event_count >= self.max_events)
        truncated = False

        if terminated:
            # End-of-day bonuses / penalties
            unassigned_count = len(self.unassigned_queue)
            reward -= 10.0 * unassigned_count
            if unassigned_count == 0 and total_delay < 5.0:
                reward += 25.0  # Perfect day bonus

        self.total_reward += reward
        obs = self._build_observation()
        info = {
            "action_name": action_name,
            "total_reward": self.total_reward,
            "unassigned": len(self.unassigned_queue),
            "max_delay_min": total_delay,
            "churn": churn,
        }

        return obs, reward, terminated, truncated, info
