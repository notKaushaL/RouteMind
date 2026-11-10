"""
simulator/disruptions/engine.py
================================
Phase 4 — Dynamic Operational Simulator & Disruption Engine

Implements live disruption events occurring across an 8-hour workday (09:00 to 17:00):
  1. TrafficSpikeEvent: localized slowdown (1.5x–3.0x duration) on H3 clusters/corridors
  2. VehicleBreakdownEvent: disables vehicle mid-shift; remaining stops must be salvaged
  3. FailedDeliveryEvent: recipient unavailable/locked gate; flags for retry or return
  4. UrgentOrderEvent: mid-day high-priority parcel arrival with tight SLA (60–90 min)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from solver.baselines.greedy import Stop

log = logging.getLogger(__name__)

WORKDAY_START_S = 9.0 * 3600.0   # 09:00 AM (32,400s)
WORKDAY_END_S = 17.0 * 3600.0    # 05:00 PM (61,200s)


class DisruptionType(str, Enum):
    TRAFFIC_SPIKE = "TRAFFIC_SPIKE"
    VEHICLE_BREAKDOWN = "VEHICLE_BREAKDOWN"
    FAILED_DELIVERY = "FAILED_DELIVERY"
    URGENT_ORDER = "URGENT_ORDER"


@dataclass
class DisruptionEvent:
    """Base class for live operational disruptions."""
    event_id: str
    timestamp_s: float
    event_type: DisruptionType = DisruptionType.TRAFFIC_SPIKE
    description: str = ""


@dataclass
class TrafficSpikeEvent(DisruptionEvent):
    """Localized congestion wave affecting specific H3 clusters/corridors."""
    end_time_s: float = 0.0
    h3_cluster_ids: List[int] = field(default_factory=list)
    duration_multiplier: float = 2.0
    event_type: DisruptionType = DisruptionType.TRAFFIC_SPIKE


@dataclass
class VehicleBreakdownEvent(DisruptionEvent):
    """Mechanical or electrical failure disabling an active vehicle mid-route."""
    vehicle_id: str = ""
    recovery_required: bool = True
    event_type: DisruptionType = DisruptionType.VEHICLE_BREAKDOWN


@dataclass
class FailedDeliveryEvent(DisruptionEvent):
    """Customer unavailable or premises locked upon driver arrival."""
    vehicle_id: str = ""
    stop_id: str = ""
    reason: str = "customer_unavailable"
    retry_allowed: bool = True
    retry_window_open_s: float = 0.0
    retry_window_close_s: float = 0.0
    event_type: DisruptionType = DisruptionType.FAILED_DELIVERY


@dataclass
class UrgentOrderEvent(DisruptionEvent):
    """High-priority parcel arriving mid-day requiring rapid insertion."""
    stop: Stop = field(default_factory=lambda: Stop("TEMP", 0.0, 0.0, 1.0, 1.0, 0, 0, 180))
    sla_window_minutes: float = 75.0
    event_type: DisruptionType = DisruptionType.URGENT_ORDER


class DisruptionEngine:
    """
    Manages and fires dynamic disruption events along the simulated workday timeline.
    """

    def __init__(self, events: Optional[List[DisruptionEvent]] = None):
        self.events: List[DisruptionEvent] = sorted(events or [], key=lambda e: e.timestamp_s)
        self._fired_events: List[DisruptionEvent] = []
        self._active_traffic_spikes: List[TrafficSpikeEvent] = []

    def get_due_events(self, current_time_s: float) -> List[DisruptionEvent]:
        """Return and activate all events whose timestamp has arrived."""
        due: List[DisruptionEvent] = []
        remaining: List[DisruptionEvent] = []

        for e in self.events:
            if e.timestamp_s <= current_time_s:
                due.append(e)
                self._fired_events.append(e)
                if isinstance(e, TrafficSpikeEvent):
                    self._active_traffic_spikes.append(e)
            else:
                remaining.append(e)

        self.events = remaining

        # Clean expired traffic spikes
        self._active_traffic_spikes = [
            spike for spike in self._active_traffic_spikes if spike.end_time_s > current_time_s
        ]

        return due

    def get_active_traffic_multiplier(self, cluster_id: int, current_time_s: float) -> float:
        """Return the highest active congestion multiplier for a specific zone."""
        mult = 1.0
        for spike in self._active_traffic_spikes:
            if spike.timestamp_s <= current_time_s < spike.end_time_s:
                if not spike.h3_cluster_ids or cluster_id in spike.h3_cluster_ids:
                    mult = max(mult, spike.duration_multiplier)
        return mult

    @classmethod
    def generate_benchmark_scenario(
        cls,
        seed: int = 42,
        num_urgent: int = 5,
        num_traffic_spikes: int = 2,
        include_breakdown: bool = True,
        vehicle_ids: Optional[List[str]] = None,
    ) -> DisruptionEngine:
        """
        Generate a standardized benchmark disruption timeline for a workday.
        """
        rng = np.random.default_rng(seed)
        vids = vehicle_ids or ["V_TW1", "V_TW2", "V_VAN"]
        events: List[DisruptionEvent] = []

        # 1. Traffic Spikes (e.g. Alkapuri/Gotri corridors during peak hours)
        spike_times = [
            (10.5 * 3600.0, 11.75 * 3600.0, [0, 1], 2.2),  # Morning congestion in Alkapuri/Gotri
            (14.5 * 3600.0, 16.0 * 3600.0, [2, 3], 2.5),   # Afternoon congestion in Akota/Vasna
        ]
        for idx in range(min(num_traffic_spikes, len(spike_times))):
            t_start, t_end, clusters, mult = spike_times[idx]
            events.append(
                TrafficSpikeEvent(
                    event_id=f"TRAFFIC_SPIKE_{idx+1}",
                    timestamp_s=t_start,
                    end_time_s=t_end,
                    h3_cluster_ids=clusters,
                    duration_multiplier=mult,
                    description=f"Traffic spike ({mult:.1f}x) in clusters {clusters}",
                )
            )

        # 2. Mid-day Vehicle Breakdown (salvage required)
        if include_breakdown and vids:
            breakdown_v = vids[0]  # First two-wheeler breaks down at 12:45 PM
            t_breakdown = 12.75 * 3600.0
            events.append(
                VehicleBreakdownEvent(
                    event_id="BREAKDOWN_1",
                    timestamp_s=t_breakdown,
                    vehicle_id=breakdown_v,
                    recovery_required=True,
                    description=f"Vehicle {breakdown_v} breakdown at 12:45 PM",
                )
            )

        # 3. Urgent On-Demand Orders
        for k in range(num_urgent):
            # Arrival between 10:00 AM and 15:30 PM
            arr_time_s = float(rng.uniform(10.0 * 3600.0, 15.5 * 3600.0))
            sla_mins = float(rng.choice([60.0, 75.0, 90.0]))
            tw_open_s = int(arr_time_s)
            tw_close_s = int(arr_time_s + sla_mins * 60.0)

            lat = float(rng.uniform(22.28, 22.34))
            lon = float(rng.uniform(73.14, 73.22))

            urgent_stop = Stop(
                stop_id=f"URGENT_{k+1:02d}",
                lat=lat,
                lon=lon,
                demand_weight=float(rng.uniform(1.0, 8.0)),
                demand_volume=float(rng.uniform(2.0, 15.0)),
                tw_open_s=tw_open_s,
                tw_close_s=tw_close_s,
                service_time_s=float(rng.uniform(120.0, 240.0)),
                is_priority=True,
                vehicle_compat=["bike", "three_wheeler", "van"],
            )

            events.append(
                UrgentOrderEvent(
                    event_id=f"URGENT_EVT_{k+1}",
                    timestamp_s=arr_time_s,
                    stop=urgent_stop,
                    sla_window_minutes=sla_mins,
                    description=f"Urgent order {urgent_stop.stop_id} (SLA: {sla_mins:.0f}m)",
                )
            )

        return cls(events=events)
