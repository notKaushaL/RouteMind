"""
ml/gnn_sequencer/schema.py
===========================
Phase 3 — Graph Schema and Data Representations for Route Sequencing

Represents a delivery dispatch instance as a directed graph G = (V, E):
  - Node features (x_i in R^8):
      [norm_lat, norm_lon, norm_tw_open, norm_tw_close,
       norm_weight, norm_volume, norm_service_time, zone_cluster_id]
  - Edge features (e_ij in R^4):
      [norm_transit_duration, norm_distance, bearing_sin, same_zone_flag]
  - Target (y_ij in {0, 1}):
      Binary indicator whether expert driver directly traversed from stop i to stop j.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

# Normalization bounds for Vadodara region & standard parcel limits
VADODARA_LAT_MIN, VADODARA_LAT_MAX = 22.20, 22.42
VADODARA_LON_MIN, VADODARA_LON_MAX = 73.10, 73.30
HORIZON_SECONDS = 24.0 * 3600.0
MAX_WEIGHT_KG = 50.0
MAX_VOLUME_L = 100.0
MAX_SERVICE_S = 600.0
MAX_DURATION_S = 3600.0
MAX_DISTANCE_M = 25000.0
N_ZONE_CLUSTERS = 16


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate forward azimuth bearing in radians from (lat1, lon1) to (lat2, lon2)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    return math.atan2(y, x)


@dataclass
class RouteGraphNode:
    """Attributes of a single node (stop or depot) in the dispatch graph."""
    node_id: str
    lat: float
    lon: float
    tw_open_s: float
    tw_close_s: float
    demand_weight: float
    demand_volume: float
    service_time_s: float
    is_depot: bool = False
    is_priority: bool = False
    h3_cluster_id: int = 0

    def to_feature_vector(self) -> np.ndarray:
        """
        Return normalized 8-dimensional node feature vector x_i in R^8.
        """
        norm_lat = (self.lat - VADODARA_LAT_MIN) / max(VADODARA_LAT_MAX - VADODARA_LAT_MIN, 1e-5)
        norm_lon = (self.lon - VADODARA_LON_MIN) / max(VADODARA_LON_MAX - VADODARA_LON_MIN, 1e-5)
        norm_tw_open = min(max(self.tw_open_s / HORIZON_SECONDS, 0.0), 1.0)
        norm_tw_close = min(max(self.tw_close_s / HORIZON_SECONDS, 0.0), 1.0)
        norm_weight = min(self.demand_weight / MAX_WEIGHT_KG, 1.0)
        norm_volume = min(self.demand_volume / MAX_VOLUME_L, 1.0)
        norm_service = min(self.service_time_s / MAX_SERVICE_S, 1.0)
        norm_zone = (self.h3_cluster_id % N_ZONE_CLUSTERS) / float(N_ZONE_CLUSTERS)

        return np.array([
            norm_lat,
            norm_lon,
            norm_tw_open,
            norm_tw_close,
            norm_weight,
            norm_volume,
            norm_service,
            norm_zone,
        ], dtype=np.float32)


@dataclass
class RouteGraphData:
    """
    In-memory representation of a single dispatch instance graph G = (V, E).
    
    Compatible with PyTorch / PyTorch Geometric loaders:
      - x: [num_nodes, 8] float32
      - edge_index: [2, num_edges] int64 (source, target)
      - edge_attr: [num_edges, 4] float32
      - y: [num_edges] float32 / int64 (1 if expert transition, 0 otherwise)
      - target_transitions: Dict[int, int] mapping node_i -> next_node_j in expert routes
      - metadata: Optional instance context (route_id, day, vehicle_count)
    """
    x: np.ndarray                            # shape (N, 8)
    edge_index: np.ndarray                   # shape (2, E)
    edge_attr: np.ndarray                    # shape (E, 4)
    y: np.ndarray                            # shape (E,)
    num_nodes: int
    target_transitions: Dict[int, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_torch(self, device: str = "cpu"):
        """Convert numpy arrays to PyTorch tensors or PyG Data object."""
        import torch
        try:
            from torch_geometric.data import Data
            return Data(
                x=torch.from_numpy(self.x).float().to(device),
                edge_index=torch.from_numpy(self.edge_index).long().to(device),
                edge_attr=torch.from_numpy(self.edge_attr).float().to(device),
                y=torch.from_numpy(self.y).float().to(device),
                num_nodes=self.num_nodes,
            )
        except ImportError:
            # Pure PyTorch dictionary fallback
            return {
                "x": torch.from_numpy(self.x).float().to(device),
                "edge_index": torch.from_numpy(self.edge_index).long().to(device),
                "edge_attr": torch.from_numpy(self.edge_attr).float().to(device),
                "y": torch.from_numpy(self.y).float().to(device),
                "num_nodes": self.num_nodes,
            }

    @classmethod
    def build_from_stops_and_routes(
        cls,
        nodes: List[RouteGraphNode],
        duration_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        expert_routes: List[List[int]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RouteGraphData:
        """
        Build fully connected directed graph G = (V, E) from nodes and observed expert routes.

        Args:
            nodes: List of N RouteGraphNode (index 0 is depot).
            duration_matrix: (N, N) array in seconds.
            distance_matrix: (N, N) array in meters.
            expert_routes: List of routes, each route is a list of node indices [0, s1, s2, ..., 0].
            metadata: Optional dictionary with route/instance details.
        """
        num_nodes = len(nodes)
        node_features = np.stack([node.to_feature_vector() for node in nodes], axis=0)  # (N, 8)

        # Collect expert transitions (u -> v)
        expert_edges = set()
        target_transitions: Dict[int, int] = {}
        for route in expert_routes:
            for k in range(len(route) - 1):
                u = route[k]
                v = route[k + 1]
                expert_edges.add((u, v))
                target_transitions[u] = v

        # Construct dense directed graph edges excluding self-loops (u != v)
        sources: List[int] = []
        targets: List[int] = []
        edge_feats: List[List[float]] = []
        labels: List[float] = []

        for u in range(num_nodes):
            u_node = nodes[u]
            for v in range(num_nodes):
                if u == v:
                    continue  # No self-loops
                v_node = nodes[v]

                dur = duration_matrix[u, v]
                dist = distance_matrix[u, v]
                norm_dur = min(dur / MAX_DURATION_S, 2.0)
                norm_dist = min(dist / MAX_DISTANCE_M, 2.0)

                bearing = calculate_bearing(u_node.lat, u_node.lon, v_node.lat, v_node.lon)
                bearing_sin = math.sin(bearing)

                same_zone = 1.0 if (u_node.h3_cluster_id == v_node.h3_cluster_id and not u_node.is_depot and not v_node.is_depot) else 0.0

                sources.append(u)
                targets.append(v)
                edge_feats.append([norm_dur, norm_dist, bearing_sin, same_zone])
                labels.append(1.0 if (u, v) in expert_edges else 0.0)

        edge_index = np.array([sources, targets], dtype=np.int64)
        edge_attr = np.array(edge_feats, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)

        return cls(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_nodes,
            target_transitions=target_transitions,
            metadata=metadata or {},
        )
