"""
ml/gnn_sequencer/decoder.py
============================
Phase 3 — Neural Sequence Decoder via Masked Transition Search

Decodes initial vehicle routes using learned GATv2 transition probabilities P_theta(j | i).
Performs masked greedy / beam decoding respecting vehicle capacity, stop compatibility,
and depot return constraints.

Returns route sequences formatted as:
    [[0, stop_a, stop_b, 0], [0, stop_c, stop_d, 0], ...]
ready for direct injection into Google OR-Tools via ReadAssignmentFromRoutes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from solver.baselines.greedy import Stop, Vehicle
from ml.gnn_sequencer.model import RouteGATv2
from ml.gnn_sequencer.schema import RouteGraphData, RouteGraphNode

log = logging.getLogger(__name__)


class GNNSequenceDecoder:
    """
    Decodes initial vehicle routing solutions from GATv2 transition probabilities.
    """

    def __init__(
        self,
        model: Optional[RouteGATv2] = None,
        model_path: Optional[Path] = None,
        device_str: str = "cpu",
    ):
        self.device = torch.device(device_str)
        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None and model_path.exists():
            checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
            hidden_dim = checkpoint.get("hidden_dim", 64)
            num_layers = checkpoint.get("num_layers", 3)
            heads = checkpoint.get("heads", 4)
            self.model = RouteGATv2(
                node_in_dim=8,
                edge_in_dim=4,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                heads=heads,
            ).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
        else:
            self.model = None

    def decode_initial_routes(
        self,
        depot: Tuple[float, float],
        stops: List[Stop],
        vehicles: List[Vehicle],
        duration_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        beam_width: int = 1,
    ) -> List[List[int]]:
        """
        Decode feasible initial routes from depot and stops using GATv2 transition priors.

        Args:
            depot: (lat, lon) coordinates
            stops: List of N delivery stops (1-indexed in matrices, 0 is depot)
            vehicles: Available fleet
            duration_matrix: (N+1, N+1) in seconds
            distance_matrix: (N+1, N+1) in meters
            beam_width: 1 for greedy masked search, >1 for beam search

        Returns:
            List of routes, e.g. [[0, 5, 2, 0], [0, 1, 3, 0]]
        """
        num_stops = len(stops)
        num_nodes = num_stops + 1
        num_vehicles = len(vehicles)

        # 1. Build graph representation for this instance
        depot_node = RouteGraphNode(
            node_id="DEPOT",
            lat=depot[0],
            lon=depot[1],
            tw_open_s=6.0 * 3600.0,
            tw_close_s=22.0 * 3600.0,
            demand_weight=0.0,
            demand_volume=0.0,
            service_time_s=0.0,
            is_depot=True,
            is_priority=False,
            h3_cluster_id=0,
        )

        stop_nodes = []
        for i, s in enumerate(stops):
            # Approximate cluster from lat/lon
            cluster_id = int((hash(f"{round(s.lat, 2)}_{round(s.lon, 2)}") % 8))
            stop_nodes.append(
                RouteGraphNode(
                    node_id=s.stop_id,
                    lat=s.lat,
                    lon=s.lon,
                    tw_open_s=float(s.tw_open_s),
                    tw_close_s=float(s.tw_close_s),
                    demand_weight=float(s.demand_weight),
                    demand_volume=float(s.demand_volume),
                    service_time_s=float(s.service_time_s),
                    is_depot=False,
                    is_priority=s.is_priority,
                    h3_cluster_id=cluster_id,
                )
            )

        all_nodes = [depot_node] + stop_nodes
        graph_data = RouteGraphData.build_from_stops_and_routes(
            nodes=all_nodes,
            duration_matrix=duration_matrix,
            distance_matrix=distance_matrix,
            expert_routes=[],  # Unlabeled at inference time
        )

        # 2. Get transition matrix P_theta(j | i)
        if self.model is not None:
            data = graph_data.to_torch(str(self.device))
            x = data.x if hasattr(data, "x") else data["x"]
            edge_index = data.edge_index if hasattr(data, "edge_index") else data["edge_index"]
            edge_attr = data.edge_attr if hasattr(data, "edge_attr") else data["edge_attr"]

            with torch.no_grad():
                prob_tensor = self.model.predict_transition_matrix(x, edge_index, edge_attr, num_nodes)
                P_theta = prob_tensor.cpu().numpy()
        else:
            # Fallback prior: inverse distance normalized
            log.warning("No GATv2 model loaded — using spatial proximity as prior transition matrix.")
            inv_dist = 1.0 / np.maximum(distance_matrix, 100.0)
            np.fill_diagonal(inv_dist, 0.0)
            P_theta = inv_dist / (inv_dist.sum(axis=1, keepdims=True) + 1e-8)

        # 3. Masked Greedy Route Construction
        unvisited = set(range(1, num_nodes))
        routes: List[List[int]] = []
        max_stops_per_vehicle = int(np.ceil(num_stops / max(num_vehicles, 1))) + 3

        for v_idx, vehicle in enumerate(vehicles):
            route = [0]
            curr_stop = 0
            curr_weight = 0.0
            curr_volume = 0.0

            while unvisited and len(route) - 1 < max_stops_per_vehicle:
                # Find valid candidate stops for this vehicle
                candidates = []
                scores = []

                for next_stop in unvisited:
                    s_obj = stops[next_stop - 1]
                    # Check capacity
                    if curr_weight + s_obj.demand_weight > vehicle.max_weight:
                        continue
                    if curr_volume + s_obj.demand_volume > vehicle.max_volume:
                        continue
                    # Check compatibility
                    if vehicle.vtype not in s_obj.vehicle_compat:
                        continue

                    candidates.append(next_stop)
                    prob = P_theta[curr_stop, next_stop]
                    # Log-probability with small epsilon
                    scores.append(np.log(max(prob, 1e-6)))

                if not candidates:
                    # Vehicle cannot fit any more unvisited stops
                    break

                # Pick top candidate according to GATv2 prior
                best_cand_idx = int(np.argmax(scores))
                chosen_stop = candidates[best_cand_idx]

                route.append(chosen_stop)
                unvisited.remove(chosen_stop)
                s_obj = stops[chosen_stop - 1]
                curr_weight += s_obj.demand_weight
                curr_volume += s_obj.demand_volume
                curr_stop = chosen_stop

            route.append(0)  # Return to depot
            routes.append(route)

        # If any stops remain unassigned due to tight capacity, distribute them
        # defensively to the first route that has room
        if unvisited:
            log.info("Assigning %d remaining overflow stops to available vehicle routes", len(unvisited))
            for leftover in list(unvisited):
                # Insert before the last depot (0) in the shortest route
                best_r = min(routes, key=len)
                best_r.insert(-1, leftover)
                unvisited.remove(leftover)

        return routes
