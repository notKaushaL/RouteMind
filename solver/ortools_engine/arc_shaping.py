"""
solver/ortools_engine/arc_shaping.py
=====================================
Phase 3 — Hybrid Arc-Cost Shaping for Neural-Guided OR-Tools Routing

Blends physical road network costs (OSRM distance/duration) with learned human driver
transition priors P_theta(j | i) from GATv2:

    C_tilde_ij = C_ij^OSRM - lambda * log(P_theta(j | i) + epsilon)

Properties:
  - If lambda = 0: Exact physical routing (cold-start baseline)
  - If lambda > 0: High human transition probability P_theta -> reward / cost reduction;
                   Low transition probability -> severe cost penalty
"""
from __future__ import annotations

import logging
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)


class ArcCostShaper:
    """
    Computes hybrid arc costs combining physical travel cost and learned human transition priors.
    """

    def __init__(
        self,
        lambda_shaping: float = 0.50,
        epsilon: float = 1e-4,
        scale_factor: float = 1000.0,
    ):
        """
        Args:
            lambda_shaping: blending weight lambda >= 0.
            epsilon: small constant to avoid log(0).
            scale_factor: scale applied to log-p prior to match physical cost magnitude (meters/seconds).
        """
        self.lambda_shaping = float(lambda_shaping)
        self.epsilon = float(epsilon)
        self.scale_factor = float(scale_factor)

    def shape_cost_matrix(
        self,
        base_cost_matrix: np.ndarray,
        transition_prob_matrix: np.ndarray,
        lambda_shaping: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply arc-cost shaping to an (N, N) physical cost matrix.

        Args:
            base_cost_matrix: (N, N) array (e.g. distance in meters or duration in seconds)
            transition_prob_matrix: (N, N) array where P[i, j] in [0, 1] is P_theta(j | i)
            lambda_shaping: optional override for lambda

        Returns:
            shaped_matrix: (N, N) float32 array
        """
        lam = self.lambda_shaping if lambda_shaping is None else float(lambda_shaping)
        if lam <= 0.0:
            return base_cost_matrix.copy()

        # Prior penalty: -log(P_theta(j | i) + epsilon) >= 0
        log_p = np.log(np.clip(transition_prob_matrix, self.epsilon, 1.0))
        # When P is close to 1: -log_p is small (~0).
        # When P is near epsilon: -log_p is large (~9.2).
        prior_penalty = -log_p * self.scale_factor

        # Set diagonal (self-loops) to 0 penalty
        np.fill_diagonal(prior_penalty, 0.0)

        # C_tilde_ij = C_ij - lambda * log(P_theta) = C_ij + lambda * (-log(P_theta))
        shaped_cost = base_cost_matrix + (lam * prior_penalty)

        # Do not allow costs to become negative
        shaped_cost = np.maximum(shaped_cost, 0.0).astype(np.float32)

        return shaped_cost
