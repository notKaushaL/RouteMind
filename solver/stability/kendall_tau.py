"""
Phase 1 — Route Stability Utilities
=====================================
Implements the Kendall-Tau sequence distance and vehicle-reassignment
count used in the route-stability penalty term w5 of the VRPTW objective:

    D_edit(R_new, R_prev) = λ_kt * kendall_tau_norm(R_new, R_prev)
                           + λ_ra * reassignment_count(R_new, R_prev)

Used by:
  - OR-Tools solver (Phase 1+): as a soft penalty term
  - MaskablePPO meta-controller (Phase 4): as part of reward shaping
  - Ablation experiments: to measure route churn across re-solves
"""
from __future__ import annotations

import numpy as np


def kendall_tau_distance(seq_a: list[str], seq_b: list[str]) -> int:
    """
    Compute the Kendall-Tau distance (number of discordant pairs)
    between two orderings of the same elements.

    Only elements present in BOTH sequences are considered.
    Elements unique to one sequence are ignored (they will be counted
    via the reassignment penalty instead).

    Args:
        seq_a: New route sequence (list of stop IDs)
        seq_b: Previous route sequence (list of stop IDs)

    Returns:
        Number of discordant pairs (integer ≥ 0).
    """
    common = set(seq_a) & set(seq_b)
    if len(common) < 2:
        return 0

    # Build rank lookup for seq_a over common stops
    rank_a: dict[str, int] = {sid: i for i, sid in enumerate(seq_a) if sid in common}
    rank_b: dict[str, int] = {sid: i for i, sid in enumerate(seq_b) if sid in common}

    items = sorted(common, key=lambda x: rank_a[x])
    n = len(items)
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Items are sorted by rank_a, so rank_a[items[i]] < rank_a[items[j]]
            # Discordant if rank_b order is reversed
            if rank_b[items[i]] > rank_b[items[j]]:
                discordant += 1

    return discordant


def kendall_tau_normalized(seq_a: list[str], seq_b: list[str]) -> float:
    """
    Normalized Kendall-Tau distance in [0, 1].
    0 = identical order, 1 = perfectly reversed.
    """
    common = set(seq_a) & set(seq_b)
    n = len(common)
    if n < 2:
        return 0.0
    max_pairs = n * (n - 1) / 2
    return kendall_tau_distance(seq_a, seq_b) / max_pairs


def reassignment_count(
    new_routes: dict[str, list[str]],
    prev_routes: dict[str, list[str]],
    unvisited_only: bool = True,
    completed_stops: set[str] | None = None,
) -> int:
    """
    Count stops that are assigned to a DIFFERENT vehicle in new_routes vs prev_routes.

    Args:
        new_routes:  vehicle_id → [stop_ids] for the new plan
        prev_routes: vehicle_id → [stop_ids] for the published plan
        unvisited_only: if True, only count stops not in completed_stops
        completed_stops: set of stop_ids already delivered (locked)

    Returns:
        Number of vehicle reassignments.
    """
    completed = completed_stops or set()
    prev_assignment: dict[str, str] = {}  # stop_id → vehicle_id
    for vid, stops in prev_routes.items():
        for sid in stops:
            if unvisited_only and sid in completed:
                continue
            prev_assignment[sid] = vid

    reassigned = 0
    for vid, stops in new_routes.items():
        for sid in stops:
            if sid in prev_assignment and prev_assignment[sid] != vid:
                reassigned += 1

    return reassigned


def route_stability_penalty(
    new_routes: dict[str, list[str]],
    prev_routes: dict[str, list[str]],
    completed_stops: set[str] | None = None,
    lambda_kt: float = 1.0,
    lambda_ra: float = 2.0,
) -> float:
    """
    Composite route-stability penalty D_edit(R_new, R_prev):

        D_edit = λ_kt * Σ_v kendall_tau_norm(R_v_new, R_v_prev)
               + λ_ra * reassignment_count(new, prev)

    Lower is better (0 = no disruption to published routes).

    Args:
        lambda_kt: weight for per-vehicle Kendall-Tau distance sum
        lambda_ra: weight for cross-vehicle reassignment count
    """
    completed = completed_stops or set()
    kt_total = 0.0

    all_vids = set(new_routes) | set(prev_routes)
    for vid in all_vids:
        seq_new = [s for s in new_routes.get(vid, []) if s not in completed]
        seq_prev = [s for s in prev_routes.get(vid, []) if s not in completed]
        kt_total += kendall_tau_normalized(seq_new, seq_prev)

    ra = reassignment_count(new_routes, prev_routes, completed_stops=completed)

    return lambda_kt * kt_total + lambda_ra * ra


# ── Vectorized convenience for large ablation runs ─────────────────────────────

def batch_stability(
    new_routes_list: list[dict[str, list[str]]],
    prev_routes: dict[str, list[str]],
    completed_stops: set[str] | None = None,
) -> np.ndarray:
    """Return stability penalties for a batch of new route plans."""
    return np.array(
        [route_stability_penalty(nr, prev_routes, completed_stops) for nr in new_routes_list],
        dtype=np.float64,
    )
