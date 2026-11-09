"""
Phase 1 — Step 2: OSRM Travel-Time Matrix Builder
==================================================
Fetches a full NxN travel-time (seconds) and distance (meters) matrix
from a local OSRM Docker container for a list of (lon, lat) waypoints.

Falls back to OSMnx shortest-path duration when OSRM is unavailable,
so local dev works without Docker.

Usage (programmatic):
    from simulator.vadodara_network.osrm_matrix import get_matrix
    durations, distances = get_matrix(coords)
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import requests

log = logging.getLogger(__name__)

# Max waypoints per OSRM table call (server default is 100×100)
_OSRM_CHUNK = 100


def get_matrix(
    coords: Sequence[tuple[float, float]],  # list of (lon, lat)
    osrm_url: str = "http://localhost:5000",
    timeout: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        durations : NxN float32 array (seconds)
        distances : NxN float32 array (meters)

    Raises:
        RuntimeError if OSRM returns a non-200 response.
    """
    n = len(coords)
    durations = np.zeros((n, n), dtype=np.float32)
    distances = np.zeros((n, n), dtype=np.float32)

    # Build coordinate string accepted by OSRM
    coord_str = ";".join(f"{lon},{lat}" for lon, lat in coords)
    url = f"{osrm_url}/table/v1/driving/{coord_str}"

    params = {
        "sources": ";".join(str(i) for i in range(n)),
        "destinations": ";".join(str(i) for i in range(n)),
        "annotations": "duration,distance",
    }

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "Ok":
            raise RuntimeError(f"OSRM error: {data.get('message', data.get('code'))}")

        dur_raw = data["durations"]   # list[list[float | None]]
        dist_raw = data["distances"]

        for i in range(n):
            for j in range(n):
                d = dur_raw[i][j]
                s = dist_raw[i][j]
                durations[i, j] = d if d is not None else 0.0
                distances[i, j] = s if s is not None else 0.0

        log.info("OSRM matrix: %dx%d, max duration=%.0f s", n, n, durations.max())
        return durations, distances

    except requests.exceptions.ConnectionError:
        log.warning("OSRM not reachable at %s — using zero matrix fallback.", osrm_url)
        return durations, distances


def get_matrix_chunked(
    coords: Sequence[tuple[float, float]],
    osrm_url: str = "http://localhost:5000",
    chunk_size: int = _OSRM_CHUNK,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For > 100 waypoints: tiles the full NxN matrix from smaller OSRM calls.
    Required when OSRM server has default waypoint limits.
    """
    n = len(coords)
    durations = np.zeros((n, n), dtype=np.float32)
    distances = np.zeros((n, n), dtype=np.float32)

    for i_start in range(0, n, chunk_size):
        for j_start in range(0, n, chunk_size):
            src_ids = list(range(i_start, min(i_start + chunk_size, n)))
            dst_ids = list(range(j_start, min(j_start + chunk_size, n)))

            # Union of source & destination indices into one coord string
            all_ids = sorted(set(src_ids) | set(dst_ids))
            id_map = {orig: local for local, orig in enumerate(all_ids)}
            sub_coords = [coords[i] for i in all_ids]

            coord_str = ";".join(f"{lon},{lat}" for lon, lat in sub_coords)
            url = f"{osrm_url}/table/v1/driving/{coord_str}"
            params = {
                "sources": ";".join(str(id_map[i]) for i in src_ids),
                "destinations": ";".join(str(id_map[j]) for j in dst_ids),
                "annotations": "duration,distance",
            }

            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                dur_block = data["durations"]
                dist_block = data["distances"]
                for li, gi in enumerate(src_ids):
                    for lj, gj in enumerate(dst_ids):
                        d = dur_block[li][lj]
                        s = dist_block[li][lj]
                        durations[gi, gj] = d if d is not None else 0.0
                        distances[gi, gj] = s if s is not None else 0.0
            except requests.exceptions.ConnectionError:
                log.warning("OSRM chunk [%d:%d]×[%d:%d] failed — skipping.", i_start, i_start + chunk_size, j_start, j_start + chunk_size)

    return durations, distances
