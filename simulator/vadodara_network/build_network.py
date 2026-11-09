"""
Phase 1 — Step 1: Vadodara Road Network Extraction & H3 Indexing
================================================================
Downloads Vadodara's drivable road network via OSMnx, builds a projected
graph, computes H3 hexagonal indices at resolution 8 and 9 for every node,
and persists the graph + H3-annotated node table for downstream use.

Usage:
    python simulator/vadodara_network/build_network.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import h3
import networkx as nx
import osmnx as ox
import pandas as pd

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent
GRAPH_PATH = OUT_DIR / "vadodara_drive.graphml"
NODES_PATH = OUT_DIR / "vadodara_nodes_h3.parquet"
EDGES_PATH = OUT_DIR / "vadodara_edges.parquet"
BBOX_PATH = OUT_DIR / "vadodara_bbox.json"


def build_network() -> nx.MultiDiGraph:
    """Download & prepare Vadodara drivable road network."""
    log.info("Configuring OSMnx cache at %s", settings.osmnx_cache_dir)
    settings.osmnx_cache_dir.mkdir(parents=True, exist_ok=True)
    ox.settings.cache_folder = str(settings.osmnx_cache_dir)
    ox.settings.use_cache = True
    ox.settings.log_console = False

    bbox = (
        settings.vadodara_north,
        settings.vadodara_south,
        settings.vadodara_east,
        settings.vadodara_west,
    )
    log.info("Downloading drivable network for bbox N=%.4f S=%.4f E=%.4f W=%.4f", *bbox)

    G = ox.graph_from_bbox(
        north=settings.vadodara_north,
        south=settings.vadodara_south,
        east=settings.vadodara_east,
        west=settings.vadodara_west,
        network_type="drive",
        simplify=True,
        retain_all=False,
    )
    log.info("Raw graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # Add speed/travel-time attributes
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    G = ox.project_graph(G, to_crs="EPSG:32643")  # UTM zone 43N (Gujarat)

    log.info("Projected to UTM 43N. Saving GraphML …")
    ox.save_graphml(G, filepath=str(GRAPH_PATH))

    return G


def annotate_h3(G: nx.MultiDiGraph) -> pd.DataFrame:
    """Add H3 cell indices at resolution 8 & 9 to every graph node."""
    nodes, _ = ox.graph_to_gdfs(G)

    # OSMnx projects; reproject back to WGS-84 for H3
    nodes_wgs = nodes.to_crs("EPSG:4326")

    log.info("Computing H3 indices for %d nodes …", len(nodes_wgs))
    nodes["h3_r8"] = nodes_wgs.apply(
        lambda r: h3.latlng_to_cell(r.geometry.y, r.geometry.x, settings.h3_res_coarse), axis=1
    )
    nodes["h3_r9"] = nodes_wgs.apply(
        lambda r: h3.latlng_to_cell(r.geometry.y, r.geometry.x, settings.h3_res_fine), axis=1
    )

    # Unique H3 cell counts
    log.info(
        "H3 resolution %d: %d unique cells | resolution %d: %d unique cells",
        settings.h3_res_coarse,
        nodes["h3_r8"].nunique(),
        settings.h3_res_fine,
        nodes["h3_r9"].nunique(),
    )

    # Persist
    nodes.drop(columns=["geometry"]).to_parquet(NODES_PATH)
    log.info("Saved annotated nodes → %s", NODES_PATH)

    return nodes


def save_edges(G: nx.MultiDiGraph) -> None:
    """Persist edge table (osmid, length_m, travel_time_s, speed_kph)."""
    _, edges = ox.graph_to_gdfs(G)
    keep = ["length", "travel_time", "speed_kph", "name", "highway"]
    edges[[c for c in keep if c in edges.columns]].to_parquet(EDGES_PATH)
    log.info("Saved edge table → %s", EDGES_PATH)


def save_bbox_meta() -> None:
    """Persist bounding-box metadata for downstream modules."""
    meta = {
        "north": settings.vadodara_north,
        "south": settings.vadodara_south,
        "east": settings.vadodara_east,
        "west": settings.vadodara_west,
        "depot_lat": settings.depot_lat,
        "depot_lon": settings.depot_lon,
        "crs": "EPSG:32643",
    }
    BBOX_PATH.write_text(json.dumps(meta, indent=2))
    log.info("Saved bbox metadata → %s", BBOX_PATH)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    G = build_network()
    annotate_h3(G)
    save_edges(G)
    save_bbox_meta()
    log.info("✅ Network build complete.")


if __name__ == "__main__":
    main()
