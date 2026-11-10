"use client";

import { useEffect, useRef, useState } from "react";
import { Layers, MapPin, ZoomIn, ZoomOut, AlertTriangle, ShieldCheck } from "lucide-react";

interface MapViewProps {
  routesGeojson?: any;
  beforeRoutesGeojson?: any;
  showDiff?: boolean;
  onSelectStop?: (stop: any) => void;
  disruptionCell?: string;
  selectedStopId?: string;
}

export function MapView({
  routesGeojson,
  beforeRoutesGeojson,
  showDiff = false,
  onSelectStop,
  disruptionCell,
  selectedStopId,
}: MapViewProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<any>(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [useFallbackSvg, setUseFallbackSvg] = useState(false);

  const VADODARA_CENTER = [73.1812, 22.3072]; // [lon, lat]

  useEffect(() => {
    if (!mapContainer.current) return;

    let maplibregl: any;
    try {
      maplibregl = require("maplibre-gl");
    } catch (e) {
      console.warn("maplibre-gl could not be loaded, using SVG fallback:", e);
      setUseFallbackSvg(true);
      return;
    }

    try {
      const map = new maplibregl.Map({
        container: mapContainer.current,
        style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        center: VADODARA_CENTER,
        zoom: 12.5,
        attributionControl: false,
      });

      map.on("load", () => {
        mapInstance.current = map;
        setMapLoaded(true);
      });

      map.on("error", (e: any) => {
        console.warn("MapLibre GL warning/fallback:", e);
      });

      return () => {
        map.remove();
        mapInstance.current = null;
      };
    } catch (err) {
      console.warn("WebGL not supported or MapLibre initialization failed, rendering SVG fallback:", err);
      setUseFallbackSvg(true);
    }
  }, []);

  // Update GeoJSON Layers
  useEffect(() => {
    if (!mapInstance.current || !mapLoaded || !routesGeojson) return;
    const map = mapInstance.current;

    try {
      // 1. Before Routes Layer (Dashed Grey Line)
      if (beforeRoutesGeojson && showDiff) {
        if (map.getSource("before-routes")) {
          map.getSource("before-routes").setData(beforeRoutesGeojson);
        } else {
          map.addSource("before-routes", {
            type: "geojson",
            data: beforeRoutesGeojson,
          });
          map.addLayer({
            id: "before-routes-layer",
            type: "line",
            source: "before-routes",
            filter: ["==", "$type", "LineString"],
            paint: {
              "line-color": "#64748b",
              "line-width": 3,
              "line-dasharray": [3, 2],
              "line-opacity": 0.7,
            },
          });
        }
      } else if (map.getLayer("before-routes-layer")) {
        map.removeLayer("before-routes-layer");
        map.removeSource("before-routes");
      }

      // 2. Active / After Routes Source
      if (map.getSource("routemind-data")) {
        map.getSource("routemind-data").setData(routesGeojson);
      } else {
        map.addSource("routemind-data", {
          type: "geojson",
          data: routesGeojson,
        });

        // Vehicle Route Lines
        map.addLayer({
          id: "route-lines",
          type: "line",
          source: "routemind-data",
          filter: ["==", "$type", "LineString"],
          paint: {
            "line-color": ["coalesce", ["get", "color"], "#38bdf8"],
            "line-width": 4,
            "line-opacity": 0.85,
          },
        });

        // Stop Circles
        map.addLayer({
          id: "stop-points",
          type: "circle",
          source: "routemind-data",
          filter: ["all", ["==", "$type", "Point"], ["!=", "type", "depot"]],
          paint: {
            "circle-radius": [
              "case",
              ["boolean", ["get", "is_priority"], false],
              9,
              7,
            ],
            "circle-color": [
              "case",
              ["boolean", ["get", "is_sla_breached"], false],
              "#ef4444", // Red breached
              ["boolean", ["get", "is_priority"], false],
              "#f59e0b", // Amber priority
              ["coalesce", ["get", "color"], "#38bdf8"],
            ],
            "circle-stroke-width": 2,
            "circle-stroke-color": "#0f172a",
          },
        });

        // Depot Point Marker
        map.addLayer({
          id: "depot-point",
          type: "circle",
          source: "routemind-data",
          filter: ["==", "type", "depot"],
          paint: {
            "circle-radius": 12,
            "circle-color": "#818cf8",
            "circle-stroke-width": 3,
            "circle-stroke-color": "#ffffff",
          },
        });

        // Click Handler on stops
        map.on("click", "stop-points", (e: any) => {
          if (e.features && e.features[0]) {
            const props = e.features[0].properties;
            if (onSelectStop) onSelectStop(props);
          }
        });

        map.on("mouseenter", "stop-points", () => {
          map.getCanvas().style.cursor = "pointer";
        });
        map.on("mouseleave", "stop-points", () => {
          map.getCanvas().style.cursor = "";
        });
      }
    } catch (e) {
      console.warn("Map layer update warning:", e);
    }
  }, [routesGeojson, beforeRoutesGeojson, showDiff, mapLoaded, onSelectStop]);

  // Extract stops & lines for SVG fallback rendering
  const features = routesGeojson?.features || [];
  const routeLines = features.filter((f: any) => f.geometry?.type === "LineString");
  const stopPoints = features.filter((f: any) => f.geometry?.type === "Point");

  // Project lat/lon to SVG 800x600 viewBox
  const minLon = 73.12, maxLon = 73.24;
  const minLat = 22.25, maxLat = 22.36;
  const project = (coords: number[]) => {
    const x = ((coords[0] - minLon) / (maxLon - minLon)) * 800;
    const y = (1.0 - (coords[1] - minLat) / (maxLat - minLat)) * 600;
    return [x, y];
  };

  return (
    <div className="relative w-full h-full min-h-[500px] bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 shadow-2xl flex flex-col">
      {/* Map Header Overlay */}
      <div className="absolute top-4 left-4 z-10 flex items-center space-x-2">
        <div className="bg-slate-900/90 backdrop-blur px-3 py-1.5 rounded-lg border border-slate-700/80 shadow-lg flex items-center space-x-2">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
          <span className="text-xs font-semibold text-slate-200">
            Vadodara Road Network (Live H3 Grid)
          </span>
        </div>
        {showDiff && (
          <div className="bg-amber-500/10 border border-amber-500/30 px-2.5 py-1 rounded-lg text-amber-300 text-xs font-mono font-medium">
            Comparing: Baseline (Dashed) vs Re-Route (Solid)
          </div>
        )}
      </div>

      {/* MapLibre Canvas Container */}
      {!useFallbackSvg ? (
        <div ref={mapContainer} className="w-full h-full flex-1" />
      ) : (
        /* High-Fidelity SVG Fallback Canvas */
        <div className="w-full h-full flex-1 relative bg-slate-950 p-4 flex items-center justify-center">
          <svg
            viewBox="0 0 800 600"
            className="w-full h-full max-h-[700px] filter drop-shadow-md select-none"
          >
            {/* Dark Grid Background */}
            <defs>
              <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" strokeWidth="0.5" />
              </pattern>
            </defs>
            <rect width="800" height="600" fill="url(#grid)" />

            {/* Before Routes (Dashed) */}
            {showDiff &&
              beforeRoutesGeojson?.features
                ?.filter((f: any) => f.geometry?.type === "LineString")
                .map((f: any, i: number) => {
                  const pts = f.geometry.coordinates.map(project).map((p: any) => `${p[0]},${p[1]}`).join(" ");
                  return (
                    <polyline
                      key={`before-${i}`}
                      points={pts}
                      fill="none"
                      stroke="#64748b"
                      strokeWidth="2.5"
                      strokeDasharray="6,4"
                      opacity="0.6"
                    />
                  );
                })}

            {/* Route Lines */}
            {routeLines.map((f: any, i: number) => {
              const pts = f.geometry.coordinates.map(project).map((p: any) => `${p[0]},${p[1]}`).join(" ");
              return (
                <polyline
                  key={`line-${i}`}
                  points={pts}
                  fill="none"
                  stroke={f.properties?.color || "#38bdf8"}
                  strokeWidth="3.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity="0.85"
                />
              );
            })}

            {/* Stop Points */}
            {stopPoints.map((f: any, i: number) => {
              const [cx, cy] = project(f.geometry.coordinates);
              const isDepot = f.properties?.type === "depot";
              const isSelected = selectedStopId === f.properties?.stop_id;

              if (isDepot) {
                return (
                  <g key={`depot-${i}`}>
                    <circle cx={cx} cy={cy} r="14" fill="#818cf8" stroke="#ffffff" strokeWidth="3" />
                    <text x={cx} y={cy + 4} textAnchor="middle" fill="#ffffff" fontSize="10" fontWeight="bold">
                      HUB
                    </text>
                  </g>
                );
              }

              const isBreached = f.properties?.is_sla_breached;
              const isPriority = f.properties?.is_priority;
              const fill = isBreached ? "#ef4444" : isPriority ? "#f59e0b" : f.properties?.color || "#38bdf8";

              return (
                <g
                  key={`stop-${i}`}
                  className="cursor-pointer transition-transform hover:scale-125"
                  onClick={() => onSelectStop && onSelectStop(f.properties)}
                >
                  {isSelected && (
                    <circle cx={cx} cy={cy} r="16" fill="none" stroke="#38bdf8" strokeWidth="2" strokeDasharray="3,3" />
                  )}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isPriority ? "9" : "7"}
                    fill={fill}
                    stroke="#0f172a"
                    strokeWidth="2"
                  />
                  <text
                    x={cx}
                    y={cy + 3}
                    textAnchor="middle"
                    fill="#0f172a"
                    fontSize="7"
                    fontWeight="bold"
                  >
                    {f.properties?.sequence_idx || ""}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      )}

      {/* Floating Bottom Legend */}
      <div className="absolute bottom-4 right-4 z-10 bg-slate-900/90 backdrop-blur px-3.5 py-2.5 rounded-xl border border-slate-700/80 shadow-xl flex items-center space-x-4 text-xs">
        <div className="flex items-center space-x-1.5">
          <div className="w-3 h-3 rounded-full bg-indigo-400 border border-white" />
          <span className="text-slate-300">Hub</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-sky-400" />
          <span className="text-slate-300">V_TW1</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
          <span className="text-slate-300">V_TW2</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-pink-400" />
          <span className="text-slate-300">V_VAN</span>
        </div>
        <div className="flex items-center space-x-1.5 border-l border-slate-700 pl-3">
          <div className="w-2.5 h-2.5 rounded-full bg-amber-400" />
          <span className="text-slate-300">Priority</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-red-400" />
          <span className="text-slate-300">Breached</span>
        </div>
      </div>
    </div>
  );
}
