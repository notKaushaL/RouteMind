"use client";

import { useEffect, useRef, useState } from "react";
import { useTheme } from "next-themes";
import "maplibre-gl/dist/maplibre-gl.css";

interface MapViewProps {
  routesGeojson?: any;
  beforeRoutesGeojson?: any;
  showDiff?: boolean;
  onSelectStop?: (stop: any) => void;
  selectedStopId?: string;
}

export default function MapView({
  routesGeojson,
  beforeRoutesGeojson,
  showDiff = false,
  onSelectStop,
  selectedStopId,
}: MapViewProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<any>(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [useFallbackSvg, setUseFallbackSvg] = useState(false);
  const { resolvedTheme } = useTheme();

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

    const isDark = resolvedTheme === "dark";
    const styleUrl = isDark
      ? "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
      : "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

    try {
      if (mapInstance.current) {
        mapInstance.current.setStyle(styleUrl);
        return;
      }

      const map = new maplibregl.Map({
        container: mapContainer.current,
        style: styleUrl,
        center: VADODARA_CENTER,
        zoom: 12.3,
        attributionControl: false,
      });

      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");

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
      console.warn("WebGL not supported or MapLibre initialization failed, using SVG fallback:", err);
      setUseFallbackSvg(true);
    }
  }, [resolvedTheme]);

  // Update GeoJSON Layers
  useEffect(() => {
    if (!mapInstance.current || !mapLoaded) return;
    const map = mapInstance.current;
    const isDark = resolvedTheme === "dark";

    try {
      // 1. Before Routes Layer (Dashed)
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
              "line-color": isDark ? "#71717a" : "#a1a1aa",
              "line-width": 2.5,
              "line-dasharray": [3, 2],
              "line-opacity": 0.8,
            },
          });
        }
      } else if (map.getLayer("before-routes-layer")) {
        map.removeLayer("before-routes-layer");
        map.removeSource("before-routes");
      }

      // 2. Active Routes Source
      if (routesGeojson) {
        if (map.getSource("routemind-data")) {
          map.getSource("routemind-data").setData(routesGeojson);
        } else {
          map.addSource("routemind-data", {
            type: "geojson",
            data: routesGeojson,
          });

          // Vehicle Route Lines (Monochrome / High Contrast)
          map.addLayer({
            id: "route-lines",
            type: "line",
            source: "routemind-data",
            filter: ["==", "$type", "LineString"],
            paint: {
              "line-color": isDark ? "#fafafa" : "#18181b",
              "line-width": 3,
              "line-opacity": 0.9,
            },
          });

          // Stop Markers
          map.addLayer({
            id: "stop-points",
            type: "circle",
            source: "routemind-data",
            filter: ["all", ["==", "$type", "Point"], ["!=", "type", "depot"]],
            paint: {
              "circle-radius": [
                "case",
                ["boolean", ["get", "is_priority"], false],
                7.5,
                6,
              ],
              "circle-color": [
                "case",
                ["boolean", ["get", "is_sla_breached"], false],
                "#e11d48", // rose-600
                ["boolean", ["get", "is_priority"], false],
                "#f59e0b", // amber-500
                isDark ? "#22c55e" : "#16a34a", // emerald-600
              ],
              "circle-stroke-width": 2,
              "circle-stroke-color": isDark ? "#09090b" : "#ffffff",
            },
          });

          // Depot Point Marker
          map.addLayer({
            id: "depot-point",
            type: "circle",
            source: "routemind-data",
            filter: ["==", "type", "depot"],
            paint: {
              "circle-radius": 10,
              "circle-color": isDark ? "#ffffff" : "#000000",
              "circle-stroke-width": 3,
              "circle-stroke-color": isDark ? "#3f3f46" : "#e4e4e7",
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
      }
    } catch (e) {
      console.warn("Map layer update warning:", e);
    }
  }, [routesGeojson, beforeRoutesGeojson, showDiff, mapLoaded, onSelectStop, resolvedTheme]);

  // Extract stops & lines for SVG fallback rendering
  const features = routesGeojson?.features || [];
  const routeLines = features.filter((f: any) => f.geometry?.type === "LineString");
  const stopPoints = features.filter((f: any) => f.geometry?.type === "Point");

  const minLon = 73.12, maxLon = 73.24;
  const minLat = 22.25, maxLat = 22.36;
  const project = (coords: number[]) => {
    const x = ((coords[0] - minLon) / (maxLon - minLon)) * 800;
    const y = (1.0 - (coords[1] - minLat) / (maxLat - minLat)) * 600;
    return [x, y];
  };

  const isDark = resolvedTheme === "dark";

  return (
    <div className="relative h-full w-full overflow-hidden select-none">
      {!useFallbackSvg ? (
        <div ref={mapContainer} className="h-full w-full overflow-hidden" />
      ) : (
        /* Minimalist High-Contrast SVG Fallback */
        <div className="h-full w-full relative p-4 flex items-center justify-center bg-zinc-50 dark:bg-zinc-950">
          <svg viewBox="0 0 800 600" className="w-full h-full max-h-[700px] select-none">
            {/* Subtle Grid */}
            <defs>
              <pattern id="clean-grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path
                  d="M 40 0 L 0 0 0 40"
                  fill="none"
                  stroke={isDark ? "#18181b" : "#f4f4f5"}
                  strokeWidth="1"
                />
              </pattern>
            </defs>
            <rect width="800" height="600" fill="url(#clean-grid)" />

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
                      stroke={isDark ? "#52525b" : "#a1a1aa"}
                      strokeWidth="2"
                      strokeDasharray="5,4"
                      opacity="0.8"
                    />
                  );
                })}

            {/* Active Route Lines */}
            {routeLines.map((f: any, i: number) => {
              const pts = f.geometry.coordinates.map(project).map((p: any) => `${p[0]},${p[1]}`).join(" ");
              return (
                <polyline
                  key={`line-${i}`}
                  points={pts}
                  fill="none"
                  stroke={isDark ? "#fafafa" : "#18181b"}
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
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
                    <circle
                      cx={cx}
                      cy={cy}
                      r="10"
                      fill={isDark ? "#ffffff" : "#000000"}
                      stroke={isDark ? "#3f3f46" : "#e4e4e7"}
                      strokeWidth="3"
                    />
                  </g>
                );
              }

              const isBreached = f.properties?.is_sla_breached;
              const isPriority = f.properties?.is_priority;
              const fill = isBreached ? "#e11d48" : isPriority ? "#f59e0b" : "#16a34a";

              return (
                <g
                  key={`stop-${i}`}
                  className="cursor-pointer transition-transform hover:scale-125"
                  onClick={() => onSelectStop && onSelectStop(f.properties)}
                >
                  {isSelected && (
                    <circle
                      cx={cx}
                      cy={cy}
                      r="14"
                      fill="none"
                      stroke={isDark ? "#ffffff" : "#000000"}
                      strokeWidth="1.5"
                      strokeDasharray="3,3"
                    />
                  )}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isPriority ? "7" : "5.5"}
                    fill={fill}
                    stroke={isDark ? "#09090b" : "#ffffff"}
                    strokeWidth="2"
                  />
                </g>
              );
            })}
          </svg>
        </div>
      )}
    </div>
  );
}
