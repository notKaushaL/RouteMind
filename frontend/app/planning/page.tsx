"use client";

import { useEffect, useState } from "react";
import { Header } from "@/components/header";
import { MapView } from "@/components/map/map_view";
import { planDispatch } from "@/lib/api";
import {
  Sliders,
  Play,
  RotateCw,
  TrendingDown,
  CheckCircle,
  AlertOctagon,
  Clock,
  Navigation,
  Sparkles,
  Zap,
} from "lucide-react";

export default function PlanningPage() {
  const [solverType, setSolverType] = useState("ortools_gatv2_warm");
  const [riskMode, setRiskMode] = useState("p90_robust");
  const [nStops, setNStops] = useState(30);
  const [daySeed, setDaySeed] = useState(42);
  const [wDistance, setWDistance] = useState(1.0);
  const [wLateness, setWLateness] = useState(5.0);
  const [wStability, setWStability] = useState(2.0);

  const [loading, setLoading] = useState(false);
  const [planData, setPlanData] = useState<any>(null);
  const [selectedStop, setSelectedStop] = useState<any>(null);

  const runPlan = async () => {
    setLoading(true);
    try {
      const data = await planDispatch({
        day_seed: daySeed,
        n_stops: nStops,
        solver_type: solverType,
        risk_mode: riskMode,
        weights: {
          w1_distance: wDistance,
          w2_lateness: wLateness,
          w3_overtime: 3.0,
          w4_unassigned: 100.0,
          w5_stability: wStability,
          w6_workload_var: 0.5,
        },
      });
      setPlanData(data);
    } catch (e) {
      console.error("Planning failed:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runPlan();
  }, [solverType, riskMode]);

  return (
    <div className="flex flex-col h-full flex-1">
      <Header
        simulationTime="09:00"
        activeVehicles={3}
        totalVehicles={3}
        onTimePct={planData?.metrics?.on_time_pct ?? 100.0}
        costJ={planData?.metrics?.composite_objective_cost}
      />

      <div className="flex-1 p-6 flex flex-col xl:flex-row gap-6 min-h-0 overflow-y-auto">
        {/* Left Control Panel: Solver Configuration */}
        <div className="w-full xl:w-80 shrink-0 space-y-5">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-5">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-sky-400" />
                <h2 className="font-semibold text-sm text-white">Solver Controls</h2>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                Phase 1–3
              </span>
            </div>

            {/* Algorithm Selector */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-300 block">
                Routing Engine
              </label>
              <div className="space-y-1.5">
                {[
                  {
                    id: "ortools_gatv2_warm",
                    name: "GATv2 Warm-Start + OR-Tools",
                    badge: "Proposed",
                    desc: "Neural zone prior + exact GLS search",
                  },
                  {
                    id: "ortools_cold",
                    name: "Cold-Start OR-Tools GLS",
                    badge: "Baseline 2",
                    desc: "Cheapest-arc seed + 2s local search",
                  },
                  {
                    id: "greedy",
                    name: "Cheapest Insertion Greedy",
                    badge: "Baseline 1",
                    desc: "Deterministic heuristic (<1ms)",
                  },
                ].map((s) => (
                  <button
                    key={s.id}
                    onClick={() => setSolverType(s.id)}
                    className={`w-full text-left p-3 rounded-xl border transition-all text-xs ${
                      solverType === s.id
                        ? "bg-sky-500/10 border-sky-500/50 text-white shadow-sm"
                        : "bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-slate-100">{s.name}</span>
                      <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-sky-400">
                        {s.badge}
                      </span>
                    </div>
                    <p className="text-[10px] text-slate-400 mt-1">{s.desc}</p>
                  </button>
                ))}
              </div>
            </div>

            {/* CQR Risk Toggle */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-300 block">
                Uncertainty & Risk Tolerance
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setRiskMode("p90_robust")}
                  className={`p-2.5 rounded-xl border text-xs font-medium text-center transition-all ${
                    riskMode === "p90_robust"
                      ? "bg-emerald-500/10 border-emerald-500/50 text-emerald-300"
                      : "bg-slate-950/60 border-slate-800 text-slate-400"
                  }`}
                >
                  <div className="font-semibold">CQR P90 Robust</div>
                  <div className="text-[10px] text-slate-400 mt-0.5">Calibrated SLA</div>
                </button>
                <button
                  onClick={() => setRiskMode("p50_median")}
                  className={`p-2.5 rounded-xl border text-xs font-medium text-center transition-all ${
                    riskMode === "p50_median"
                      ? "bg-slate-800 border-slate-600 text-white"
                      : "bg-slate-950/60 border-slate-800 text-slate-400"
                  }`}
                >
                  <div className="font-semibold">P50 Median</div>
                  <div className="text-[10px] text-slate-400 mt-0.5">Nominal ETA</div>
                </button>
              </div>
            </div>

            {/* Objective Weight Sliders */}
            <div className="space-y-3 pt-2 border-t border-slate-800">
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>SLA Lateness Penalty (w2)</span>
                  <span className="font-mono text-sky-400">{wLateness.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="15"
                  step="0.5"
                  value={wLateness}
                  onChange={(e) => setWLateness(parseFloat(e.target.value))}
                  className="w-full accent-sky-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Transit Distance Penalty (w1)</span>
                  <span className="font-mono text-sky-400">{wDistance.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="5"
                  step="0.5"
                  value={wDistance}
                  onChange={(e) => setWDistance(parseFloat(e.target.value))}
                  className="w-full accent-sky-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>
            </div>

            {/* Run Button */}
            <button
              onClick={runPlan}
              disabled={loading}
              className="w-full py-3 bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-semibold rounded-xl text-xs flex items-center justify-center space-x-2 shadow-lg shadow-sky-500/20 transition-all disabled:opacity-50"
            >
              {loading ? (
                <>
                  <RotateCw className="w-4 h-4 animate-spin" />
                  <span>Solving Combinatorial Graph...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  <span>Execute Dispatch Optimization</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Center: Interactive Map */}
        <div className="flex-1 flex flex-col min-h-[500px]">
          <MapView
            routesGeojson={planData?.routes_geojson}
            onSelectStop={(s) => setSelectedStop(s)}
            selectedStopId={selectedStop?.stop_id}
          />
        </div>

        {/* Right Panel: Solution Quality & Schedule */}
        <div className="w-full xl:w-96 shrink-0 space-y-5">
          {/* Solution Quality Summary Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h2 className="font-semibold text-sm text-white flex items-center space-x-2">
                <Sparkles className="w-4 h-4 text-sky-400" />
                <span>Solution Quality Metrics</span>
              </h2>
              <span className="text-[11px] font-mono text-emerald-400 font-bold">
                {planData?.metrics?.solve_time_seconds}s Solve
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                <span className="text-[11px] text-slate-400">Total Distance</span>
                <div className="text-lg font-bold text-white font-mono mt-0.5">
                  {planData?.metrics?.total_distance_km ?? 0} <span className="text-xs text-slate-400 font-normal">km</span>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                <span className="text-[11px] text-slate-400">Total Duration</span>
                <div className="text-lg font-bold text-white font-mono mt-0.5">
                  {planData?.metrics?.total_duration_min ?? 0} <span className="text-xs text-slate-400 font-normal">min</span>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                <span className="text-[11px] text-slate-400">SLA Breaches</span>
                <div className="text-lg font-bold text-emerald-400 font-mono mt-0.5">
                  {planData?.metrics?.sla_violations_count ?? 0} <span className="text-xs text-slate-400 font-normal">stops</span>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                <span className="text-[11px] text-slate-400">Composite Cost J</span>
                <div className="text-lg font-bold text-sky-400 font-mono mt-0.5">
                  {planData?.metrics?.composite_objective_cost ?? 0}
                </div>
              </div>
            </div>

            {/* Vehicle Fleet Breakdown */}
            <div className="space-y-2 pt-2 border-t border-slate-800">
              <span className="text-xs font-semibold text-slate-300 block">Fleet Allocation</span>
              {planData?.vehicles?.map((v: any) => (
                <div
                  key={v.vehicle_id}
                  className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950/40 border border-slate-800/80 text-xs"
                >
                  <div className="flex items-center space-x-2">
                    <div
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: v.color }}
                    />
                    <span className="font-semibold text-slate-200">{v.vehicle_id}</span>
                    <span className="text-[10px] text-slate-400 uppercase font-mono">({v.vtype})</span>
                  </div>
                  <div className="flex items-center space-x-3 text-right font-mono">
                    <span className="text-slate-300">{v.total_stops} stops</span>
                    <span className="text-slate-400">{v.capacity_utilization_pct}% cap</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Stop Inspection Drawer */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-xs text-slate-200 flex items-center space-x-2">
                <Navigation className="w-3.5 h-3.5 text-sky-400" />
                <span>Selected Stop Inspector</span>
              </h3>
              {selectedStop && (
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-300">
                  {selectedStop.stop_id}
                </span>
              )}
            </div>

            {selectedStop ? (
              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-2 text-xs">
                <div className="flex justify-between text-slate-400">
                  <span>Sequence Index:</span>
                  <span className="font-mono text-white font-semibold">#{selectedStop.sequence_idx}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Assigned Vehicle:</span>
                  <span className="font-mono text-sky-400 font-semibold">{selectedStop.vehicle_id}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>SLA Time Window:</span>
                  <span className="font-mono text-slate-200">
                    {selectedStop.tw_open_time} – {selectedStop.tw_close_time}
                  </span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Estimated Arrival:</span>
                  <span className="font-mono font-semibold text-emerald-400">
                    {selectedStop.arrival_time}
                  </span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>H3 Zone (Res 8):</span>
                  <span className="font-mono text-[10px] text-slate-300">{selectedStop.h3_cell}</span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-400 py-3 text-center">
                Click any stop marker on the map to inspect arrival window, cargo weight, and vehicle sequence.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
