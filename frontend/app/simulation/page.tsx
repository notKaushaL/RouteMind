"use client";

import { useEffect, useState } from "react";
import { Header } from "@/components/header";
import { MapView } from "@/components/map/map_view";
import { initSimulation, injectDisruption, explainDelay } from "@/lib/api";
import {
  AlertTriangle,
  Flame,
  Truck,
  PackagePlus,
  Play,
  RotateCcw,
  Sparkles,
  Layers,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  ShieldCheck,
  Check,
  Info,
} from "lucide-react";

export default function SimulationPage() {
  const [loading, setLoading] = useState(false);
  const [initState, setInitState] = useState<any>(null);
  const [disruptResult, setDisruptResult] = useState<any>(null);
  const [selectedDisruptedStop, setSelectedDisruptedStop] = useState<any>(null);
  const [shapData, setShapData] = useState<any>(null);
  const [showDiff, setShowDiff] = useState(true);
  const [planApproved, setPlanApproved] = useState(false);

  const startSimulation = async () => {
    setLoading(true);
    try {
      const data = await initSimulation(3000, 30);
      setInitState(data);
      setDisruptResult(null);
      setShapData(null);
      setSelectedDisruptedStop(null);
      setPlanApproved(false);
    } catch (e) {
      console.error("Init simulation failed:", e);
    } finally {
      setLoading(false);
    }
  };

  const triggerShock = async (shockType: string, severity = 2.5, vid?: string) => {
    setLoading(true);
    setPlanApproved(false);
    try {
      const data = await injectDisruption({
        type: shockType,
        severity,
        vehicle_id: vid,
      });
      setDisruptResult(data);
    } catch (e) {
      console.error("Disruption failed:", e);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectStopForExplain = async (stop: any) => {
    setSelectedDisruptedStop(stop);
    try {
      const shap = await explainDelay({
        h3_cell: stop.h3_cell || "88619b4563fffff",
        cluster_id: stop.cluster_id || 0,
        osrm_base_time_s: 480.0,
        hour: 17,
        vehicle_class: "three_wheeler",
        package_weight_kg: 10.5,
      });
      setShapData(shap);
    } catch (e) {
      console.error("TreeSHAP explainability fetch failed:", e);
    }
  };

  useEffect(() => {
    startSimulation();
  }, []);

  const activeRoutes = disruptResult?.after_routes_geojson || initState?.routes_geojson;
  const beforeRoutes = disruptResult?.before_routes_geojson || initState?.routes_geojson;
  const stateSummary = disruptResult?.state || initState?.state;

  return (
    <div className="flex flex-col h-full flex-1">
      <Header
        simulationTime={stateSummary?.sim_time_str ?? "09:00"}
        activeVehicles={stateSummary?.active_vehicles_count ?? 3}
        totalVehicles={3}
        onTimePct={100.0}
      />

      <div className="flex-1 p-6 flex flex-col xl:flex-row gap-6 min-h-0 overflow-y-auto">
        {/* Left Column: Disruption Trigger Suite & Alert Banner */}
        <div className="w-full xl:w-96 shrink-0 space-y-5">
          {/* Disruption Trigger Palette */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2">
                <Flame className="w-4 h-4 text-rose-400" />
                <h2 className="font-semibold text-sm text-white">Live Disruption Suite</h2>
              </div>
              <button
                onClick={startSimulation}
                className="text-[11px] text-slate-400 hover:text-white flex items-center space-x-1"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset Sim</span>
              </button>
            </div>

            <p className="text-xs text-slate-400">
              Inject real-time operational shocks into the simulated Vadodara workday. MaskablePPO will dynamically select meta-actions to resolve delays.
            </p>

            <div className="space-y-2.5">
              {/* Trigger 1: Traffic Spike */}
              <button
                onClick={() => triggerShock("TRAFFIC_SPIKE", 2.5)}
                disabled={loading}
                className="w-full p-3.5 rounded-xl border border-rose-500/30 bg-rose-500/10 hover:bg-rose-500/20 text-rose-200 transition-all text-left flex items-start space-x-3 group"
              >
                <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5 group-hover:scale-110 transition-transform" />
                <div>
                  <div className="font-semibold text-xs text-white">
                    Simulate Traffic Spike (2.5x Delay)
                  </div>
                  <p className="text-[11px] text-rose-300/80 mt-0.5">
                    Alkapuri corridor congestion surge (+15m delay wave)
                  </p>
                </div>
              </button>

              {/* Trigger 2: Vehicle Breakdown */}
              <button
                onClick={() => triggerShock("BREAKDOWN", 1.0, "V_TW1")}
                disabled={loading}
                className="w-full p-3.5 rounded-xl border border-amber-500/30 bg-amber-500/10 hover:bg-amber-500/20 text-amber-200 transition-all text-left flex items-start space-x-3 group"
              >
                <Truck className="w-5 h-5 text-amber-400 shrink-0 mt-0.5 group-hover:scale-110 transition-transform" />
                <div>
                  <div className="font-semibold text-xs text-white">
                    Simulate Vehicle Breakdown (V_TW1)
                  </div>
                  <p className="text-[11px] text-amber-300/80 mt-0.5">
                    Mechanical failure; remaining stops salvaged & re-assigned
                  </p>
                </div>
              </button>

              {/* Trigger 3: Urgent Parcel */}
              <button
                onClick={() => triggerShock("URGENT_ORDER", 1.0)}
                disabled={loading}
                className="w-full p-3.5 rounded-xl border border-sky-500/30 bg-sky-500/10 hover:bg-sky-500/20 text-sky-200 transition-all text-left flex items-start space-x-3 group"
              >
                <PackagePlus className="w-5 h-5 text-sky-400 shrink-0 mt-0.5 group-hover:scale-110 transition-transform" />
                <div>
                  <div className="font-semibold text-xs text-white">
                    Inject Urgent High-Priority Parcel
                  </div>
                  <p className="text-[11px] text-sky-300/80 mt-0.5">
                    Dynamic mid-day arrival with strict 60-min SLA window
                  </p>
                </div>
              </button>
            </div>
          </div>

          {/* MaskablePPO Decision Banner */}
          {disruptResult && (
            <div className="bg-slate-900 border border-sky-500/40 rounded-2xl p-5 shadow-2xl space-y-3 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-sky-500/10 rounded-full blur-2xl pointer-events-none" />

              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono font-bold tracking-wider uppercase px-2 py-0.5 rounded bg-sky-500/20 text-sky-300 border border-sky-500/30">
                  RL META-CONTROLLER ACTION
                </span>
                <span className="text-xs font-mono text-emerald-400 font-semibold">
                  {disruptResult.decision_latency_ms} ms Latency
                </span>
              </div>

              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1.5">
                <div className="text-xs text-slate-400 font-medium">Selected Policy Action:</div>
                <div className="text-sm font-mono font-bold text-sky-400">
                  {disruptResult.agent_action_name}
                </div>
                <p className="text-[11px] text-slate-300 leading-relaxed pt-1">
                  {disruptResult.description}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <span className="text-[10px] text-slate-400">Route Churn (KT)</span>
                  <div className="font-mono font-bold text-emerald-400 mt-0.5">
                    {disruptResult.route_churn_score} <span className="text-[10px] text-slate-400 font-normal">(Stable)</span>
                  </div>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <span className="text-[10px] text-slate-400">Unassigned Queue</span>
                  <div className="font-mono font-bold text-slate-200 mt-0.5">
                    {disruptResult.state?.unassigned_queue_count ?? 0} stops
                  </div>
                </div>
              </div>

              {/* Action Bar: Approve & Dispatch */}
              <div className="pt-2">
                <button
                  onClick={() => setPlanApproved(true)}
                  disabled={planApproved}
                  className={`w-full py-2.5 rounded-xl font-semibold text-xs flex items-center justify-center space-x-2 transition-all ${
                    planApproved
                      ? "bg-emerald-600/30 text-emerald-300 border border-emerald-500/40 cursor-default"
                      : "bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-lg shadow-emerald-500/20"
                  }`}
                >
                  {planApproved ? (
                    <>
                      <Check className="w-4 h-4 text-emerald-300" />
                      <span>Re-plan Dispatched to Driver Telemetry</span>
                    </>
                  ) : (
                    <>
                      <ShieldCheck className="w-4 h-4" />
                      <span>Approve & Dispatch Plan</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Center: Interactive Map with Before/After Diff */}
        <div className="flex-1 flex flex-col min-h-[500px] space-y-3">
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center space-x-2">
              <span className="text-xs font-semibold text-slate-300">Route Diff Mode:</span>
              <button
                onClick={() => setShowDiff(!showDiff)}
                className={`px-3 py-1 rounded-lg text-xs font-medium border transition-all ${
                  showDiff
                    ? "bg-sky-500/20 border-sky-500/50 text-sky-300"
                    : "bg-slate-900 border-slate-800 text-slate-400"
                }`}
              >
                {showDiff ? "Active (Dashed = Before, Solid = After)" : "Diff Overlay Hidden"}
              </button>
            </div>

            <span className="text-xs text-slate-400">
              Click any stop to inspect TreeSHAP Delay Attribution
            </span>
          </div>

          <div className="flex-1 min-h-[500px]">
            <MapView
              routesGeojson={activeRoutes}
              beforeRoutesGeojson={beforeRoutes}
              showDiff={showDiff}
              onSelectStop={handleSelectStopForExplain}
              selectedStopId={selectedDisruptedStop?.stop_id}
            />
          </div>
        </div>

        {/* Right Column: Disrupted Stops Table & TreeSHAP Waterfall */}
        <div className="w-full xl:w-96 shrink-0 space-y-5">
          {/* TreeSHAP Explainability Drawer */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2">
                <Sparkles className="w-4 h-4 text-amber-400" />
                <h3 className="font-semibold text-xs text-white">TreeSHAP Delay Inspector</h3>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-amber-400">
                LightGBM CQR
              </span>
            </div>

            {shapData ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-400">Nominal OSRM</span>
                    <div className="font-mono font-bold text-slate-200 mt-0.5">
                      {shapData.osrm_nominal_min} min
                    </div>
                  </div>
                  <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-400">P90 Calibrated ETA</span>
                    <div className="font-mono font-bold text-amber-400 mt-0.5">
                      {shapData.cqr_p90_robust_min} min
                    </div>
                  </div>
                </div>

                {/* Horizontal Waterfall Attributions */}
                <div className="space-y-2">
                  <span className="text-[11px] font-semibold text-slate-300 block">
                    Feature Delay Waterfall (Minutes):
                  </span>
                  <div className="space-y-2 text-xs">
                    {shapData.top_factors?.map((f: any, idx: number) => {
                      const isPositive = f.direction === "increase";
                      return (
                        <div
                          key={idx}
                          className="p-2.5 rounded-xl bg-slate-950 border border-slate-800/80 space-y-1"
                        >
                          <div className="flex justify-between items-center">
                            <span className="font-semibold text-slate-200">{f.feature}</span>
                            <span
                              className={`font-mono font-bold ${
                                isPositive ? "text-rose-400" : "text-emerald-400"
                              }`}
                            >
                              {isPositive ? `+${f.impact_min.toFixed(2)}` : `${f.impact_min.toFixed(2)}`} m
                            </span>
                          </div>
                          <p className="text-[10px] text-slate-400">{f.description}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-[11px] text-slate-300 leading-relaxed">
                  <div className="font-semibold text-sky-400 mb-1 flex items-center space-x-1.5">
                    <Info className="w-3.5 h-3.5" />
                    <span>Explainability Summary:</span>
                  </div>
                  {shapData.narrative}
                </div>
              </div>
            ) : (
              <div className="p-6 text-center text-xs text-slate-400 border border-dashed border-slate-800 rounded-xl">
                Click any stop marker on the map to trigger TreeSHAP delay decomposition.
              </div>
            )}
          </div>

          {/* Disrupted Orders Schedule Table */}
          {disruptResult?.disrupted_orders?.length > 0 && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
              <h3 className="font-semibold text-xs text-white">Disrupted Stops Re-sequenced</h3>
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {disruptResult.disrupted_orders.map((o: any) => (
                  <div
                    key={o.stop_id}
                    onClick={() => handleSelectStopForExplain(o)}
                    className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800 hover:border-sky-500/50 cursor-pointer transition-all text-xs flex items-center justify-between"
                  >
                    <div>
                      <div className="font-semibold text-slate-200">{o.stop_id}</div>
                      <div className="text-[10px] text-slate-400">
                        ETA: {o.original_eta_str} → <span className="text-amber-400">{o.new_eta_str}</span>
                      </div>
                    </div>
                    <div className="text-right font-mono">
                      <span className="text-rose-400 font-bold">+{o.delay_delta_min}m</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
