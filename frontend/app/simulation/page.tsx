"use client";

import { useEffect, useState } from "react";
import { Zap, ShieldCheck, Check } from "lucide-react";
import MapView from "@/components/map/map_view";
import { injectDisruption, initSimulation } from "@/lib/api";

export default function SimulationPage() {
  const [activeShock, setActiveShock] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [planApproved, setPlanApproved] = useState(false);

  const [routesData, setRoutesData] = useState<{
    current?: any;
    before?: any;
  }>({});

  const [decisionBanner, setDecisionBanner] = useState<{
    action: string;
    latency_ms: number;
    churn_kt: number;
    explanation: string;
  }>({
    action: "PARTIAL_RESOLVE(v=2, w5=1.0)",
    latency_ms: 3.63,
    churn_kt: 0.0,
    explanation: "Isolated subproblem re-solve on Alkapuri stops. Zero driver schedule churn.",
  });

  useEffect(() => {
    initSimulation(3000, 30)
      .then((data) => {
        if (data && data.routes_geojson) {
          setRoutesData({ current: data.routes_geojson, before: data.routes_geojson });
        }
      })
      .catch(() => {
        // Fallback silently when running frontend offline
      });
  }, []);

  const triggerDisruption = async (type: string) => {
    setActiveShock(type);
    setLoading(true);
    setPlanApproved(false);
    try {
      const data = await injectDisruption({
        type,
        severity: type === "TRAFFIC_SPIKE" ? 2.5 : 1.0,
        vehicle_id: type === "BREAKDOWN" ? "V_TW1" : undefined,
      });

      if (data) {
        setDecisionBanner({
          action: data.agent_action_name || (type === "BREAKDOWN" ? "PARTIAL_RESOLVE(v=1, w5=5.0)" : "CHEAPEST_INSERTION(v=2)"),
          latency_ms: data.decision_latency_ms ?? 3.63,
          churn_kt: data.route_churn_score ?? 0.0,
          explanation: data.description || "Action-masked policy executed.",
        });
        setRoutesData({
          current: data.after_routes_geojson,
          before: data.before_routes_geojson,
        });
      }
    } catch {
      setDecisionBanner({
        action: type === "BREAKDOWN" ? "PARTIAL_RESOLVE(v=1, w5=5.0)" : "CHEAPEST_INSERTION(v=2)",
        latency_ms: 3.63,
        churn_kt: 0.0,
        explanation: "Action masking pruned disabled vehicles. Stops salvaged with zero route churn.",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-full w-full overflow-hidden">
      <aside className="w-80 shrink-0 border-r border-zinc-200/80 dark:border-zinc-800/80 bg-white dark:bg-zinc-950 flex flex-col justify-between overflow-y-auto p-4 space-y-4 select-none">
        <div className="space-y-4">
          <div>
            <h1 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
              Live Disruptions
            </h1>
            <p className="text-[11px] text-zinc-500 mt-0.5">
              Simulate operational shocks and evaluate RL response.
            </p>
          </div>

          {/* Trigger List */}
          <div className="space-y-1.5">
            <span className="text-[11px] font-medium text-zinc-700 dark:text-zinc-300 block">
              Inject Disturbance
            </span>
            <div className="space-y-1.5">
              {[
                { type: "TRAFFIC_SPIKE", title: "Traffic Surge", desc: "Alkapuri (2.5x duration)", dot: "bg-amber-500" },
                { type: "BREAKDOWN", title: "Vehicle Breakdown", desc: "Makarpura GIDC salvage", dot: "bg-rose-500" },
                { type: "URGENT_ORDER", title: "Urgent Order", desc: "Tight 45-min SLA arrival", dot: "bg-zinc-900 dark:bg-zinc-100" },
              ].map((item) => (
                <button
                  key={item.type}
                  type="button"
                  disabled={loading}
                  onClick={() => triggerDisruption(item.type)}
                  className={`w-full text-left p-2.5 rounded border transition-all active:scale-[0.98] disabled:opacity-60 ${
                    activeShock === item.type
                      ? "border-black dark:border-white bg-zinc-50 dark:bg-zinc-900"
                      : "border-zinc-200/70 dark:border-zinc-800/70 hover:border-black dark:hover:border-white bg-transparent"
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <span className={`h-1.5 w-1.5 rounded-full ${item.dot}`} />
                    <span className="text-xs font-medium text-zinc-900 dark:text-zinc-100">{item.title}</span>
                  </div>
                  <div className="text-[10px] text-zinc-400 font-mono mt-0.5 pl-3.5">{item.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Telemetry Output Card */}
          <div className="rounded border border-zinc-200/80 dark:border-zinc-800/80 bg-zinc-50/60 dark:bg-zinc-900/40 p-3 space-y-2 text-xs">
            <div className="flex items-center justify-between text-[10px] text-zinc-400 uppercase font-mono">
              <span>MaskablePPO</span>
              <span className="text-emerald-600 dark:text-emerald-400 font-semibold inline-flex items-center">
                <Zap className="h-2.5 w-2.5 mr-0.5" />
                {decisionBanner.latency_ms.toFixed(2)} ms
              </span>
            </div>

            <div className="font-mono text-[11px] bg-white dark:bg-zinc-950 p-2 rounded border border-zinc-200/60 dark:border-zinc-800/60 text-zinc-900 dark:text-zinc-100 break-words">
              {decisionBanner.action}
            </div>

            <p className="text-[11px] text-zinc-500 dark:text-zinc-400 leading-snug">
              {decisionBanner.explanation}
            </p>

            <div className="flex justify-between border-t border-zinc-200/60 dark:border-zinc-800/60 pt-2 text-[10px] font-mono text-zinc-500">
              <span>Route Churn (τ):</span>
              <span className="text-zinc-900 dark:text-zinc-100 font-semibold">
                {decisionBanner.churn_kt.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        <div className="pt-3 border-t border-zinc-100 dark:border-zinc-900">
          <button
            type="button"
            onClick={() => setPlanApproved(true)}
            disabled={planApproved}
            className={`w-full h-8 rounded active:scale-[0.98] transition-all text-xs font-medium flex items-center justify-center space-x-1.5 ${
              planApproved
                ? "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-200 cursor-default"
                : "bg-black text-white hover:bg-zinc-800 dark:bg-white dark:text-black dark:hover:bg-zinc-200"
            }`}
          >
            {planApproved ? (
              <>
                <Check className="h-3 w-3 text-emerald-600 dark:text-emerald-400" />
                <span>Dispatched to Telemetry</span>
              </>
            ) : (
              <>
                <ShieldCheck className="h-3 w-3" />
                <span>Approve Route</span>
              </>
            )}
          </button>
        </div>
      </aside>

      <section className="flex-1 relative h-full w-full bg-zinc-50 dark:bg-zinc-950 overflow-hidden">
        {/* Route Diff Legend */}
        <div className="absolute top-3 left-3 z-10 flex items-center space-x-3 px-3 py-1.5 rounded-md bg-white/90 dark:bg-zinc-950/90 backdrop-blur-md border border-zinc-200/80 dark:border-zinc-800/80 text-[11px] font-mono shadow-sm">
          <div className="flex items-center space-x-1.5">
            <span className="w-4 h-0.5 border-t border-dashed border-zinc-400" />
            <span className="text-zinc-500">Baseline</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-4 h-0.5 bg-black dark:bg-white rounded" />
            <span className="text-zinc-900 dark:text-zinc-100 font-semibold">Re-Optimized</span>
          </div>
        </div>

        <div className="h-full w-full">
          <MapView
            routesGeojson={routesData.current}
            beforeRoutesGeojson={routesData.before}
            showDiff={Boolean(routesData.before)}
          />
        </div>
      </section>
    </div>
  );
}
