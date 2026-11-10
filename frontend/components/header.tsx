"use client";

import { Clock, Truck, ShieldAlert, Sparkles, CheckCircle2 } from "lucide-react";

interface HeaderProps {
  simulationTime?: string;
  activeVehicles?: number;
  totalVehicles?: number;
  onTimePct?: number;
  costJ?: number;
}

export function Header({
  simulationTime = "09:00",
  activeVehicles = 3,
  totalVehicles = 3,
  onTimePct = 100.0,
  costJ,
}: HeaderProps) {
  return (
    <header className="h-16 bg-slate-900/90 backdrop-blur border-b border-slate-800 px-6 flex items-center justify-between shrink-0 sticky top-0 z-20">
      {/* Simulation Clock & Shift */}
      <div className="flex items-center space-x-6">
        <div className="flex items-center space-x-2.5 bg-slate-800/80 px-3.5 py-1.5 rounded-lg border border-slate-700/60 shadow-inner">
          <Clock className="w-4 h-4 text-sky-400 shrink-0" />
          <span className="text-xs text-slate-400 font-medium">Shift Clock:</span>
          <span className="font-mono font-bold text-sm text-white tracking-wider">
            {simulationTime}
          </span>
          <span className="text-[10px] text-slate-400 font-mono">/ 17:00</span>
        </div>

        {/* Fleet Operational Readiness */}
        <div className="hidden sm:flex items-center space-x-2 text-xs">
          <Truck className="w-4 h-4 text-emerald-400" />
          <span className="text-slate-400">Active Fleet:</span>
          <span className="font-semibold text-slate-200">
            {activeVehicles}/{totalVehicles} Operational
          </span>
        </div>
      </div>

      {/* SLA & Objective Cost Stat Badges */}
      <div className="flex items-center space-x-4">
        {costJ !== undefined && (
          <div className="hidden md:flex items-center space-x-1.5 text-xs bg-slate-800/50 px-3 py-1 rounded-md border border-slate-700/50">
            <span className="text-slate-400">Composite Cost J:</span>
            <span className="font-mono font-bold text-sky-400">{costJ.toFixed(1)}</span>
          </div>
        )}

        <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs font-semibold">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          <span>SLA: {onTimePct.toFixed(1)}%</span>
        </div>

        <div className="hidden lg:flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-sky-500/10 border border-sky-500/30 text-sky-300 text-xs font-medium">
          <Sparkles className="w-3.5 h-3.5 text-sky-400" />
          <span>RL Meta-Active</span>
        </div>
      </div>
    </header>
  );
}
