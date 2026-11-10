"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  MapPin,
  Route,
  Zap,
  BarChart3,
  Cpu,
  ShieldCheck,
  Radio,
  Layers,
} from "lucide-react";

export function Sidebar() {
  const pathname = usePathname();

  const navItems = [
    {
      name: "Planning Studio",
      href: "/planning",
      icon: Route,
      badge: "GATv2 Warm-Start",
      description: "Morning routing & solver comparison",
    },
    {
      name: "Simulation Lab",
      href: "/simulation",
      icon: Zap,
      badge: "MaskablePPO",
      description: "Real-time disruption & route diffs",
    },
    {
      name: "MLOps Governance",
      href: "/mlops",
      icon: BarChart3,
      badge: "CQR + Evidently",
      description: "Ablation tables & TreeSHAP",
    },
  ];

  return (
    <aside className="w-72 bg-slate-900 border-r border-slate-800 flex flex-col shrink-0 h-screen sticky top-0 select-none">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20">
            <Cpu className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-1.5">
              <h1 className="font-bold text-white text-lg tracking-tight">RouteMind</h1>
              <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20">
                v1.0
              </span>
            </div>
            <p className="text-xs text-slate-400">Decision-Intelligence Hub</p>
          </div>
        </div>
      </div>

      {/* Target Deployment Geography */}
      <div className="mx-4 mt-4 px-3.5 py-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80 flex items-center justify-between">
        <div className="flex items-center space-x-2 text-xs text-slate-300">
          <MapPin className="w-3.5 h-3.5 text-rose-400 shrink-0" />
          <span>Vadodara Hub (H3 Res-8)</span>
        </div>
        <span className="inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          Tier-II Core
        </span>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 px-3 py-4 space-y-1.5 overflow-y-auto">
        <p className="px-3 text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
          OPERATIONAL MODULES
        </p>
        {navItems.map((item) => {
          const isActive = pathname === item.href || (item.href === "/planning" && pathname === "/");
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-start space-x-3 px-3.5 py-3 rounded-xl transition-all duration-150 ${
                isActive
                  ? "bg-sky-500/10 border border-sky-500/30 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border border-transparent"
              }`}
            >
              <Icon
                className={`w-5 h-5 mt-0.5 shrink-0 ${
                  isActive ? "text-sky-400" : "text-slate-400"
                }`}
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-sm leading-tight">{item.name}</span>
                  {item.badge && (
                    <span
                      className={`text-[9px] px-1.5 py-0.5 rounded font-mono font-medium ${
                        isActive
                          ? "bg-sky-500/20 text-sky-300"
                          : "bg-slate-800 text-slate-400"
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                </div>
                <p className="text-[11px] text-slate-400 mt-0.5 truncate">
                  {item.description}
                </p>
              </div>
            </Link>
          );
        })}
      </nav>

      {/* Engine Status Footer */}
      <div className="p-4 border-t border-slate-800/80 bg-slate-950/40 text-xs space-y-2">
        <div className="flex items-center justify-between text-slate-400">
          <span className="flex items-center space-x-1.5">
            <Radio className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
            <span>FastAPI Engine</span>
          </span>
          <span className="font-mono text-[11px] text-emerald-400">:8000</span>
        </div>
        <div className="flex items-center justify-between text-slate-400">
          <span className="flex items-center space-x-1.5">
            <Layers className="w-3.5 h-3.5 text-sky-400" />
            <span>OR-Tools & CQR</span>
          </span>
          <span className="text-[10px] text-slate-300 font-mono">GLS + P90</span>
        </div>
      </div>
    </aside>
  );
}
