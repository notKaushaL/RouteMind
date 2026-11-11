"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import { Moon, Sun, ArrowUpRight } from "lucide-react";

export function Navbar() {
  const pathname = usePathname();
  const { resolvedTheme, setTheme } = useTheme();

  const navItems = [
    { name: "Planning", href: "/planning" },
    { name: "Simulation", href: "/simulation" },
    { name: "MLOps", href: "/mlops" },
  ];

  return (
    <header className="h-11 border-b border-zinc-200/80 dark:border-zinc-800/80 bg-white/80 dark:bg-zinc-950/80 backdrop-blur-md px-4 flex items-center justify-between shrink-0 select-none z-30">
      <div className="flex items-center space-x-6">
        <Link href="/planning" className="flex items-center space-x-2">
          <span className="h-5 w-5 rounded bg-black dark:bg-white text-white dark:text-black flex items-center justify-center text-[11px] font-bold tracking-tighter">
            R
          </span>
          <span className="font-semibold text-xs tracking-tight text-zinc-900 dark:text-zinc-100">
            RouteMind
          </span>
        </Link>

        {/* Segmented Control */}
        <nav className="flex items-center p-0.5 rounded-md bg-zinc-100 dark:bg-zinc-900 border border-zinc-200/60 dark:border-zinc-800/60">
          {navItems.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`px-2.5 py-1 text-[11px] font-medium rounded transition-all duration-150 ${
                  active
                    ? "bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-sm"
                    : "text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
                }`}
              >
                {item.name}
              </Link>
            );
          })}
        </nav>
      </div>

      <div className="flex items-center space-x-2.5">
        <div className="hidden sm:flex items-center space-x-1.5 px-2 py-0.5 rounded-full border border-zinc-200/60 dark:border-zinc-800/60 text-[10px] text-zinc-500 dark:text-zinc-400 font-mono">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
          <span>Vadodara Hub</span>
        </div>

        <button
          type="button"
          onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
          className="h-7 w-7 rounded border border-zinc-200/80 dark:border-zinc-800/80 flex items-center justify-center text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-zinc-50 dark:hover:bg-zinc-900 transition-colors"
          aria-label="Toggle theme"
        >
          <Sun className="h-3.5 w-3.5 hidden dark:block" />
          <Moon className="h-3.5 w-3.5 block dark:hidden" />
        </button>

        <a
          href="https://github.com/notKaushaL/RouteMind"
          target="_blank"
          rel="noreferrer"
          aria-label="GitHub Repository"
          className="h-7 w-7 rounded flex items-center justify-center text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors"
        >
          <ArrowUpRight className="h-3.5 w-3.5" />
        </a>
      </div>
    </header>
  );
}
