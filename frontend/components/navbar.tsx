"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import { Moon, Sun, ArrowUpRight } from "lucide-react";
import { useEffect, useState } from "react";

export function Navbar() {
  const pathname = usePathname();
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const navItems = [
    { name: "Planning", href: "/planning" },
    { name: "Live Simulation", href: "/simulation" },
    { name: "MLOps & Drift", href: "/mlops" },
  ];

  return (
    <header className="h-14 border-b border-zinc-200 bg-white px-6 dark:border-zinc-800 dark:bg-zinc-950 flex items-center justify-between shrink-0 select-none">
      <div className="flex items-center space-x-8">
        <Link href="/planning" className="flex items-center space-x-2.5 group">
          <div className="h-6 w-6 rounded-md bg-black dark:bg-white flex items-center justify-center transition-transform group-hover:scale-105">
            <span className="text-xs font-bold text-white dark:text-black">R</span>
          </div>
          <span className="font-semibold tracking-tight text-sm text-zinc-950 dark:text-white">
            RouteMind
          </span>
        </Link>

        <nav className="flex space-x-1">
          {navItems.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                  active
                    ? "bg-zinc-100 text-black dark:bg-zinc-800 dark:text-white"
                    : "text-zinc-500 hover:text-black dark:text-zinc-400 dark:hover:text-white"
                }`}
              >
                {item.name}
              </Link>
            );
          })}
        </nav>
      </div>

      <div className="flex items-center space-x-3">
        <div className="hidden sm:flex items-center space-x-2 rounded-full border border-zinc-200 dark:border-zinc-800 px-2.5 py-1 text-[11px] font-medium text-zinc-600 dark:text-zinc-400">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
          <span>Vadodara Hub: Operational</span>
        </div>

        {mounted && (
          <button
            onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
            className="h-8 w-8 rounded-md border border-zinc-200 dark:border-zinc-800 flex items-center justify-center text-zinc-600 dark:text-zinc-400 hover:border-black dark:hover:border-white transition-colors"
            aria-label="Toggle theme"
          >
            {resolvedTheme === "dark" ? (
              <Sun className="h-3.5 w-3.5" />
            ) : (
              <Moon className="h-3.5 w-3.5" />
            )}
          </button>
        )}

        <a
          href="https://github.com/notKaushaL/RouteMind"
          target="_blank"
          rel="noreferrer"
          className="hidden md:flex items-center space-x-1 text-xs font-medium text-zinc-500 hover:text-black dark:hover:text-white transition-colors"
        >
          <span>GitHub</span>
          <ArrowUpRight className="h-3 w-3" />
        </a>
      </div>
    </header>
  );
}
