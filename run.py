"""
run.py
======
Root orchestration launcher for RouteMind.
Simultaneously runs the FastAPI Decision Engine (:8000) and Next.js Control Tower (:3000).

Usage:
    python run.py
"""
import os
import sys
import time
import signal
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"

# Ensure local Node.js is on PATH
node_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "nodejs"
if node_dir.exists():
    os.environ["PATH"] = str(node_dir) + os.pathsep + os.environ.get("PATH", "")

BANNER = r"""
========================================================================================
  ____             _       __  __ _           _ 
 |  _ \ ___  _   _| |_ ___|  \/  (_)_ __   __| |
 | |_) / _ \| | | | __/ _ \ |\/| | | '_ \ / _` |
 |  _ < (_) | |_| | ||  __/ |  | | | | | | (_| |
 |_| \_\___/ \__,_|\__\___|_|  |_|_|_| |_|\__,_|
   Hybrid AI & Operations Research Last-Mile Dispatch Engine
========================================================================================
  [Backend API]       http://127.0.0.1:8000
  [Interactive Docs]  http://127.0.0.1:8000/docs
  [Control Tower UI]  http://localhost:3000
  [Simulation Lab]    http://localhost:3000/simulation
  [MLOps Dashboard]   http://localhost:3000/mlops
========================================================================================
"""


def main():
    print(BANNER)
    procs = []

    try:
        # 1. Start FastAPI Backend
        print("[1/2] Starting FastAPI Decision-Intelligence Backend on port 8000...")
        backend_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "info",
        ]
        p_back = subprocess.Popen(backend_cmd, cwd=str(ROOT_DIR))
        procs.append(p_back)

        # 2. Start Next.js Frontend
        print("[2/2] Starting Next.js Control Tower Dashboard on port 3000...")
        npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
        frontend_cmd = [npm_cmd, "run", "dev"]
        p_front = subprocess.Popen(frontend_cmd, cwd=str(FRONTEND_DIR), env=os.environ.copy())
        procs.append(p_front)

        print("\n--> RouteMind is live! Press Ctrl+C to stop all services.\n")

        # Keep running
        while True:
            time.sleep(1)
            for p in procs:
                if p.poll() is not None:
                    print(f"Warning: A child process exited with code {p.returncode}")
                    return

    except KeyboardInterrupt:
        print("\nShutting down RouteMind services cleanly...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except Exception:
                    p.kill()
        print("All RouteMind services stopped.")


if __name__ == "__main__":
    main()
