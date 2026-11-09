# RouteMind 🚚🧠

**AI-Powered Last-Mile Delivery Dispatch | Hybrid Neural-OR Optimization | Dynamic Fleet Simulation**

> *Modeled on a regional parcel hub in Vadodara, Gujarat, India — combining Operations Research, Graph Neural Networks, Conformal Prediction, and Reinforcement Learning.*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RouteMind System Architecture                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Data & Geospatial Layer                                                    │
│  OSMnx Graph → OSRM Matrix → Uber H3 Hexagons → DuckDB Feature Store      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ML Intelligence Core                                                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │ CQR ETA Model    │  │ GATv2 Sequencer  │  │ MaskablePPO Dispatcher   │  │
│  │ LightGBM + MAPIE │  │ PyTorch Geometric │  │ Gymnasium + sb3-contrib  │  │
│  │ [P10, P50, P90]  │  │ Arc-Cost Shaping │  │ Rolling-Horizon LNS      │  │
│  └────────┬─────────┘  └────────┬─────────┘  └───────────┬──────────────┘  │
│           │                     │                         │                  │
├───────────▼─────────────────────▼─────────────────────────▼─────────────────┤
│  Hybrid OR-Tools VRPTW Solver                                                │
│  P90 risk-buffered TW → GATv2 warm-start → GLS → Stability penalty w5      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Decision-Intelligence Control Tower (Next.js + MapLibre GL)                │
│  Planning Studio | Simulation Lab | MLOps & Drift Monitor                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Objective (Risk-Aware Multi-Objective VRPTW)

$$
\min J = w_1 \sum_{v \in V} \sum_{(i,j) \in R_v} c_v \cdot d_{ij}
       + w_2 \sum_{i \in N} \max(0, \hat{A}_i(\alpha_i) - l_i)
       + w_3 \sum_{v \in V} \max(0, T_v - S_v^{\max})
       + w_4 \sum_{i \in U} p_i
       + w_5 \sum_{v \in V} D_{\text{edit}}(R_v^{\text{new}}, R_v^{\text{prev}})
       + w_6 \cdot \text{Var}(\{L_v\}_{v \in V})
$$

Where $\hat{A}_i(\alpha_i)$ uses **P90** quantile for SLA-critical stops and **P50** for standard stops.

**Route Stability Penalty:**
$$
D_{\text{edit}} = \lambda_{\text{kt}} \cdot \tau_{\text{norm}}(R^{\text{new}}, R^{\text{prev}}) + \lambda_{\text{ra}} \cdot \text{reassignment\_count}
$$

---

## Ablation Results

### Experiment 1: Static Dispatch (Vadodara, 30 simulated workdays)

| Algorithm | Total Cost (m) | Unassigned (%) | Solve Time (s) |
|-----------|---------------|----------------|----------------|
| NN Greedy | — | — | ~0.001 |
| Cheapest Insertion | — | — | ~0.003 |
| OR-Tools GLS (10s) | — | — | 10.0 |
| PyVRP HGS (60s) | — | — | 60.0 |

*Run `scripts/benchmark_phase1.py` to populate with real metrics.*

### Experiment 2: Uncertainty Routing (Phase 2)

| Method | On-time SLA % | P90 Lateness (min) | Empirical Coverage |
|--------|--------------|--------------------|--------------------|
| OSRM Static | — | — | — |
| LightGBM P50 | — | — | — |
| **LightGBM CQR P90** | — | — | ≥90% |

### Experiment 3: Neural-Guided VRP (Phase 3)

| Method | Primal Integral | Obj Gap @10s | Zone-Cohesion |
|--------|----------------|--------------|---------------|
| Cold-Start OR-Tools | — | — | — |
| GATv2 Arc-Cost Shaping | — | — | — |
| **GATv2 Warm-Start + GLS** | — | — | — |

### Experiment 4: Dynamic Re-Planning (Phase 4)

| Method | SLA Recovery % | Route Churn (τ) | Compute Time |
|--------|---------------|-----------------|--------------|
| Full Re-Solve | — | — | — |
| Greedy Insertion | — | — | — |
| **MaskablePPO + w5** | — | — | — |

---

## Project Structure

```
routemind/
├── ml/
│   ├── eta_cqr/          # LightGBM + MAPIE Conformal Quantile Regression
│   ├── gnn_sequencer/    # PyTorch Geometric GATv2 on Amazon Last-Mile
│   ├── rl_dispatcher/    # Gymnasium + MaskablePPO meta-controller
│   ├── feature_store/    # DuckDB offline + Redis online feature sync
│   └── evaluation/       # Ablation scripts & Evidently drift reports
├── solver/
│   ├── baselines/        # NN Greedy & Cheapest Insertion
│   ├── ortools_engine/   # Full VRPTW solver with CQR + warm-start
│   ├── pyvrp_benchmark/  # PyVRP HGS comparison
│   └── stability/        # Kendall-Tau & reassignment penalties
├── simulator/
│   ├── vadodara_network/ # OSMnx graph + H3 indexing + OSRM matrix
│   ├── demand_generator/ # Synthetic parcel workdays
│   └── disruptions/      # Traffic surge, breakdown, urgent order events
├── backend/              # FastAPI + Celery async solver workers
├── frontend/             # Next.js Control Tower + Simulation Lab
├── configs/              # Pydantic settings
├── scripts/              # Benchmark & experiment scripts
├── dvc.yaml              # End-to-end ML pipeline DAG
└── docker-compose.yml    # Full stack: OSRM, PostGIS, Redis, MLflow
```

---

## 5-Phase Roadmap

| Phase | Weeks | Focus | Deliverable |
|-------|-------|-------|-------------|
| **1** | 1–2 | Geospatial Engine & Solver Benchmarks | Reproducible ablation on 30 Vadodara workdays |
| **2** | 3–4 | CQR ETA & Dwell Intelligence + MLOps | CQR P90 → SLA breach reduction vs. static |
| **3** | 5–6 | GATv2 Neural-Guided Routing | Convergence curves under 10s budget |
| **4** | 7–8 | MaskablePPO Disruption Dispatcher | Benchmark across 100 disruption episodes |
| **5** | 9–10 | Control Tower UI & Packaging | Live Before/After diff demo |

---

## Setup

```bash
# 1. Install Phase 1 dependencies
pip install -e ".[phase1]"

# 2. Build Vadodara road network (downloads from OpenStreetMap)
python simulator/vadodara_network/build_network.py

# 3. Generate synthetic workdays
python simulator/demand_generator/generate_workday.py --days 30

# 4. Run Phase 1 benchmark
python scripts/benchmark_phase1.py --mode vadodara --days 5

# 5. Full pipeline via DVC
dvc repro benchmark_phase1
```

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **OR Engine** | Google OR-Tools, PyVRP (HGS), OSMnx, OSRM, Uber H3 |
| **ML** | LightGBM, CatBoost, MAPIE (CQR), PyTorch Geometric (GATv2), SHAP |
| **RL** | Gymnasium, Stable-Baselines3, sb3-contrib (MaskablePPO) |
| **MLOps** | DVC, MLflow, Evidently, DuckDB, Redis |
| **Backend** | FastAPI, Celery, PostgreSQL/PostGIS |
| **Frontend** | Next.js, TypeScript, MapLibre GL JS, Recharts, shadcn/ui |

---

## Datasets

| Dataset | Scale | License | Usage |
|---------|-------|---------|-------|
| **Amazon Last-Mile Routing** | 9,184 routes | CC BY-NC 4.0 | GATv2 training |
| **CVRPLib (Solomon/Homberger)** | 56+ instances | Public | Solver benchmarking |
| **Vadodara OSMnx Simulator** | Synthetic | Generated | End-to-end evaluation |

---

## Resume Bullets (Fill in [X], [Y], [Z], [W] after running experiments)

> **RouteMind** — Hybrid AI & Operations Research Last-Mile Dispatch Engine  
> *(Python · PyTorch Geometric · LightGBM · Google OR-Tools · Gymnasium · FastAPI · PostGIS · MLflow · DVC)*

- Engineered a Risk-Aware VRPTW platform for a 25-vehicle heterogeneous fleet over Vadodara's OSM/OSRM network, combining **Conformalized Quantile Regression** (LightGBM + MAPIE) and **Uber H3** spatial features to predict calibrated [P10, P90] intervals, cutting simulated SLA violations by **[X]%** over static routing baselines.
- Trained a **GATv2** (PyTorch Geometric) on 9,184 Amazon Last-Mile routes to learn human driver sequencing priors; integrated via warm-started Guided Local Search and arc-cost shaping, reducing convergence time by **[Y]%** under a 10-second dispatch budget.
- Built a discrete-event **Gymnasium** fleet simulator and trained an action-masked **MaskablePPO** meta-controller with a Kendall-Tau route-stability penalty, recovering **[Z]%** of disrupted deliveries while reducing driver route churn by **[W]%**.
- Architected a reproducible **MLOps** pipeline with DVC, DuckDB/Redis offline-to-online feature serving, MLflow Champion/Challenger registry, and Evidently drift detection.
