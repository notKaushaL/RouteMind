"""
backend/app/main.py
===================
RouteMind Decision-Intelligence & Operations Research Engine API.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.services.engine_registry import registry
from backend.app.api.dispatch import router as dispatch_router
from backend.app.api.simulation import router as simulation_router
from backend.app.api.mlops import router as mlops_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("routemind.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown events."""
    log.info("Starting up RouteMind Decision Engine...")
    # Pre-load LightGBM CQR, GATv2, and MaskablePPO in memory
    registry.load_engines()
    yield
    log.info("Shutting down RouteMind Decision Engine...")


app = FastAPI(
    title="RouteMind — Decision-Intelligence Engine",
    description="Hybrid Operations Research & Machine Learning Dispatch Control Tower",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for local Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes under /api/v1
app.include_router(dispatch_router, prefix="/api/v1")
app.include_router(simulation_router, prefix="/api/v1")
app.include_router(mlops_router, prefix="/api/v1")


@app.get("/", tags=["Health"])
async def root():
    return {
        "engine": "RouteMind",
        "status": "online",
        "version": "1.0.0",
        "city": "Vadodara, India",
        "docs_url": "/docs",
    }


@app.get("/health", tags=["Health"])
@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "models": {
            "cqr_lightgbm": registry.cqr_model is not None,
            "gatv2_sequencer": registry.gatv2_model is not None,
            "maskable_ppo": registry.rl_agent is not None,
        },
    }
