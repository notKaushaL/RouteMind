"""
backend/app/api/simulation.py
=============================
FastAPI router for interactive disruption simulation and MaskablePPO meta-controller actions.
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.simulation import (
    SimulationInitRequest,
    SimulationInitResponse,
    DisruptionInjectRequest,
    DisruptionInjectResponse,
)
from backend.app.services.simulation_service import simulation_lab

log = logging.getLogger(__name__)

router = APIRouter(prefix="/simulation", tags=["Simulation Lab"])


@router.post(
    "/init",
    response_model=SimulationInitResponse,
    status_code=status.HTTP_200_OK,
    summary="Initialize active 8-hour dispatch simulation",
)
async def initialize_simulation(request: SimulationInitRequest) -> SimulationInitResponse:
    try:
        return simulation_lab.init_simulation(day_seed=request.day_seed, n_stops=request.n_stops)
    except Exception as e:
        log.error("Simulation initialization failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation initialization failed: {str(e)}",
        )


@router.post(
    "/inject-disruption",
    response_model=DisruptionInjectResponse,
    status_code=status.HTTP_200_OK,
    summary="Inject operational shock and trigger MaskablePPO action",
)
async def inject_disruption(request: DisruptionInjectRequest) -> DisruptionInjectResponse:
    try:
        return simulation_lab.inject_disruption(request)
    except Exception as e:
        log.error("Disruption injection failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Disruption processing failed: {str(e)}",
        )
