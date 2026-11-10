"""
backend/app/api/dispatch.py
===========================
FastAPI router for static morning dispatch planning and route optimization.
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.dispatch import PlanRequest, PlanResponse
from backend.app.services.dispatch_service import generate_dispatch_plan

log = logging.getLogger(__name__)

router = APIRouter(prefix="/dispatch", tags=["Dispatch Planning"])


@router.post(
    "/plan",
    response_model=PlanResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate vehicle dispatch route plan",
    description="Optimizes last-mile delivery routes across Vadodara using OR-Tools, GATv2 warm-start, and CQR bounds.",
)
async def create_dispatch_plan(request: PlanRequest) -> PlanResponse:
    try:
        return generate_dispatch_plan(request)
    except Exception as e:
        log.error("Dispatch plan generation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Route planning failed: {str(e)}",
        )
