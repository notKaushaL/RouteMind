"""
backend/app/api/mlops.py
========================
FastAPI router for TreeSHAP explainability and MLOps metrics.
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.mlops import (
    ExplainDelayRequest,
    ExplainDelayResponse,
    MLOpsMetricsResponse,
)
from backend.app.services.mlops_service import explain_stop_delay, get_mlops_benchmarks

log = logging.getLogger(__name__)

router = APIRouter(tags=["MLOps & Explainability"])


@router.post(
    "/explain/delay",
    response_model=ExplainDelayResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute TreeSHAP delay feature attributions for a delivery stop",
)
async def explain_delay(request: ExplainDelayRequest) -> ExplainDelayResponse:
    try:
        return explain_stop_delay(request)
    except Exception as e:
        log.error("Delay explainability attribution failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explainability computation failed: {str(e)}",
        )


@router.get(
    "/mlops/metrics",
    response_model=MLOpsMetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Phase 1-4 benchmark metrics and drift reports",
)
async def get_metrics() -> MLOpsMetricsResponse:
    try:
        return get_mlops_benchmarks()
    except Exception as e:
        log.error("Failed to retrieve MLOps metrics: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics retrieval failed: {str(e)}",
        )
