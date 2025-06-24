"""Health check router."""

from fastapi import APIRouter, Depends
from typing import Dict, Any

from ...config.config import settings

router = APIRouter()


@router.get("/", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        Dict containing health status information
    """
    return {
        "status": "healthy",
        "version": settings.app_version,
        "env": settings.app_env,
        "dspy": {
            "model": settings.dspy_model,
            "openai_configured": settings.openai_api_key is not None,
        },
    }


@router.get("/liveness", response_model=Dict[str, Any])
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness probe endpoint.

    Returns:
        Dict containing liveness status
    """
    return {"status": "alive"}


@router.get("/readiness", response_model=Dict[str, Any])
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness probe endpoint.

    Returns:
        Dict containing readiness status
    """
    return {"status": "ready"}
