"""Health check API endpoints."""

from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncEngine
from redis.asyncio import Redis

from dspy_meme_gen.health.checks import HealthCheck

router = APIRouter(prefix="/health", tags=["health"])

async def get_health_check(
    db: AsyncEngine = Depends(),
    redis: Redis = Depends(),
) -> HealthCheck:
    """Get HealthCheck instance.
    
    Args:
        db: Database engine.
        redis: Redis client.
        
    Returns:
        HealthCheck: Health check instance.
    """
    return HealthCheck(db, redis)

@router.get("/")
async def check_health(
    health_check: HealthCheck = Depends(get_health_check)
) -> Dict[str, str]:
    """Check overall health of the service.
    
    Args:
        health_check: Health check instance.
        
    Returns:
        Dict[str, str]: Health status details.
        
    Raises:
        HTTPException: If service is unhealthy.
    """
    is_healthy, details = await health_check.check_health()
    if not is_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=details
        )
    return details

@router.get("/liveness")
async def liveness() -> Dict[str, str]:
    """Liveness probe endpoint.
    
    Returns:
        Dict[str, str]: Status message.
    """
    return {"status": "alive"}

@router.get("/readiness")
async def readiness(
    health_check: HealthCheck = Depends(get_health_check)
) -> Dict[str, str]:
    """Readiness probe endpoint.
    
    Args:
        health_check: Health check instance.
        
    Returns:
        Dict[str, str]: Readiness details.
        
    Raises:
        HTTPException: If service is not ready.
    """
    is_ready, details = await health_check.check_readiness()
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=details
        )
    return details 