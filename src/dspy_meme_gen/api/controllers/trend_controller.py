"""Controllers for trend-related endpoints."""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any

from ...services.meme_service import meme_service

# Create router
router = APIRouter()


@router.get("/")
async def get_trending_topics() -> Dict[str, Any]:
    """
    Get all trending topics.
    
    Returns:
        Dictionary with items and count.
    """
    trends = await meme_service.get_trending_topics()
    return {
        "items": trends,
        "count": len(trends)
    }


@router.get("/{trend_id}")
async def get_trending_topic(trend_id: str) -> Dict[str, Any]:
    """
    Get a trending topic by ID.
    
    Args:
        trend_id: ID of the trending topic
        
    Returns:
        Trending topic data
        
    Raises:
        HTTPException: If trending topic not found
    """
    trend = await meme_service.get_trending_topic(trend_id)
    if not trend:
        raise HTTPException(status_code=404, detail=f"Trending topic with ID {trend_id} not found")
    return trend 