"""API routes for trending meme topics."""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from ...agents.trend_analyzer import trend_analyzer


router = APIRouter(
    prefix="/api/v1/trends",
    tags=["trends"],
)


@router.get("/", response_model=Dict[str, Any])
async def get_trending_topics(query: Optional[str] = None) -> Dict[str, Any]:
    """
    Get a list of trending topics for meme generation.
    
    Args:
        query: Optional query to filter trending topics.
        
    Returns:
        Dictionary containing a list of trending topics and total count.
    """
    topics = await trend_analyzer.get_trending_topics(query=query)
    return {
        "items": topics,
        "total": len(topics)
    }


@router.get("/{trend_id}", response_model=Dict[str, Any])
async def get_trending_topic(trend_id: str) -> Dict[str, Any]:
    """
    Get a specific trending topic by ID.
    
    Args:
        trend_id: ID of the trending topic to retrieve.
        
    Returns:
        Trending topic data.
        
    Raises:
        HTTPException: If trending topic is not found.
    """
    topic = await trend_analyzer.get_trending_topic(trend_id)
    if not topic:
        raise HTTPException(status_code=404, detail=f"Trending topic with ID {trend_id} not found")
    return topic 