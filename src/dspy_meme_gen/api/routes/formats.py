"""API routes for meme formats."""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from ...agents.format_generator import format_generator


router = APIRouter(
    prefix="/api/v1/formats",
    tags=["formats"],
)


@router.get("/", response_model=Dict[str, Any])
async def get_formats(topic: Optional[str] = None) -> Dict[str, Any]:
    """
    Get a list of available meme formats.
    
    Args:
        topic: Optional topic to suggest formats for.
        
    Returns:
        Dictionary containing a list of formats and total count.
    """
    if topic:
        # Get suggested formats for the given topic
        formats = await format_generator.suggest_formats(topic)
    else:
        # Get all available formats
        formats = await format_generator.get_formats()
    
    return {
        "items": formats,
        "total": len(formats)
    }


@router.get("/{format_id}", response_model=Dict[str, Any])
async def get_format(format_id: str) -> Dict[str, Any]:
    """
    Get a specific meme format by ID.
    
    Args:
        format_id: ID of the format to retrieve.
        
    Returns:
        Format data.
        
    Raises:
        HTTPException: If format is not found.
    """
    format_data = await format_generator.get_format(format_id)
    if not format_data:
        raise HTTPException(status_code=404, detail=f"Format with ID {format_id} not found")
    return format_data 