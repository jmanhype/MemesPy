"""Controllers for format-related endpoints."""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any

from ...services.meme_service import meme_service

# Create router
router = APIRouter()


@router.get("/")
async def get_formats() -> Dict[str, Any]:
    """
    Get all meme formats.
    
    Returns:
        Dictionary with items and count.
    """
    formats = await meme_service.get_formats()
    return {
        "items": formats,
        "count": len(formats)
    }


@router.get("/{format_id}")
async def get_format(format_id: str) -> Dict[str, Any]:
    """
    Get a meme format by ID.
    
    Args:
        format_id: ID of the meme format
        
    Returns:
        Format data
        
    Raises:
        HTTPException: If format not found
    """
    format_data = await meme_service.get_format(format_id)
    if not format_data:
        raise HTTPException(status_code=404, detail=f"Format with ID {format_id} not found")
    return format_data 