"""Controllers for meme-related endpoints."""

from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, List, Any

from ...services.meme_service import meme_service
from ..models.requests import CreateMemeRequest

# Create router
router = APIRouter()


@router.get("/")
async def get_memes() -> Dict[str, Any]:
    """
    Get all memes.
    
    Returns:
        Dictionary with items and count.
    """
    memes = await meme_service.get_memes()
    return {
        "items": memes,
        "count": len(memes)
    }


@router.get("/{meme_id}")
async def get_meme(meme_id: str) -> Dict[str, Any]:
    """
    Get a meme by ID.
    
    Args:
        meme_id: ID of the meme
        
    Returns:
        Meme data
        
    Raises:
        HTTPException: If meme not found
    """
    meme = await meme_service.get_meme(meme_id)
    if not meme:
        raise HTTPException(status_code=404, detail=f"Meme with ID {meme_id} not found")
    return meme


@router.post("/")
async def create_meme(request: CreateMemeRequest) -> Dict[str, Any]:
    """
    Create a new meme.
    
    Args:
        request: Create meme request
        
    Returns:
        Created meme data
    """
    return await meme_service.create_meme(
        topic=request.topic,
        format_id=request.format
    ) 