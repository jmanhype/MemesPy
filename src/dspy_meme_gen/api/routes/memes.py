"""API routes for meme-related endpoints."""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
import logging

from ...services.meme_service import meme_service
from ...models.meme import CreateMemeRequest, MemeData


router = APIRouter(
    prefix="/api/v1/memes",
    tags=["memes"],
)

# Set up logging
logger = logging.getLogger(__name__)


@router.get("/", response_model=Dict[str, Any])
async def get_memes() -> Dict[str, Any]:
    """
    Get a list of all memes.
    
    Returns:
        Dictionary containing a list of memes and total count.
    """
    try:
        memes = await meme_service.get_memes()
        return {
            "items": memes,
            "total": len(memes)
        }
    except Exception as e:
        logger.error(f"Error retrieving memes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memes: {str(e)}"
        )


@router.get("/{meme_id}", response_model=Dict[str, Any])
async def get_meme(meme_id: str) -> Dict[str, Any]:
    """
    Get a specific meme by ID.
    
    Args:
        meme_id: ID of the meme to retrieve.
        
    Returns:
        Meme data.
        
    Raises:
        HTTPException: If meme is not found.
    """
    try:
        meme = await meme_service.get_meme(meme_id)
        if not meme:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Meme with ID {meme_id} not found"
            )
        return meme
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving meme {meme_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve meme: {str(e)}"
        )


@router.post("/", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_meme(request: CreateMemeRequest) -> Dict[str, Any]:
    """
    Create a new meme using DSPy.
    
    Args:
        request: Meme creation request data.
        
    Returns:
        Created meme data.
    """
    try:
        # Log the request for debugging
        logger.info(f"Creating meme with topic: {request.topic}, format: {request.format}")
        
        # Generate the meme using DSPy
        meme = await meme_service.create_meme(
            topic=request.topic,
            format=request.format,
            parameters=request.parameters
        )
        
        # Check if there was an error in generation (fallback case)
        if "error" in meme:
            logger.warning(f"Meme created with fallback due to error: {meme.get('error')}")
            return JSONResponse(
                status_code=status.HTTP_207_MULTI_STATUS,
                content=meme
            )
        
        return meme
    except Exception as e:
        logger.error(f"Error creating meme: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating meme: {str(e)}"
        ) 