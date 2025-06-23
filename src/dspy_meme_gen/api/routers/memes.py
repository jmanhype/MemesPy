"""Router for meme-related endpoints."""

import uuid
import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from typing import List, Optional

from ...config.config import settings
from ...models.schemas.memes import (
    MemeGenerationRequest,
    MemeResponse,
    MemeListResponse,
)
from ...models.database.memes import MemeDB
from ...dspy_modules.meme_predictor import MemePredictor
from ...dspy_modules.image_generator import ImageGenerator
from ...database.connection import get_session
from ..dependencies import get_cache

router = APIRouter()

# Initialize our DSPy modules
meme_predictor = MemePredictor()
image_generator = ImageGenerator(provider=settings.image_provider)

@router.post("/", response_model=MemeResponse, status_code=status.HTTP_201_CREATED)
async def generate_meme(
    request: MemeGenerationRequest,
    db: Session = Depends(get_session),
    cache = Depends(get_cache)
) -> MemeResponse:
    """
    Generate a new meme based on the provided request.
    
    Args:
        request: The meme generation request containing topic and format
        db: Database session
        cache: Cache connection
        
    Returns:
        The generated meme with text and image
    
    Raises:
        HTTPException: If meme generation fails
    """
    # Check cache first
    cache_key = f"meme:{request.topic}:{request.format}"
    cached_meme = await cache.get(cache_key)
    if cached_meme:
        return MemeResponse(**json.loads(cached_meme))
    
    try:
        # Generate meme content using DSPy
        meme_text, image_prompt = meme_predictor.forward(
            topic=request.topic,
            format=request.format
        )
        
        # Check if generation failed (fallback might return None)
        if meme_text is None or image_prompt is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Meme generation failed (predictor returned None)"
            )
        
        # Generate image using the configured provider
        image_url = image_generator.generate(
            prompt=image_prompt,
            meme_text=meme_text  # Pass meme text for providers that support text overlay
        )
        
        if not image_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate meme image"
            )
        
        # Create meme response
        meme = MemeResponse(
            id=str(uuid.uuid4()),
            topic=request.topic,
            format=request.format,
            text=meme_text,
            image_url=image_url,
            created_at=datetime.utcnow().isoformat()
        )
        
        # Store in cache
        await cache.set(cache_key, json.dumps(meme.model_dump()), ex=settings.cache_ttl)
        
        # Store in database
        db_meme = MemeDB(
            id=meme.id,
            topic=meme.topic,
            format=meme.format,
            text=meme.text,
            image_url=meme.image_url,
            created_at=datetime.fromisoformat(meme.created_at),
            score=0.8  # Default score for now
        )
        db.add(db_meme)
        db.commit()
        
        return meme
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate meme: {str(e)}"
    )

@router.get("/", response_model=MemeListResponse)
async def list_memes(
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_session),
    cache = Depends(get_cache)
) -> MemeListResponse:
    """
    List all generated memes.
    
    Args:
        limit: Maximum number of memes to return
        offset: Offset for pagination
        db: Database session
        cache: Cache connection
        
    Returns:
        List of memes with pagination info
    """
    # Check cache first
    cache_key = f"memes:list:{limit}:{offset}"
    cached_list = await cache.get(cache_key)
    if cached_list:
        return MemeListResponse(**json.loads(cached_list))
    
    try:
        # Query database for memes
        memes = db.execute(
            select(MemeDB).order_by(MemeDB.created_at.desc()).offset(offset).limit(limit)
        ).scalars().all()
        
        # Get total count
        total = db.query(MemeDB).count()
        
        # Convert to response models
        meme_responses = [
            MemeResponse(
                id=meme.id,
                topic=meme.topic,
                format=meme.format,
                text=meme.text,
                image_url=meme.image_url,
                created_at=meme.created_at.isoformat()
            ) for meme in memes
        ]
        
        response = MemeListResponse(
            items=meme_responses,
            total=total,
        limit=limit,
        offset=offset
    )
        
        # Store in cache
        await cache.set(cache_key, json.dumps(response.model_dump()), ex=settings.cache_ttl)
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list memes: {str(e)}"
        )

@router.get("/{meme_id}", response_model=MemeResponse)
async def get_meme(
    meme_id: str,
    db: Session = Depends(get_session),
    cache = Depends(get_cache)
) -> MemeResponse:
    """
    Get a specific meme by ID.
    
    Args:
        meme_id: The ID of the meme to retrieve
        db: Database session
        cache: Cache connection
        
    Returns:
        The requested meme
        
    Raises:
        HTTPException: If meme is not found
    """
    # Check cache first
    cache_key = f"meme:{meme_id}"
    cached_meme = await cache.get(cache_key)
    if cached_meme:
        return MemeResponse(**json.loads(cached_meme))
    
    try:
        # Query database for meme
        meme = db.query(MemeDB).filter(MemeDB.id == meme_id).first()
        
        if not meme:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Meme not found: {meme_id}"
            )
        
        # Convert to response model
        response = MemeResponse(
            id=meme.id,
            topic=meme.topic,
            format=meme.format,
            text=meme.text,
            image_url=meme.image_url,
            created_at=meme.created_at.isoformat()
        )
        
        # Store in cache
        await cache.set(cache_key, json.dumps(response.model_dump()), ex=settings.cache_ttl)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get meme: {str(e)}"
    ) 