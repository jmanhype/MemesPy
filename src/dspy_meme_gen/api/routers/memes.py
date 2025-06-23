"""Router for meme-related endpoints with async event sourcing."""

import uuid
import json
import time
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy import select
from sqlalchemy.orm import Session
from typing import List, Optional

from ...config.config import settings
from ...models.schemas.memes import (
    MemeGenerationRequest,
    MemeResponse,
    MemeListResponse,
)
# Legacy imports for backward compatibility
from ...models.database.memes import MemeDB
from ...models.database.metadata import MemeMetadata, GenerationLog
from ...dspy_modules.meme_predictor import MemePredictor
from ...dspy_modules.image_generator import ImageGenerator
from ...services.metadata_collector import MetadataCollector
from ...database.connection import get_session
from ..dependencies import get_cache

# New async event sourcing imports
# Temporarily disable async service to avoid CQRS import issues
# from ...services.meme_service import async_meme_service
async_meme_service = None
# Temporarily disable system setup
# from ...system_setup import get_system, health_check
get_system = lambda: None
health_check = lambda: {"status": "ok"}

router = APIRouter()

# Initialize our DSPy modules (legacy)
meme_predictor = MemePredictor()
image_generator = ImageGenerator(provider=settings.image_provider)

@router.post("/", response_model=MemeResponse, status_code=status.HTTP_201_CREATED)
async def generate_meme(
    request: MemeGenerationRequest,
    http_request: Request,
    use_legacy: bool = False,
    db: Session = Depends(get_session),
    cache = Depends(get_cache)
) -> MemeResponse:
    """
    Generate a new meme using async event sourcing or legacy system.
    
    Args:
        request: The meme generation request containing topic and format
        http_request: The HTTP request object for metadata
        use_legacy: If True, use legacy generation system
        db: Database session
        cache: Cache connection
        
    Returns:
        The generated meme with text and image
    
    Raises:
        HTTPException: If meme generation fails
    """
    # Determine which system to use
    if not use_legacy:
        try:
            # Use new async event sourcing system
            user_id = http_request.headers.get("x-user-id", "anonymous")
            
            result = await async_meme_service.create_meme(
                topic=request.topic,
                format_id=request.format,
                style=getattr(request, 'style', None),
                parameters={
                    'client_ip': http_request.client.host if http_request.client else "unknown",
                    'user_agent': http_request.headers.get("user-agent", "unknown"),
                    'api_version': settings.app_version
                },
                user_id=user_id
            )
            
            # Convert to legacy response format
            return MemeResponse(
                id=result["meme_id"],
                topic=request.topic,
                format=request.format,
                text="Generating...",  # Text will be available once generation completes
                image_url="",  # Image URL will be available once generation completes
                created_at=datetime.utcnow().isoformat(),
                status=result["status"],
                request_id=result["request_id"],
                message=result["message"]
            )
            
        except Exception as e:
            # Fall back to legacy system on error
            use_legacy = True
    
    if use_legacy:
        # Legacy generation system (existing code)
        # Initialize metadata collector
        metadata_collector = MetadataCollector()
        generation_id = metadata_collector.start_generation(
            topic=request.topic,
            format=request.format,
            request_id=str(uuid.uuid4()),
            client_ip=http_request.client.host if http_request.client else "unknown",
            user_agent=http_request.headers.get("user-agent", "unknown")
        )
        
        # Check cache first
        cache_key = f"meme:{request.topic}:{request.format}"
        cached_meme = await cache.get(cache_key)
        if cached_meme:
            metadata_collector.metadata['cache_hit'] = True
            return MemeResponse(**json.loads(cached_meme))
        
        metadata_collector.metadata['cache_hit'] = False
        
        try:
            # Track DSPy generation
            dspy_start = time.time()
            
            # Generate meme content using DSPy
            meme_text, image_prompt = meme_predictor.forward(
                topic=request.topic,
                format=request.format
            )
            
            dspy_duration = (time.time() - dspy_start) * 1000
            
            # Track DSPy metadata
            metadata_collector.track_dspy_generation(
                predictor_class=meme_predictor.__class__.__name__,
                inputs={"topic": request.topic, "format": request.format},
                outputs={"text": meme_text, "image_prompt": image_prompt},
                duration_ms=dspy_duration,
                model_info={
                    "model": settings.dspy_model,
                    "temperature": getattr(settings, 'dspy_temperature', 0.7)
                }
            )
            
            # Check if generation failed (fallback might return None)
            if meme_text is None or image_prompt is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Meme generation failed (predictor returned None)"
                )
            
            # Set metadata collector on image generator
            image_generator.metadata_collector = metadata_collector
            
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
            
            # Calculate quality score based on metadata
            score = _calculate_quality_score(metadata_collector.metadata)
            
            # Create meme response
            meme = MemeResponse(
                id=str(uuid.uuid4()),
                topic=request.topic,
                format=request.format,
                text=meme_text,
                image_url=image_url,
                created_at=datetime.utcnow().isoformat()
            )
            
            # Finalize metadata
            final_metadata = metadata_collector.finalize(
                score=score,
                image_url=image_url,
                meme_text=meme_text,
                success=True
            )
            
            # Store in cache
            await cache.set(cache_key, json.dumps(meme.model_dump()), ex=settings.cache_ttl)
            
            # Store in database with enhanced metadata
            db_meme = MemeMetadata(
                id=meme.id,
                topic=meme.topic,
                format=meme.format,
                text=meme.text,
                image_url=meme.image_url,
                created_at=datetime.fromisoformat(meme.created_at),
                score=score,
                generation_metadata=final_metadata.get('generation_metadata', {}),
                image_metadata=final_metadata.get('image_metadata', {}),
                dspy_metadata=final_metadata.get('dspy_metadata', {}),
                technical_metadata={
                    'api_version': settings.app_version,
                    'client_ip': http_request.client.host if http_request.client else "unknown",
                    'user_agent': http_request.headers.get("user-agent", "unknown"),
                    'request_id': generation_id,
                    'server_region': settings.app_env,
                    'cache_hit': False,
                    'response_time_ms': final_metadata.get('total_duration_ms', 0),
                    'legacy_generation': True
                },
                cost_metadata=final_metadata.get('cost_metadata', {})
            )
            db.add(db_meme)
            
            # Also log the generation
            generation_log = GenerationLog(
                meme_id=meme.id,
                request_type='full_generation_legacy',
                request_payload={
                    'topic': request.topic,
                    'format': request.format
                },
                response_status='success',
                response_payload={'meme_id': meme.id},
                response_time_ms=int(final_metadata.get('total_duration_ms', 0)),
                model_name=settings.dspy_model,
                model_provider='openai'
            )
            db.add(generation_log)
            
            db.commit()
            
            return meme
            
        except Exception as e:
            db.rollback()
            
            # Finalize metadata with error
            final_metadata = metadata_collector.finalize(
                score=0.0,
                image_url="",
                meme_text="",
                success=False,
                error=str(e)
            )
            
            # Log the failed generation
            try:
                generation_log = GenerationLog(
                    meme_id=None,
                    request_type='full_generation_legacy',
                    request_payload={
                        'topic': request.topic,
                        'format': request.format
                    },
                    response_status='error',
                    response_time_ms=int(final_metadata.get('total_duration_ms', 0)),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    model_name=settings.dspy_model,
                    model_provider='openai'
                )
                db.add(generation_log)
                db.commit()
            except:
                pass
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate meme: {str(e)}"
            )


def _calculate_quality_score(metadata: dict) -> float:
    """Calculate quality score based on generation metadata."""
    score = 0.5  # Base score
    
    # Efficiency bonus
    if metadata.get('generation_metadata'):
        efficiency = metadata['generation_metadata'].get('efficiency_score', 0)
        score += efficiency * 0.2
    
    # Success bonus
    if metadata.get('success', False):
        score += 0.2
    
    # No retry bonus
    if metadata.get('generation_metadata', {}).get('retry_count', 0) == 0:
        score += 0.1
    
    return min(1.0, score)

@router.get("/", response_model=MemeListResponse)
async def list_memes(
    limit: int = 10,
    offset: int = 0,
    status: Optional[str] = None,
    topic: Optional[str] = None,
    format: Optional[str] = None,
    use_legacy: bool = False,
    db: Session = Depends(get_session),
    cache = Depends(get_cache)
) -> MemeListResponse:
    """
    List generated memes using async event sourcing or legacy system.
    
    Args:
        limit: Maximum number of memes to return
        offset: Offset for pagination
        status: Filter by status (for event sourcing)
        topic: Filter by topic
        format: Filter by format
        use_legacy: If True, use legacy database
        db: Database session
        cache: Cache connection
        
    Returns:
        List of memes with pagination info
    """
    # Use async event sourcing projections if not legacy
    if not use_legacy:
        try:
            projections = await async_meme_service.get_memes(
                status=status,
                topic=topic,
                format=format,
                limit=limit,
                offset=offset
            )
            
            # Convert to legacy response format
            meme_responses = [
                MemeResponse(
                    id=meme["meme_id"],
                    topic=meme["topic"],
                    format=meme["format"],
                    text=meme.get("text", ""),
                    image_url=meme.get("image_url", ""),
                    created_at=meme["created_at"],
                    status=meme.get("status", "unknown"),
                    request_id=meme.get("request_id"),
                    **{k: v for k, v in meme.items() if k not in ["meme_id", "topic", "format", "text", "image_url", "created_at"]}
                ) for meme in projections
            ]
            
            return MemeListResponse(
                items=meme_responses,
                total=len(meme_responses),  # Note: actual count would need additional query
                limit=limit,
                offset=offset
            )
            
        except Exception as e:
            # Fall back to legacy on error
            use_legacy = True
    
    if use_legacy:
        # Check cache first
        cache_key = f"memes:list:{limit}:{offset}:{topic}:{format}"
        cached_list = await cache.get(cache_key)
        if cached_list:
            return MemeListResponse(**json.loads(cached_list))
        
        try:
            # Query database for memes
            query = select(MemeDB).order_by(MemeDB.created_at.desc())
            
            # Apply filters
            if topic:
                query = query.where(MemeDB.topic.contains(topic))
            if format:
                query = query.where(MemeDB.format == format)
            
            memes = db.execute(query.offset(offset).limit(limit)).scalars().all()
            
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


# New async event sourcing endpoints
@router.get("/generation/{request_id}/status")
async def get_generation_status(request_id: str):
    """Get the status of a meme generation request."""
    try:
        status = await async_meme_service.get_generation_status(request_id)
        if not status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Generation request not found: {request_id}"
            )
        return status
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get generation status: {str(e)}"
        )


@router.post("/{meme_id}/share")
async def share_meme(
    meme_id: str,
    platform: str,
    method: str = "api",
    http_request: Request = None
):
    """Record a meme share event."""
    try:
        user_id = http_request.headers.get("x-user-id") if http_request else None
        result = await async_meme_service.share_meme(meme_id, platform, method, user_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record share: {str(e)}"
        )


@router.post("/{meme_id}/score")
async def score_meme(
    meme_id: str,
    http_request: Request = None
):
    """Score an existing meme."""
    try:
        user_id = http_request.headers.get("x-user-id") if http_request else None
        result = await async_meme_service.score_meme(meme_id, user_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to score meme: {str(e)}"
        )


@router.delete("/{meme_id}")
async def delete_meme(
    meme_id: str,
    reason: str,
    soft_delete: bool = True,
    http_request: Request = None
):
    """Delete a meme."""
    try:
        user_id = http_request.headers.get("x-user-id") if http_request else None
        result = await async_meme_service.delete_meme(meme_id, reason, user_id, soft_delete)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete meme: {str(e)}"
        )


@router.get("/metrics/daily")
async def get_daily_metrics(date: Optional[str] = None):
    """Get daily metrics for meme generation."""
    try:
        metrics = await async_meme_service.get_metrics(date)
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No metrics found for the specified date"
            )
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.get("/system/health")
async def system_health():
    """Get system health status."""
    try:
        health = await health_check()
        if not health["overall"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System is not healthy"
            )
        return health
    except Exception as e:
        return {
            "overall": False,
            "error": str(e),
            "system_initialized": False,
            "actor_system": False,
            "event_store": False,
            "event_bus": False,
            "actors": {}
        } 