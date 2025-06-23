"""Analytics and metadata endpoints."""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import Session
import pandas as pd
from io import BytesIO
import csv

from ...models.database import get_db
from ...models.db_models.metadata import MemeMetadata, GenerationLog, PerformanceMetrics
from ...models.schemas.metadata import (
    MemeMetadataResponse,
    MemeAnalyticsResponse,
    GenerationStatsResponse,
    MetadataSearchRequest,
    MetadataExportRequest
)
from ..dependencies import get_cache

router = APIRouter()


@router.get("/memes/{meme_id}/metadata", response_model=MemeMetadataResponse)
async def get_meme_metadata(
    meme_id: str,
    db: Session = Depends(get_db)
) -> MemeMetadataResponse:
    """Get comprehensive metadata for a specific meme."""
    
    meme = db.query(MemeMetadata).filter(MemeMetadata.id == meme_id).first()
    
    if not meme:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meme not found: {meme_id}"
        )
    
    # Calculate additional metrics
    total_cost = 0.0
    if meme.cost_metadata:
        total_cost = meme.cost_metadata.get('total_cost', 0.0)
    
    return MemeMetadataResponse(
        id=meme.id,
        topic=meme.topic,
        format=meme.format,
        text=meme.text,
        image_url=meme.image_url,
        created_at=meme.created_at.isoformat(),
        score=meme.score,
        confidence=meme.confidence,
        relevance_score=meme.relevance_score,
        humor_score=meme.humor_score,
        virality_score=meme.virality_score,
        generation_metadata=meme.generation_metadata or {},
        image_metadata=meme.image_metadata or {},
        dspy_metadata=meme.dspy_metadata or {},
        total_duration_ms=meme.generation_metadata.get('total_duration_ms') if meme.generation_metadata else None,
        cache_hit=meme.technical_metadata.get('cache_hit', False) if meme.technical_metadata else False,
        generation_cost=total_cost
    )


@router.get("/memes/{meme_id}/analytics", response_model=MemeAnalyticsResponse)
async def get_meme_analytics(
    meme_id: str,
    db: Session = Depends(get_db),
    cache = Depends(get_cache)
) -> MemeAnalyticsResponse:
    """Get analytics data for a specific meme."""
    
    # Check cache first
    cache_key = f"analytics:{meme_id}"
    cached_data = await cache.get(cache_key)
    if cached_data:
        return MemeAnalyticsResponse(**json.loads(cached_data))
    
    meme = db.query(MemeMetadata).filter(MemeMetadata.id == meme_id).first()
    
    if not meme:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meme not found: {meme_id}"
        )
    
    # Get interaction data
    interactions = meme.interaction_metadata or {}
    
    # Calculate engagement rate
    total_views = interactions.get('views', 0)
    total_engagements = (
        interactions.get('likes', 0) +
        interactions.get('shares', 0) +
        interactions.get('downloads', 0)
    )
    engagement_rate = (total_engagements / max(1, total_views)) * 100 if total_views > 0 else 0
    
    # Calculate cost metrics
    total_cost = meme.cost_metadata.get('total_cost', 0.0) if meme.cost_metadata else 0.0
    cost_per_view = total_cost / max(1, total_views) if total_views > 0 else total_cost
    
    analytics = MemeAnalyticsResponse(
        meme_id=meme_id,
        views=interactions.get('views', 0),
        likes=interactions.get('likes', 0),
        shares=interactions.get('shares', 0),
        downloads=interactions.get('downloads', 0),
        engagement_rate=round(engagement_rate, 2),
        is_viral=meme.is_viral,
        total_cost=total_cost,
        cost_per_view=round(cost_per_view, 4)
    )
    
    # Cache the result
    await cache.set(cache_key, json.dumps(analytics.dict()), ex=300)  # 5 minute cache
    
    return analytics


@router.get("/stats/{time_period}", response_model=GenerationStatsResponse)
async def get_generation_stats(
    time_period: str,
    db: Session = Depends(get_db),
    cache = Depends(get_cache)
) -> GenerationStatsResponse:
    """Get aggregate generation statistics for a time period."""
    
    # Validate time period
    if time_period not in ["hourly", "daily", "weekly"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="time_period must be one of: hourly, daily, weekly"
        )
    
    # Calculate time range
    now = datetime.utcnow()
    if time_period == "hourly":
        start_time = now - timedelta(hours=1)
        start_time = start_time.replace(minute=0, second=0, microsecond=0)
    elif time_period == "daily":
        start_time = now - timedelta(days=1)
        start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    else:  # weekly
        start_time = now - timedelta(days=7)
        start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Check cache
    cache_key = f"stats:{time_period}:{start_time.isoformat()}"
    cached_data = await cache.get(cache_key)
    if cached_data:
        return GenerationStatsResponse(**json.loads(cached_data))
    
    # Query generation logs
    logs = db.query(GenerationLog).filter(
        GenerationLog.timestamp >= start_time
    ).all()
    
    # Calculate metrics
    total_generations = len(logs)
    successful_generations = sum(1 for log in logs if log.response_status == 'success')
    failed_generations = total_generations - successful_generations
    success_rate = (successful_generations / max(1, total_generations)) * 100
    
    # Performance metrics
    response_times = [log.response_time_ms for log in logs if log.response_time_ms]
    if response_times:
        response_times.sort()
        avg_time = sum(response_times) / len(response_times)
        p50_time = response_times[len(response_times) // 2]
        p95_time = response_times[int(len(response_times) * 0.95)]
        p99_time = response_times[int(len(response_times) * 0.99)]
    else:
        avg_time = p50_time = p95_time = p99_time = 0
    
    # Model breakdown
    model_breakdown = {}
    provider_breakdown = {}
    error_breakdown = {}
    
    for log in logs:
        if log.model_name:
            model_breakdown[log.model_name] = model_breakdown.get(log.model_name, 0) + 1
        if log.model_provider:
            provider_breakdown[log.model_provider] = provider_breakdown.get(log.model_provider, 0) + 1
        if log.error_type:
            error_breakdown[log.error_type] = error_breakdown.get(log.error_type, 0) + 1
    
    # Query memes for quality metrics
    memes = db.query(MemeMetadata).filter(
        MemeMetadata.created_at >= start_time
    ).all()
    
    if memes:
        avg_score = sum(m.score for m in memes) / len(memes)
        avg_virality = sum(m.virality_score or 0 for m in memes) / len(memes)
        viral_count = sum(1 for m in memes if m.is_viral)
        
        # Cost calculations
        total_cost = sum(
            m.cost_metadata.get('total_cost', 0) 
            for m in memes 
            if m.cost_metadata
        )
        avg_cost = total_cost / len(memes)
    else:
        avg_score = avg_virality = viral_count = total_cost = avg_cost = 0
    
    # Cache hit rate
    cache_hits = sum(
        1 for m in memes 
        if m.technical_metadata and m.technical_metadata.get('cache_hit', False)
    )
    cache_hit_rate = (cache_hits / max(1, len(memes))) * 100 if memes else 0
    
    error_rate = (failed_generations / max(1, total_generations)) * 100
    
    stats = GenerationStatsResponse(
        time_period=time_period,
        start_time=start_time.isoformat(),
        end_time=now.isoformat(),
        total_generations=total_generations,
        successful_generations=successful_generations,
        failed_generations=failed_generations,
        success_rate=round(success_rate, 2),
        avg_generation_time_ms=round(avg_time, 2),
        p50_generation_time_ms=round(p50_time, 2),
        p95_generation_time_ms=round(p95_time, 2),
        p99_generation_time_ms=round(p99_time, 2),
        model_breakdown=model_breakdown,
        provider_breakdown=provider_breakdown,
        avg_score=round(avg_score, 3),
        avg_virality_score=round(avg_virality, 3),
        viral_meme_count=viral_count,
        total_cost=round(total_cost, 2),
        avg_cost_per_meme=round(avg_cost, 3),
        cache_hit_rate=round(cache_hit_rate, 2),
        error_rate=round(error_rate, 2),
        error_breakdown=error_breakdown
    )
    
    # Cache for 5 minutes
    await cache.set(cache_key, json.dumps(stats.dict()), ex=300)
    
    return stats


@router.post("/search", response_model=List[MemeMetadataResponse])
async def search_memes_by_metadata(
    search_request: MetadataSearchRequest,
    db: Session = Depends(get_db)
) -> List[MemeMetadataResponse]:
    """Search memes using metadata filters."""
    
    query = db.query(MemeMetadata)
    
    # Apply filters
    if search_request.topic:
        query = query.filter(MemeMetadata.topic.ilike(f"%{search_request.topic}%"))
    
    if search_request.format:
        query = query.filter(MemeMetadata.format == search_request.format)
    
    if search_request.min_score is not None:
        query = query.filter(MemeMetadata.score >= search_request.min_score)
    
    if search_request.min_virality_score is not None:
        query = query.filter(MemeMetadata.virality_score >= search_request.min_virality_score)
    
    if search_request.created_after:
        query = query.filter(MemeMetadata.created_at >= datetime.fromisoformat(search_request.created_after))
    
    if search_request.created_before:
        query = query.filter(MemeMetadata.created_at <= datetime.fromisoformat(search_request.created_before))
    
    if search_request.is_viral is not None:
        if search_request.is_viral:
            query = query.filter(
                or_(
                    MemeMetadata.interaction_metadata['views'].astext.cast(Integer) > 10000,
                    MemeMetadata.interaction_metadata['shares'].astext.cast(Integer) > 1000,
                    MemeMetadata.virality_score > 0.8
                )
            )
    
    # Sorting
    sort_field = getattr(MemeMetadata, search_request.sort_by, MemeMetadata.created_at)
    if search_request.sort_order == "asc":
        query = query.order_by(sort_field.asc())
    else:
        query = query.order_by(sort_field.desc())
    
    # Pagination
    query = query.offset(search_request.offset).limit(search_request.limit)
    
    memes = query.all()
    
    return [
        MemeMetadataResponse(
            id=meme.id,
            topic=meme.topic,
            format=meme.format,
            text=meme.text,
            image_url=meme.image_url,
            created_at=meme.created_at.isoformat(),
            score=meme.score,
            confidence=meme.confidence,
            relevance_score=meme.relevance_score,
            humor_score=meme.humor_score,
            virality_score=meme.virality_score,
            generation_metadata=meme.generation_metadata or {},
            image_metadata=meme.image_metadata or {},
            dspy_metadata=meme.dspy_metadata or {},
            total_duration_ms=meme.generation_metadata.get('total_duration_ms') if meme.generation_metadata else None,
            generation_cost=meme.cost_metadata.get('total_cost', 0) if meme.cost_metadata else 0
        )
        for meme in memes
    ]


@router.post("/export")
async def export_metadata(
    export_request: MetadataExportRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Export metadata in various formats."""
    
    # Build query
    query = db.query(MemeMetadata)
    
    # Apply filters if provided
    if export_request.filters:
        # Apply same filters as search
        # ... (filter logic same as search_memes_by_metadata)
        pass
    
    memes = query.all()
    
    # Convert to dictionaries
    data = [meme.to_dict() for meme in memes]
    
    # Filter fields if specified
    if export_request.fields:
        data = [
            {k: v for k, v in item.items() if k in export_request.fields}
            for item in data
        ]
    
    # Handle different export formats
    if export_request.format == "json":
        return {
            "format": "json",
            "count": len(data),
            "data": data
        }
    
    elif export_request.format == "csv":
        # Flatten nested dictionaries for CSV
        flattened_data = []
        for item in data:
            flat_item = {}
            for key, value in item.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_item[f"{key}.{sub_key}"] = str(sub_value)
                else:
                    flat_item[key] = str(value)
            flattened_data.append(flat_item)
        
        # Convert to CSV
        if flattened_data:
            csv_buffer = BytesIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=flattened_data[0].keys())
            writer.writeheader()
            writer.writerows(flattened_data)
            csv_content = csv_buffer.getvalue().decode('utf-8')
        else:
            csv_content = ""
        
        return {
            "format": "csv",
            "count": len(data),
            "data": csv_content
        }
    
    elif export_request.format == "parquet":
        # Convert to DataFrame and then to parquet
        df = pd.DataFrame(data)
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, engine='pyarrow')
        parquet_bytes = parquet_buffer.getvalue()
        
        return {
            "format": "parquet",
            "count": len(data),
            "size_bytes": len(parquet_bytes),
            "message": "Parquet data ready for download"
        }
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported export format: {export_request.format}"
        )


@router.get("/trending")
async def get_trending_metadata(
    hours: int = Query(24, ge=1, le=168),  # Max 1 week
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    cache = Depends(get_cache)
) -> List[Dict[str, Any]]:
    """Get trending memes based on engagement and virality scores."""
    
    cache_key = f"trending:{hours}:{limit}"
    cached_data = await cache.get(cache_key)
    if cached_data:
        return json.loads(cached_data)
    
    # Calculate time threshold
    threshold = datetime.utcnow() - timedelta(hours=hours)
    
    # Query memes with engagement data
    memes = db.query(MemeMetadata).filter(
        MemeMetadata.created_at >= threshold
    ).all()
    
    # Calculate trending score
    trending_memes = []
    for meme in memes:
        interactions = meme.interaction_metadata or {}
        
        # Trending score formula
        views = interactions.get('views', 0)
        likes = interactions.get('likes', 0)
        shares = interactions.get('shares', 0)
        
        # Time decay factor (newer = higher score)
        age_hours = (datetime.utcnow() - meme.created_at).total_seconds() / 3600
        time_factor = 1 / (1 + age_hours / 24)  # Decay over 24 hours
        
        # Calculate trending score
        trending_score = (
            (views * 0.1 + likes * 0.3 + shares * 0.6) * 
            time_factor * 
            (meme.virality_score or 0.5)
        )
        
        trending_memes.append({
            'id': meme.id,
            'topic': meme.topic,
            'format': meme.format,
            'text': meme.text,
            'image_url': meme.image_url,
            'created_at': meme.created_at.isoformat(),
            'trending_score': round(trending_score, 2),
            'views': views,
            'likes': likes,
            'shares': shares,
            'virality_score': meme.virality_score
        })
    
    # Sort by trending score
    trending_memes.sort(key=lambda x: x['trending_score'], reverse=True)
    
    result = trending_memes[:limit]
    
    # Cache for 15 minutes
    await cache.set(cache_key, json.dumps(result), ex=900)
    
    return result