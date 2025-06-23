"""Enhanced meme schemas with metadata support."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class MemeMetadataResponse(BaseModel):
    """Comprehensive meme response with all metadata."""
    
    # Core fields
    id: str = Field(..., description="Unique identifier for the meme")
    topic: str = Field(..., description="The topic of the meme")
    format: str = Field(..., description="The format of the meme")
    text: str = Field(..., description="The generated meme text")
    image_url: str = Field(..., description="URL to the generated meme image")
    created_at: str = Field(..., description="ISO timestamp of creation")
    
    # Quality metrics
    score: float = Field(..., description="Overall quality score (0-1)")
    confidence: Optional[float] = Field(None, description="Model confidence in generation")
    relevance_score: Optional[float] = Field(None, description="Topic relevance score")
    humor_score: Optional[float] = Field(None, description="Predicted humor level")
    virality_score: Optional[float] = Field(None, description="Predicted viral potential")
    
    # Generation metadata
    generation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation process metadata")
    image_metadata: Dict[str, Any] = Field(default_factory=dict, description="Image technical metadata")
    dspy_metadata: Dict[str, Any] = Field(default_factory=dict, description="DSPy model metadata")
    
    # Performance metrics
    total_duration_ms: Optional[float] = Field(None, description="Total generation time in milliseconds")
    cache_hit: bool = Field(False, description="Whether this was served from cache")
    
    # Cost information
    generation_cost: Optional[float] = Field(None, description="Total cost of generation in USD")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "b1691d8c-c16d-4128-bf29-010557116f1c",
                "topic": "Python Programming",
                "format": "Drake meme",
                "text": "Writing code in other languages vs Writing code in Python",
                "image_url": "/static/images/memes/abc123.png",
                "created_at": "2025-03-28T18:15:35.647677",
                "score": 0.85,
                "confidence": 0.92,
                "generation_metadata": {
                    "model_used": "gpt-image-1",
                    "generation_time_ms": 3240,
                    "retry_count": 0,
                    "fallback_used": False
                },
                "image_metadata": {
                    "width": 1024,
                    "height": 1024,
                    "format": "png",
                    "size_bytes": 1245632,
                    "storage_location": "local"
                },
                "total_duration_ms": 4567,
                "cache_hit": False,
                "generation_cost": 0.022
            }
        }


class MemeAnalyticsResponse(BaseModel):
    """Analytics data for a specific meme."""
    
    meme_id: str
    
    # Engagement metrics
    views: int = Field(0, description="Total view count")
    likes: int = Field(0, description="Total likes")
    shares: int = Field(0, description="Total shares")
    downloads: int = Field(0, description="Total downloads")
    engagement_rate: float = Field(0.0, description="Engagement rate percentage")
    
    # Performance over time
    hourly_views: List[int] = Field(default_factory=list, description="Views per hour (last 24h)")
    daily_views: List[int] = Field(default_factory=list, description="Views per day (last 7d)")
    
    # Virality metrics
    is_viral: bool = Field(False, description="Whether meme has gone viral")
    viral_timestamp: Optional[str] = Field(None, description="When meme went viral")
    peak_views_per_hour: int = Field(0, description="Peak hourly views")
    
    # Cost analysis
    total_cost: float = Field(0.0, description="Total cost including generation and serving")
    cost_per_view: float = Field(0.0, description="Cost per view")
    revenue_generated: float = Field(0.0, description="Revenue from ads/premium")
    profit_margin: float = Field(0.0, description="Profit margin percentage")


class GenerationStatsResponse(BaseModel):
    """Aggregate generation statistics."""
    
    time_period: str = Field(..., description="Time period for stats (hourly, daily, weekly)")
    start_time: str = Field(..., description="Start of period")
    end_time: str = Field(..., description="End of period")
    
    # Volume metrics
    total_generations: int = Field(0, description="Total memes generated")
    successful_generations: int = Field(0, description="Successful generations")
    failed_generations: int = Field(0, description="Failed generations")
    success_rate: float = Field(0.0, description="Success rate percentage")
    
    # Performance metrics
    avg_generation_time_ms: float = Field(0.0, description="Average generation time")
    p50_generation_time_ms: float = Field(0.0, description="50th percentile generation time")
    p95_generation_time_ms: float = Field(0.0, description="95th percentile generation time")
    p99_generation_time_ms: float = Field(0.0, description="99th percentile generation time")
    
    # Model usage
    model_breakdown: Dict[str, int] = Field(default_factory=dict, description="Generations by model")
    provider_breakdown: Dict[str, int] = Field(default_factory=dict, description="Generations by provider")
    
    # Quality metrics
    avg_score: float = Field(0.0, description="Average quality score")
    avg_virality_score: float = Field(0.0, description="Average virality score")
    viral_meme_count: int = Field(0, description="Number of viral memes")
    
    # Cost metrics
    total_cost: float = Field(0.0, description="Total generation cost")
    avg_cost_per_meme: float = Field(0.0, description="Average cost per meme")
    
    # Cache metrics
    cache_hit_rate: float = Field(0.0, description="Cache hit rate percentage")
    cache_savings: float = Field(0.0, description="Cost savings from cache")
    
    # Error analysis
    error_rate: float = Field(0.0, description="Error rate percentage")
    error_breakdown: Dict[str, int] = Field(default_factory=dict, description="Errors by type")


class MetadataSearchRequest(BaseModel):
    """Request for searching memes by metadata."""
    
    # Basic filters
    topic: Optional[str] = Field(None, description="Filter by topic")
    format: Optional[str] = Field(None, description="Filter by format")
    
    # Quality filters
    min_score: Optional[float] = Field(None, description="Minimum quality score")
    min_virality_score: Optional[float] = Field(None, description="Minimum virality score")
    
    # Time filters
    created_after: Optional[str] = Field(None, description="Created after ISO timestamp")
    created_before: Optional[str] = Field(None, description="Created before ISO timestamp")
    
    # Model filters
    model_used: Optional[str] = Field(None, description="Filter by model used")
    provider: Optional[str] = Field(None, description="Filter by provider")
    
    # Performance filters
    max_generation_time_ms: Optional[float] = Field(None, description="Maximum generation time")
    no_retries: bool = Field(False, description="Only memes generated without retries")
    
    # Engagement filters
    min_views: Optional[int] = Field(None, description="Minimum view count")
    is_viral: Optional[bool] = Field(None, description="Filter viral memes")
    
    # Pagination
    limit: int = Field(10, ge=1, le=100, description="Results per page")
    offset: int = Field(0, ge=0, description="Offset for pagination")
    
    # Sorting
    sort_by: str = Field("created_at", description="Field to sort by")
    sort_order: str = Field("desc", description="Sort order (asc/desc)")


class MetadataExportRequest(BaseModel):
    """Request for exporting metadata."""
    
    format: str = Field("json", description="Export format (json, csv, parquet)")
    include_images: bool = Field(False, description="Include base64 encoded images")
    filters: Optional[MetadataSearchRequest] = Field(None, description="Optional filters")
    fields: Optional[List[str]] = Field(None, description="Specific fields to export")