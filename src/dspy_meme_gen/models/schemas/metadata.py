"""Enhanced meme schemas with metadata support and Pydantic models for JSON columns."""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


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


# Pydantic models for JSON columns to replace raw JSON storage

class ModerationStatus(str, Enum):
    """Moderation status enumeration."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED = "flagged"
    HUMAN_REVIEW = "human_review"


class GenerationStatus(str, Enum):
    """Generation status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    PARTIAL = "partial"


class StorageLocation(str, Enum):
    """Storage location enumeration."""
    LOCAL = "local"
    S3 = "s3"
    CLOUDINARY = "cloudinary"
    CDN = "cdn"


class ModelProvider(str, Enum):
    """Model provider enumeration."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class GenerationMetadata(BaseModel):
    """Metadata for meme generation process."""
    
    model_used: str = Field(..., description="Model used for generation")
    model_version: Optional[str] = Field(None, description="Version of the model")
    model_provider: ModelProvider = Field(default=ModelProvider.OPENAI, description="Model provider")
    temperature: Optional[float] = Field(None, description="Generation temperature", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="Maximum tokens", gt=0)
    prompt_tokens: Optional[int] = Field(None, description="Tokens in prompt", ge=0)
    completion_tokens: Optional[int] = Field(None, description="Tokens in completion", ge=0)
    total_tokens: Optional[int] = Field(None, description="Total tokens used", ge=0)
    generation_time_ms: Optional[int] = Field(None, description="Generation time in milliseconds", ge=0)
    retry_count: int = Field(default=0, description="Number of retries", ge=0)
    fallback_used: bool = Field(default=False, description="Whether fallback model was used")
    fallback_reason: Optional[str] = Field(None, description="Reason for fallback")
    cache_hit: bool = Field(default=False, description="Whether result was cached")
    
    @validator('total_tokens')
    def validate_total_tokens(cls, v, values):
        """Validate total tokens equals prompt + completion."""
        if v is not None:
            prompt = values.get('prompt_tokens', 0) or 0
            completion = values.get('completion_tokens', 0) or 0
            if prompt + completion > 0 and v != prompt + completion:
                raise ValueError('total_tokens must equal prompt_tokens + completion_tokens')
        return v


class ImageMetadata(BaseModel):
    """Metadata for generated images."""
    
    width: Optional[int] = Field(None, description="Image width in pixels", gt=0)
    height: Optional[int] = Field(None, description="Image height in pixels", gt=0)
    format: Optional[str] = Field(None, description="Image format (png, jpg, etc.)")
    size_bytes: Optional[int] = Field(None, description="Image size in bytes", ge=0)
    color_profile: Optional[str] = Field(None, description="Color profile")
    has_transparency: bool = Field(default=False, description="Whether image has transparency")
    dominant_colors: List[str] = Field(default_factory=list, description="Dominant color hex codes")
    brightness: Optional[float] = Field(None, description="Average brightness", ge=0.0, le=1.0)
    contrast: Optional[float] = Field(None, description="Contrast measure", ge=0.0)
    sharpness: Optional[float] = Field(None, description="Sharpness measure", ge=0.0)
    compression_ratio: Optional[float] = Field(None, description="Compression ratio", ge=0.0, le=1.0)
    generation_provider: Optional[str] = Field(None, description="Image generation provider")
    storage_location: StorageLocation = Field(default=StorageLocation.LOCAL, description="Where image is stored")
    cdn_url: Optional[str] = Field(None, description="CDN URL for the image")
    
    @validator('dominant_colors')
    def validate_hex_colors(cls, v):
        """Validate hex color codes."""
        import re
        hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
        for color in v:
            if not hex_pattern.match(color):
                raise ValueError(f'Invalid hex color code: {color}')
        return v


class DSPyMetadata(BaseModel):
    """Metadata for DSPy operations."""
    
    predictor_class: Optional[str] = Field(None, description="DSPy predictor class used")
    signature_used: Optional[str] = Field(None, description="DSPy signature used")
    chain_of_thought: List[str] = Field(default_factory=list, description="Chain of thought steps")
    reasoning_steps: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed reasoning steps")
    temperature: Optional[float] = Field(None, description="DSPy temperature", ge=0.0, le=2.0)
    prompts_used: Dict[str, str] = Field(default_factory=dict, description="Prompts used in generation")
    model_outputs: Dict[str, Any] = Field(default_factory=dict, description="Raw model outputs")
    optimization_metrics: Dict[str, float] = Field(default_factory=dict, description="Optimization metrics")
    trace_id: Optional[str] = Field(None, description="DSPy trace identifier")


class ContentAnalysis(BaseModel):
    """Content analysis results."""
    
    sentiment: Optional[str] = Field(None, description="Overall sentiment")
    emotions: Dict[str, float] = Field(default_factory=dict, description="Emotion scores")
    topics_detected: List[str] = Field(default_factory=list, description="Detected topics")
    entities_mentioned: List[Dict[str, str]] = Field(default_factory=list, description="Named entities")
    language: Optional[str] = Field(None, description="Detected language code")
    profanity_score: Optional[float] = Field(None, description="Profanity score", ge=0.0, le=1.0)
    toxicity_score: Optional[float] = Field(None, description="Toxicity score", ge=0.0, le=1.0)
    readability_score: Optional[float] = Field(None, description="Readability score", ge=0.0, le=1.0)
    meme_elements: List[Dict[str, str]] = Field(default_factory=list, description="Meme visual elements")
    
    @validator('sentiment')
    def validate_sentiment(cls, v):
        """Validate sentiment value."""
        if v and v not in ['positive', 'negative', 'neutral']:
            raise ValueError('sentiment must be positive, negative, or neutral')
        return v


class InteractionMetadata(BaseModel):
    """User interaction metrics."""
    
    views: int = Field(default=0, description="Number of views", ge=0)
    likes: int = Field(default=0, description="Number of likes", ge=0)
    shares: int = Field(default=0, description="Number of shares", ge=0)
    downloads: int = Field(default=0, description="Number of downloads", ge=0)
    report_count: int = Field(default=0, description="Number of reports", ge=0)
    avg_view_duration: Optional[float] = Field(None, description="Average view duration in seconds", ge=0.0)
    click_through_rate: Optional[float] = Field(None, description="Click-through rate", ge=0.0, le=1.0)
    engagement_rate: Optional[float] = Field(None, description="Engagement rate", ge=0.0, le=1.0)
    unique_viewers: Optional[int] = Field(None, description="Number of unique viewers", ge=0)
    bounce_rate: Optional[float] = Field(None, description="Bounce rate", ge=0.0, le=1.0)


class TechnicalMetadata(BaseModel):
    """Technical request metadata."""
    
    api_version: Optional[str] = Field(None, description="API version used")
    client_ip: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    request_id: Optional[str] = Field(None, description="Unique request ID")
    session_id: Optional[str] = Field(None, description="Session identifier")
    server_region: Optional[str] = Field(None, description="Server region")
    processing_node: Optional[str] = Field(None, description="Processing node ID")
    cache_hit: bool = Field(default=False, description="Whether request hit cache")
    response_time_ms: Optional[int] = Field(None, description="Response time in milliseconds", ge=0)
    error_logs: List[Dict[str, str]] = Field(default_factory=list, description="Error log entries")


class ExperimentMetadata(BaseModel):
    """A/B testing and experiment metadata."""
    
    experiment_id: Optional[str] = Field(None, description="Experiment identifier")
    variant: Optional[str] = Field(None, description="Experiment variant")
    control_group: bool = Field(default=False, description="Whether in control group")
    feature_flags: Dict[str, bool] = Field(default_factory=dict, description="Active feature flags")
    experiment_start: Optional[datetime] = Field(None, description="Experiment start time")
    experiment_end: Optional[datetime] = Field(None, description="Experiment end time")
    cohort: Optional[str] = Field(None, description="User cohort")


class ModerationMetadata(BaseModel):
    """Content moderation metadata."""
    
    moderation_status: ModerationStatus = Field(default=ModerationStatus.PENDING, description="Moderation status")
    flagged_categories: List[str] = Field(default_factory=list, description="Flagged content categories")
    moderation_scores: Dict[str, float] = Field(default_factory=dict, description="Moderation scores")
    human_reviewed: bool = Field(default=False, description="Whether human reviewed")
    reviewer_id: Optional[str] = Field(None, description="Human reviewer ID")
    review_timestamp: Optional[datetime] = Field(None, description="Review timestamp")
    moderation_actions: List[Dict[str, str]] = Field(default_factory=list, description="Actions taken")
    auto_moderation_confidence: Optional[float] = Field(None, description="Auto-moderation confidence", ge=0.0, le=1.0)


class CostMetadata(BaseModel):
    """Cost tracking metadata."""
    
    text_generation_cost: float = Field(default=0.0, description="Text generation cost", ge=0.0)
    image_generation_cost: float = Field(default=0.0, description="Image generation cost", ge=0.0)
    storage_cost: float = Field(default=0.0, description="Storage cost", ge=0.0)
    bandwidth_cost: float = Field(default=0.0, description="Bandwidth cost", ge=0.0)
    total_cost: float = Field(default=0.0, description="Total cost", ge=0.0)
    cost_per_view: Optional[float] = Field(None, description="Cost per view", ge=0.0)
    currency: str = Field(default="USD", description="Currency code")
    billing_period: Optional[str] = Field(None, description="Billing period")
    
    @validator('total_cost')
    def validate_total_cost(cls, v, values):
        """Validate total cost equals sum of individual costs."""
        individual_costs = (
            values.get('text_generation_cost', 0) +
            values.get('image_generation_cost', 0) +
            values.get('storage_cost', 0) +
            values.get('bandwidth_cost', 0)
        )
        if individual_costs > 0 and abs(v - individual_costs) > 0.01:
            raise ValueError('total_cost should equal sum of individual costs')
        return v


class MetadataConverter:
    """Converter between Pydantic models and JSON for database storage."""
    
    @staticmethod
    def to_json(model: BaseModel) -> Dict[str, Any]:
        """Convert Pydantic model to JSON-serializable dict."""
        return model.dict(exclude_none=False, by_alias=True)
    
    @staticmethod
    def from_json(data: Dict[str, Any], model_class: type) -> BaseModel:
        """Convert JSON dict to Pydantic model."""
        return model_class(**data)
    
    @staticmethod
    def validate_and_convert(data: Dict[str, Any], model_class: type) -> Dict[str, Any]:
        """Validate data with Pydantic model and return JSON dict."""
        validated_model = model_class(**data)
        return MetadataConverter.to_json(validated_model)


# Mapping of JSON column names to their Pydantic models
METADATA_MODELS = {
    'generation_metadata': GenerationMetadata,
    'image_metadata': ImageMetadata,
    'dspy_metadata': DSPyMetadata,
    'content_analysis': ContentAnalysis,
    'interaction_metadata': InteractionMetadata,
    'technical_metadata': TechnicalMetadata,
    'experiment_metadata': ExperimentMetadata,
    'moderation_metadata': ModerationMetadata,
    'cost_metadata': CostMetadata,
}