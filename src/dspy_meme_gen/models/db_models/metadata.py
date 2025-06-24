"""Enhanced database models with comprehensive metadata tracking."""

import json
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import Column, String, Float, DateTime, JSON, Integer, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property

Base = declarative_base()


class MemeMetadata(Base):
    """
    Enhanced meme model with comprehensive metadata tracking.

    This model tracks EVERYTHING about meme generation including:
    - Generation parameters and settings
    - Performance metrics
    - Model information
    - Error handling and retries
    - User interaction data
    - Image technical details
    """

    __tablename__ = "meme_metadata"

    # Core identifiers
    id = Column(String, primary_key=True, index=True)

    # Basic meme data
    topic = Column(String, nullable=False, index=True)
    format = Column(String, nullable=False, index=True)
    text = Column(Text, nullable=False)
    image_url = Column(String, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Quality metrics
    score = Column(Float, nullable=False)
    confidence = Column(Float)  # Model confidence in generation
    relevance_score = Column(Float)  # How well it matches the topic
    humor_score = Column(Float)  # Predicted humor level
    virality_score = Column(Float)  # Predicted viral potential

    # Generation metadata
    generation_metadata = Column(JSON, default={})
    """
    Includes:
    - model_used: str (gpt-image-1, dall-e-3, etc.)
    - model_version: str
    - temperature: float
    - max_tokens: int
    - prompt_tokens: int
    - completion_tokens: int
    - total_tokens: int
    - generation_time_ms: int
    - retry_count: int
    - fallback_used: bool
    - fallback_reason: str
    """

    # Image metadata
    image_metadata = Column(JSON, default={})
    """
    Includes:
    - width: int
    - height: int
    - format: str (png, jpg, etc.)
    - size_bytes: int
    - color_profile: str
    - has_transparency: bool
    - dominant_colors: list
    - brightness: float
    - contrast: float
    - sharpness: float
    - compression_ratio: float
    - generation_provider: str
    - storage_location: str (local, s3, cloudinary)
    - cdn_url: str
    """

    # DSPy metadata
    dspy_metadata = Column(JSON, default={})
    """
    Includes:
    - predictor_class: str
    - signature_used: str
    - chain_of_thought: list
    - reasoning_steps: list
    - temperature: float
    - prompts_used: dict
    - model_outputs: dict
    - optimization_metrics: dict
    """

    # Content analysis
    content_analysis = Column(JSON, default={})
    """
    Includes:
    - sentiment: str (positive, negative, neutral)
    - emotions: dict (joy, anger, surprise, etc.)
    - topics_detected: list
    - entities_mentioned: list
    - language: str
    - profanity_score: float
    - toxicity_score: float
    - readability_score: float
    - meme_elements: list (text placement, font, style)
    """

    # User interaction metadata
    interaction_metadata = Column(JSON, default={})
    """
    Includes:
    - views: int
    - likes: int
    - shares: int
    - downloads: int
    - report_count: int
    - avg_view_duration: float
    - click_through_rate: float
    - engagement_rate: float
    """

    # Technical metadata
    technical_metadata = Column(JSON, default={})
    """
    Includes:
    - api_version: str
    - client_ip: str
    - user_agent: str
    - request_id: str
    - session_id: str
    - server_region: str
    - processing_node: str
    - cache_hit: bool
    - response_time_ms: int
    - error_logs: list
    """

    # A/B testing metadata
    experiment_metadata = Column(JSON, default={})
    """
    Includes:
    - experiment_id: str
    - variant: str
    - control_group: bool
    - feature_flags: dict
    """

    # Moderation metadata
    moderation_metadata = Column(JSON, default={})
    """
    Includes:
    - moderation_status: str
    - flagged_categories: list
    - moderation_scores: dict
    - human_reviewed: bool
    - reviewer_id: str
    - review_timestamp: datetime
    - moderation_actions: list
    """

    # Cost tracking
    cost_metadata = Column(JSON, default={})
    """
    Includes:
    - text_generation_cost: float
    - image_generation_cost: float
    - storage_cost: float
    - bandwidth_cost: float
    - total_cost: float
    - cost_per_view: float
    """

    # Computed properties
    @hybrid_property
    def total_engagement(self) -> int:
        """Calculate total engagement from interaction metadata."""
        if not self.interaction_metadata:
            return 0
        return (
            self.interaction_metadata.get("likes", 0)
            + self.interaction_metadata.get("shares", 0)
            + self.interaction_metadata.get("downloads", 0)
        )

    @hybrid_property
    def is_viral(self) -> bool:
        """Determine if meme has gone viral based on metrics."""
        if not self.interaction_metadata:
            return False
        return (
            self.interaction_metadata.get("views", 0) > 10000
            or self.interaction_metadata.get("shares", 0) > 1000
            or self.virality_score
            and self.virality_score > 0.8
        )

    @hybrid_property
    def generation_efficiency(self) -> float:
        """Calculate generation efficiency score."""
        if not self.generation_metadata:
            return 0.0

        time_score = max(0, 1 - (self.generation_metadata.get("generation_time_ms", 10000) / 10000))
        retry_penalty = 1 / (1 + self.generation_metadata.get("retry_count", 0))
        token_efficiency = min(1, 500 / max(1, self.generation_metadata.get("total_tokens", 500)))

        return time_score * 0.4 + retry_penalty * 0.3 + token_efficiency * 0.3

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary with all metadata."""
        return {
            "id": self.id,
            "topic": self.topic,
            "format": self.format,
            "text": self.text,
            "image_url": self.image_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "score": self.score,
            "confidence": self.confidence,
            "relevance_score": self.relevance_score,
            "humor_score": self.humor_score,
            "virality_score": self.virality_score,
            "generation_metadata": self.generation_metadata,
            "image_metadata": self.image_metadata,
            "dspy_metadata": self.dspy_metadata,
            "content_analysis": self.content_analysis,
            "interaction_metadata": self.interaction_metadata,
            "technical_metadata": self.technical_metadata,
            "experiment_metadata": self.experiment_metadata,
            "moderation_metadata": self.moderation_metadata,
            "cost_metadata": self.cost_metadata,
            "total_engagement": self.total_engagement,
            "is_viral": self.is_viral,
            "generation_efficiency": self.generation_efficiency,
        }


class GenerationLog(Base):
    """
    Detailed log of every generation attempt for debugging and analytics.
    """

    __tablename__ = "generation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    meme_id = Column(String, index=True)  # Links to MemeMetadata
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Request details
    request_type = Column(String)  # text_generation, image_generation
    request_payload = Column(JSON)

    # Response details
    response_status = Column(String)  # success, error, timeout
    response_payload = Column(JSON)
    response_time_ms = Column(Integer)

    # Error tracking
    error_type = Column(String)
    error_message = Column(Text)
    error_traceback = Column(Text)

    # Model details
    model_name = Column(String)
    model_provider = Column(String)

    # Cost
    estimated_cost = Column(Float)
    actual_cost = Column(Float)


class PerformanceMetrics(Base):
    """
    Aggregated performance metrics for monitoring and optimization.
    """

    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    metric_type = Column(String, index=True)  # hourly, daily, weekly

    # Generation metrics
    total_generations = Column(Integer, default=0)
    successful_generations = Column(Integer, default=0)
    failed_generations = Column(Integer, default=0)

    # Performance metrics
    avg_generation_time_ms = Column(Float)
    p50_generation_time_ms = Column(Float)
    p95_generation_time_ms = Column(Float)
    p99_generation_time_ms = Column(Float)

    # Model usage
    model_usage = Column(JSON)  # {model_name: count}

    # Cost metrics
    total_cost = Column(Float)
    avg_cost_per_meme = Column(Float)

    # Quality metrics
    avg_score = Column(Float)
    avg_virality_score = Column(Float)
    viral_meme_count = Column(Integer)

    # Error metrics
    error_rate = Column(Float)
    error_breakdown = Column(JSON)  # {error_type: count}

    # Cache metrics
    cache_hit_rate = Column(Float)
    cache_size_mb = Column(Float)
