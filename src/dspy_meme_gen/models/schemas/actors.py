"""
Message schema validation for actors in the MemesPy system.
Provides comprehensive validation for all actor message types with proper schema definitions.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator
from pydantic.json import pydantic_encoder


class MessagePriority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessageStatus(str, Enum):
    """Message processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class BaseMessage(BaseModel):
    """Base message model with common fields."""
    
    id: str = Field(..., description="Unique message identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")
    sender: Optional[str] = Field(None, description="Message sender identifier")
    receiver: Optional[str] = Field(None, description="Message receiver identifier")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="Message priority")
    timeout_ms: Optional[int] = Field(None, description="Message timeout in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('id')
    def validate_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Message id cannot be empty')
        return v.strip()
    
    @validator('timeout_ms')
    def validate_timeout(cls, v):
        if v is not None and v <= 0:
            raise ValueError('timeout_ms must be positive')
        return v


class Request(BaseMessage):
    """Base request message."""
    expects_response: bool = Field(default=True, description="Whether this request expects a response")
    response_timeout_ms: Optional[int] = Field(None, description="Response timeout in milliseconds")


class Response(BaseMessage):
    """Base response message."""
    request_id: str = Field(..., description="ID of the original request")
    success: bool = Field(default=True, description="Whether the request was successful")
    error_message: Optional[str] = Field(None, description="Error message if request failed")
    processing_duration_ms: Optional[float] = Field(None, description="Processing duration in milliseconds")
    
    @validator('request_id')
    def validate_request_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('request_id cannot be empty')
        return v.strip()


class Event(BaseMessage):
    """Base event message."""
    event_type: str = Field(..., description="Type of event")
    source_actor: Optional[str] = Field(None, description="Actor that generated the event")
    
    @validator('event_type')
    def validate_event_type(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('event_type cannot be empty')
        return v.strip()


# System Messages
class PingMessage(BaseMessage):
    """Health check message."""
    message_type: str = Field(default="ping", description="Message type")


class PongMessage(Response):
    """Health check response."""
    message_type: str = Field(default="pong", description="Message type")
    system_status: Optional[Dict[str, Any]] = Field(None, description="System status information")


class TerminateMessage(BaseMessage):
    """Request to terminate an actor."""
    message_type: str = Field(default="terminate", description="Message type")
    reason: str = Field(..., description="Reason for termination")
    graceful: bool = Field(default=True, description="Whether to terminate gracefully")
    
    @validator('reason')
    def validate_reason(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('reason cannot be empty')
        return v.strip()


class RestartMessage(BaseMessage):
    """Request to restart an actor."""
    message_type: str = Field(default="restart", description="Message type")
    reason: str = Field(..., description="Reason for restart")
    clear_state: bool = Field(default=False, description="Whether to clear actor state")
    
    @validator('reason')
    def validate_reason(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('reason cannot be empty')
        return v.strip()


# Meme Generation Messages
class GenerateMemeRequest(Request):
    """Request to generate a meme."""
    message_type: str = Field(default="generate_meme", description="Message type")
    prompt: str = Field(..., description="Meme generation prompt")
    style: Optional[str] = Field(None, description="Meme style preference")
    user_id: str = Field(default="anonymous", description="User identifier")
    format_preference: Optional[str] = Field(None, description="Preferred meme format")
    verification_level: str = Field(default="standard", description="Level of verification required")
    max_refinement_iterations: int = Field(default=3, description="Maximum refinement iterations")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('prompt cannot be empty')
        if len(v) > 2000:
            raise ValueError('prompt cannot exceed 2000 characters')
        return v.strip()
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('user_id cannot be empty')
        return v.strip()
    
    @validator('max_refinement_iterations')
    def validate_max_iterations(cls, v):
        if v < 0 or v > 10:
            raise ValueError('max_refinement_iterations must be between 0 and 10')
        return v


class GenerateMemeResponse(Response):
    """Response from meme generation."""
    message_type: str = Field(default="generate_meme_response", description="Message type")
    meme_id: Optional[str] = Field(None, description="Generated meme identifier")
    image_url: Optional[str] = Field(None, description="URL to generated image")
    caption: Optional[str] = Field(None, description="Meme caption text")
    format_type: Optional[str] = Field(None, description="Actual meme format used")
    quality_score: Optional[float] = Field(None, description="Quality score (0.0 to 10.0)")
    verification_results: Dict[str, bool] = Field(default_factory=dict, description="Verification results")
    refinement_iterations: int = Field(default=0, description="Number of refinement iterations performed")
    generation_details: Dict[str, Any] = Field(default_factory=dict, description="Generation process details")
    
    @validator('quality_score')
    def validate_quality_score(cls, v):
        if v is not None and not (0.0 <= v <= 10.0):
            raise ValueError('quality_score must be between 0.0 and 10.0')
        return v
    
    @validator('refinement_iterations')
    def validate_refinement_iterations(cls, v):
        if v < 0:
            raise ValueError('refinement_iterations cannot be negative')
        return v


class MemeGenerationProgressEvent(Event):
    """Progress update for meme generation."""
    message_type: str = Field(default="meme_generation_progress", description="Message type")
    meme_id: str = Field(..., description="Meme identifier")
    stage: str = Field(..., description="Current generation stage")
    progress: float = Field(..., description="Progress percentage (0.0 to 1.0)")
    stage_message: str = Field(..., description="Human-readable progress message")
    estimated_completion_ms: Optional[int] = Field(None, description="Estimated time to completion")
    
    @validator('meme_id', 'stage', 'stage_message')
    def validate_non_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Field cannot be empty')
        return v.strip()
    
    @validator('progress')
    def validate_progress(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('progress must be between 0.0 and 1.0')
        return v


# Verification Messages
class VerifyContentRequest(Request):
    """Request to verify content."""
    message_type: str = Field(default="verify_content", description="Message type")
    content: str = Field(..., description="Content to verify")
    meme_id: str = Field(..., description="Associated meme identifier")
    verification_type: str = Field(..., description="Type of verification")
    verification_config: Dict[str, Any] = Field(default_factory=dict, description="Verification configuration")
    
    @validator('content', 'meme_id', 'verification_type')
    def validate_non_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Field cannot be empty')
        return v.strip()
    
    @validator('verification_type')
    def validate_verification_type(cls, v):
        valid_types = ['appropriateness', 'factuality', 'instructions', 'quality', 'safety']
        if v not in valid_types:
            raise ValueError(f'verification_type must be one of: {valid_types}')
        return v


class VerifyContentResponse(Response):
    """Response from content verification."""
    message_type: str = Field(default="verify_content_response", description="Message type")
    passed: bool = Field(..., description="Whether verification passed")
    confidence: float = Field(default=1.0, description="Confidence in verification result")
    issues: List[str] = Field(default_factory=list, description="List of identified issues")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    verification_details: Dict[str, Any] = Field(default_factory=dict, description="Detailed verification results")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('confidence must be between 0.0 and 1.0')
        return v


class VerificationCompleteEvent(Event):
    """Event when all verifications are complete."""
    message_type: str = Field(default="verification_complete", description="Message type")
    meme_id: str = Field(..., description="Meme identifier")
    all_passed: bool = Field(..., description="Whether all verifications passed")
    verification_results: Dict[str, Dict[str, Any]] = Field(..., description="Complete verification results")
    overall_confidence: float = Field(..., description="Overall confidence score")
    
    @validator('meme_id')
    def validate_meme_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('meme_id cannot be empty')
        return v.strip()
    
    @validator('overall_confidence')
    def validate_confidence(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('overall_confidence must be between 0.0 and 1.0')
        return v


# Scoring Messages
class ScoreMemeRequest(Request):
    """Request to score a meme."""
    message_type: str = Field(default="score_meme", description="Message type")
    meme_id: str = Field(..., description="Meme identifier")
    content: str = Field(..., description="Meme content to score")
    image_url: Optional[str] = Field(None, description="URL to meme image")
    verification_results: Dict[str, Any] = Field(default_factory=dict, description="Previous verification results")
    scoring_criteria: List[str] = Field(default_factory=list, description="Specific criteria to evaluate")
    
    @validator('meme_id', 'content')
    def validate_non_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Field cannot be empty')
        return v.strip()


class ScoreMemeResponse(Response):
    """Response from meme scoring."""
    message_type: str = Field(default="score_meme_response", description="Message type")
    overall_score: float = Field(..., description="Overall quality score (0.0 to 10.0)")
    score_breakdown: Dict[str, float] = Field(..., description="Detailed score breakdown")
    feedback: Optional[str] = Field(None, description="Human-readable feedback")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    confidence: float = Field(default=1.0, description="Confidence in scoring")
    
    @validator('overall_score')
    def validate_overall_score(cls, v):
        if not (0.0 <= v <= 10.0):
            raise ValueError('overall_score must be between 0.0 and 10.0')
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('confidence must be between 0.0 and 1.0')
        return v


# Refinement Messages
class RefineMemeRequest(Request):
    """Request to refine a meme."""
    message_type: str = Field(default="refine_meme", description="Message type")
    meme_id: str = Field(..., description="Meme identifier")
    original_content: str = Field(..., description="Original meme content")
    issues: List[str] = Field(..., description="Issues to address")
    current_score: Optional[float] = Field(None, description="Current quality score")
    target_score: float = Field(default=7.0, description="Target quality score")
    refinement_strategy: str = Field(default="iterative", description="Refinement approach")
    max_attempts: int = Field(default=3, description="Maximum refinement attempts")
    
    @validator('meme_id', 'original_content')
    def validate_non_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Field cannot be empty')
        return v.strip()
    
    @validator('issues')
    def validate_issues(cls, v):
        if not v:
            raise ValueError('At least one issue must be specified')
        return v
    
    @validator('target_score')
    def validate_target_score(cls, v):
        if not (0.0 <= v <= 10.0):
            raise ValueError('target_score must be between 0.0 and 10.0')
        return v
    
    @validator('max_attempts')
    def validate_max_attempts(cls, v):
        if v < 1 or v > 10:
            raise ValueError('max_attempts must be between 1 and 10')
        return v


class RefineMemeResponse(Response):
    """Response from meme refinement."""
    message_type: str = Field(default="refine_meme_response", description="Message type")
    refined_content: str = Field(..., description="Refined meme content")
    changes_made: List[str] = Field(..., description="List of changes made")
    new_score: Optional[float] = Field(None, description="New quality score after refinement")
    improvement_delta: Optional[float] = Field(None, description="Score improvement amount")
    refinement_successful: bool = Field(..., description="Whether refinement was successful")
    refinement_details: Dict[str, Any] = Field(default_factory=dict, description="Detailed refinement information")
    
    @validator('refined_content')
    def validate_refined_content(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('refined_content cannot be empty')
        return v.strip()
    
    @validator('changes_made')
    def validate_changes_made(cls, v):
        if not v:
            raise ValueError('At least one change must be documented')
        return v
    
    @validator('new_score')
    def validate_new_score(cls, v):
        if v is not None and not (0.0 <= v <= 10.0):
            raise ValueError('new_score must be between 0.0 and 10.0')
        return v


# Router Messages
class RouteRequest(Request):
    """Request to route a meme generation request."""
    message_type: str = Field(default="route_request", description="Message type")
    user_request: str = Field(..., description="Original user request")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    routing_preferences: Dict[str, Any] = Field(default_factory=dict, description="Routing preferences")
    
    @validator('user_request')
    def validate_user_request(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('user_request cannot be empty')
        return v.strip()


class RouteResponse(Response):
    """Response from routing."""
    message_type: str = Field(default="route_response", description="Message type")
    topic: str = Field(..., description="Determined topic")
    format_type: str = Field(..., description="Recommended format")
    verification_needs: Dict[str, bool] = Field(..., description="Required verifications")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Generation constraints")
    generation_approach: str = Field(..., description="Recommended generation approach")
    confidence: float = Field(default=1.0, description="Confidence in routing decision")
    
    @validator('topic', 'format_type', 'generation_approach')
    def validate_non_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Field cannot be empty')
        return v.strip()
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('confidence must be between 0.0 and 1.0')
        return v


# Storage Messages
class StoreMemeRequest(Request):
    """Request to store a meme."""
    message_type: str = Field(default="store_meme", description="Message type")
    meme_id: str = Field(..., description="Meme identifier")
    content: Dict[str, Any] = Field(..., description="Meme content to store")
    user_id: str = Field(..., description="User identifier")
    storage_options: Dict[str, Any] = Field(default_factory=dict, description="Storage configuration")
    
    @validator('meme_id', 'user_id')
    def validate_non_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Field cannot be empty')
        return v.strip()
    
    @validator('content')
    def validate_content(cls, v):
        if not v:
            raise ValueError('content cannot be empty')
        return v


class StoreMemeResponse(Response):
    """Response from meme storage."""
    message_type: str = Field(default="store_meme_response", description="Message type")
    stored: bool = Field(..., description="Whether storage was successful")
    storage_url: Optional[str] = Field(None, description="URL where meme is stored")
    storage_id: Optional[str] = Field(None, description="Storage system identifier")
    storage_metadata: Dict[str, Any] = Field(default_factory=dict, description="Storage metadata")


class RetrieveMemeRequest(Request):
    """Request to retrieve a meme."""
    message_type: str = Field(default="retrieve_meme", description="Message type")
    meme_id: str = Field(..., description="Meme identifier")
    retrieval_options: Dict[str, Any] = Field(default_factory=dict, description="Retrieval options")
    
    @validator('meme_id')
    def validate_meme_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('meme_id cannot be empty')
        return v.strip()


class RetrieveMemeResponse(Response):
    """Response from meme retrieval."""
    message_type: str = Field(default="retrieve_meme_response", description="Message type")
    found: bool = Field(..., description="Whether meme was found")
    content: Optional[Dict[str, Any]] = Field(None, description="Retrieved meme content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Meme metadata")


# Cache Messages
class CacheGetRequest(Request):
    """Request to get from cache."""
    message_type: str = Field(default="cache_get", description="Message type")
    key: str = Field(..., description="Cache key")
    cache_type: str = Field(default="default", description="Cache type/namespace")
    
    @validator('key')
    def validate_key(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('key cannot be empty')
        return v.strip()


class CacheGetResponse(Response):
    """Response from cache get."""
    message_type: str = Field(default="cache_get_response", description="Message type")
    found: bool = Field(..., description="Whether key was found in cache")
    value: Optional[Any] = Field(None, description="Cached value")
    ttl_seconds: Optional[int] = Field(None, description="Time to live in seconds")
    cache_metadata: Dict[str, Any] = Field(default_factory=dict, description="Cache metadata")


class CacheSetRequest(Request):
    """Request to set in cache."""
    message_type: str = Field(default="cache_set", description="Message type")
    key: str = Field(..., description="Cache key")
    value: Any = Field(..., description="Value to cache")
    ttl_seconds: Optional[int] = Field(None, description="Time to live in seconds")
    cache_type: str = Field(default="default", description="Cache type/namespace")
    
    @validator('key')
    def validate_key(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('key cannot be empty')
        return v.strip()
    
    @validator('ttl_seconds')
    def validate_ttl(cls, v):
        if v is not None and v <= 0:
            raise ValueError('ttl_seconds must be positive')
        return v


class CacheSetResponse(Response):
    """Response from cache set."""
    message_type: str = Field(default="cache_set_response", description="Message type")
    stored: bool = Field(..., description="Whether value was stored successfully")
    cache_metadata: Dict[str, Any] = Field(default_factory=dict, description="Cache metadata")


# Metrics Messages
class CollectMetricsRequest(Request):
    """Request to collect metrics."""
    message_type: str = Field(default="collect_metrics", description="Message type")
    metric_types: List[str] = Field(..., description="Types of metrics to collect")
    collection_config: Dict[str, Any] = Field(default_factory=dict, description="Collection configuration")
    
    @validator('metric_types')
    def validate_metric_types(cls, v):
        if not v:
            raise ValueError('At least one metric type must be specified')
        return v


class CollectMetricsResponse(Response):
    """Response with collected metrics."""
    message_type: str = Field(default="collect_metrics_response", description="Message type")
    metrics: Dict[str, Any] = Field(..., description="Collected metrics data")
    collection_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When metrics were collected")
    collection_duration_ms: Optional[float] = Field(None, description="Time taken to collect metrics")


class MetricUpdateEvent(Event):
    """Event for metric updates."""
    message_type: str = Field(default="metric_update", description="Message type")
    metric_name: str = Field(..., description="Name of the metric")
    metric_value: float = Field(..., description="Metric value")
    metric_tags: Dict[str, str] = Field(default_factory=dict, description="Metric tags")
    metric_type: str = Field(default="gauge", description="Type of metric")
    
    @validator('metric_name')
    def validate_metric_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('metric_name cannot be empty')
        return v.strip()


class MessageValidator:
    """Validator for actor message schemas."""
    
    # Mapping of message types to their corresponding Pydantic models
    MESSAGE_TYPE_MODELS = {
        # System messages
        "ping": PingMessage,
        "pong": PongMessage,
        "terminate": TerminateMessage,
        "restart": RestartMessage,
        
        # Meme generation messages
        "generate_meme": GenerateMemeRequest,
        "generate_meme_response": GenerateMemeResponse,
        "meme_generation_progress": MemeGenerationProgressEvent,
        
        # Verification messages
        "verify_content": VerifyContentRequest,
        "verify_content_response": VerifyContentResponse,
        "verification_complete": VerificationCompleteEvent,
        
        # Scoring messages
        "score_meme": ScoreMemeRequest,
        "score_meme_response": ScoreMemeResponse,
        
        # Refinement messages
        "refine_meme": RefineMemeRequest,
        "refine_meme_response": RefineMemeResponse,
        
        # Router messages
        "route_request": RouteRequest,
        "route_response": RouteResponse,
        
        # Storage messages
        "store_meme": StoreMemeRequest,
        "store_meme_response": StoreMemeResponse,
        "retrieve_meme": RetrieveMemeRequest,
        "retrieve_meme_response": RetrieveMemeResponse,
        
        # Cache messages
        "cache_get": CacheGetRequest,
        "cache_get_response": CacheGetResponse,
        "cache_set": CacheSetRequest,
        "cache_set_response": CacheSetResponse,
        
        # Metrics messages
        "collect_metrics": CollectMetricsRequest,
        "collect_metrics_response": CollectMetricsResponse,
        "metric_update": MetricUpdateEvent,
    }
    
    @classmethod
    def validate_message(cls, message_data: Dict[str, Any]) -> BaseMessage:
        """
        Validate message data against the appropriate schema.
        
        Args:
            message_data: Raw message data dictionary
        
        Returns:
            Validated message instance
        
        Raises:
            ValueError: If validation fails
        """
        if 'message_type' not in message_data:
            raise ValueError("message_type is required")
        
        message_type = message_data['message_type']
        model_class = cls.MESSAGE_TYPE_MODELS.get(message_type, BaseMessage)
        
        try:
            return model_class(**message_data)
        except Exception as e:
            raise ValueError(f"Message validation failed for {message_type}: {str(e)}")
    
    @classmethod
    def validate_message_json(cls, message_json: str) -> BaseMessage:
        """
        Validate message JSON string against the appropriate schema.
        
        Args:
            message_json: JSON string containing message data
        
        Returns:
            Validated message instance
        
        Raises:
            ValueError: If validation fails
        """
        try:
            message_data = json.loads(message_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {str(e)}")
        
        return cls.validate_message(message_data)
    
    @classmethod
    def get_schema(cls, message_type: str) -> Dict[str, Any]:
        """
        Get JSON schema for a specific message type.
        
        Args:
            message_type: Type of message
        
        Returns:
            JSON schema dictionary
        """
        model_class = cls.MESSAGE_TYPE_MODELS.get(message_type, BaseMessage)
        return model_class.schema()
    
    @classmethod
    def list_message_types(cls) -> List[str]:
        """
        List all supported message types.
        
        Returns:
            List of message type strings
        """
        return list(cls.MESSAGE_TYPE_MODELS.keys())


# Export public API
__all__ = [
    "MessagePriority",
    "MessageStatus",
    "BaseMessage",
    "Request",
    "Response",
    "Event",
    "PingMessage",
    "PongMessage",
    "TerminateMessage",
    "RestartMessage",
    "GenerateMemeRequest",
    "GenerateMemeResponse",
    "MemeGenerationProgressEvent",
    "VerifyContentRequest",
    "VerifyContentResponse",
    "VerificationCompleteEvent",
    "ScoreMemeRequest",
    "ScoreMemeResponse",
    "RefineMemeRequest",
    "RefineMemeResponse",
    "RouteRequest",
    "RouteResponse",
    "StoreMemeRequest",
    "StoreMemeResponse",
    "RetrieveMemeRequest",
    "RetrieveMemeResponse",
    "CacheGetRequest",
    "CacheGetResponse",
    "CacheSetRequest",
    "CacheSetResponse",
    "CollectMetricsRequest",
    "CollectMetricsResponse",
    "MetricUpdateEvent",
    "MessageValidator"
]