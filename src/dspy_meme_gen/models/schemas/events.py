"""
JSON schema validation for events in the MemesPy system.
Provides comprehensive validation for all event types with proper schema definitions.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator
from pydantic.json import pydantic_encoder


class EventType(str, Enum):
    """Event type enumeration."""
    # Lifecycle events
    ACTOR_STARTED = "actor.started"
    ACTOR_STOPPED = "actor.stopped"
    ACTOR_CRASHED = "actor.crashed"
    ACTOR_RESTARTED = "actor.restarted"
    
    # Message events
    MESSAGE_SENT = "message.sent"
    MESSAGE_RECEIVED = "message.received"
    MESSAGE_PROCESSED = "message.processed"
    MESSAGE_FAILED = "message.failed"
    
    # Pipeline events
    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_STAGE_COMPLETED = "pipeline.stage.completed"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"
    
    # Meme generation events
    MEME_GENERATION_STARTED = "meme.generation.started"
    MEME_GENERATION_PROGRESS = "meme.generation.progress"
    MEME_GENERATION_COMPLETED = "meme.generation.completed"
    MEME_GENERATION_FAILED = "meme.generation.failed"
    MEME_VERIFICATION_COMPLETED = "meme.verification.completed"
    MEME_SCORING_COMPLETED = "meme.scoring.completed"
    MEME_REFINEMENT_STARTED = "meme.refinement.started"
    
    # System events
    SYSTEM_HEALTH_CHECK = "system.health_check"
    SYSTEM_METRICS_COLLECTED = "system.metrics.collected"
    SYSTEM_CACHE_CLEARED = "system.cache.cleared"
    SYSTEM_ERROR = "system.error"
    
    # External service events
    EXTERNAL_API_CALLED = "external.api.called"
    EXTERNAL_API_FAILED = "external.api.failed"
    EXTERNAL_API_TIMEOUT = "external.api.timeout"


class EventSeverity(str, Enum):
    """Event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class BaseEvent(BaseModel):
    """Base event model with common fields."""
    
    event_id: str = Field(..., description="Unique event identifier")
    event_type: EventType = Field(..., description="Type of event")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    severity: EventSeverity = Field(default=EventSeverity.INFO, description="Event severity")
    source: str = Field(..., description="Source component/actor that generated the event")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")
    trace_id: Optional[str] = Field(None, description="OpenTelemetry trace ID")
    span_id: Optional[str] = Field(None, description="OpenTelemetry span ID")
    tags: Dict[str, str] = Field(default_factory=dict, description="Additional tags for the event")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('event_id')
    def validate_event_id(cls, v):
        if not v or len(v) < 8:
            raise ValueError('event_id must be at least 8 characters long')
        return v
    
    @validator('source')
    def validate_source(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('source cannot be empty')
        return v.strip()


class ActorLifecycleEvent(BaseEvent):
    """Event for actor lifecycle changes."""
    
    actor_name: str = Field(..., description="Name of the actor")
    actor_type: str = Field(..., description="Type/class of the actor")
    previous_state: Optional[str] = Field(None, description="Previous actor state")
    current_state: str = Field(..., description="Current actor state")
    reason: Optional[str] = Field(None, description="Reason for state change")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional actor metadata")
    
    @validator('actor_name', 'current_state')
    def validate_non_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError(f'{v} cannot be empty')
        return v.strip()


class MessageEvent(BaseEvent):
    """Event for message-related operations."""
    
    message_id: str = Field(..., description="Unique message identifier")
    message_type: str = Field(..., description="Type of message")
    sender: str = Field(..., description="Message sender")
    receiver: str = Field(..., description="Message receiver")
    message_size: Optional[int] = Field(None, description="Message size in bytes")
    processing_duration_ms: Optional[float] = Field(None, description="Processing duration in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Message metadata")
    
    @validator('message_id', 'message_type', 'sender', 'receiver')
    def validate_non_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Field cannot be empty')
        return v.strip()
    
    @validator('processing_duration_ms')
    def validate_positive_duration(cls, v):
        if v is not None and v < 0:
            raise ValueError('processing_duration_ms must be non-negative')
        return v


class PipelineEvent(BaseEvent):
    """Event for pipeline execution."""
    
    pipeline_name: str = Field(..., description="Name of the pipeline")
    pipeline_id: str = Field(..., description="Unique pipeline execution ID")
    stage: Optional[str] = Field(None, description="Current pipeline stage")
    progress: Optional[float] = Field(None, description="Pipeline progress (0.0 to 1.0)")
    total_stages: Optional[int] = Field(None, description="Total number of stages")
    completed_stages: Optional[int] = Field(None, description="Number of completed stages")
    execution_duration_ms: Optional[float] = Field(None, description="Execution duration in milliseconds")
    error_details: Optional[str] = Field(None, description="Error details if pipeline failed")
    stage_results: Dict[str, Any] = Field(default_factory=dict, description="Results from individual stages")
    
    @validator('pipeline_name', 'pipeline_id')
    def validate_non_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Field cannot be empty')
        return v.strip()
    
    @validator('progress')
    def validate_progress_range(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError('progress must be between 0.0 and 1.0')
        return v
    
    @validator('total_stages', 'completed_stages')
    def validate_positive_int(cls, v):
        if v is not None and v < 0:
            raise ValueError('Stage counts must be non-negative')
        return v


class MemeGenerationEvent(BaseEvent):
    """Event for meme generation operations."""
    
    meme_id: str = Field(..., description="Unique meme identifier")
    user_id: str = Field(..., description="User who requested the meme")
    topic: Optional[str] = Field(None, description="Meme topic")
    format_type: Optional[str] = Field(None, description="Meme format")
    generation_stage: Optional[str] = Field(None, description="Current generation stage")
    quality_score: Optional[float] = Field(None, description="Meme quality score")
    verification_results: Dict[str, bool] = Field(default_factory=dict, description="Verification results")
    refinement_count: Optional[int] = Field(None, description="Number of refinement iterations")
    generation_duration_ms: Optional[float] = Field(None, description="Generation duration in milliseconds")
    error_details: Optional[str] = Field(None, description="Error details if generation failed")
    prompt: Optional[str] = Field(None, description="Original generation prompt")
    
    @validator('meme_id', 'user_id')
    def validate_non_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Field cannot be empty')
        return v.strip()
    
    @validator('quality_score')
    def validate_score_range(cls, v):
        if v is not None and not (0.0 <= v <= 10.0):
            raise ValueError('quality_score must be between 0.0 and 10.0')
        return v
    
    @validator('refinement_count')
    def validate_non_negative(cls, v):
        if v is not None and v < 0:
            raise ValueError('refinement_count must be non-negative')
        return v


class SystemEvent(BaseEvent):
    """Event for system-level operations."""
    
    component: str = Field(..., description="System component name")
    operation: str = Field(..., description="Operation performed")
    resource_usage: Optional[Dict[str, float]] = Field(None, description="Resource usage metrics")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Configuration data")
    health_status: Optional[str] = Field(None, description="System health status")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    error_details: Optional[str] = Field(None, description="Error details if operation failed")
    
    @validator('component', 'operation')
    def validate_non_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Field cannot be empty')
        return v.strip()


class ExternalServiceEvent(BaseEvent):
    """Event for external service interactions."""
    
    service_name: str = Field(..., description="Name of external service")
    operation: str = Field(..., description="Operation performed")
    endpoint_url: Optional[str] = Field(None, description="Service endpoint URL")
    request_method: Optional[str] = Field(None, description="HTTP request method")
    response_status: Optional[int] = Field(None, description="HTTP response status code")
    request_duration_ms: Optional[float] = Field(None, description="Request duration in milliseconds")
    request_size_bytes: Optional[int] = Field(None, description="Request size in bytes")
    response_size_bytes: Optional[int] = Field(None, description="Response size in bytes")
    retry_count: Optional[int] = Field(None, description="Number of retries attempted")
    error_details: Optional[str] = Field(None, description="Error details if request failed")
    rate_limit_info: Optional[Dict[str, Any]] = Field(None, description="Rate limiting information")
    
    @validator('service_name', 'operation')
    def validate_non_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Field cannot be empty')
        return v.strip()
    
    @validator('response_status')
    def validate_http_status(cls, v):
        if v is not None and not (100 <= v <= 599):
            raise ValueError('response_status must be a valid HTTP status code')
        return v
    
    @validator('request_duration_ms', 'request_size_bytes', 'response_size_bytes')
    def validate_non_negative(cls, v):
        if v is not None and v < 0:
            raise ValueError('Size and duration values must be non-negative')
        return v


class EventValidator:
    """Validator for event schemas."""
    
    # Mapping of event types to their corresponding Pydantic models
    EVENT_TYPE_MODELS = {
        EventType.ACTOR_STARTED: ActorLifecycleEvent,
        EventType.ACTOR_STOPPED: ActorLifecycleEvent,
        EventType.ACTOR_CRASHED: ActorLifecycleEvent,
        EventType.ACTOR_RESTARTED: ActorLifecycleEvent,
        
        EventType.MESSAGE_SENT: MessageEvent,
        EventType.MESSAGE_RECEIVED: MessageEvent,
        EventType.MESSAGE_PROCESSED: MessageEvent,
        EventType.MESSAGE_FAILED: MessageEvent,
        
        EventType.PIPELINE_STARTED: PipelineEvent,
        EventType.PIPELINE_STAGE_COMPLETED: PipelineEvent,
        EventType.PIPELINE_COMPLETED: PipelineEvent,
        EventType.PIPELINE_FAILED: PipelineEvent,
        
        EventType.MEME_GENERATION_STARTED: MemeGenerationEvent,
        EventType.MEME_GENERATION_PROGRESS: MemeGenerationEvent,
        EventType.MEME_GENERATION_COMPLETED: MemeGenerationEvent,
        EventType.MEME_GENERATION_FAILED: MemeGenerationEvent,
        EventType.MEME_VERIFICATION_COMPLETED: MemeGenerationEvent,
        EventType.MEME_SCORING_COMPLETED: MemeGenerationEvent,
        EventType.MEME_REFINEMENT_STARTED: MemeGenerationEvent,
        
        EventType.SYSTEM_HEALTH_CHECK: SystemEvent,
        EventType.SYSTEM_METRICS_COLLECTED: SystemEvent,
        EventType.SYSTEM_CACHE_CLEARED: SystemEvent,
        EventType.SYSTEM_ERROR: SystemEvent,
        
        EventType.EXTERNAL_API_CALLED: ExternalServiceEvent,
        EventType.EXTERNAL_API_FAILED: ExternalServiceEvent,
        EventType.EXTERNAL_API_TIMEOUT: ExternalServiceEvent,
    }
    
    @classmethod
    def validate_event(cls, event_data: Dict[str, Any]) -> BaseEvent:
        """
        Validate event data against the appropriate schema.
        
        Args:
            event_data: Raw event data dictionary
        
        Returns:
            Validated event instance
        
        Raises:
            ValueError: If validation fails
        """
        if 'event_type' not in event_data:
            raise ValueError("event_type is required")
        
        event_type = EventType(event_data['event_type'])
        model_class = cls.EVENT_TYPE_MODELS.get(event_type, BaseEvent)
        
        try:
            return model_class(**event_data)
        except Exception as e:
            raise ValueError(f"Event validation failed for {event_type}: {str(e)}")
    
    @classmethod
    def validate_event_json(cls, event_json: str) -> BaseEvent:
        """
        Validate event JSON string against the appropriate schema.
        
        Args:
            event_json: JSON string containing event data
        
        Returns:
            Validated event instance
        
        Raises:
            ValueError: If validation fails
        """
        try:
            event_data = json.loads(event_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {str(e)}")
        
        return cls.validate_event(event_data)
    
    @classmethod
    def get_schema(cls, event_type: EventType) -> Dict[str, Any]:
        """
        Get JSON schema for a specific event type.
        
        Args:
            event_type: Type of event
        
        Returns:
            JSON schema dictionary
        """
        model_class = cls.EVENT_TYPE_MODELS.get(event_type, BaseEvent)
        return model_class.schema()
    
    @classmethod
    def list_event_types(cls) -> List[str]:
        """
        List all supported event types.
        
        Returns:
            List of event type strings
        """
        return [et.value for et in EventType]


class EventBuilder:
    """Builder for creating validated events."""
    
    def __init__(self, event_type: EventType, source: str):
        self.event_type = event_type
        self.source = source
        self.data = {
            'event_type': event_type,
            'source': source
        }
    
    def with_correlation_id(self, correlation_id: str) -> 'EventBuilder':
        """Add correlation ID to the event."""
        self.data['correlation_id'] = correlation_id
        return self
    
    def with_trace_context(self, trace_id: str, span_id: str) -> 'EventBuilder':
        """Add trace context to the event."""
        self.data['trace_id'] = trace_id
        self.data['span_id'] = span_id
        return self
    
    def with_severity(self, severity: EventSeverity) -> 'EventBuilder':
        """Set event severity."""
        self.data['severity'] = severity
        return self
    
    def with_tags(self, **tags) -> 'EventBuilder':
        """Add tags to the event."""
        if 'tags' not in self.data:
            self.data['tags'] = {}
        self.data['tags'].update(tags)
        return self
    
    def with_data(self, **kwargs) -> 'EventBuilder':
        """Add additional data to the event."""
        self.data.update(kwargs)
        return self
    
    def build(self) -> BaseEvent:
        """Build and validate the event."""
        import uuid
        
        # Add event_id if not provided
        if 'event_id' not in self.data:
            self.data['event_id'] = str(uuid.uuid4())
        
        return EventValidator.validate_event(self.data)


# Convenience functions for creating common events
def create_actor_lifecycle_event(
    event_type: EventType,
    actor_name: str,
    current_state: str,
    source: str,
    previous_state: Optional[str] = None,
    reason: Optional[str] = None,
    **kwargs
) -> ActorLifecycleEvent:
    """Create an actor lifecycle event."""
    builder = EventBuilder(event_type, source)
    return builder.with_data(
        actor_name=actor_name,
        actor_type=actor_name.split('_')[0] if '_' in actor_name else actor_name,
        current_state=current_state,
        previous_state=previous_state,
        reason=reason,
        **kwargs
    ).build()


def create_message_event(
    event_type: EventType,
    message_id: str,
    message_type: str,
    sender: str,
    receiver: str,
    source: str,
    **kwargs
) -> MessageEvent:
    """Create a message event."""
    builder = EventBuilder(event_type, source)
    return builder.with_data(
        message_id=message_id,
        message_type=message_type,
        sender=sender,
        receiver=receiver,
        **kwargs
    ).build()


def create_meme_generation_event(
    event_type: EventType,
    meme_id: str,
    user_id: str,
    source: str,
    **kwargs
) -> MemeGenerationEvent:
    """Create a meme generation event."""
    builder = EventBuilder(event_type, source)
    return builder.with_data(
        meme_id=meme_id,
        user_id=user_id,
        **kwargs
    ).build()


def create_system_event(
    event_type: EventType,
    component: str,
    operation: str,
    source: str,
    **kwargs
) -> SystemEvent:
    """Create a system event."""
    builder = EventBuilder(event_type, source)
    return builder.with_data(
        component=component,
        operation=operation,
        **kwargs
    ).build()


# Export public API
__all__ = [
    "EventType",
    "EventSeverity",
    "BaseEvent",
    "ActorLifecycleEvent",
    "MessageEvent",
    "PipelineEvent",
    "MemeGenerationEvent",
    "SystemEvent",
    "ExternalServiceEvent",
    "EventValidator",
    "EventBuilder",
    "create_actor_lifecycle_event",
    "create_message_event",
    "create_meme_generation_event",
    "create_system_event"
]