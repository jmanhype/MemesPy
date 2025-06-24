"""Event definitions for meme generation pipeline.

This module provides all domain events for the CQRS event sourcing system.
All events are defined here to avoid circular import issues.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from uuid import UUID, uuid4
import json
from enum import Enum


class EventType(Enum):
    """Enumeration of all event types in the system."""

    # Meme Events
    MEME_GENERATION_REQUESTED = "meme.generation.requested"
    MEME_GENERATED = "meme.generated"
    MEME_GENERATION_FAILED = "meme.generation.failed"
    MEME_SCORED = "meme.scored"
    MEME_REFINED = "meme.refined"
    MEME_APPROVED = "meme.approved"
    MEME_REJECTED = "meme.rejected"
    MEME_VIEWED = "meme.viewed"
    MEME_SHARED = "meme.shared"
    MEME_DELETED = "meme.deleted"

    # Template Events
    TEMPLATE_CREATED = "template.created"
    TEMPLATE_UPDATED = "template.updated"
    TEMPLATE_DELETED = "template.deleted"
    TEMPLATE_POPULARITY_UPDATED = "template.popularity.updated"

    # Trend Events
    TREND_DISCOVERED = "trend.discovered"
    TREND_UPDATED = "trend.updated"
    TREND_EXPIRED = "trend.expired"

    # Verification Events
    VERIFICATION_REQUESTED = "verification.requested"
    VERIFICATION_COMPLETED = "verification.completed"
    VERIFICATION_FAILED = "verification.failed"

    # System Events
    SNAPSHOT_CREATED = "system.snapshot.created"
    PROJECTION_REBUILT = "system.projection.rebuilt"
    MIGRATION_STARTED = "system.migration.started"
    MIGRATION_COMPLETED = "system.migration.completed"


@dataclass
class EventMetadata:
    """Metadata attached to every event."""

    event_id: UUID = field(default_factory=uuid4)
    event_type: str = ""
    aggregate_id: UUID = field(default_factory=uuid4)
    aggregate_type: str = ""
    aggregate_version: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    correlation_id: Optional[UUID] = None
    causation_id: Optional[UUID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainEvent(ABC):
    """Base class for all domain events."""

    metadata: EventMetadata = field(default_factory=EventMetadata, init=False)

    def __post_init__(self):
        """Initialize event metadata after dataclass creation."""
        if not self.metadata.event_type:
            self.metadata.event_type = self.get_event_type()

    @classmethod
    @abstractmethod
    def get_event_type(cls) -> str:
        """Return the event type string."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for storage."""
        # Get all fields except metadata
        data_fields = {}
        for field_info in self.__dataclass_fields__.values():
            if field_info.name != "metadata":
                value = getattr(self, field_info.name)
                # Convert UUIDs to strings
                if isinstance(value, UUID):
                    data_fields[field_info.name] = str(value)
                elif isinstance(value, datetime):
                    data_fields[field_info.name] = value.isoformat()
                else:
                    data_fields[field_info.name] = value

        return {
            "metadata": {
                "event_id": str(self.metadata.event_id),
                "event_type": self.metadata.event_type,
                "aggregate_id": str(self.metadata.aggregate_id),
                "aggregate_type": self.metadata.aggregate_type,
                "aggregate_version": self.metadata.aggregate_version,
                "timestamp": self.metadata.timestamp.isoformat(),
                "user_id": self.metadata.user_id,
                "correlation_id": (
                    str(self.metadata.correlation_id) if self.metadata.correlation_id else None
                ),
                "causation_id": (
                    str(self.metadata.causation_id) if self.metadata.causation_id else None
                ),
                "metadata": self.metadata.metadata,
            },
            "data": data_fields,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainEvent":
        """Reconstruct event from dictionary."""
        # Convert string UUIDs back to UUID objects
        event_data = data["data"].copy()
        for field_name, field_info in cls.__dataclass_fields__.items():
            if field_name in event_data:
                field_type = field_info.type
                value = event_data[field_name]

                # Handle UUID conversion
                if field_type == UUID and isinstance(value, str):
                    event_data[field_name] = UUID(value)
                # Handle Optional[UUID] conversion
                elif hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                    args = field_type.__args__
                    if UUID in args and isinstance(value, str):
                        event_data[field_name] = UUID(value)
                # Handle datetime conversion
                elif field_type == datetime and isinstance(value, str):
                    event_data[field_name] = datetime.fromisoformat(value)

        # Create event instance
        event = cls(**event_data)

        # Reconstruct metadata
        meta_data = data["metadata"]
        event.metadata = EventMetadata(
            event_id=UUID(meta_data["event_id"]),
            event_type=meta_data["event_type"],
            aggregate_id=UUID(meta_data["aggregate_id"]),
            aggregate_type=meta_data["aggregate_type"],
            aggregate_version=meta_data["aggregate_version"],
            timestamp=datetime.fromisoformat(meta_data["timestamp"]),
            user_id=meta_data.get("user_id"),
            correlation_id=(
                UUID(meta_data["correlation_id"]) if meta_data.get("correlation_id") else None
            ),
            causation_id=UUID(meta_data["causation_id"]) if meta_data.get("causation_id") else None,
            metadata=meta_data.get("metadata", {}),
        )

        return event

    def with_metadata(self, **kwargs) -> "DomainEvent":
        """Update event metadata."""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
        return self

    def to_json(self) -> str:
        """Serialize event to JSON."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "DomainEvent":
        """Deserialize event from JSON."""
        return cls.from_dict(json.loads(json_str))


class EventRegistry:
    """Registry for mapping event types to classes."""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, event_class: type):
        """Register an event class decorator."""
        if hasattr(event_class, "get_event_type"):
            event_type = event_class.get_event_type()
            cls._registry[event_type] = event_class
        return event_class

    @classmethod
    def get_class(cls, event_type: str) -> Optional[type]:
        """Get event class by type."""
        return cls._registry.get(event_type)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional[DomainEvent]:
        """Create event instance from dictionary."""
        event_type = data["metadata"]["event_type"]
        event_class = cls.get_class(event_type)
        if event_class:
            return event_class.from_dict(data)
        return None

    @classmethod
    def list_registered_events(cls) -> Dict[str, type]:
        """List all registered event types."""
        return cls._registry.copy()


# ============================================================================
# MEME EVENTS
# ============================================================================


@EventRegistry.register
@dataclass
class MemeGenerationRequested(DomainEvent):
    """Event emitted when meme generation is requested."""

    request_id: UUID = field(default_factory=lambda: uuid4())
    topic: str = ""
    format: str = ""
    style: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.event_type:
            self.event_type = "meme.generation.requested"

    @classmethod
    def get_event_type(cls) -> str:
        return "meme.generation.requested"


@EventRegistry.register
@dataclass
class MemeGenerated(DomainEvent):
    """Event emitted when a meme is successfully generated."""

    meme_id: UUID
    request_id: UUID
    topic: str
    format: str
    text: str
    image_url: str
    template_id: Optional[UUID] = None
    generation_time_ms: int = 0
    model_used: str = ""

    @classmethod
    def get_event_type(cls) -> str:
        return EventType.MEME_GENERATED.value


@EventRegistry.register
@dataclass
class MemeGenerationFailed(DomainEvent):
    """Event emitted when meme generation fails."""

    request_id: UUID
    error_code: str
    error_message: str
    retry_count: int = 0

    @classmethod
    def get_event_type(cls) -> str:
        return EventType.MEME_GENERATION_FAILED.value


@EventRegistry.register
@dataclass
class MemeScored(DomainEvent):
    """Event emitted when a meme is scored."""

    meme_id: UUID
    score: float
    humor_score: float
    relevance_score: float
    appropriateness_score: float
    scoring_model: str
    scorer_agent_id: str

    @classmethod
    def get_event_type(cls) -> str:
        return EventType.MEME_SCORED.value


@EventRegistry.register
@dataclass
class MemeRefined(DomainEvent):
    """Event emitted when a meme is refined."""

    original_meme_id: UUID
    refined_meme_id: UUID
    refinement_reason: str
    changes_made: List[str]
    refinement_iteration: int

    @classmethod
    def get_event_type(cls) -> str:
        return EventType.MEME_REFINED.value


@EventRegistry.register
@dataclass
class MemeApproved(DomainEvent):
    """Event emitted when a meme passes verification."""

    meme_id: UUID
    verification_scores: Dict[str, float]
    approval_timestamp: datetime
    approver_agent_id: str

    @classmethod
    def get_event_type(cls) -> str:
        return EventType.MEME_APPROVED.value


@EventRegistry.register
@dataclass
class MemeRejected(DomainEvent):
    """Event emitted when a meme fails verification."""

    meme_id: UUID
    rejection_reasons: List[str]
    violation_categories: List[str]
    rejection_timestamp: datetime
    rejector_agent_id: str

    @classmethod
    def get_event_type(cls) -> str:
        return EventType.MEME_REJECTED.value


@EventRegistry.register
@dataclass
class MemeViewed(DomainEvent):
    """Event emitted when a meme is viewed."""

    meme_id: UUID
    viewer_id: Optional[str] = None
    view_duration_ms: Optional[int] = None
    view_source: str = "web"

    @classmethod
    def get_event_type(cls) -> str:
        return EventType.MEME_VIEWED.value


@EventRegistry.register
@dataclass
class MemeShared(DomainEvent):
    """Event emitted when a meme is shared."""

    meme_id: UUID
    sharer_id: Optional[str] = None
    share_platform: str = ""
    share_method: str = ""

    @classmethod
    def get_event_type(cls) -> str:
        return EventType.MEME_SHARED.value


@EventRegistry.register
@dataclass
class MemeDeleted(DomainEvent):
    """Event emitted when a meme is deleted."""

    meme_id: UUID
    deletion_reason: str
    deleted_by: str
    soft_delete: bool = True

    @classmethod
    def get_event_type(cls) -> str:
        return EventType.MEME_DELETED.value


# ============================================================================
# PIPELINE EVENTS
# ============================================================================


@EventRegistry.register
@dataclass
class TextGenerated(DomainEvent):
    """Event emitted when meme text is generated."""

    meme_id: UUID
    text: str
    generation_method: str
    model_used: str
    prompt_used: str
    generation_time_ms: int
    confidence_score: float
    alternatives_considered: List[str] = field(default_factory=list)

    @classmethod
    def get_event_type(cls) -> str:
        return "meme.text.generated"


@EventRegistry.register
@dataclass
class ImageGenerated(DomainEvent):
    """Event emitted when meme image is generated."""

    meme_id: UUID
    image_url: str
    image_type: str
    template_used: Optional[str] = None
    generation_time_ms: int = 0
    dimensions: Dict[str, int] = field(default_factory=lambda: {"width": 0, "height": 0})
    file_size_bytes: int = 0

    @classmethod
    def get_event_type(cls) -> str:
        return "meme.image.generated"


@EventRegistry.register
@dataclass
class QualityScored(DomainEvent):
    """Event emitted when meme quality is scored."""

    meme_id: UUID
    overall_score: float
    humor_score: float
    relevance_score: float
    originality_score: float
    visual_appeal_score: float
    appropriateness_score: float
    scoring_agent: str
    scoring_model: str
    criteria_details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def get_event_type(cls) -> str:
        return "meme.quality.scored"


@EventRegistry.register
@dataclass
class MemeCompleted(DomainEvent):
    """Event emitted when meme generation pipeline is completed."""

    meme_id: UUID
    request_id: UUID
    final_score: float
    status: str  # "approved", "rejected", "needs_refinement"
    completion_time_ms: int
    pipeline_stages: List[str]
    final_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def get_event_type(cls) -> str:
        return "meme.completed"


@EventRegistry.register
@dataclass
class MemeGenerationStarted(DomainEvent):
    """Event emitted when meme generation pipeline starts."""

    request_id: UUID
    topic: str
    format: str
    parameters: Dict[str, Any]
    pipeline_config: Dict[str, Any]
    expected_stages: List[str]
    priority: int = 0

    @classmethod
    def get_event_type(cls) -> str:
        return "meme.generation.started"


@EventRegistry.register
@dataclass
class VerificationRequested(DomainEvent):
    """Event emitted when meme verification is requested."""

    meme_id: UUID
    verification_type: str  # "content", "appropriateness", "factuality"
    content_to_verify: Dict[str, Any]
    verification_criteria: List[str]
    priority: int = 0

    @classmethod
    def get_event_type(cls) -> str:
        return "meme.verification.requested"


@EventRegistry.register
@dataclass
class VerificationCompleted(DomainEvent):
    """Event emitted when meme verification is completed."""

    meme_id: UUID
    verification_type: str
    verification_result: str  # "passed", "failed", "warning"
    confidence_score: float
    details: Dict[str, Any]
    verification_agent: str
    flags_raised: List[str] = field(default_factory=list)

    @classmethod
    def get_event_type(cls) -> str:
        return "meme.verification.completed"


@EventRegistry.register
@dataclass
class RefinementRequested(DomainEvent):
    """Event emitted when meme refinement is requested."""

    original_meme_id: UUID
    refinement_reason: str
    current_score: float
    target_score: float
    refinement_suggestions: List[str]
    refinement_type: str  # "text", "image", "both"
    iteration_count: int = 0

    @classmethod
    def get_event_type(cls) -> str:
        return "meme.refinement.requested"


@EventRegistry.register
@dataclass
class ActorTaskStarted(DomainEvent):
    """Event emitted when an actor starts a task."""

    actor_id: str
    task_type: str
    task_id: UUID
    input_data: Dict[str, Any]
    estimated_duration_ms: int

    @classmethod
    def get_event_type(cls) -> str:
        return "actor.task.started"


@EventRegistry.register
@dataclass
class ActorTaskCompleted(DomainEvent):
    """Event emitted when an actor completes a task."""

    actor_id: str
    task_type: str
    task_id: UUID
    output_data: Dict[str, Any]
    actual_duration_ms: int
    success: bool
    error_message: Optional[str] = None

    @classmethod
    def get_event_type(cls) -> str:
        return "actor.task.completed"


@EventRegistry.register
@dataclass
class PipelineStageStarted(DomainEvent):
    """Event emitted when a pipeline stage starts."""

    pipeline_id: UUID
    stage_name: str
    stage_order: int
    input_data: Dict[str, Any]
    expected_outputs: List[str]

    @classmethod
    def get_event_type(cls) -> str:
        return "pipeline.stage.started"


@EventRegistry.register
@dataclass
class PipelineStageCompleted(DomainEvent):
    """Event emitted when a pipeline stage completes."""

    pipeline_id: UUID
    stage_name: str
    stage_order: int
    output_data: Dict[str, Any]
    success: bool
    duration_ms: int
    next_stage: Optional[str] = None

    @classmethod
    def get_event_type(cls) -> str:
        return "pipeline.stage.completed"


# Export all event classes for easy importing
__all__ = [
    # Base classes
    "DomainEvent",
    "EventMetadata",
    "EventRegistry",
    "EventType",
    # Meme events
    "MemeGenerationRequested",
    "MemeGenerated",
    "MemeGenerationFailed",
    "MemeScored",
    "MemeRefined",
    "MemeApproved",
    "MemeRejected",
    "MemeViewed",
    "MemeShared",
    "MemeDeleted",
    # Pipeline events
    "TextGenerated",
    "ImageGenerated",
    "QualityScored",
    "MemeCompleted",
    "MemeGenerationStarted",
    "VerificationRequested",
    "VerificationCompleted",
    "RefinementRequested",
    # Actor events
    "ActorTaskStarted",
    "ActorTaskCompleted",
    # Pipeline stage events
    "PipelineStageStarted",
    "PipelineStageCompleted",
]
