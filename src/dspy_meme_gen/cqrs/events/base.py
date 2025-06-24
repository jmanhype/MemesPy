"""Base event definitions for event sourcing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, TypeVar, Generic, Union
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

    metadata: EventMetadata = field(default_factory=EventMetadata)

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
