"""Template-specific domain events."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from uuid import UUID

from .base import DomainEvent, EventRegistry, EventType


@EventRegistry.register
@dataclass
class TemplateCreated(DomainEvent):
    """Event emitted when a new meme template is created."""
    
    template_id: UUID
    name: str
    description: str
    image_url: str
    text_areas: List[Dict[str, Any]]
    tags: List[str]
    created_by: str
    
    @classmethod
    def get_event_type(cls) -> str:
        return EventType.TEMPLATE_CREATED.value


@EventRegistry.register
@dataclass
class TemplateUpdated(DomainEvent):
    """Event emitted when a template is updated."""
    
    template_id: UUID
    changes: Dict[str, Any]
    updated_by: str
    update_reason: Optional[str] = None
    
    @classmethod
    def get_event_type(cls) -> str:
        return EventType.TEMPLATE_UPDATED.value


@EventRegistry.register
@dataclass
class TemplateDeleted(DomainEvent):
    """Event emitted when a template is deleted."""
    
    template_id: UUID
    deletion_reason: str
    deleted_by: str
    
    @classmethod
    def get_event_type(cls) -> str:
        return EventType.TEMPLATE_DELETED.value


@EventRegistry.register
@dataclass
class TemplatePopularityUpdated(DomainEvent):
    """Event emitted when template popularity changes."""
    
    template_id: UUID
    old_popularity: float
    new_popularity: float
    usage_count: int
    success_rate: float
    
    @classmethod
    def get_event_type(cls) -> str:
        return EventType.TEMPLATE_POPULARITY_UPDATED.value