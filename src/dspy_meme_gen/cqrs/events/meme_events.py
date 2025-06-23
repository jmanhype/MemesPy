"""Meme-specific domain events."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

from .base import DomainEvent, EventRegistry, EventType


@EventRegistry.register
@dataclass
class MemeGenerationRequested(DomainEvent):
    """Event emitted when meme generation is requested."""
    
    request_id: UUID
    topic: str
    format: str
    style: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def get_event_type(cls) -> str:
        return EventType.MEME_GENERATION_REQUESTED.value


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
    share_platform: str
    share_method: str
    
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