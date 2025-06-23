"""Trend-specific domain events."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List
from uuid import UUID

from .base import DomainEvent, EventRegistry, EventType


@EventRegistry.register
@dataclass
class TrendDiscovered(DomainEvent):
    """Event emitted when a new trend is discovered."""
    
    trend_id: UUID
    name: str
    description: str
    relevance_score: float
    source: str
    keywords: List[str]
    discovered_at: datetime
    
    @classmethod
    def get_event_type(cls) -> str:
        return EventType.TREND_DISCOVERED.value


@EventRegistry.register
@dataclass
class TrendUpdated(DomainEvent):
    """Event emitted when trend metrics are updated."""
    
    trend_id: UUID
    old_relevance_score: float
    new_relevance_score: float
    momentum: float
    engagement_metrics: Dict[str, Any]
    
    @classmethod
    def get_event_type(cls) -> str:
        return EventType.TREND_UPDATED.value


@EventRegistry.register
@dataclass
class TrendExpired(DomainEvent):
    """Event emitted when a trend is no longer relevant."""
    
    trend_id: UUID
    expiration_reason: str
    final_relevance_score: float
    lifetime_hours: int
    
    @classmethod
    def get_event_type(cls) -> str:
        return EventType.TREND_EXPIRED.value