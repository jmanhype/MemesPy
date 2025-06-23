"""Message definitions for the actor system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .base_messages import Message, Request, Response, Event


# System Messages
@dataclass
class Ping(Message):
    """Health check message."""
    pass


@dataclass
class Pong(Message):
    """Health check response."""
    pass


@dataclass
class Terminate(Message):
    """Request to terminate an actor."""
    reason: str = "shutdown"
    graceful: bool = True


@dataclass
class Restart(Message):
    """Request to restart an actor."""
    reason: str = "restart"
    clear_state: bool = False


# Meme Generation Messages
@dataclass
class GenerateMemeRequest(Request):
    """Request to generate a meme."""
    prompt: str = ""
    style: Optional[str] = None
    user_id: str = "anonymous"
    priority: str = "normal"  # low, normal, high
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GenerateMemeResponse(Response):
    """Response from meme generation."""
    meme_id: Optional[str] = None
    image_url: Optional[str] = None
    caption: Optional[str] = None
    format: Optional[str] = None
    score: Optional[float] = None


@dataclass
class MemeGenerationProgress(Event):
    """Progress update for meme generation."""
    meme_id: str = ""
    stage: str = ""  # routing, generating, verifying, refining, complete
    progress: float = 0.0  # 0.0 to 1.0
    message: str = ""


# Verification Messages
@dataclass
class VerifyContentRequest(Request):
    """Request to verify content."""
    content: str = ""
    meme_id: str = ""
    verification_type: str = ""  # appropriateness, factuality, instructions


@dataclass
class VerifyContentResponse(Response):
    """Response from content verification."""
    passed: bool = False
    issues: List[str] = field(default_factory=list)
    confidence: float = 1.0
    suggestions: Optional[List[str]] = None


@dataclass
class VerificationComplete(Event):
    """Event when all verifications are complete."""
    meme_id: str = ""
    all_passed: bool = False
    results: Dict[str, Any] = field(default_factory=dict)


# Scoring Messages
@dataclass
class ScoreMemeRequest(Request):
    """Request to score a meme."""
    meme_id: str = ""
    content: str = ""
    image_url: Optional[str] = None
    verification_results: Optional[Dict[str, Any]] = None


@dataclass
class ScoreMemeResponse(Response):
    """Response from meme scoring."""
    score: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    feedback: Optional[str] = None


# Refinement Messages
@dataclass
class RefineMemeRequest(Request):
    """Request to refine a meme."""
    meme_id: str = ""
    original_content: str = ""
    issues: List[str] = field(default_factory=list)
    score: Optional[float] = None


@dataclass
class RefineMemeResponse(Response):
    """Response from meme refinement."""
    refined_content: str = ""
    changes_made: List[str] = field(default_factory=list)
    new_score: Optional[float] = None


# Router Messages
@dataclass
class RouteRequest(Request):
    """Request to route a meme generation request."""
    user_request: str = ""
    context: Optional[Dict[str, Any]] = None


@dataclass
class RouteResponse(Response):
    """Response from routing."""
    topic: str = ""
    format: str = ""
    verification_needs: Dict[str, bool] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    generation_approach: str = ""


# Storage Messages
@dataclass
class StoreMemeRequest(Request):
    """Request to store a meme."""
    meme_id: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    user_id: str = ""


@dataclass
class StoreMemeResponse(Response):
    """Response from meme storage."""
    stored: bool = False
    storage_url: Optional[str] = None


@dataclass
class RetrieveMemeRequest(Request):
    """Request to retrieve a meme."""
    meme_id: str = ""


@dataclass
class RetrieveMemeResponse(Response):
    """Response from meme retrieval."""
    found: bool = False
    content: Optional[Dict[str, Any]] = None


# Cache Messages
@dataclass
class CacheGetRequest(Request):
    """Request to get from cache."""
    key: str = ""


@dataclass
class CacheGetResponse(Response):
    """Response from cache get."""
    found: bool = False
    value: Optional[Any] = None


@dataclass
class CacheSetRequest(Request):
    """Request to set in cache."""
    key: str = ""
    value: Any = None
    ttl: Optional[int] = None  # seconds


@dataclass
class CacheSetResponse(Response):
    """Response from cache set."""
    success: bool = False


# Metrics Messages
@dataclass
class CollectMetricsRequest(Request):
    """Request to collect metrics."""
    metric_types: List[str] = field(default_factory=list)


@dataclass
class CollectMetricsResponse(Response):
    """Response with collected metrics."""
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricUpdate(Event):
    """Event for metric updates."""
    metric_name: str = ""
    value: float = 0.0
    tags: Optional[Dict[str, str]] = None


# Text Generation Messages
@dataclass
class GenerateTextRequest(Request):
    """Request to generate meme text."""
    topic: str = ""
    format: str = ""
    style: Optional[str] = None
    max_length: int = 100
    context: Optional[Dict[str, Any]] = None


@dataclass
class GenerateTextResponse(Response):
    """Response from text generation."""
    text: str = ""
    model_used: str = ""
    prompt_used: str = ""
    confidence_score: float = 0.0
    alternatives: List[str] = field(default_factory=list)


# Image Generation Messages
@dataclass
class GenerateImageRequest(Request):
    """Request to generate meme image."""
    text: str = ""
    format: str = ""
    template_id: Optional[str] = None
    style: str = "meme"
    dimensions: Optional[Dict[str, int]] = None


@dataclass
class GenerateImageResponse(Response):
    """Response from image generation."""
    image_url: str = ""
    template_id: Optional[str] = None
    width: int = 0
    height: int = 0
    file_size: int = 0


# Enhanced Scoring Messages
@dataclass
class EnhancedScoreMemeRequest(Request):
    """Enhanced request to score a meme."""
    text: str = ""
    image_url: str = ""
    topic: str = ""
    format: str = ""
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EnhancedScoreMemeResponse(Response):
    """Enhanced response from meme scoring."""
    overall_score: float = 0.0
    humor_score: float = 0.0
    relevance_score: float = 0.0
    originality_score: float = 0.0
    visual_appeal_score: float = 0.0
    appropriateness_score: float = 0.0
    agent_id: str = ""
    model_used: str = ""
    scores: Dict[str, float] = field(default_factory=dict)
    criteria_details: Dict[str, Any] = field(default_factory=dict)


# Enhanced Verification Messages
@dataclass
class EnhancedVerifyContentRequest(Request):
    """Enhanced request to verify content."""
    content: Dict[str, Any] = field(default_factory=dict)
    verification_type: str = ""  # content, appropriateness, factuality
    criteria: List[str] = field(default_factory=list)


@dataclass
class EnhancedVerifyContentResponse(Response):
    """Enhanced response from content verification."""
    result: str = ""  # passed, failed, warning
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    agent_id: str = ""
    flags: List[str] = field(default_factory=list)


# Event Bus Integration Messages
@dataclass
class EventMessage(Message):
    """Message containing a domain event."""
    event: Any = None  # DomainEvent


@dataclass
class SubscribeToEventsRequest(Request):
    """Request to subscribe actor to events."""
    event_types: List[str] = field(default_factory=list)
    filters: Optional[Dict[str, Any]] = None


@dataclass
class SubscribeToEventsResponse(Response):
    """Response from event subscription."""
    subscription_ids: List[str] = field(default_factory=list)


@dataclass
class UnsubscribeFromEventsRequest(Request):
    """Request to unsubscribe from events."""
    subscription_ids: Optional[List[str]] = None  # None means all


@dataclass
class UnsubscribeFromEventsResponse(Response):
    """Response from event unsubscription."""
    unsubscribed_count: int = 0