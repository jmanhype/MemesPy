"""Meme aggregate for event sourcing."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4

from ..events import (
    DomainEvent, MemeGenerationRequested, MemeGenerated, MemeGenerationFailed,
    MemeScored, MemeRefined, MemeApproved, MemeRejected,
    MemeViewed, MemeShared, MemeDeleted
)


@dataclass
class MemeState:
    """Current state of a meme aggregate."""
    meme_id: UUID
    request_id: Optional[UUID] = None
    topic: Optional[str] = None
    format: Optional[str] = None
    text: Optional[str] = None
    image_url: Optional[str] = None
    template_id: Optional[UUID] = None
    
    # Status
    status: str = "pending"  # pending, generated, failed, approved, rejected, deleted
    
    # Scores
    score: Optional[float] = None
    humor_score: Optional[float] = None
    relevance_score: Optional[float] = None
    appropriateness_score: Optional[float] = None
    
    # Verification
    verification_scores: Dict[str, float] = field(default_factory=dict)
    rejection_reasons: List[str] = field(default_factory=list)
    violation_categories: List[str] = field(default_factory=list)
    
    # Refinement
    refinement_count: int = 0
    refinement_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metrics
    view_count: int = 0
    share_count: int = 0
    
    # Timestamps
    created_at: Optional[datetime] = None
    generated_at: Optional[datetime] = None
    scored_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    
    # Metadata
    generation_time_ms: int = 0
    model_used: Optional[str] = None
    deleted: bool = False
    deletion_reason: Optional[str] = None


class MemeAggregate:
    """Aggregate root for meme domain."""
    
    def __init__(self, aggregate_id: Optional[UUID] = None):
        self.id = aggregate_id or uuid4()
        self.version = 0
        self.state = MemeState(meme_id=self.id)
        self._uncommitted_events: List[DomainEvent] = []
    
    def apply(self, event: DomainEvent) -> None:
        """Apply an event to update aggregate state."""
        if isinstance(event, MemeGenerationRequested):
            self._apply_generation_requested(event)
        elif isinstance(event, MemeGenerated):
            self._apply_generated(event)
        elif isinstance(event, MemeGenerationFailed):
            self._apply_generation_failed(event)
        elif isinstance(event, MemeScored):
            self._apply_scored(event)
        elif isinstance(event, MemeRefined):
            self._apply_refined(event)
        elif isinstance(event, MemeApproved):
            self._apply_approved(event)
        elif isinstance(event, MemeRejected):
            self._apply_rejected(event)
        elif isinstance(event, MemeViewed):
            self._apply_viewed(event)
        elif isinstance(event, MemeShared):
            self._apply_shared(event)
        elif isinstance(event, MemeDeleted):
            self._apply_deleted(event)
        
        self.version = event.metadata.aggregate_version
    
    def _apply_generation_requested(self, event: MemeGenerationRequested) -> None:
        """Apply generation requested event."""
        self.state.request_id = event.request_id
        self.state.topic = event.topic
        self.state.format = event.format
        self.state.status = "pending"
        self.state.created_at = event.metadata.timestamp
    
    def _apply_generated(self, event: MemeGenerated) -> None:
        """Apply generated event."""
        self.state.text = event.text
        self.state.image_url = event.image_url
        self.state.template_id = event.template_id
        self.state.status = "generated"
        self.state.generated_at = event.metadata.timestamp
        self.state.generation_time_ms = event.generation_time_ms
        self.state.model_used = event.model_used
    
    def _apply_generation_failed(self, event: MemeGenerationFailed) -> None:
        """Apply generation failed event."""
        self.state.status = "failed"
    
    def _apply_scored(self, event: MemeScored) -> None:
        """Apply scored event."""
        self.state.score = event.score
        self.state.humor_score = event.humor_score
        self.state.relevance_score = event.relevance_score
        self.state.appropriateness_score = event.appropriateness_score
        self.state.scored_at = event.metadata.timestamp
    
    def _apply_refined(self, event: MemeRefined) -> None:
        """Apply refined event."""
        self.state.refinement_count += 1
        self.state.refinement_history.append({
            "refined_meme_id": event.refined_meme_id,
            "reason": event.refinement_reason,
            "changes": event.changes_made,
            "iteration": event.refinement_iteration,
            "timestamp": event.metadata.timestamp
        })
    
    def _apply_approved(self, event: MemeApproved) -> None:
        """Apply approved event."""
        self.state.status = "approved"
        self.state.verification_scores = event.verification_scores
        self.state.approved_at = event.approval_timestamp
    
    def _apply_rejected(self, event: MemeRejected) -> None:
        """Apply rejected event."""
        self.state.status = "rejected"
        self.state.rejection_reasons = event.rejection_reasons
        self.state.violation_categories = event.violation_categories
        self.state.rejected_at = event.rejection_timestamp
    
    def _apply_viewed(self, event: MemeViewed) -> None:
        """Apply viewed event."""
        self.state.view_count += 1
    
    def _apply_shared(self, event: MemeShared) -> None:
        """Apply shared event."""
        self.state.share_count += 1
    
    def _apply_deleted(self, event: MemeDeleted) -> None:
        """Apply deleted event."""
        self.state.deleted = True
        self.state.deletion_reason = event.deletion_reason
        self.state.deleted_at = event.metadata.timestamp
        if not event.soft_delete:
            self.state.status = "deleted"
    
    def get_uncommitted_events(self) -> List[DomainEvent]:
        """Get events that haven't been persisted yet."""
        return self._uncommitted_events
    
    def mark_events_committed(self) -> None:
        """Clear uncommitted events after persistence."""
        self._uncommitted_events.clear()
    
    def to_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of current state."""
        return {
            "meme_id": str(self.state.meme_id),
            "request_id": str(self.state.request_id) if self.state.request_id else None,
            "topic": self.state.topic,
            "format": self.state.format,
            "text": self.state.text,
            "image_url": self.state.image_url,
            "template_id": str(self.state.template_id) if self.state.template_id else None,
            "status": self.state.status,
            "score": self.state.score,
            "humor_score": self.state.humor_score,
            "relevance_score": self.state.relevance_score,
            "appropriateness_score": self.state.appropriateness_score,
            "verification_scores": self.state.verification_scores,
            "rejection_reasons": self.state.rejection_reasons,
            "violation_categories": self.state.violation_categories,
            "refinement_count": self.state.refinement_count,
            "refinement_history": self.state.refinement_history,
            "view_count": self.state.view_count,
            "share_count": self.state.share_count,
            "created_at": self.state.created_at.isoformat() if self.state.created_at else None,
            "generated_at": self.state.generated_at.isoformat() if self.state.generated_at else None,
            "scored_at": self.state.scored_at.isoformat() if self.state.scored_at else None,
            "approved_at": self.state.approved_at.isoformat() if self.state.approved_at else None,
            "rejected_at": self.state.rejected_at.isoformat() if self.state.rejected_at else None,
            "deleted_at": self.state.deleted_at.isoformat() if self.state.deleted_at else None,
            "generation_time_ms": self.state.generation_time_ms,
            "model_used": self.state.model_used,
            "deleted": self.state.deleted,
            "deletion_reason": self.state.deletion_reason
        }
    
    def from_snapshot(self, snapshot: Dict[str, Any], version: int) -> None:
        """Restore state from snapshot."""
        self.version = version
        self.state = MemeState(
            meme_id=UUID(snapshot["meme_id"]),
            request_id=UUID(snapshot["request_id"]) if snapshot.get("request_id") else None,
            topic=snapshot.get("topic"),
            format=snapshot.get("format"),
            text=snapshot.get("text"),
            image_url=snapshot.get("image_url"),
            template_id=UUID(snapshot["template_id"]) if snapshot.get("template_id") else None,
            status=snapshot.get("status", "pending"),
            score=snapshot.get("score"),
            humor_score=snapshot.get("humor_score"),
            relevance_score=snapshot.get("relevance_score"),
            appropriateness_score=snapshot.get("appropriateness_score"),
            verification_scores=snapshot.get("verification_scores", {}),
            rejection_reasons=snapshot.get("rejection_reasons", []),
            violation_categories=snapshot.get("violation_categories", []),
            refinement_count=snapshot.get("refinement_count", 0),
            refinement_history=snapshot.get("refinement_history", []),
            view_count=snapshot.get("view_count", 0),
            share_count=snapshot.get("share_count", 0),
            created_at=datetime.fromisoformat(snapshot["created_at"]) if snapshot.get("created_at") else None,
            generated_at=datetime.fromisoformat(snapshot["generated_at"]) if snapshot.get("generated_at") else None,
            scored_at=datetime.fromisoformat(snapshot["scored_at"]) if snapshot.get("scored_at") else None,
            approved_at=datetime.fromisoformat(snapshot["approved_at"]) if snapshot.get("approved_at") else None,
            rejected_at=datetime.fromisoformat(snapshot["rejected_at"]) if snapshot.get("rejected_at") else None,
            deleted_at=datetime.fromisoformat(snapshot["deleted_at"]) if snapshot.get("deleted_at") else None,
            generation_time_ms=snapshot.get("generation_time_ms", 0),
            model_used=snapshot.get("model_used"),
            deleted=snapshot.get("deleted", False),
            deletion_reason=snapshot.get("deletion_reason")
        )