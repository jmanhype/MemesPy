"""Read model projections for meme generation system."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from uuid import UUID
from dataclasses import dataclass, field
from collections import defaultdict

from .events import (
    DomainEvent,
    MemeGenerationStarted,
    TextGenerated,
    ImageGenerated,
    QualityScored,
    MemeCompleted,
    MemeGenerationFailed,
    MemeViewed,
    MemeShared,
    VerificationCompleted,
    PipelineStageStarted,
    PipelineStageCompleted,
)
from .event_store import get_event_store


logger = logging.getLogger(__name__)


@dataclass
class MemeProjection:
    """Read model projection for a meme."""

    meme_id: UUID
    request_id: UUID
    topic: str
    format: str
    text: Optional[str] = None
    image_url: Optional[str] = None
    status: str = "pending"  # pending, generating, completed, failed, rejected
    overall_score: float = 0.0
    humor_score: float = 0.0
    relevance_score: float = 0.0
    appropriateness_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    generation_time_ms: int = 0
    refinement_count: int = 0
    view_count: int = 0
    share_count: int = 0
    pipeline_stages: List[str] = field(default_factory=list)
    verification_status: str = "pending"  # pending, passed, failed
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "meme_id": str(self.meme_id),
            "request_id": str(self.request_id),
            "topic": self.topic,
            "format": self.format,
            "text": self.text,
            "image_url": self.image_url,
            "status": self.status,
            "scores": {
                "overall": self.overall_score,
                "humor": self.humor_score,
                "relevance": self.relevance_score,
                "appropriateness": self.appropriateness_score,
            },
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "generation_time_ms": self.generation_time_ms,
            "refinement_count": self.refinement_count,
            "view_count": self.view_count,
            "share_count": self.share_count,
            "verification_status": self.verification_status,
            "metadata": self.metadata,
        }


@dataclass
class PipelineStageProjection:
    """Projection for pipeline stage tracking."""

    pipeline_id: UUID
    meme_id: UUID
    stage_name: str
    stage_order: int
    status: str = "pending"  # pending, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: int = 0
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class MemeMetricsProjection:
    """Aggregated metrics projection."""

    date: datetime
    total_requests: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    completed_pipelines: int = 0
    average_score: float = 0.0
    average_generation_time_ms: int = 0
    top_topics: List[str] = field(default_factory=list)
    top_formats: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    completion_rate: float = 0.0


class MemeProjectionStore:
    """In-memory store for meme projections with Redis-like functionality."""

    def __init__(self):
        self.memes: Dict[UUID, MemeProjection] = {}
        self.pipeline_stages: Dict[UUID, List[PipelineStageProjection]] = defaultdict(list)
        self.metrics_by_date: Dict[str, MemeMetricsProjection] = {}
        self.topic_index: Dict[str, Set[UUID]] = defaultdict(set)
        self.format_index: Dict[str, Set[UUID]] = defaultdict(set)
        self.status_index: Dict[str, Set[UUID]] = defaultdict(set)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def upsert_meme(self, projection: MemeProjection) -> None:
        """Insert or update meme projection."""
        async with self._locks[f"meme_{projection.meme_id}"]:
            old_projection = self.memes.get(projection.meme_id)

            # Update indexes
            if old_projection:
                self.topic_index[old_projection.topic].discard(projection.meme_id)
                self.format_index[old_projection.format].discard(projection.meme_id)
                self.status_index[old_projection.status].discard(projection.meme_id)

            self.memes[projection.meme_id] = projection
            self.topic_index[projection.topic].add(projection.meme_id)
            self.format_index[projection.format].add(projection.meme_id)
            self.status_index[projection.status].add(projection.meme_id)

    async def get_meme(self, meme_id: UUID) -> Optional[MemeProjection]:
        """Get meme projection by ID."""
        return self.memes.get(meme_id)

    async def list_memes(
        self,
        status: Optional[str] = None,
        topic: Optional[str] = None,
        format: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[MemeProjection]:
        """List memes with filtering."""
        # Get candidate IDs based on filters
        candidate_ids = set(self.memes.keys())

        if status:
            candidate_ids &= self.status_index[status]
        if topic:
            candidate_ids &= self.topic_index[topic]
        if format:
            candidate_ids &= self.format_index[format]

        # Get projections and sort by created_at
        memes = [self.memes[meme_id] for meme_id in candidate_ids]
        memes.sort(key=lambda m: m.created_at, reverse=True)

        # Apply pagination
        return memes[offset : offset + limit]

    async def upsert_pipeline_stage(self, stage: PipelineStageProjection) -> None:
        """Insert or update pipeline stage projection."""
        async with self._locks[f"pipeline_{stage.pipeline_id}"]:
            stages = self.pipeline_stages[stage.pipeline_id]

            # Find existing stage
            for i, existing_stage in enumerate(stages):
                if existing_stage.stage_name == stage.stage_name:
                    stages[i] = stage
                    return

            # Add new stage
            stages.append(stage)
            stages.sort(key=lambda s: s.stage_order)

    async def get_pipeline_stages(self, pipeline_id: UUID) -> List[PipelineStageProjection]:
        """Get pipeline stages for a pipeline."""
        return self.pipeline_stages.get(pipeline_id, [])

    async def update_metrics(self, date: datetime, updates: Dict[str, Any]) -> None:
        """Update daily metrics."""
        date_key = date.strftime("%Y-%m-%d")
        async with self._locks[f"metrics_{date_key}"]:
            if date_key not in self.metrics_by_date:
                self.metrics_by_date[date_key] = MemeMetricsProjection(date=date)

            metrics = self.metrics_by_date[date_key]
            for key, value in updates.items():
                if hasattr(metrics, key):
                    if key in [
                        "total_requests",
                        "successful_generations",
                        "failed_generations",
                        "completed_pipelines",
                    ]:
                        setattr(metrics, key, getattr(metrics, key) + value)
                    else:
                        setattr(metrics, key, value)

    async def get_metrics(self, date: datetime) -> Optional[MemeMetricsProjection]:
        """Get metrics for a specific date."""
        date_key = date.strftime("%Y-%m-%d")
        return self.metrics_by_date.get(date_key)


# Global projection store
_projection_store = MemeProjectionStore()


async def get_projection_store() -> MemeProjectionStore:
    """Get the global projection store."""
    return _projection_store


class MemeProjectionHandler:
    """Handler for updating meme projections based on events."""

    def __init__(self):
        self.store = _projection_store

    async def handle_event(self, event: DomainEvent) -> None:
        """Handle domain event and update projections."""
        event_type = event.metadata.event_type

        handlers = {
            "meme.generation.started": self._handle_generation_started,
            "meme.text.generated": self._handle_text_generated,
            "meme.image.generated": self._handle_image_generated,
            "meme.quality.scored": self._handle_quality_scored,
            "meme.completed": self._handle_meme_completed,
            "meme.generation.failed": self._handle_generation_failed,
            "meme.viewed": self._handle_meme_viewed,
            "meme.shared": self._handle_meme_shared,
            "meme.verification.completed": self._handle_verification_completed,
            "pipeline.stage.started": self._handle_pipeline_stage_started,
            "pipeline.stage.completed": self._handle_pipeline_stage_completed,
        }

        handler = handlers.get(event_type)
        if handler:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error handling event {event.metadata.event_id}: {e}", exc_info=True)

    async def _handle_generation_started(self, event: MemeGenerationStarted) -> None:
        """Handle meme generation started event."""
        projection = MemeProjection(
            meme_id=event.metadata.aggregate_id,
            request_id=event.request_id,
            topic=event.topic,
            format=event.format,
            status="generating",
            created_at=event.metadata.timestamp,
            pipeline_stages=event.expected_stages,
            metadata={"parameters": event.parameters, "pipeline_config": event.pipeline_config},
        )

        await self.store.upsert_meme(projection)

        # Update metrics
        await self.store.update_metrics(event.metadata.timestamp, {"total_requests": 1})

    async def _handle_text_generated(self, event: TextGenerated) -> None:
        """Handle text generated event."""
        projection = await self.store.get_meme(event.meme_id)
        if projection:
            projection.text = event.text
            projection.generation_time_ms += event.generation_time_ms
            projection.metadata.update(
                {
                    "text_generation": {
                        "method": event.generation_method,
                        "model": event.model_used,
                        "confidence": event.confidence_score,
                        "alternatives": event.alternatives_considered,
                    }
                }
            )
            await self.store.upsert_meme(projection)

    async def _handle_image_generated(self, event: ImageGenerated) -> None:
        """Handle image generated event."""
        projection = await self.store.get_meme(event.meme_id)
        if projection:
            projection.image_url = event.image_url
            projection.generation_time_ms += event.generation_time_ms
            projection.metadata.update(
                {
                    "image_generation": {
                        "type": event.image_type,
                        "template": event.template_used,
                        "dimensions": event.dimensions,
                        "file_size": event.file_size_bytes,
                    }
                }
            )
            await self.store.upsert_meme(projection)

    async def _handle_quality_scored(self, event: QualityScored) -> None:
        """Handle quality scored event."""
        projection = await self.store.get_meme(event.meme_id)
        if projection:
            projection.overall_score = event.overall_score
            projection.humor_score = event.humor_score
            projection.relevance_score = event.relevance_score
            projection.appropriateness_score = event.appropriateness_score
            projection.metadata.update(
                {
                    "scoring": {
                        "agent": event.scoring_agent,
                        "model": event.scoring_model,
                        "criteria": event.criteria_details,
                    }
                }
            )
            await self.store.upsert_meme(projection)

    async def _handle_meme_completed(self, event: MemeCompleted) -> None:
        """Handle meme completed event."""
        projection = await self.store.get_meme(event.meme_id)
        if projection:
            projection.status = event.status
            projection.completed_at = event.metadata.timestamp
            projection.overall_score = event.final_score
            projection.generation_time_ms = event.completion_time_ms
            projection.metadata.update(event.final_metadata)
            await self.store.upsert_meme(projection)

            # Update metrics
            if event.status == "completed":
                await self.store.update_metrics(
                    event.metadata.timestamp,
                    {"successful_generations": 1, "completed_pipelines": 1},
                )

    async def _handle_generation_failed(self, event: MemeGenerationFailed) -> None:
        """Handle generation failed event."""
        # Find meme by request_id
        meme_id = event.metadata.aggregate_id
        projection = await self.store.get_meme(meme_id)
        if projection:
            projection.status = "failed"
            projection.failed_at = event.metadata.timestamp
            projection.error_message = event.error_message
            await self.store.upsert_meme(projection)

            # Update metrics
            await self.store.update_metrics(event.metadata.timestamp, {"failed_generations": 1})

    async def _handle_meme_viewed(self, event: MemeViewed) -> None:
        """Handle meme viewed event."""
        projection = await self.store.get_meme(event.meme_id)
        if projection:
            projection.view_count += 1
            await self.store.upsert_meme(projection)

    async def _handle_meme_shared(self, event: MemeShared) -> None:
        """Handle meme shared event."""
        projection = await self.store.get_meme(event.meme_id)
        if projection:
            projection.share_count += 1
            await self.store.upsert_meme(projection)

    async def _handle_verification_completed(self, event: VerificationCompleted) -> None:
        """Handle verification completed event."""
        projection = await self.store.get_meme(event.meme_id)
        if projection:
            projection.verification_status = event.verification_result
            projection.metadata.update(
                {
                    "verification": {
                        "type": event.verification_type,
                        "result": event.verification_result,
                        "confidence": event.confidence_score,
                        "agent": event.verification_agent,
                        "flags": event.flags_raised,
                    }
                }
            )
            await self.store.upsert_meme(projection)

    async def _handle_pipeline_stage_started(self, event: PipelineStageStarted) -> None:
        """Handle pipeline stage started event."""
        stage = PipelineStageProjection(
            pipeline_id=event.pipeline_id,
            meme_id=event.metadata.aggregate_id,
            stage_name=event.stage_name,
            stage_order=event.stage_order,
            status="running",
            started_at=event.metadata.timestamp,
            input_data=event.input_data,
        )
        await self.store.upsert_pipeline_stage(stage)

    async def _handle_pipeline_stage_completed(self, event: PipelineStageCompleted) -> None:
        """Handle pipeline stage completed event."""
        stage = PipelineStageProjection(
            pipeline_id=event.pipeline_id,
            meme_id=event.metadata.aggregate_id,
            stage_name=event.stage_name,
            stage_order=event.stage_order,
            status="completed" if event.success else "failed",
            completed_at=event.metadata.timestamp,
            duration_ms=event.duration_ms,
            output_data=event.output_data,
        )
        await self.store.upsert_pipeline_stage(stage)


# Global projection handler
_projection_handler = MemeProjectionHandler()


async def initialize_projections() -> None:
    """Initialize projection system."""
    event_store = await get_event_store()

    # Register projection handler for all relevant events
    event_types = [
        "meme.generation.started",
        "meme.text.generated",
        "meme.image.generated",
        "meme.quality.scored",
        "meme.completed",
        "meme.generation.failed",
        "meme.viewed",
        "meme.shared",
        "meme.verification.completed",
        "pipeline.stage.started",
        "pipeline.stage.completed",
    ]

    for event_type in event_types:
        await event_store.register_projection_handler(event_type, _projection_handler.handle_event)

    logger.info("Projection system initialized")


async def rebuild_projections_from_events() -> None:
    """Rebuild all projections from events."""
    logger.info("Starting projection rebuild")

    event_store = await get_event_store()
    await event_store.rebuild_projections()

    logger.info("Projection rebuild completed")


# API helper functions
async def get_meme_projection(meme_id: UUID) -> Optional[Dict[str, Any]]:
    """Get meme projection for API."""
    store = await get_projection_store()
    projection = await store.get_meme(meme_id)
    return projection.to_dict() if projection else None


async def list_meme_projections(
    status: Optional[str] = None,
    topic: Optional[str] = None,
    format: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """List meme projections for API."""
    store = await get_projection_store()
    projections = await store.list_memes(status, topic, format, limit, offset)
    return [p.to_dict() for p in projections]


async def get_pipeline_status(pipeline_id: UUID) -> Dict[str, Any]:
    """Get pipeline status for API."""
    store = await get_projection_store()
    stages = await store.get_pipeline_stages(pipeline_id)

    return {
        "pipeline_id": str(pipeline_id),
        "total_stages": len(stages),
        "completed_stages": len([s for s in stages if s.status == "completed"]),
        "current_stage": next((s.stage_name for s in stages if s.status == "running"), None),
        "stages": [
            {
                "name": s.stage_name,
                "order": s.stage_order,
                "status": s.status,
                "duration_ms": s.duration_ms,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
            }
            for s in stages
        ],
    }


async def get_daily_metrics(date: datetime) -> Optional[Dict[str, Any]]:
    """Get daily metrics for API."""
    store = await get_projection_store()
    metrics = await store.get_metrics(date)

    if not metrics:
        return None

    return {
        "date": metrics.date.strftime("%Y-%m-%d"),
        "total_requests": metrics.total_requests,
        "successful_generations": metrics.successful_generations,
        "failed_generations": metrics.failed_generations,
        "completed_pipelines": metrics.completed_pipelines,
        "success_rate": metrics.success_rate,
        "completion_rate": metrics.completion_rate,
        "average_score": metrics.average_score,
        "average_generation_time_ms": metrics.average_generation_time_ms,
    }
