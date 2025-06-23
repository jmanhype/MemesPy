"""Meme-specific projections for read models."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID
import json
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, JSON,
    text, select, update, delete, Index, func
)
from sqlalchemy.orm import declarative_base
import redis.asyncio as redis

from .base import Projection
from ..events.base import DomainEvent, EventType
from ..events.meme_events import (
    MemeGenerated, MemeScored, MemeApproved, MemeRejected,
    MemeViewed, MemeShared, MemeDeleted
)


Base = declarative_base()


class MemeReadModel(Base):
    """Read model for meme data."""
    __tablename__ = 'meme_read_model'
    
    meme_id = Column(String(36), primary_key=True)
    topic = Column(String(500), nullable=False, index=True)
    format = Column(String(100), nullable=False, index=True)
    text = Column(String(1000), nullable=False)
    image_url = Column(String(500), nullable=False)
    
    # Status and scores
    status = Column(String(50), nullable=False, index=True)
    score = Column(Float, nullable=True, index=True)
    humor_score = Column(Float, nullable=True)
    relevance_score = Column(Float, nullable=True)
    appropriateness_score = Column(Float, nullable=True)
    
    # Metrics
    view_count = Column(Integer, default=0, index=True)
    share_count = Column(Integer, default=0, index=True)
    engagement_score = Column(Float, default=0.0, index=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, index=True)
    scored_at = Column(DateTime, nullable=True)
    approved_at = Column(DateTime, nullable=True)
    rejected_at = Column(DateTime, nullable=True)
    
    # Metadata
    model_used = Column(String(100), nullable=True)
    generation_time_ms = Column(Integer, nullable=True)
    is_deleted = Column(Boolean, default=False, index=True)
    
    __table_args__ = (
        Index('idx_topic_status', 'topic', 'status'),
        Index('idx_format_score', 'format', 'score'),
        Index('idx_created_status', 'created_at', 'status'),
        Index('idx_engagement', 'engagement_score', 'created_at'),
    )


class TrendingMemesModel(Base):
    """Read model for trending memes."""
    __tablename__ = 'trending_memes'
    
    meme_id = Column(String(36), primary_key=True)
    trend_score = Column(Float, nullable=False, index=True)
    time_window = Column(String(20), primary_key=True)  # hour, day, week
    topic = Column(String(500), nullable=False)
    format = Column(String(100), nullable=False)
    view_count = Column(Integer, default=0)
    share_count = Column(Integer, default=0)
    calculated_at = Column(DateTime, nullable=False)
    
    __table_args__ = (
        Index('idx_trending_window_score', 'time_window', 'trend_score'),
    )


class MemeListProjection(Projection):
    """Projection for meme list/search queries."""
    
    def __init__(self, connection_string: str):
        super().__init__("MemeListProjection")
        self.engine = create_async_engine(connection_string)
        
        # Subscribe to relevant events
        self.subscribe(
            EventType.MEME_GENERATED.value,
            EventType.MEME_SCORED.value,
            EventType.MEME_APPROVED.value,
            EventType.MEME_REJECTED.value,
            EventType.MEME_VIEWED.value,
            EventType.MEME_SHARED.value,
            EventType.MEME_DELETED.value
        )
    
    async def initialize(self) -> None:
        """Create tables and indexes."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def reset(self) -> None:
        """Clear all data."""
        async with self.engine.begin() as conn:
            await conn.execute(text("TRUNCATE TABLE meme_read_model CASCADE"))
    
    async def handle(self, event: DomainEvent, position: int) -> None:
        """Handle events and update read model."""
        async with AsyncSession(self.engine) as session:
            if isinstance(event, MemeGenerated):
                await self._handle_generated(session, event)
            elif isinstance(event, MemeScored):
                await self._handle_scored(session, event)
            elif isinstance(event, MemeApproved):
                await self._handle_approved(session, event)
            elif isinstance(event, MemeRejected):
                await self._handle_rejected(session, event)
            elif isinstance(event, MemeViewed):
                await self._handle_viewed(session, event)
            elif isinstance(event, MemeShared):
                await self._handle_shared(session, event)
            elif isinstance(event, MemeDeleted):
                await self._handle_deleted(session, event)
            
            await session.commit()
    
    async def _handle_generated(self, session: AsyncSession, event: MemeGenerated) -> None:
        """Handle meme generated event."""
        await session.execute(
            text("""
                INSERT INTO meme_read_model (
                    meme_id, topic, format, text, image_url,
                    status, created_at, model_used, generation_time_ms
                ) VALUES (
                    :meme_id, :topic, :format, :text, :image_url,
                    'generated', :created_at, :model_used, :generation_time_ms
                )
            """),
            {
                "meme_id": str(event.meme_id),
                "topic": event.topic,
                "format": event.format,
                "text": event.text,
                "image_url": event.image_url,
                "created_at": event.metadata.timestamp,
                "model_used": event.model_used,
                "generation_time_ms": event.generation_time_ms
            }
        )
    
    async def _handle_scored(self, session: AsyncSession, event: MemeScored) -> None:
        """Handle meme scored event."""
        await session.execute(
            text("""
                UPDATE meme_read_model SET
                    score = :score,
                    humor_score = :humor_score,
                    relevance_score = :relevance_score,
                    appropriateness_score = :appropriateness_score,
                    scored_at = :scored_at
                WHERE meme_id = :meme_id
            """),
            {
                "meme_id": str(event.meme_id),
                "score": event.score,
                "humor_score": event.humor_score,
                "relevance_score": event.relevance_score,
                "appropriateness_score": event.appropriateness_score,
                "scored_at": event.metadata.timestamp
            }
        )
    
    async def _handle_approved(self, session: AsyncSession, event: MemeApproved) -> None:
        """Handle meme approved event."""
        await session.execute(
            text("""
                UPDATE meme_read_model SET
                    status = 'approved',
                    approved_at = :approved_at
                WHERE meme_id = :meme_id
            """),
            {
                "meme_id": str(event.meme_id),
                "approved_at": event.approval_timestamp
            }
        )
    
    async def _handle_rejected(self, session: AsyncSession, event: MemeRejected) -> None:
        """Handle meme rejected event."""
        await session.execute(
            text("""
                UPDATE meme_read_model SET
                    status = 'rejected',
                    rejected_at = :rejected_at
                WHERE meme_id = :meme_id
            """),
            {
                "meme_id": str(event.meme_id),
                "rejected_at": event.rejection_timestamp
            }
        )
    
    async def _handle_viewed(self, session: AsyncSession, event: MemeViewed) -> None:
        """Handle meme viewed event."""
        await session.execute(
            text("""
                UPDATE meme_read_model SET
                    view_count = view_count + 1,
                    engagement_score = (view_count + 1) + (share_count * 5)
                WHERE meme_id = :meme_id
            """),
            {"meme_id": str(event.meme_id)}
        )
    
    async def _handle_shared(self, session: AsyncSession, event: MemeShared) -> None:
        """Handle meme shared event."""
        await session.execute(
            text("""
                UPDATE meme_read_model SET
                    share_count = share_count + 1,
                    engagement_score = view_count + ((share_count + 1) * 5)
                WHERE meme_id = :meme_id
            """),
            {"meme_id": str(event.meme_id)}
        )
    
    async def _handle_deleted(self, session: AsyncSession, event: MemeDeleted) -> None:
        """Handle meme deleted event."""
        if event.soft_delete:
            await session.execute(
                text("""
                    UPDATE meme_read_model SET
                        is_deleted = true,
                        status = 'deleted'
                    WHERE meme_id = :meme_id
                """),
                {"meme_id": str(event.meme_id)}
            )
        else:
            await session.execute(
                text("DELETE FROM meme_read_model WHERE meme_id = :meme_id"),
                {"meme_id": str(event.meme_id)}
            )


class TrendingProjection(Projection):
    """Projection for trending memes calculation."""
    
    def __init__(self, connection_string: str):
        super().__init__("TrendingProjection")
        self.engine = create_async_engine(connection_string)
        
        # Subscribe to engagement events
        self.subscribe(
            EventType.MEME_VIEWED.value,
            EventType.MEME_SHARED.value,
            EventType.MEME_APPROVED.value
        )
    
    async def initialize(self) -> None:
        """Create tables and indexes."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def reset(self) -> None:
        """Clear all data."""
        async with self.engine.begin() as conn:
            await conn.execute(text("TRUNCATE TABLE trending_memes CASCADE"))
    
    async def handle(self, event: DomainEvent, position: int) -> None:
        """Update trending calculations."""
        # For efficiency, we'll batch process trending updates
        # In production, this would run periodically
        if position % 100 == 0:  # Every 100 events
            await self._recalculate_trending()
    
    async def _recalculate_trending(self) -> None:
        """Recalculate trending memes."""
        async with self.engine.begin() as conn:
            # Calculate trending for different time windows
            for window in ['hour', 'day', 'week']:
                interval = {
                    'hour': '1 hour',
                    'day': '1 day',
                    'week': '7 days'
                }[window]
                
                await conn.execute(text(f"""
                    INSERT INTO trending_memes (
                        meme_id, trend_score, time_window, topic, format,
                        view_count, share_count, calculated_at
                    )
                    SELECT 
                        meme_id,
                        (view_count + share_count * 5) / 
                        EXTRACT(EPOCH FROM (NOW() - created_at)) * 3600 AS trend_score,
                        :window,
                        topic,
                        format,
                        view_count,
                        share_count,
                        NOW()
                    FROM meme_read_model
                    WHERE created_at > NOW() - INTERVAL '{interval}'
                    AND status = 'approved'
                    AND is_deleted = false
                    ORDER BY trend_score DESC
                    LIMIT 100
                    ON CONFLICT (meme_id, time_window) DO UPDATE SET
                        trend_score = EXCLUDED.trend_score,
                        view_count = EXCLUDED.view_count,
                        share_count = EXCLUDED.share_count,
                        calculated_at = EXCLUDED.calculated_at
                """), {"window": window})


class MemeAnalyticsProjection(Projection):
    """Projection for analytics and reporting."""
    
    def __init__(self, redis_url: str):
        super().__init__("MemeAnalyticsProjection")
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        
        # Subscribe to all meme events
        self.subscribe(
            EventType.MEME_GENERATED.value,
            EventType.MEME_SCORED.value,
            EventType.MEME_APPROVED.value,
            EventType.MEME_REJECTED.value,
            EventType.MEME_VIEWED.value,
            EventType.MEME_SHARED.value
        )
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        self.redis = await redis.from_url(self.redis_url)
    
    async def reset(self) -> None:
        """Clear analytics data."""
        if self.redis:
            await self.redis.flushdb()
    
    async def handle(self, event: DomainEvent, position: int) -> None:
        """Update analytics metrics."""
        if not self.redis:
            return
        
        # Update counters
        date_key = event.metadata.timestamp.strftime("%Y-%m-%d")
        hour_key = event.metadata.timestamp.strftime("%Y-%m-%d:%H")
        
        if isinstance(event, MemeGenerated):
            await self.redis.hincrby(f"stats:{date_key}", "generated", 1)
            await self.redis.hincrby(f"stats:{hour_key}", "generated", 1)
            await self.redis.hincrby(f"format:{event.format}:stats", "generated", 1)
            
        elif isinstance(event, MemeApproved):
            await self.redis.hincrby(f"stats:{date_key}", "approved", 1)
            await self.redis.hincrby(f"stats:{hour_key}", "approved", 1)
            
        elif isinstance(event, MemeRejected):
            await self.redis.hincrby(f"stats:{date_key}", "rejected", 1)
            await self.redis.hincrby(f"stats:{hour_key}", "rejected", 1)
            
        elif isinstance(event, MemeViewed):
            await self.redis.hincrby(f"stats:{date_key}", "views", 1)
            await self.redis.hincrby(f"stats:{hour_key}", "views", 1)
            
        elif isinstance(event, MemeShared):
            await self.redis.hincrby(f"stats:{date_key}", "shares", 1)
            await self.redis.hincrby(f"stats:{hour_key}", "shares", 1)
            await self.redis.hincrby(f"platform:{event.share_platform}:stats", "shares", 1)
        
        # Update real-time metrics
        await self.redis.zadd(
            "recent_activity",
            {json.dumps({
                "event_type": event.metadata.event_type,
                "timestamp": event.metadata.timestamp.isoformat(),
                "aggregate_id": str(event.metadata.aggregate_id)
            }): event.metadata.timestamp.timestamp()},
            nx=True
        )
        
        # Trim old activity (keep last 1000)
        await self.redis.zremrangebyrank("recent_activity", 0, -1001)