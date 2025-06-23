"""Event store implementation for persisting and retrieving events."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, AsyncIterator, Callable
from uuid import UUID
import json
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import (
    Column, String, Integer, DateTime, JSON, Index, 
    text, select, and_, or_, func
)
from sqlalchemy.orm import declarative_base
import asyncpg

from ..events import DomainEvent, EventRegistry


Base = declarative_base()


class EventEntity(Base):
    """Database entity for storing events."""
    __tablename__ = 'event_store'
    
    # Primary fields
    global_position = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(36), unique=True, nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    
    # Aggregate fields
    aggregate_id = Column(String(36), nullable=False, index=True)
    aggregate_type = Column(String(100), nullable=False, index=True)
    aggregate_version = Column(Integer, nullable=False)
    
    # Metadata fields
    timestamp = Column(DateTime, nullable=False, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    correlation_id = Column(String(36), nullable=True, index=True)
    causation_id = Column(String(36), nullable=True, index=True)
    
    # Event data
    event_data = Column(JSON, nullable=False)
    event_meta = Column(JSON, nullable=True)
    
    # Create composite indexes for common queries
    __table_args__ = (
        Index('idx_aggregate_stream', 'aggregate_id', 'aggregate_version'),
        Index('idx_event_type_timestamp', 'event_type', 'timestamp'),
        Index('idx_correlation', 'correlation_id', 'timestamp'),
    )


@dataclass
class EventStream:
    """Represents a stream of events for an aggregate."""
    aggregate_id: UUID
    aggregate_type: str
    version: int
    events: List[DomainEvent]


@dataclass
class Snapshot:
    """Aggregate snapshot for performance optimization."""
    aggregate_id: UUID
    aggregate_type: str
    version: int
    state: Dict[str, Any]
    timestamp: datetime


class EventStore(ABC):
    """Abstract base class for event stores."""
    
    @abstractmethod
    async def append(self, event: DomainEvent) -> int:
        """Append an event to the store. Returns global position."""
        pass
    
    @abstractmethod
    async def append_batch(self, events: List[DomainEvent]) -> List[int]:
        """Append multiple events atomically. Returns global positions."""
        pass
    
    @abstractmethod
    async def get_events(
        self,
        aggregate_id: UUID,
        from_version: int = 0,
        to_version: Optional[int] = None
    ) -> List[DomainEvent]:
        """Get events for an aggregate within version range."""
        pass
    
    @abstractmethod
    async def get_all_events(
        self,
        from_position: int = 0,
        limit: Optional[int] = None
    ) -> AsyncIterator[DomainEvent]:
        """Get all events from a global position."""
        pass
    
    @abstractmethod
    async def get_events_by_type(
        self,
        event_type: str,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[DomainEvent]:
        """Get events by type within time range."""
        pass
    
    @abstractmethod
    async def get_snapshot(
        self,
        aggregate_id: UUID,
        max_version: Optional[int] = None
    ) -> Optional[Snapshot]:
        """Get latest snapshot for an aggregate."""
        pass
    
    @abstractmethod
    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """Save an aggregate snapshot."""
        pass


class PostgresEventStore(EventStore):
    """PostgreSQL-based event store implementation."""
    
    def __init__(self, connection_string: str):
        self.engine = create_async_engine(
            connection_string,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True
        )
        self._subscribers: Dict[str, List[Callable]] = {}
    
    async def initialize(self):
        """Initialize database schema."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
            # Create snapshot table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS aggregate_snapshots (
                    aggregate_id VARCHAR(36) NOT NULL,
                    aggregate_type VARCHAR(100) NOT NULL,
                    version INTEGER NOT NULL,
                    state JSONB NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    PRIMARY KEY (aggregate_id, version)
                );
                
                CREATE INDEX IF NOT EXISTS idx_snapshot_lookup 
                ON aggregate_snapshots (aggregate_id, version DESC);
            """))
    
    async def append(self, event: DomainEvent) -> int:
        """Append an event to the store."""
        async with AsyncSession(self.engine) as session:
            # Check for version conflict
            existing = await session.execute(
                select(EventEntity).where(
                    and_(
                        EventEntity.aggregate_id == str(event.metadata.aggregate_id),
                        EventEntity.aggregate_version == event.metadata.aggregate_version
                    )
                )
            )
            if existing.scalar():
                raise ValueError(f"Version conflict for aggregate {event.metadata.aggregate_id}")
            
            # Create entity
            event_dict = event.to_dict()
            entity = EventEntity(
                event_id=str(event.metadata.event_id),
                event_type=event.metadata.event_type,
                aggregate_id=str(event.metadata.aggregate_id),
                aggregate_type=event.metadata.aggregate_type,
                aggregate_version=event.metadata.aggregate_version,
                timestamp=event.metadata.timestamp,
                user_id=event.metadata.user_id,
                correlation_id=str(event.metadata.correlation_id) if event.metadata.correlation_id else None,
                causation_id=str(event.metadata.causation_id) if event.metadata.causation_id else None,
                event_data=event_dict["data"],
                event_meta=event.metadata
            )
            
            session.add(entity)
            await session.commit()
            
            # Notify subscribers
            await self._notify_subscribers(event)
            
            return entity.global_position
    
    async def append_batch(self, events: List[DomainEvent]) -> List[int]:
        """Append multiple events atomically."""
        if not events:
            return []
        
        async with AsyncSession(self.engine) as session:
            entities = []
            
            # Check for version conflicts
            aggregate_versions = {}
            for event in events:
                key = str(event.metadata.aggregate_id)
                if key in aggregate_versions:
                    aggregate_versions[key] = max(
                        aggregate_versions[key],
                        event.metadata.aggregate_version
                    )
                else:
                    aggregate_versions[key] = event.metadata.aggregate_version
            
            # Verify no conflicts
            for aggregate_id, max_version in aggregate_versions.items():
                existing = await session.execute(
                    select(func.max(EventEntity.aggregate_version)).where(
                        EventEntity.aggregate_id == aggregate_id
                    )
                )
                current_version = existing.scalar() or 0
                if current_version >= max_version:
                    raise ValueError(f"Version conflict for aggregate {aggregate_id}")
            
            # Create entities
            for event in events:
                event_dict = event.to_dict()
                entity = EventEntity(
                    event_id=str(event.metadata.event_id),
                    event_type=event.metadata.event_type,
                    aggregate_id=str(event.metadata.aggregate_id),
                    aggregate_type=event.metadata.aggregate_type,
                    aggregate_version=event.metadata.aggregate_version,
                    timestamp=event.metadata.timestamp,
                    user_id=event.metadata.user_id,
                    correlation_id=str(event.metadata.correlation_id) if event.metadata.correlation_id else None,
                    causation_id=str(event.metadata.causation_id) if event.metadata.causation_id else None,
                    event_data=event_dict["data"],
                    event_meta=event.metadata
                )
                entities.append(entity)
                session.add(entity)
            
            await session.commit()
            
            # Notify subscribers for all events
            for event in events:
                await self._notify_subscribers(event)
            
            return [e.global_position for e in entities]
    
    async def get_events(
        self,
        aggregate_id: UUID,
        from_version: int = 0,
        to_version: Optional[int] = None
    ) -> List[DomainEvent]:
        """Get events for an aggregate within version range."""
        async with AsyncSession(self.engine) as session:
            query = select(EventEntity).where(
                and_(
                    EventEntity.aggregate_id == str(aggregate_id),
                    EventEntity.aggregate_version >= from_version
                )
            )
            
            if to_version is not None:
                query = query.where(EventEntity.aggregate_version <= to_version)
            
            query = query.order_by(EventEntity.aggregate_version)
            
            result = await session.execute(query)
            entities = result.scalars().all()
            
            events = []
            for entity in entities:
                event_dict = {
                    "metadata": {
                        "event_id": entity.event_id,
                        "event_type": entity.event_type,
                        "aggregate_id": entity.aggregate_id,
                        "aggregate_type": entity.aggregate_type,
                        "aggregate_version": entity.aggregate_version,
                        "timestamp": entity.timestamp.isoformat(),
                        "user_id": entity.user_id,
                        "correlation_id": entity.correlation_id,
                        "causation_id": entity.causation_id,
                        "metadata": entity.event_meta or {}
                    },
                    "data": entity.event_data
                }
                
                event = EventRegistry.from_dict(event_dict)
                if event:
                    events.append(event)
            
            return events
    
    async def get_all_events(
        self,
        from_position: int = 0,
        limit: Optional[int] = None
    ) -> AsyncIterator[DomainEvent]:
        """Get all events from a global position."""
        batch_size = min(limit or 1000, 1000)
        current_position = from_position
        total_yielded = 0
        
        while True:
            async with AsyncSession(self.engine) as session:
                query = select(EventEntity).where(
                    EventEntity.global_position >= current_position
                ).order_by(EventEntity.global_position).limit(batch_size)
                
                result = await session.execute(query)
                entities = result.scalars().all()
                
                if not entities:
                    break
                
                for entity in entities:
                    if limit and total_yielded >= limit:
                        return
                    
                    event_dict = {
                        "metadata": {
                            "event_id": entity.event_id,
                            "event_type": entity.event_type,
                            "aggregate_id": entity.aggregate_id,
                            "aggregate_type": entity.aggregate_type,
                            "aggregate_version": entity.aggregate_version,
                            "timestamp": entity.timestamp.isoformat(),
                            "user_id": entity.user_id,
                            "correlation_id": entity.correlation_id,
                            "causation_id": entity.causation_id,
                            "metadata": entity.event_meta or {}
                        },
                        "data": entity.event_data
                    }
                    
                    event = EventRegistry.from_dict(event_dict)
                    if event:
                        yield event
                        total_yielded += 1
                
                current_position = entities[-1].global_position + 1
    
    async def get_events_by_type(
        self,
        event_type: str,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[DomainEvent]:
        """Get events by type within time range."""
        async with AsyncSession(self.engine) as session:
            query = select(EventEntity).where(
                EventEntity.event_type == event_type
            )
            
            if from_timestamp:
                query = query.where(EventEntity.timestamp >= from_timestamp)
            
            if to_timestamp:
                query = query.where(EventEntity.timestamp <= to_timestamp)
            
            query = query.order_by(EventEntity.timestamp)
            
            if limit:
                query = query.limit(limit)
            
            result = await session.execute(query)
            entities = result.scalars().all()
            
            events = []
            for entity in entities:
                event_dict = {
                    "metadata": {
                        "event_id": entity.event_id,
                        "event_type": entity.event_type,
                        "aggregate_id": entity.aggregate_id,
                        "aggregate_type": entity.aggregate_type,
                        "aggregate_version": entity.aggregate_version,
                        "timestamp": entity.timestamp.isoformat(),
                        "user_id": entity.user_id,
                        "correlation_id": entity.correlation_id,
                        "causation_id": entity.causation_id,
                        "metadata": entity.event_meta or {}
                    },
                    "data": entity.event_data
                }
                
                event = EventRegistry.from_dict(event_dict)
                if event:
                    events.append(event)
            
            return events
    
    async def get_snapshot(
        self,
        aggregate_id: UUID,
        max_version: Optional[int] = None
    ) -> Optional[Snapshot]:
        """Get latest snapshot for an aggregate."""
        async with AsyncSession(self.engine) as session:
            query = text("""
                SELECT aggregate_id, aggregate_type, version, state, timestamp
                FROM aggregate_snapshots
                WHERE aggregate_id = :aggregate_id
                AND (:max_version IS NULL OR version <= :max_version)
                ORDER BY version DESC
                LIMIT 1
            """)
            
            result = await session.execute(
                query,
                {
                    "aggregate_id": str(aggregate_id),
                    "max_version": max_version
                }
            )
            
            row = result.first()
            if row:
                return Snapshot(
                    aggregate_id=UUID(row.aggregate_id),
                    aggregate_type=row.aggregate_type,
                    version=row.version,
                    state=row.state,
                    timestamp=row.timestamp
                )
            
            return None
    
    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """Save an aggregate snapshot."""
        async with AsyncSession(self.engine) as session:
            await session.execute(
                text("""
                    INSERT INTO aggregate_snapshots 
                    (aggregate_id, aggregate_type, version, state, timestamp)
                    VALUES (:aggregate_id, :aggregate_type, :version, :state, :timestamp)
                    ON CONFLICT (aggregate_id, version) DO UPDATE
                    SET state = EXCLUDED.state, timestamp = EXCLUDED.timestamp
                """),
                {
                    "aggregate_id": str(snapshot.aggregate_id),
                    "aggregate_type": snapshot.aggregate_type,
                    "version": snapshot.version,
                    "state": json.dumps(snapshot.state),
                    "timestamp": snapshot.timestamp
                }
            )
            await session.commit()
    
    async def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to events of a specific type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
    
    async def _notify_subscribers(self, event: DomainEvent) -> None:
        """Notify subscribers of new events."""
        handlers = self._subscribers.get(event.metadata.event_type, [])
        handlers.extend(self._subscribers.get("*", []))  # Wildcard subscribers
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                # Log error but don't fail event append
                print(f"Subscriber error: {e}")