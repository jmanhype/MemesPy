"""Base projection definitions for CQRS read models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from uuid import UUID
import asyncio
from enum import Enum

from ..events.base import DomainEvent, EventType
from ..store.event_store import EventStore


class ProjectionStatus(Enum):
    """Status of a projection."""
    IDLE = "idle"
    BUILDING = "building"
    CATCHING_UP = "catching_up"
    LIVE = "live"
    ERROR = "error"
    REBUILDING = "rebuilding"


@dataclass
class ProjectionState:
    """State of a projection."""
    name: str
    status: ProjectionStatus
    last_position: int
    last_updated: datetime
    error: Optional[str] = None
    events_processed: int = 0
    processing_rate: float = 0.0  # events per second


class Projection(ABC):
    """Base class for all projections."""
    
    def __init__(self, name: str):
        self.name = name
        self.state = ProjectionState(
            name=name,
            status=ProjectionStatus.IDLE,
            last_position=0,
            last_updated=datetime.utcnow()
        )
        self._subscribed_events: Set[str] = set()
    
    @abstractmethod
    async def handle(self, event: DomainEvent, position: int) -> None:
        """Handle an event and update the projection."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the projection (create tables, indexes, etc)."""
        pass
    
    @abstractmethod
    async def reset(self) -> None:
        """Reset the projection to initial state."""
        pass
    
    def subscribes_to(self) -> Set[str]:
        """Return set of event types this projection handles."""
        return self._subscribed_events
    
    def subscribe(self, *event_types: str) -> None:
        """Subscribe to event types."""
        self._subscribed_events.update(event_types)
    
    async def get_checkpoint(self) -> int:
        """Get the last processed position."""
        return self.state.last_position
    
    async def save_checkpoint(self, position: int) -> None:
        """Save the last processed position."""
        self.state.last_position = position
        self.state.last_updated = datetime.utcnow()


class ProjectionManager:
    """Manages projections and event distribution."""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.projections: Dict[str, Projection] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    def register(self, projection: Projection) -> None:
        """Register a projection."""
        self.projections[projection.name] = projection
    
    async def initialize_all(self) -> None:
        """Initialize all registered projections."""
        for projection in self.projections.values():
            await projection.initialize()
    
    async def start(self) -> None:
        """Start processing events for all projections."""
        self._running = True
        
        # Start a task for each projection
        for projection in self.projections.values():
            task = asyncio.create_task(self._process_projection(projection))
            self._tasks.append(task)
    
    async def stop(self) -> None:
        """Stop processing events."""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
    
    async def rebuild_projection(self, projection_name: str) -> None:
        """Rebuild a specific projection from scratch."""
        projection = self.projections.get(projection_name)
        if not projection:
            raise ValueError(f"Projection {projection_name} not found")
        
        # Update status
        projection.state.status = ProjectionStatus.REBUILDING
        
        try:
            # Reset projection
            await projection.reset()
            
            # Process all events from the beginning
            position = 0
            async for event in self.event_store.get_all_events(from_position=0):
                if event.metadata.event_type in projection.subscribes_to():
                    await projection.handle(event, position)
                position += 1
            
            # Save final position
            await projection.save_checkpoint(position)
            projection.state.status = ProjectionStatus.LIVE
            
        except Exception as e:
            projection.state.status = ProjectionStatus.ERROR
            projection.state.error = str(e)
            raise
    
    async def _process_projection(self, projection: Projection) -> None:
        """Process events for a single projection."""
        projection.state.status = ProjectionStatus.CATCHING_UP
        
        try:
            while self._running:
                # Get checkpoint
                from_position = await projection.get_checkpoint()
                
                # Track processing metrics
                events_in_batch = 0
                batch_start = datetime.utcnow()
                
                # Process events
                has_events = False
                async for event in self.event_store.get_all_events(
                    from_position=from_position + 1,
                    limit=1000
                ):
                    has_events = True
                    
                    # Check if projection subscribes to this event
                    if event.metadata.event_type in projection.subscribes_to():
                        await projection.handle(event, from_position + events_in_batch + 1)
                        projection.state.events_processed += 1
                    
                    events_in_batch += 1
                    
                    # Save checkpoint periodically
                    if events_in_batch % 100 == 0:
                        await projection.save_checkpoint(from_position + events_in_batch)
                
                # Save final checkpoint
                if events_in_batch > 0:
                    await projection.save_checkpoint(from_position + events_in_batch)
                    
                    # Calculate processing rate
                    batch_duration = (datetime.utcnow() - batch_start).total_seconds()
                    if batch_duration > 0:
                        projection.state.processing_rate = events_in_batch / batch_duration
                
                # Update status
                if not has_events:
                    projection.state.status = ProjectionStatus.LIVE
                    # Wait before checking for new events
                    await asyncio.sleep(1)
                else:
                    projection.state.status = ProjectionStatus.CATCHING_UP
                    
        except asyncio.CancelledError:
            projection.state.status = ProjectionStatus.IDLE
            raise
        except Exception as e:
            projection.state.status = ProjectionStatus.ERROR
            projection.state.error = str(e)
            raise


class CompositeProjection(Projection):
    """Projection that delegates to multiple sub-projections."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self._sub_projections: List[Projection] = []
    
    def add_projection(self, projection: Projection) -> None:
        """Add a sub-projection."""
        self._sub_projections.append(projection)
        # Subscribe to all events that sub-projections handle
        for event_type in projection.subscribes_to():
            self.subscribe(event_type)
    
    async def handle(self, event: DomainEvent, position: int) -> None:
        """Delegate event to all sub-projections."""
        tasks = []
        for projection in self._sub_projections:
            if event.metadata.event_type in projection.subscribes_to():
                tasks.append(projection.handle(event, position))
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def initialize(self) -> None:
        """Initialize all sub-projections."""
        for projection in self._sub_projections:
            await projection.initialize()
    
    async def reset(self) -> None:
        """Reset all sub-projections."""
        for projection in self._sub_projections:
            await projection.reset()