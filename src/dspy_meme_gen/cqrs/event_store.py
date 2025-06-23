"""Async event store implementation for meme generation pipeline."""

import asyncio
from typing import Dict, List, Optional, AsyncIterator, Callable, Any
from uuid import UUID
from datetime import datetime
import logging

from .store.event_store import PostgresEventStore, EventStore
from .events import DomainEvent
from .events import *


logger = logging.getLogger(__name__)


class MemeEventStore(PostgresEventStore):
    """Extended event store with meme-specific functionality."""
    
    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self._projection_handlers: Dict[str, List[Callable]] = {}
        
    async def get_meme_generation_stream(
        self, 
        request_id: UUID,
        from_position: int = 0
    ) -> List[DomainEvent]:
        """Get all events for a specific meme generation request."""
        async with self.engine.begin() as conn:
            query = """
                SELECT * FROM event_store 
                WHERE event_data->>'request_id' = %s
                AND global_position >= %s
                ORDER BY global_position
            """
            result = await conn.execute(query, (str(request_id), from_position))
            
            events = []
            for row in result:
                event_dict = {
                    "metadata": {
                        "event_id": row.event_id,
                        "event_type": row.event_type,
                        "aggregate_id": row.aggregate_id,
                        "aggregate_type": row.aggregate_type,
                        "aggregate_version": row.aggregate_version,
                        "timestamp": row.timestamp.isoformat(),
                        "user_id": row.user_id,
                        "correlation_id": row.correlation_id,
                        "causation_id": row.causation_id,
                        "metadata": row.metadata or {}
                    },
                    "data": row.event_data
                }
                
                event = EventRegistry.from_dict(event_dict)
                if event:
                    events.append(event)
            
            return events
    
    async def get_meme_pipeline_events(
        self,
        meme_id: UUID,
        stage_filter: Optional[List[str]] = None
    ) -> List[DomainEvent]:
        """Get all pipeline events for a specific meme."""
        pipeline_event_types = [
            "meme.generation.started",
            "meme.text.generated", 
            "meme.image.generated",
            "meme.quality.scored",
            "meme.verification.requested",
            "meme.verification.completed",
            "meme.refinement.requested",
            "meme.completed",
            "pipeline.stage.started",
            "pipeline.stage.completed"
        ]
        
        if stage_filter:
            pipeline_event_types = [t for t in pipeline_event_types if any(s in t for s in stage_filter)]
        
        events = []
        for event_type in pipeline_event_types:
            type_events = await self.get_events_by_type(event_type)
            # Filter by meme_id
            filtered_events = [
                e for e in type_events 
                if hasattr(e, 'meme_id') and e.meme_id == meme_id
            ]
            events.extend(filtered_events)
        
        # Sort by timestamp
        events.sort(key=lambda e: e.metadata.timestamp)
        return events
    
    async def get_active_meme_generations(self) -> List[Dict[str, Any]]:
        """Get all currently active meme generations."""
        # Find all started generations
        started_events = await self.get_events_by_type("meme.generation.started")
        
        # Find all completed/failed generations  
        completed_events = await self.get_events_by_type("meme.completed")
        failed_events = await self.get_events_by_type("meme.generation.failed")
        
        completed_request_ids = {
            getattr(e, 'request_id', None) for e in completed_events + failed_events
        }
        
        # Return active ones
        active = []
        for event in started_events:
            if hasattr(event, 'request_id') and event.request_id not in completed_request_ids:
                active.append({
                    'request_id': event.request_id,
                    'topic': getattr(event, 'topic', 'unknown'),
                    'format': getattr(event, 'format', 'unknown'),
                    'started_at': event.metadata.timestamp,
                    'pipeline_config': getattr(event, 'pipeline_config', {})
                })
        
        return active
    
    async def get_meme_metrics(
        self,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get metrics about meme generation."""
        # Count different event types
        metrics = {
            'total_requests': len(await self.get_events_by_type(
                "meme.generation.started", from_timestamp, to_timestamp
            )),
            'successful_generations': len(await self.get_events_by_type(
                "meme.generated", from_timestamp, to_timestamp
            )),
            'failed_generations': len(await self.get_events_by_type(
                "meme.generation.failed", from_timestamp, to_timestamp
            )),
            'completed_pipelines': len(await self.get_events_by_type(
                "meme.completed", from_timestamp, to_timestamp
            )),
            'approvals': len(await self.get_events_by_type(
                "meme.approved", from_timestamp, to_timestamp
            )),
            'rejections': len(await self.get_events_by_type(
                "meme.rejected", from_timestamp, to_timestamp
            ))
        }
        
        # Calculate success rate
        if metrics['total_requests'] > 0:
            metrics['success_rate'] = metrics['successful_generations'] / metrics['total_requests']
            metrics['completion_rate'] = metrics['completed_pipelines'] / metrics['total_requests']
        else:
            metrics['success_rate'] = 0.0
            metrics['completion_rate'] = 0.0
        
        return metrics
    
    async def register_projection_handler(
        self, 
        event_type: str, 
        handler: Callable[[DomainEvent], None]
    ) -> None:
        """Register a projection handler for specific event types."""
        if event_type not in self._projection_handlers:
            self._projection_handlers[event_type] = []
        self._projection_handlers[event_type].append(handler)
        
        # Also subscribe to the event store
        await self.subscribe(event_type, handler)
    
    async def rebuild_projections(
        self, 
        from_position: int = 0,
        event_types: Optional[List[str]] = None
    ) -> None:
        """Rebuild projections from events."""
        logger.info(f"Rebuilding projections from position {from_position}")
        
        processed_count = 0
        async for event in self.get_all_events(from_position):
            if event_types and event.metadata.event_type not in event_types:
                continue
                
            handlers = self._projection_handlers.get(event.metadata.event_type, [])
            handlers.extend(self._projection_handlers.get("*", []))  # Wildcard handlers
            
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Projection handler error for event {event.metadata.event_id}: {e}")
            
            processed_count += 1
            
            if processed_count % 1000 == 0:
                logger.info(f"Processed {processed_count} events for projection rebuild")
        
        logger.info(f"Completed rebuilding projections. Processed {processed_count} events")


class EventStoreManager:
    """Manager for event store lifecycle and configuration."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._event_store: Optional[MemeEventStore] = None
        
    async def get_event_store(self) -> MemeEventStore:
        """Get or create event store instance."""
        if self._event_store is None:
            self._event_store = MemeEventStore(self.connection_string)
            await self._event_store.initialize()
            logger.info("Event store initialized")
        return self._event_store
    
    async def shutdown(self) -> None:
        """Shutdown event store connections."""
        if self._event_store:
            await self._event_store.engine.dispose()
            logger.info("Event store shutdown complete")


# Global event store manager instance
_event_store_manager: Optional[EventStoreManager] = None


async def initialize_event_store(connection_string: str) -> MemeEventStore:
    """Initialize the global event store."""
    global _event_store_manager
    _event_store_manager = EventStoreManager(connection_string)
    return await _event_store_manager.get_event_store()


async def get_event_store() -> MemeEventStore:
    """Get the global event store instance."""
    if _event_store_manager is None:
        raise RuntimeError("Event store not initialized. Call initialize_event_store() first.")
    return await _event_store_manager.get_event_store()


async def shutdown_event_store() -> None:
    """Shutdown the global event store."""
    global _event_store_manager
    if _event_store_manager:
        await _event_store_manager.shutdown()
        _event_store_manager = None