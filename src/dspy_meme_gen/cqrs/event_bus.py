"""Event bus for publishing/subscribing to domain events."""

import asyncio
import logging
from typing import Dict, List, Callable, Optional, Any, Set
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass, field
from collections import defaultdict
import weakref

from .events import DomainEvent
from .event_store import get_event_store


logger = logging.getLogger(__name__)


@dataclass
class Subscription:
    """Event subscription information."""
    
    subscriber_id: str
    event_type: str
    handler: Callable[[DomainEvent], None]
    is_async: bool
    priority: int = 0
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True


class EventBus:
    """Async event bus for domain events."""
    
    def __init__(self, buffer_size: int = 10000):
        self._subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self._subscriber_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._event_buffer: asyncio.Queue = asyncio.Queue(maxsize=buffer_size)
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "failed_deliveries": 0,
            "active_subscriptions": 0
        }
        self._dead_letter_queue: List[Dict[str, Any]] = []
        
    async def start(self) -> None:
        """Start the event bus processing."""
        if self._running:
            return
            
        self._running = True
        self._processing_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
    
    async def stop(self) -> None:
        """Stop the event bus processing."""
        if not self._running:
            return
            
        self._running = False
        
        # Wait for current events to process
        if self._processing_task:
            try:
                await asyncio.wait_for(self._processing_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._processing_task.cancel()
                logger.warning("Event bus processing task timed out during shutdown")
        
        logger.info("Event bus stopped")
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable[[DomainEvent], None],
        subscriber_id: Optional[str] = None,
        priority: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Subscribe to events of a specific type."""
        if subscriber_id is None:
            subscriber_id = f"subscriber_{len(self._subscriptions)}_{id(handler)}"
        
        is_async = asyncio.iscoroutinefunction(handler)
        subscription = Subscription(
            subscriber_id=subscriber_id,
            event_type=event_type,
            handler=handler,
            is_async=is_async,
            priority=priority,
            filters=filters or {}
        )
        
        self._subscriptions[event_type].append(subscription)
        # Sort by priority (higher priority first)
        self._subscriptions[event_type].sort(key=lambda s: s.priority, reverse=True)
        
        # Store weak reference to subscriber for cleanup
        self._subscriber_refs[subscriber_id] = subscription
        
        self._stats["active_subscriptions"] += 1
        
        logger.info(f"Subscribed {subscriber_id} to {event_type} (async={is_async}, priority={priority})")
        return subscriber_id
    
    def unsubscribe(self, subscriber_id: str, event_type: Optional[str] = None) -> None:
        """Unsubscribe from events."""
        removed_count = 0
        
        if event_type:
            # Remove from specific event type
            subscriptions = self._subscriptions[event_type]
            original_count = len(subscriptions)
            self._subscriptions[event_type] = [
                s for s in subscriptions if s.subscriber_id != subscriber_id
            ]
            removed_count = original_count - len(self._subscriptions[event_type])
        else:
            # Remove from all event types
            for event_type_key in list(self._subscriptions.keys()):
                subscriptions = self._subscriptions[event_type_key]
                original_count = len(subscriptions)
                self._subscriptions[event_type_key] = [
                    s for s in subscriptions if s.subscriber_id != subscriber_id
                ]
                removed_count += original_count - len(self._subscriptions[event_type_key])
        
        # Remove from weak references
        self._subscriber_refs.pop(subscriber_id, None)
        
        self._stats["active_subscriptions"] -= removed_count
        
        if removed_count > 0:
            logger.info(f"Unsubscribed {subscriber_id} from {removed_count} subscriptions")
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish an event to the bus."""
        if not self._running:
            logger.warning("Event bus not running, event will be dropped")
            return
        
        try:
            await self._event_buffer.put(event)
            self._stats["events_published"] += 1
        except asyncio.QueueFull:
            logger.error(f"Event buffer full, dropping event {event.metadata.event_id}")
            self._dead_letter_queue.append({
                "event": event.to_dict(),
                "reason": "buffer_full",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def publish_batch(self, events: List[DomainEvent]) -> None:
        """Publish multiple events."""
        for event in events:
            await self.publish(event)
    
    async def _process_events(self) -> None:
        """Main event processing loop."""
        logger.info("Event processing started")
        
        while self._running or not self._event_buffer.empty():
            try:
                # Get event with timeout to allow graceful shutdown
                try:
                    event = await asyncio.wait_for(
                        self._event_buffer.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                await self._deliver_event(event)
                self._stats["events_processed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing event: {e}", exc_info=True)
                self._stats["failed_deliveries"] += 1
        
        logger.info("Event processing stopped")
    
    async def _deliver_event(self, event: DomainEvent) -> None:
        """Deliver event to subscribers."""
        event_type = event.metadata.event_type
        
        # Get subscribers for specific event type and wildcard subscribers
        subscribers = []
        subscribers.extend(self._subscriptions.get(event_type, []))
        subscribers.extend(self._subscriptions.get("*", []))
        
        if not subscribers:
            return
        
        # Group by priority and execute
        priority_groups = defaultdict(list)
        for subscription in subscribers:
            if subscription.active and self._matches_filters(event, subscription.filters):
                priority_groups[subscription.priority].append(subscription)
        
        # Execute in priority order (highest first)
        for priority in sorted(priority_groups.keys(), reverse=True):
            # Execute all handlers in the same priority group concurrently
            tasks = []
            for subscription in priority_groups[priority]:
                task = self._execute_handler(event, subscription)
                tasks.append(task)
            
            if tasks:
                # Wait for all handlers in this priority group to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log any exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        subscription = priority_groups[priority][i]
                        logger.error(
                            f"Handler {subscription.subscriber_id} failed for event {event.metadata.event_id}: {result}",
                            exc_info=result
                        )
                        self._stats["failed_deliveries"] += 1
    
    async def _execute_handler(self, event: DomainEvent, subscription: Subscription) -> None:
        """Execute a single event handler."""
        try:
            if subscription.is_async:
                await subscription.handler(event)
            else:
                # Run sync handler in executor to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, subscription.handler, event)
        except Exception as e:
            # Re-raise to be caught by gather
            raise e
    
    def _matches_filters(self, event: DomainEvent, filters: Dict[str, Any]) -> bool:
        """Check if event matches subscription filters."""
        if not filters:
            return True
        
        for key, expected_value in filters.items():
            if key.startswith("metadata."):
                # Filter on metadata fields
                metadata_key = key[9:]  # Remove "metadata." prefix
                actual_value = getattr(event.metadata, metadata_key, None)
            elif key.startswith("data."):
                # Filter on event data fields
                data_key = key[5:]  # Remove "data." prefix
                actual_value = getattr(event, data_key, None)
            else:
                # Filter on top-level event fields
                actual_value = getattr(event, key, None)
            
            if actual_value != expected_value:
                return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            **self._stats,
            "buffer_size": self._event_buffer.qsize(),
            "dead_letter_count": len(self._dead_letter_queue),
            "subscription_count": sum(len(subs) for subs in self._subscriptions.values()),
            "event_types": list(self._subscriptions.keys())
        }
    
    def get_subscriptions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get current subscriptions."""
        result = {}
        for event_type, subscriptions in self._subscriptions.items():
            result[event_type] = [
                {
                    "subscriber_id": s.subscriber_id,
                    "is_async": s.is_async,
                    "priority": s.priority,
                    "filters": s.filters,
                    "created_at": s.created_at.isoformat(),
                    "active": s.active
                }
                for s in subscriptions
            ]
        return result
    
    def get_dead_letters(self) -> List[Dict[str, Any]]:
        """Get dead letter queue."""
        return self._dead_letter_queue.copy()
    
    def clear_dead_letters(self) -> None:
        """Clear dead letter queue."""
        self._dead_letter_queue.clear()


class IntegratedEventBus(EventBus):
    """Event bus integrated with event store."""
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish event and persist to event store."""
        # First persist to event store
        event_store = await get_event_store()
        await event_store.append(event)
        
        # Then publish to bus for real-time subscribers
        await super().publish(event)
    
    async def publish_batch(self, events: List[DomainEvent]) -> None:
        """Publish batch of events and persist to event store."""
        # First persist to event store
        event_store = await get_event_store()
        await event_store.append_batch(events)
        
        # Then publish to bus
        await super().publish_batch(events)


# Global event bus instance
_event_bus: Optional[IntegratedEventBus] = None


async def get_event_bus() -> IntegratedEventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = IntegratedEventBus()
        await _event_bus.start()
        logger.info("Global event bus initialized")
    return _event_bus


async def shutdown_event_bus() -> None:
    """Shutdown the global event bus."""
    global _event_bus
    if _event_bus:
        await _event_bus.stop()
        _event_bus = None
        logger.info("Global event bus shutdown")


# Convenience functions
async def publish_event(event: DomainEvent) -> None:
    """Publish a single event."""
    bus = await get_event_bus()
    await bus.publish(event)


async def publish_events(events: List[DomainEvent]) -> None:
    """Publish multiple events."""
    bus = await get_event_bus()
    await bus.publish_batch(events)


def subscribe_to_events(
    event_type: str,
    handler: Callable[[DomainEvent], None],
    subscriber_id: Optional[str] = None,
    priority: int = 0,
    filters: Optional[Dict[str, Any]] = None
) -> str:
    """Subscribe to events (sync function for easier use)."""
    async def _subscribe():
        bus = await get_event_bus()
        return bus.subscribe(event_type, handler, subscriber_id, priority, filters)
    
    # If we're in an async context, await directly
    try:
        loop = asyncio.get_running_loop()
        # Create a task and run it
        task = loop.create_task(_subscribe())
        return task
    except RuntimeError:
        # Not in async context, create new event loop
        return asyncio.run(_subscribe())


def unsubscribe_from_events(subscriber_id: str, event_type: Optional[str] = None) -> None:
    """Unsubscribe from events (sync function for easier use)."""
    async def _unsubscribe():
        bus = await get_event_bus()
        bus.unsubscribe(subscriber_id, event_type)
    
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_unsubscribe())
    except RuntimeError:
        asyncio.run(_unsubscribe())


# Actor integration
class ActorEventSubscriber:
    """Helper for actors to subscribe to events."""
    
    def __init__(self, actor_ref, event_bus: EventBus):
        self.actor_ref = actor_ref
        self.event_bus = event_bus
        self.subscriptions: Set[str] = set()
    
    async def subscribe(
        self,
        event_type: str,
        priority: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Subscribe actor to events."""
        async def forward_to_actor(event: DomainEvent):
            """Forward event to actor as a message."""
            from ..actors.messages import EventMessage
            event_msg = EventMessage(event=event)
            await self.actor_ref.tell(event_msg)
        
        subscriber_id = f"actor_{self.actor_ref.path}_{event_type}"
        actual_id = self.event_bus.subscribe(
            event_type, forward_to_actor, subscriber_id, priority, filters
        )
        self.subscriptions.add(actual_id)
        return actual_id
    
    async def unsubscribe_all(self) -> None:
        """Unsubscribe actor from all events."""
        for subscription_id in self.subscriptions:
            self.event_bus.unsubscribe(subscription_id)
        self.subscriptions.clear()


async def subscribe_actor_to_events(
    actor_ref,
    event_types: List[str],
    priority: int = 0,
    filters: Optional[Dict[str, Any]] = None
) -> ActorEventSubscriber:
    """Subscribe an actor to multiple event types."""
    event_bus = await get_event_bus()
    subscriber = ActorEventSubscriber(actor_ref, event_bus)
    
    for event_type in event_types:
        await subscriber.subscribe(event_type, priority, filters)
    
    return subscriber