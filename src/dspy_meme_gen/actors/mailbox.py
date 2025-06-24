"""Actor mailbox implementation with backpressure."""

import asyncio
import logging
from collections import deque
from enum import Enum
from typing import Deque, Optional

from .base_messages import Message


class OverflowStrategy(Enum):
    """Strategy for handling mailbox overflow."""
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"
    ERROR = "error"


class ActorMailbox:
    """Thread-safe mailbox for actor message queuing."""
    
    def __init__(
        self,
        capacity: int = 1000,
        overflow_strategy: OverflowStrategy = OverflowStrategy.DROP_OLDEST
    ):
        self.capacity = capacity
        self.overflow_strategy = overflow_strategy
        self._queue: Optional[asyncio.Queue] = None
        self._backpressure_event: Optional[asyncio.Event] = None
        self.high_water_mark = int(capacity * 0.8)
        self.low_water_mark = int(capacity * 0.2)
        self.logger = logging.getLogger("mailbox")
        
        # Metrics
        self.messages_received = 0
        self.messages_dropped = 0
        self.messages_processed = 0
        
    def _ensure_initialized(self):
        """Ensure queue is initialized in the correct event loop."""
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.capacity)
        if self._backpressure_event is None:
            self._backpressure_event = asyncio.Event()
            self._backpressure_event.set()  # Not under pressure initially
        
    async def put(self, message: Message) -> bool:
        """Put a message in the mailbox."""
        self._ensure_initialized()
        self.messages_received += 1
        
        if self._queue.qsize() >= self.capacity:
            return await self._handle_overflow(message)
            
        try:
            self._queue.put_nowait(message)
            
            # Check for backpressure
            if self._queue.qsize() >= self.high_water_mark:
                self._signal_backpressure()
                
            return True
            
        except asyncio.QueueFull:
            return await self._handle_overflow(message)
            
    async def get(self) -> Optional[Message]:
        """Get a message from the mailbox."""
        self._ensure_initialized()
        try:
            message = await self._queue.get()
            self.messages_processed += 1
            
            # Check if we can relieve backpressure
            if self._queue.qsize() <= self.low_water_mark:
                self._relieve_backpressure()
                
            return message
            
        except asyncio.CancelledError:
            return None
            
    async def _handle_overflow(self, message: Message) -> bool:
        """Handle mailbox overflow based on strategy."""
        if self.overflow_strategy == OverflowStrategy.DROP_OLDEST:
            try:
                # Remove oldest message
                self._queue.get_nowait()
                self.messages_dropped += 1
                # Add new message
                self._queue.put_nowait(message)
                return True
            except asyncio.QueueEmpty:
                # Race condition, try again
                return await self.put(message)
                
        elif self.overflow_strategy == OverflowStrategy.DROP_NEWEST:
            self.messages_dropped += 1
            self.logger.warning(f"Dropping message {message.id} due to overflow")
            return False
            
        elif self.overflow_strategy == OverflowStrategy.BLOCK:
            # Block until space is available
            await self._queue.put(message)
            return True
            
        elif self.overflow_strategy == OverflowStrategy.ERROR:
            raise OverflowError(f"Mailbox full, cannot accept message {message.id}")
            
    def _signal_backpressure(self) -> None:
        """Signal that mailbox is under pressure."""
        self._ensure_initialized()
        if self._backpressure_event.is_set():
            self.logger.warning("Mailbox under pressure, signaling backpressure")
            self._backpressure_event.clear()
            
    def _relieve_backpressure(self) -> None:
        """Signal that backpressure is relieved."""
        self._ensure_initialized()
        if not self._backpressure_event.is_set():
            self.logger.info("Mailbox pressure relieved")
            self._backpressure_event.set()
            
    async def wait_for_capacity(self) -> None:
        """Wait until mailbox has capacity."""
        self._ensure_initialized()
        await self._backpressure_event.wait()
        
    def is_under_pressure(self) -> bool:
        """Check if mailbox is under pressure."""
        self._ensure_initialized()
        return not self._backpressure_event.is_set()
        
    def size(self) -> int:
        """Get current mailbox size."""
        self._ensure_initialized()
        return self._queue.qsize()
        
    def is_empty(self) -> bool:
        """Check if mailbox is empty."""
        self._ensure_initialized()
        return self._queue.empty()
        
    def is_full(self) -> bool:
        """Check if mailbox is full."""
        self._ensure_initialized()
        return self._queue.qsize() >= self.capacity
        
    @property
    def size(self) -> int:
        """Get current mailbox size."""
        self._ensure_initialized()
        return self._queue.qsize()
        
    @property
    def is_empty(self) -> bool:
        """Check if mailbox is empty."""
        self._ensure_initialized()
        return self._queue.empty()
        
    @property
    def is_full(self) -> bool:
        """Check if mailbox is full."""
        self._ensure_initialized()
        return self._queue.qsize() >= self.capacity
        
    @property
    def remaining_capacity(self) -> int:
        """Get remaining mailbox capacity."""
        self._ensure_initialized()
        return self.capacity - self._queue.qsize()