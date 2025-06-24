"""Flow control and backpressure mechanisms for the actor system."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
from collections import deque

from .core import Actor, Message


class FlowControlStrategy(Enum):
    """Flow control strategies."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    ADAPTIVE_WINDOW = "adaptive_window"
    CIRCUIT_BREAKER = "circuit_breaker"


class PressureLevel(Enum):
    """Backpressure levels."""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class FlowControlMetrics:
    """Metrics for flow control monitoring."""

    requests_total: int = 0
    requests_allowed: int = 0
    requests_rejected: int = 0
    current_pressure: PressureLevel = PressureLevel.NONE
    average_response_time: float = 0.0
    throughput: float = 0.0
    last_updated: float = field(default_factory=time.time)


class FlowController(ABC):
    """Abstract base class for flow controllers."""

    def __init__(self, name: str):
        self.name = name
        self.metrics = FlowControlMetrics()
        self.logger = logging.getLogger(f"flow_control.{name}")

    @abstractmethod
    async def should_allow_request(self, message: Message) -> bool:
        """Determine if a request should be allowed."""
        pass

    @abstractmethod
    async def on_request_completed(self, message: Message, success: bool, duration: float) -> None:
        """Called when a request completes."""
        pass

    @abstractmethod
    def get_pressure_level(self) -> PressureLevel:
        """Get current pressure level."""
        pass


class TokenBucketFlowController(FlowController):
    """Token bucket rate limiter with burst capacity."""

    def __init__(
        self,
        name: str,
        rate: float,  # tokens per second
        burst_capacity: int,
        refill_interval: float = 0.1,
    ):
        super().__init__(name)
        self.rate = rate
        self.burst_capacity = burst_capacity
        self.refill_interval = refill_interval

        self.tokens = burst_capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

        # Start refill task
        self._refill_task = asyncio.create_task(self._refill_tokens())

    async def should_allow_request(self, message: Message) -> bool:
        """Check if request should be allowed based on token availability."""
        async with self._lock:
            self.metrics.requests_total += 1

            if self.tokens >= 1:
                self.tokens -= 1
                self.metrics.requests_allowed += 1
                return True
            else:
                self.metrics.requests_rejected += 1
                self.logger.debug(f"Request {message.id} rejected - no tokens available")
                return False

    async def on_request_completed(self, message: Message, success: bool, duration: float) -> None:
        """Update metrics on request completion."""
        # Update average response time using exponential moving average
        alpha = 0.1
        self.metrics.average_response_time = (
            alpha * duration + (1 - alpha) * self.metrics.average_response_time
        )

    def get_pressure_level(self) -> PressureLevel:
        """Get pressure level based on token availability."""
        ratio = self.tokens / self.burst_capacity

        if ratio > 0.8:
            return PressureLevel.NONE
        elif ratio > 0.6:
            return PressureLevel.LOW
        elif ratio > 0.3:
            return PressureLevel.MEDIUM
        elif ratio > 0.1:
            return PressureLevel.HIGH
        else:
            return PressureLevel.CRITICAL

    async def _refill_tokens(self) -> None:
        """Periodically refill tokens."""
        while True:
            try:
                await asyncio.sleep(self.refill_interval)

                async with self._lock:
                    now = time.time()
                    elapsed = now - self.last_refill
                    tokens_to_add = self.rate * elapsed

                    self.tokens = min(self.burst_capacity, self.tokens + tokens_to_add)
                    self.last_refill = now

            except asyncio.CancelledError:
                break

    async def stop(self) -> None:
        """Stop the token refill task."""
        if self._refill_task:
            self._refill_task.cancel()
            try:
                await self._refill_task
            except asyncio.CancelledError:
                pass


class SlidingWindowFlowController(FlowController):
    """Sliding window rate limiter."""

    def __init__(
        self,
        name: str,
        window_size: float,  # seconds
        max_requests: int,
        cleanup_interval: float = 1.0,
    ):
        super().__init__(name)
        self.window_size = window_size
        self.max_requests = max_requests
        self.cleanup_interval = cleanup_interval

        self.request_times: deque = deque()
        self._lock = asyncio.Lock()

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_old_requests())

    async def should_allow_request(self, message: Message) -> bool:
        """Check if request should be allowed based on sliding window."""
        async with self._lock:
            now = time.time()
            self.metrics.requests_total += 1

            # Remove old requests outside the window
            while self.request_times and self.request_times[0] <= now - self.window_size:
                self.request_times.popleft()

            if len(self.request_times) < self.max_requests:
                self.request_times.append(now)
                self.metrics.requests_allowed += 1
                return True
            else:
                self.metrics.requests_rejected += 1
                return False

    async def on_request_completed(self, message: Message, success: bool, duration: float) -> None:
        """Update metrics on request completion."""
        alpha = 0.1
        self.metrics.average_response_time = (
            alpha * duration + (1 - alpha) * self.metrics.average_response_time
        )

        # Update throughput
        now = time.time()
        if now - self.metrics.last_updated > 1.0:
            async with self._lock:
                current_requests = len(self.request_times)
                self.metrics.throughput = current_requests / self.window_size
                self.metrics.last_updated = now

    def get_pressure_level(self) -> PressureLevel:
        """Get pressure level based on window utilization."""
        utilization = len(self.request_times) / self.max_requests

        if utilization <= 0.5:
            return PressureLevel.NONE
        elif utilization <= 0.7:
            return PressureLevel.LOW
        elif utilization <= 0.85:
            return PressureLevel.MEDIUM
        elif utilization <= 0.95:
            return PressureLevel.HIGH
        else:
            return PressureLevel.CRITICAL

    async def _cleanup_old_requests(self) -> None:
        """Periodically clean up old request timestamps."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)

                async with self._lock:
                    now = time.time()
                    while self.request_times and self.request_times[0] <= now - self.window_size:
                        self.request_times.popleft()

            except asyncio.CancelledError:
                break

    async def stop(self) -> None:
        """Stop the cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


class AdaptiveWindowFlowController(FlowController):
    """Adaptive flow controller that adjusts limits based on system performance."""

    def __init__(
        self,
        name: str,
        initial_limit: int = 100,
        min_limit: int = 10,
        max_limit: int = 1000,
        adjustment_interval: float = 5.0,
        target_response_time: float = 1.0,  # seconds
    ):
        super().__init__(name)
        self.current_limit = initial_limit
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.adjustment_interval = adjustment_interval
        self.target_response_time = target_response_time

        self.active_requests = 0
        self.recent_response_times: deque = deque(maxlen=100)
        self.recent_errors: deque = deque(maxlen=100)

        self._lock = asyncio.Lock()
        self._adjustment_task = asyncio.create_task(self._adjust_limits())

    async def should_allow_request(self, message: Message) -> bool:
        """Check if request should be allowed based on current limit."""
        async with self._lock:
            self.metrics.requests_total += 1

            if self.active_requests < self.current_limit:
                self.active_requests += 1
                self.metrics.requests_allowed += 1
                return True
            else:
                self.metrics.requests_rejected += 1
                return False

    async def on_request_completed(self, message: Message, success: bool, duration: float) -> None:
        """Update metrics and active request count."""
        async with self._lock:
            self.active_requests = max(0, self.active_requests - 1)

            self.recent_response_times.append(duration)
            self.recent_errors.append(not success)

            # Update average response time
            if self.recent_response_times:
                self.metrics.average_response_time = sum(self.recent_response_times) / len(
                    self.recent_response_times
                )

    def get_pressure_level(self) -> PressureLevel:
        """Get pressure level based on active requests."""
        utilization = self.active_requests / self.current_limit

        if utilization <= 0.5:
            return PressureLevel.NONE
        elif utilization <= 0.7:
            return PressureLevel.LOW
        elif utilization <= 0.85:
            return PressureLevel.MEDIUM
        elif utilization <= 0.95:
            return PressureLevel.HIGH
        else:
            return PressureLevel.CRITICAL

    async def _adjust_limits(self) -> None:
        """Periodically adjust limits based on performance."""
        while True:
            try:
                await asyncio.sleep(self.adjustment_interval)

                async with self._lock:
                    if not self.recent_response_times:
                        continue

                    avg_response_time = sum(self.recent_response_times) / len(
                        self.recent_response_times
                    )
                    error_rate = (
                        sum(self.recent_errors) / len(self.recent_errors)
                        if self.recent_errors
                        else 0
                    )

                    # Adjust limits based on performance
                    if avg_response_time < self.target_response_time * 0.8 and error_rate < 0.01:
                        # Performance is good, increase limit
                        new_limit = min(self.max_limit, int(self.current_limit * 1.1))
                        if new_limit != self.current_limit:
                            self.logger.info(
                                f"Increasing limit from {self.current_limit} to {new_limit}"
                            )
                            self.current_limit = new_limit

                    elif avg_response_time > self.target_response_time * 1.2 or error_rate > 0.05:
                        # Performance is poor, decrease limit
                        new_limit = max(self.min_limit, int(self.current_limit * 0.9))
                        if new_limit != self.current_limit:
                            self.logger.info(
                                f"Decreasing limit from {self.current_limit} to {new_limit}"
                            )
                            self.current_limit = new_limit

            except asyncio.CancelledError:
                break

    async def stop(self) -> None:
        """Stop the adjustment task."""
        if self._adjustment_task:
            self._adjustment_task.cancel()
            try:
                await self._adjustment_task
            except asyncio.CancelledError:
                pass


class BackpressureManager:
    """Manages backpressure across multiple actors and flow controllers."""

    def __init__(self):
        self.controllers: Dict[str, FlowController] = {}
        self.subscribers: Dict[PressureLevel, List[Callable[[PressureLevel], None]]] = {
            level: [] for level in PressureLevel
        }
        self.global_pressure = PressureLevel.NONE
        self.logger = logging.getLogger("backpressure_manager")

    def register_controller(self, controller: FlowController) -> None:
        """Register a flow controller."""
        self.controllers[controller.name] = controller
        self.logger.info(f"Registered flow controller: {controller.name}")

    def unregister_controller(self, name: str) -> None:
        """Unregister a flow controller."""
        if name in self.controllers:
            del self.controllers[name]
            self.logger.info(f"Unregistered flow controller: {name}")

    def subscribe_to_pressure_changes(
        self, pressure_level: PressureLevel, callback: Callable[[PressureLevel], None]
    ) -> None:
        """Subscribe to pressure level changes."""
        self.subscribers[pressure_level].append(callback)

    def calculate_global_pressure(self) -> PressureLevel:
        """Calculate global pressure from all controllers."""
        if not self.controllers:
            return PressureLevel.NONE

        max_pressure = PressureLevel.NONE
        for controller in self.controllers.values():
            pressure = controller.get_pressure_level()
            if pressure.value > max_pressure.value:
                max_pressure = pressure

        # Update global pressure and notify subscribers
        if max_pressure != self.global_pressure:
            old_pressure = self.global_pressure
            self.global_pressure = max_pressure

            self.logger.info(f"Global pressure changed from {old_pressure} to {max_pressure}")

            # Notify subscribers
            for callback in self.subscribers[max_pressure]:
                try:
                    callback(max_pressure)
                except Exception as e:
                    self.logger.error(f"Error notifying pressure subscriber: {e}")

        return max_pressure

    def get_controller_metrics(self) -> Dict[str, FlowControlMetrics]:
        """Get metrics from all controllers."""
        return {name: controller.metrics for name, controller in self.controllers.items()}

    async def stop_all_controllers(self) -> None:
        """Stop all flow controllers."""
        tasks = []
        for controller in self.controllers.values():
            if hasattr(controller, "stop"):
                tasks.append(controller.stop())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class FlowControlledActor(Actor):
    """Actor with built-in flow control."""

    def __init__(
        self,
        name: str,
        flow_controller: FlowController,
        mailbox_size: int = 1000,
        overflow_strategy=None,
    ):
        super().__init__(name, mailbox_size, overflow_strategy)
        self.flow_controller = flow_controller
        self._active_requests: Set[str] = set()

    async def receive(self, message: Message) -> None:
        """Receive message with flow control check."""
        # Check flow control before accepting message
        if not await self.flow_controller.should_allow_request(message):
            self.logger.warning(f"Message {message.id} rejected by flow control")
            return

        # Track active request
        self._active_requests.add(message.id)
        await super().receive(message)

    async def _handle_message(self, message: Message) -> None:
        """Handle message with timing for flow control."""
        start_time = time.time()
        success = True

        try:
            await super()._handle_message(message)
        except Exception as e:
            success = False
            raise
        finally:
            # Remove from active requests and notify flow controller
            self._active_requests.discard(message.id)
            duration = time.time() - start_time
            await self.flow_controller.on_request_completed(message, success, duration)

    def get_active_request_count(self) -> int:
        """Get number of active requests."""
        return len(self._active_requests)
