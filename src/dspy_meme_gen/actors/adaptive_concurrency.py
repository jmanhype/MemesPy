"""Adaptive concurrency control using Little's Law and performance metrics."""

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable, Any
from statistics import mean, median, stdev

from .core import Actor, Message
from .flow_control import FlowController, PressureLevel


class ConcurrencyStrategy(Enum):
    """Concurrency control strategies."""

    LITTLES_LAW = "littles_law"
    GRADIENT_DESCENT = "gradient_descent"
    ADDITIVE_INCREASE = "additive_increase"
    MULTIPLICATIVE_DECREASE = "multiplicative_decrease"
    VEGAS = "vegas"  # TCP Vegas-like algorithm
    CUBIC = "cubic"  # CUBIC-like algorithm


class AdaptationSignal(Enum):
    """Signals for concurrency adaptation."""

    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"
    BACKOFF = "backoff"


@dataclass
class PerformanceWindow:
    """Performance measurement window."""

    start_time: float
    end_time: float
    requests_completed: int
    total_latency: float
    error_count: int
    max_latency: float
    min_latency: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def throughput(self) -> float:
        return self.requests_completed / self.duration if self.duration > 0 else 0.0

    @property
    def average_latency(self) -> float:
        return self.total_latency / self.requests_completed if self.requests_completed > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.requests_completed if self.requests_completed > 0 else 0.0


@dataclass
class ConcurrencyMetrics:
    """Metrics for concurrency control."""

    current_concurrency: int = 1
    target_concurrency: int = 1
    max_concurrency: int = 100
    min_concurrency: int = 1

    # Performance metrics
    average_latency: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0

    # Little's Law metrics
    arrival_rate: float = 0.0  # λ (lambda)
    service_time: float = 0.0  # W (average time in system)
    queue_length: float = 0.0  # L (average queue length)

    # Adaptation metrics
    adaptation_count: int = 0
    last_adaptation: float = 0.0
    adaptation_direction: Optional[AdaptationSignal] = None

    # Control loop metrics
    proportional_error: float = 0.0
    integral_error: float = 0.0
    derivative_error: float = 0.0


class LittlesLawCalculator:
    """Calculator for Little's Law: L = λ × W."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.arrival_times: deque = deque(maxlen=window_size)
        self.completion_times: deque = deque(maxlen=window_size)
        self.service_times: deque = deque(maxlen=window_size)
        self.queue_lengths: deque = deque(maxlen=window_size)

    def record_arrival(self, timestamp: float) -> None:
        """Record a request arrival."""
        self.arrival_times.append(timestamp)

    def record_completion(self, arrival_time: float, completion_time: float) -> None:
        """Record a request completion."""
        service_time = completion_time - arrival_time
        self.completion_times.append(completion_time)
        self.service_times.append(service_time)

    def record_queue_length(self, length: int) -> None:
        """Record current queue length."""
        self.queue_lengths.append(length)

    def calculate_metrics(self) -> Tuple[float, float, float]:
        """Calculate Little's Law metrics: (λ, W, L)."""
        now = time.time()
        window_start = now - 60.0  # 1-minute window

        # Calculate arrival rate (λ)
        recent_arrivals = [t for t in self.arrival_times if t >= window_start]
        arrival_rate = len(recent_arrivals) / 60.0 if recent_arrivals else 0.0

        # Calculate average service time (W)
        if self.service_times:
            avg_service_time = mean(self.service_times)
        else:
            avg_service_time = 0.0

        # Calculate average queue length (L)
        if self.queue_lengths:
            avg_queue_length = mean(self.queue_lengths)
        else:
            avg_queue_length = 0.0

        return arrival_rate, avg_service_time, avg_queue_length

    def calculate_optimal_concurrency(self, target_latency: float) -> int:
        """Calculate optimal concurrency using Little's Law."""
        arrival_rate, avg_service_time, _ = self.calculate_metrics()

        if arrival_rate <= 0 or avg_service_time <= 0:
            return 1

        # Little's Law: L = λ × W
        # For optimal performance, we want W ≈ target_latency
        # So optimal concurrency ≈ λ × target_latency
        optimal = arrival_rate * target_latency

        return max(1, int(optimal))


class AdaptiveConcurrencyController(FlowController):
    """Adaptive concurrency controller using various strategies."""

    def __init__(
        self,
        name: str,
        strategy: ConcurrencyStrategy = ConcurrencyStrategy.LITTLES_LAW,
        initial_concurrency: int = 10,
        min_concurrency: int = 1,
        max_concurrency: int = 200,
        target_latency: float = 1.0,  # seconds
        adaptation_interval: float = 5.0,  # seconds
        measurement_window: int = 100,
    ):
        super().__init__(name)
        self.strategy = strategy
        self.target_latency = target_latency
        self.adaptation_interval = adaptation_interval

        # Concurrency limits
        self.metrics = ConcurrencyMetrics(
            current_concurrency=initial_concurrency,
            target_concurrency=initial_concurrency,
            min_concurrency=min_concurrency,
            max_concurrency=max_concurrency,
        )

        # Little's Law calculator
        self.littles_law = LittlesLawCalculator(measurement_window)

        # Performance tracking
        self.active_requests: Dict[str, float] = {}  # request_id -> start_time
        self.performance_windows: deque = deque(maxlen=50)
        self.current_window_start = time.time()
        self.current_window_stats = {
            "completed": 0,
            "total_latency": 0.0,
            "errors": 0,
            "max_latency": 0.0,
            "min_latency": float("inf"),
        }

        # PID controller for gradient-based strategies
        self.pid_integral = 0.0
        self.pid_last_error = 0.0
        self.pid_kp = 1.0  # Proportional gain
        self.pid_ki = 0.1  # Integral gain
        self.pid_kd = 0.05  # Derivative gain

        # Vegas-style congestion detection
        self.rtt_min = float("inf")
        self.rtt_measurements: deque = deque(maxlen=10)

        # Start adaptation task
        self._adaptation_task = asyncio.create_task(self._adaptation_loop())

    async def should_allow_request(self, message: Message) -> bool:
        """Check if request should be allowed based on current concurrency."""
        current_active = len(self.active_requests)

        if current_active >= self.metrics.current_concurrency:
            self.metrics.requests_rejected += 1
            return False

        # Record arrival for Little's Law
        now = time.time()
        self.littles_law.record_arrival(now)
        self.active_requests[message.id] = now

        self.metrics.requests_total += 1
        self.metrics.requests_allowed += 1

        return True

    async def on_request_completed(self, message: Message, success: bool, duration: float) -> None:
        """Update metrics when request completes."""
        if message.id not in self.active_requests:
            return

        start_time = self.active_requests.pop(message.id)
        completion_time = time.time()

        # Record completion for Little's Law
        self.littles_law.record_completion(start_time, completion_time)
        self.littles_law.record_queue_length(len(self.active_requests))

        # Update current window stats
        self.current_window_stats["completed"] += 1
        self.current_window_stats["total_latency"] += duration

        if not success:
            self.current_window_stats["errors"] += 1

        self.current_window_stats["max_latency"] = max(
            self.current_window_stats["max_latency"], duration
        )
        self.current_window_stats["min_latency"] = min(
            self.current_window_stats["min_latency"], duration
        )

        # Update Vegas RTT measurements
        self.rtt_measurements.append(duration)
        self.rtt_min = min(self.rtt_min, duration)

        # Update exponential moving averages
        alpha = 0.1
        self.metrics.average_response_time = (
            alpha * duration + (1 - alpha) * self.metrics.average_response_time
        )

    def get_pressure_level(self) -> PressureLevel:
        """Get pressure level based on concurrency utilization."""
        utilization = len(self.active_requests) / self.metrics.current_concurrency

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

    async def _adaptation_loop(self) -> None:
        """Main adaptation loop."""
        while True:
            try:
                await asyncio.sleep(self.adaptation_interval)
                await self._adapt_concurrency()
                await self._finalize_performance_window()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in adaptation loop: {e}")

    async def _adapt_concurrency(self) -> None:
        """Adapt concurrency based on current strategy."""
        if self.strategy == ConcurrencyStrategy.LITTLES_LAW:
            await self._adapt_littles_law()
        elif self.strategy == ConcurrencyStrategy.GRADIENT_DESCENT:
            await self._adapt_gradient_descent()
        elif self.strategy == ConcurrencyStrategy.ADDITIVE_INCREASE:
            await self._adapt_additive_increase()
        elif self.strategy == ConcurrencyStrategy.MULTIPLICATIVE_DECREASE:
            await self._adapt_multiplicative_decrease()
        elif self.strategy == ConcurrencyStrategy.VEGAS:
            await self._adapt_vegas()
        elif self.strategy == ConcurrencyStrategy.CUBIC:
            await self._adapt_cubic()

    async def _adapt_littles_law(self) -> None:
        """Adapt using Little's Law calculations."""
        optimal = self.littles_law.calculate_optimal_concurrency(self.target_latency)

        # Update Little's Law metrics
        lambda_val, w_val, l_val = self.littles_law.calculate_metrics()
        self.metrics.arrival_rate = lambda_val
        self.metrics.service_time = w_val
        self.metrics.queue_length = l_val

        # Smooth the adaptation
        old_concurrency = self.metrics.current_concurrency
        if optimal > old_concurrency:
            # Increase gradually
            new_concurrency = min(
                self.metrics.max_concurrency,
                old_concurrency + max(1, (optimal - old_concurrency) // 4),
            )
        elif optimal < old_concurrency:
            # Decrease more aggressively
            new_concurrency = max(
                self.metrics.min_concurrency,
                old_concurrency - max(1, (old_concurrency - optimal) // 2),
            )
        else:
            new_concurrency = old_concurrency

        await self._update_concurrency(new_concurrency, "Little's Law")

    async def _adapt_gradient_descent(self) -> None:
        """Adapt using gradient descent on latency."""
        if not self.performance_windows:
            return

        current_latency = self.performance_windows[-1].average_latency
        error = current_latency - self.target_latency

        # PID controller
        self.pid_integral += error
        derivative = error - self.pid_last_error

        self.metrics.proportional_error = error
        self.metrics.integral_error = self.pid_integral
        self.metrics.derivative_error = derivative

        # Calculate adjustment
        adjustment = -(
            self.pid_kp * error + self.pid_ki * self.pid_integral + self.pid_kd * derivative
        )

        new_concurrency = max(
            self.metrics.min_concurrency,
            min(self.metrics.max_concurrency, self.metrics.current_concurrency + int(adjustment)),
        )

        self.pid_last_error = error
        await self._update_concurrency(new_concurrency, "Gradient Descent")

    async def _adapt_additive_increase(self) -> None:
        """Additive increase / multiplicative decrease (AIMD)."""
        if not self.performance_windows:
            return

        latest_window = self.performance_windows[-1]

        if (
            latest_window.error_rate > 0.01
            or latest_window.average_latency > self.target_latency * 1.5
        ):
            # Multiplicative decrease
            new_concurrency = max(
                self.metrics.min_concurrency, int(self.metrics.current_concurrency * 0.7)
            )
            signal = AdaptationSignal.DECREASE
        else:
            # Additive increase
            new_concurrency = min(
                self.metrics.max_concurrency, self.metrics.current_concurrency + 1
            )
            signal = AdaptationSignal.INCREASE

        await self._update_concurrency(new_concurrency, "AIMD", signal)

    async def _adapt_multiplicative_decrease(self) -> None:
        """Multiplicative increase / multiplicative decrease."""
        if not self.performance_windows:
            return

        latest_window = self.performance_windows[-1]

        if (
            latest_window.error_rate > 0.01
            or latest_window.average_latency > self.target_latency * 1.2
        ):
            # Multiplicative decrease
            new_concurrency = max(
                self.metrics.min_concurrency, int(self.metrics.current_concurrency * 0.8)
            )
            signal = AdaptationSignal.DECREASE
        elif latest_window.average_latency < self.target_latency * 0.8:
            # Multiplicative increase
            new_concurrency = min(
                self.metrics.max_concurrency, int(self.metrics.current_concurrency * 1.2)
            )
            signal = AdaptationSignal.INCREASE
        else:
            new_concurrency = self.metrics.current_concurrency
            signal = AdaptationSignal.MAINTAIN

        await self._update_concurrency(new_concurrency, "MIMD", signal)

    async def _adapt_vegas(self) -> None:
        """TCP Vegas-like congestion control."""
        if len(self.rtt_measurements) < 3:
            return

        # Calculate expected and actual throughput
        current_rtt = mean(list(self.rtt_measurements)[-3:])
        expected_throughput = self.metrics.current_concurrency / self.rtt_min
        actual_throughput = self.metrics.current_concurrency / current_rtt

        diff = expected_throughput - actual_throughput

        # Vegas thresholds (α, β)
        alpha = self.metrics.current_concurrency * 0.1
        beta = self.metrics.current_concurrency * 0.3

        if diff < alpha:
            # Increase concurrency
            new_concurrency = min(
                self.metrics.max_concurrency, self.metrics.current_concurrency + 1
            )
            signal = AdaptationSignal.INCREASE
        elif diff > beta:
            # Decrease concurrency
            new_concurrency = max(
                self.metrics.min_concurrency, self.metrics.current_concurrency - 1
            )
            signal = AdaptationSignal.DECREASE
        else:
            new_concurrency = self.metrics.current_concurrency
            signal = AdaptationSignal.MAINTAIN

        await self._update_concurrency(new_concurrency, "Vegas", signal)

    async def _adapt_cubic(self) -> None:
        """CUBIC-like congestion control."""
        if not self.performance_windows:
            return

        latest_window = self.performance_windows[-1]

        # Detect congestion
        if (
            latest_window.error_rate > 0.01
            or latest_window.average_latency > self.target_latency * 1.3
        ):
            # Congestion detected - multiplicative decrease
            new_max = int(self.metrics.current_concurrency * 0.7)
            new_concurrency = max(self.metrics.min_concurrency, new_max)
            signal = AdaptationSignal.BACKOFF
        else:
            # CUBIC growth function
            # W(t) = C(t - K)³ + W_max
            # where K is the time to reach W_max

            time_since_last = time.time() - self.metrics.last_adaptation
            C = 0.4  # CUBIC parameter

            # Simplified CUBIC growth
            if latest_window.average_latency < self.target_latency * 0.9:
                cubic_increment = C * (time_since_last**3)
                new_concurrency = min(
                    self.metrics.max_concurrency,
                    self.metrics.current_concurrency + max(1, int(cubic_increment)),
                )
                signal = AdaptationSignal.INCREASE
            else:
                new_concurrency = self.metrics.current_concurrency
                signal = AdaptationSignal.MAINTAIN

        await self._update_concurrency(new_concurrency, "CUBIC", signal)

    async def _update_concurrency(
        self, new_concurrency: int, reason: str, signal: Optional[AdaptationSignal] = None
    ) -> None:
        """Update the concurrency limit."""
        old_concurrency = self.metrics.current_concurrency

        if new_concurrency != old_concurrency:
            self.metrics.current_concurrency = new_concurrency
            self.metrics.target_concurrency = new_concurrency
            self.metrics.adaptation_count += 1
            self.metrics.last_adaptation = time.time()
            self.metrics.adaptation_direction = signal

            self.logger.info(
                f"Concurrency adapted: {old_concurrency} -> {new_concurrency} "
                f"({reason}, active: {len(self.active_requests)})"
            )

    async def _finalize_performance_window(self) -> None:
        """Finalize the current performance window."""
        now = time.time()
        duration = now - self.current_window_start

        if self.current_window_stats["completed"] > 0:
            window = PerformanceWindow(
                start_time=self.current_window_start,
                end_time=now,
                requests_completed=self.current_window_stats["completed"],
                total_latency=self.current_window_stats["total_latency"],
                error_count=self.current_window_stats["errors"],
                max_latency=self.current_window_stats["max_latency"],
                min_latency=self.current_window_stats["min_latency"],
            )

            self.performance_windows.append(window)

            # Update metrics
            self.metrics.throughput = window.throughput
            self.metrics.error_rate = window.error_rate

        # Reset window
        self.current_window_start = now
        self.current_window_stats = {
            "completed": 0,
            "total_latency": 0.0,
            "errors": 0,
            "max_latency": 0.0,
            "min_latency": float("inf"),
        }

    async def stop(self) -> None:
        """Stop the concurrency controller."""
        if self._adaptation_task:
            self._adaptation_task.cancel()
            try:
                await self._adaptation_task
            except asyncio.CancelledError:
                pass

    def get_concurrency_metrics(self) -> Dict[str, Any]:
        """Get detailed concurrency metrics."""
        return {
            "strategy": self.strategy.value,
            "concurrency": {
                "current": self.metrics.current_concurrency,
                "target": self.metrics.target_concurrency,
                "min": self.metrics.min_concurrency,
                "max": self.metrics.max_concurrency,
                "active_requests": len(self.active_requests),
            },
            "performance": {
                "average_latency": self.metrics.average_latency,
                "throughput": self.metrics.throughput,
                "error_rate": self.metrics.error_rate,
                "target_latency": self.target_latency,
            },
            "littles_law": {
                "arrival_rate": self.metrics.arrival_rate,
                "service_time": self.metrics.service_time,
                "queue_length": self.metrics.queue_length,
            },
            "adaptation": {
                "count": self.metrics.adaptation_count,
                "last_adaptation": self.metrics.last_adaptation,
                "direction": (
                    self.metrics.adaptation_direction.value
                    if self.metrics.adaptation_direction
                    else None
                ),
            },
            "pid_control": {
                "proportional_error": self.metrics.proportional_error,
                "integral_error": self.metrics.integral_error,
                "derivative_error": self.metrics.derivative_error,
            },
        }


class ConcurrencyLimitedActor(Actor):
    """Actor with adaptive concurrency control."""

    def __init__(
        self,
        name: str,
        concurrency_controller: AdaptiveConcurrencyController,
        mailbox_size: int = 1000,
    ):
        super().__init__(name, mailbox_size)
        self.concurrency_controller = concurrency_controller
        self._processing_semaphore: Optional[asyncio.Semaphore] = None

    async def on_start(self) -> None:
        """Initialize the concurrency-limited actor."""
        initial_concurrency = self.concurrency_controller.metrics.current_concurrency
        self._processing_semaphore = asyncio.Semaphore(initial_concurrency)

        # Start a task to update semaphore when concurrency changes
        self._update_task = asyncio.create_task(self._update_semaphore_loop())

    async def on_stop(self) -> None:
        """Stop the concurrency controller."""
        if hasattr(self, "_update_task"):
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        await self.concurrency_controller.stop()

    async def on_error(self, error: Exception) -> None:
        """Handle errors with concurrency awareness."""
        self.logger.error(f"Concurrency-limited actor {self.name} error: {error}")

    async def receive(self, message: Message) -> None:
        """Receive message with concurrency control."""
        # Check if request should be allowed by the controller
        if not await self.concurrency_controller.should_allow_request(message):
            self.logger.debug(f"Message {message.id} rejected by concurrency control")
            return

        await super().receive(message)

    async def _handle_message(self, message: Message) -> None:
        """Handle message with concurrency limiting."""
        async with self._processing_semaphore:
            start_time = time.time()
            success = True

            try:
                await super()._handle_message(message)
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                await self.concurrency_controller.on_request_completed(message, success, duration)

    async def _update_semaphore_loop(self) -> None:
        """Update semaphore capacity when concurrency changes."""
        last_concurrency = self.concurrency_controller.metrics.current_concurrency

        while True:
            try:
                await asyncio.sleep(1.0)  # Check every second

                current_concurrency = self.concurrency_controller.metrics.current_concurrency

                if current_concurrency != last_concurrency:
                    # Recreate semaphore with new capacity
                    old_semaphore = self._processing_semaphore
                    self._processing_semaphore = asyncio.Semaphore(current_concurrency)

                    self.logger.info(
                        f"Updated processing semaphore: {last_concurrency} -> {current_concurrency}"
                    )

                    last_concurrency = current_concurrency

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error updating semaphore: {e}")

    def get_concurrency_status(self) -> Dict[str, Any]:
        """Get current concurrency status."""
        # Use a custom counter instead of accessing private attributes
        semaphore_available = (
            self.concurrency_controller.metrics.current_concurrency
            - len(self.concurrency_controller.active_requests)
            if self._processing_semaphore
            else 0
        )

        return {
            "actor_name": self.name,
            "controller_metrics": self.concurrency_controller.get_concurrency_metrics(),
            "semaphore_available": semaphore_available,
            "mailbox_size": self.mailbox.size(),
        }
