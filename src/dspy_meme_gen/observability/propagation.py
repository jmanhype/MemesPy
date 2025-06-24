"""
Trace context propagation utilities for distributed tracing across actors and services.
"""

import json
import pickle
from typing import Dict, Any, Optional, Union, TypeVar, Callable
from functools import wraps
import asyncio
from concurrent.futures import Future
from dataclasses import dataclass
from contextlib import contextmanager

from opentelemetry import trace, baggage
from opentelemetry.context import attach, detach, get_current
from opentelemetry.propagate import inject, extract
from opentelemetry.trace import SpanContext, NonRecordingSpan, Link
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

import redis.asyncio as aioredis
from celery import Task, current_task
from kombu import Producer, Consumer
import pika

from structlog import get_logger

logger = get_logger(__name__)

# Type definitions
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class PropagatedContext:
    """Container for propagated trace context."""

    trace_id: str
    span_id: str
    trace_flags: str
    baggage: Dict[str, str]
    attributes: Dict[str, Any]


class ContextPropagator:
    """Base class for context propagation strategies."""

    def inject(self, carrier: Any) -> None:
        """Inject current context into carrier."""
        raise NotImplementedError

    def extract(self, carrier: Any) -> Optional[PropagatedContext]:
        """Extract context from carrier."""
        raise NotImplementedError


class HTTPPropagator(ContextPropagator):
    """HTTP header-based context propagation."""

    def __init__(self):
        self.propagator = TraceContextTextMapPropagator()

    def inject(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into HTTP headers."""
        inject(headers)

        # Also inject baggage
        for key, value in baggage.get_all().items():
            headers[f"baggage-{key}"] = value

        return headers

    def extract(self, headers: Dict[str, str]) -> Optional[PropagatedContext]:
        """Extract trace context from HTTP headers."""
        ctx = extract(headers)
        span = trace.get_current_span(ctx)

        if span and span.get_span_context().is_valid:
            span_ctx = span.get_span_context()

            # Extract baggage
            extracted_baggage = {}
            for key, value in headers.items():
                if key.startswith("baggage-"):
                    baggage_key = key[8:]  # Remove "baggage-" prefix
                    extracted_baggage[baggage_key] = value

            return PropagatedContext(
                trace_id=format(span_ctx.trace_id, "032x"),
                span_id=format(span_ctx.span_id, "016x"),
                trace_flags=format(span_ctx.trace_flags, "02x"),
                baggage=extracted_baggage,
                attributes={},
            )

        return None


class RedisPropagator(ContextPropagator):
    """Redis-based context propagation for async tasks."""

    def __init__(self, redis_client: aioredis.Redis, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl

    async def inject(self, task_id: str) -> None:
        """Store trace context in Redis for task."""
        span = trace.get_current_span()
        if not span or not span.get_span_context().is_valid:
            return

        ctx = span.get_span_context()
        context_data = {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
            "trace_flags": format(ctx.trace_flags, "02x"),
            "baggage": dict(baggage.get_all()),
            "attributes": {},
        }

        key = f"trace:context:{task_id}"
        await self.redis.setex(key, self.ttl, json.dumps(context_data))

    async def extract(self, task_id: str) -> Optional[PropagatedContext]:
        """Retrieve trace context from Redis."""
        key = f"trace:context:{task_id}"
        data = await self.redis.get(key)

        if data:
            context_data = json.loads(data)
            return PropagatedContext(**context_data)

        return None


class CeleryPropagator(ContextPropagator):
    """Celery task context propagation."""

    def inject(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Inject trace context into Celery task kwargs."""
        span = trace.get_current_span()
        if not span or not span.get_span_context().is_valid:
            return kwargs

        ctx = span.get_span_context()
        kwargs["_trace_context"] = {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
            "trace_flags": format(ctx.trace_flags, "02x"),
            "baggage": dict(baggage.get_all()),
        }

        return kwargs

    def extract(self, kwargs: Dict[str, Any]) -> Optional[PropagatedContext]:
        """Extract trace context from Celery task kwargs."""
        context_data = kwargs.pop("_trace_context", None)

        if context_data:
            return PropagatedContext(
                trace_id=context_data["trace_id"],
                span_id=context_data["span_id"],
                trace_flags=context_data["trace_flags"],
                baggage=context_data.get("baggage", {}),
                attributes={},
            )

        return None


class MessageQueuePropagator(ContextPropagator):
    """Message queue context propagation (RabbitMQ, Kafka, etc.)."""

    def inject(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Inject trace context into message."""
        span = trace.get_current_span()
        if not span or not span.get_span_context().is_valid:
            return message

        ctx = span.get_span_context()
        message["_trace_context"] = {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
            "trace_flags": format(ctx.trace_flags, "02x"),
            "baggage": dict(baggage.get_all()),
        }

        return message

    def extract(self, message: Dict[str, Any]) -> Optional[PropagatedContext]:
        """Extract trace context from message."""
        context_data = message.get("_trace_context")

        if context_data:
            return PropagatedContext(
                trace_id=context_data["trace_id"],
                span_id=context_data["span_id"],
                trace_flags=context_data["trace_flags"],
                baggage=context_data.get("baggage", {}),
                attributes={},
            )

        return None


class ActorPropagator(ContextPropagator):
    """Context propagation for actor-based systems."""

    def inject(self, actor_message: Any) -> Any:
        """Inject trace context into actor message."""
        span = trace.get_current_span()
        if not span or not span.get_span_context().is_valid:
            return actor_message

        ctx = span.get_span_context()

        # If message is a dict, add context directly
        if isinstance(actor_message, dict):
            actor_message["_trace_context"] = {
                "trace_id": format(ctx.trace_id, "032x"),
                "span_id": format(ctx.span_id, "016x"),
                "trace_flags": format(ctx.trace_flags, "02x"),
                "baggage": dict(baggage.get_all()),
            }
        else:
            # For other types, wrap in a container
            return {
                "payload": actor_message,
                "_trace_context": {
                    "trace_id": format(ctx.trace_id, "032x"),
                    "span_id": format(ctx.span_id, "016x"),
                    "trace_flags": format(ctx.trace_flags, "02x"),
                    "baggage": dict(baggage.get_all()),
                },
            }

        return actor_message

    def extract(self, actor_message: Any) -> Optional[PropagatedContext]:
        """Extract trace context from actor message."""
        context_data = None

        if isinstance(actor_message, dict):
            context_data = actor_message.get("_trace_context")

            # Check if this is a wrapped message
            if "payload" in actor_message and "_trace_context" in actor_message:
                # Unwrap the payload
                actor_message = actor_message["payload"]

        if context_data:
            return PropagatedContext(
                trace_id=context_data["trace_id"],
                span_id=context_data["span_id"],
                trace_flags=context_data["trace_flags"],
                baggage=context_data.get("baggage", {}),
                attributes={},
            )

        return None


@contextmanager
def propagated_context(context: PropagatedContext):
    """Context manager for activating propagated context."""
    if not context:
        yield
        return

    # Create span context from propagated data
    span_context = SpanContext(
        trace_id=int(context.trace_id, 16),
        span_id=int(context.span_id, 16),
        is_remote=True,
        trace_flags=int(context.trace_flags, 16),
    )

    # Create non-recording span with the context
    span = NonRecordingSpan(span_context)
    ctx = trace.set_span_in_context(span)

    # Attach baggage
    for key, value in context.baggage.items():
        ctx = baggage.set_baggage(key, value, context=ctx)

    token = attach(ctx)
    try:
        yield
    finally:
        detach(token)


def trace_async_task(
    name: str, propagator: ContextPropagator, kind: trace.SpanKind = trace.SpanKind.CONSUMER
) -> Callable[[F], F]:
    """
    Decorator for tracing async tasks with context propagation.

    Args:
        name: Span name
        propagator: Context propagator to use
        kind: Span kind

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract context if available
            extracted_context = None

            # Try to extract from various sources
            if hasattr(propagator, "extract"):
                if args and isinstance(args[0], dict):
                    extracted_context = propagator.extract(args[0])
                elif kwargs:
                    extracted_context = propagator.extract(kwargs)

            # Use propagated context if available
            if extracted_context:
                with propagated_context(extracted_context):
                    tracer = trace.get_tracer(__name__)
                    with tracer.start_as_current_span(name, kind=kind) as span:
                        try:
                            result = await func(*args, **kwargs)
                            span.set_status(trace.Status(trace.StatusCode.OK))
                            return result
                        except Exception as e:
                            span.record_exception(e)
                            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                            raise
            else:
                # No context to propagate, create new span
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(name, kind=kind) as span:
                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        raise

        return wrapper

    return decorator


def trace_celery_task(task_cls: Type[Task]) -> Type[Task]:
    """
    Decorator for tracing Celery tasks.

    Args:
        task_cls: Celery task class

    Returns:
        Decorated task class
    """
    original_run = task_cls.run
    propagator = CeleryPropagator()

    def traced_run(self, *args, **kwargs):
        # Extract context
        extracted_context = propagator.extract(kwargs)

        if extracted_context:
            with propagated_context(extracted_context):
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(
                    f"celery.task.{self.name}", kind=trace.SpanKind.CONSUMER
                ) as span:
                    span.set_attribute("celery.task.name", self.name)
                    span.set_attribute("celery.task.id", current_task.request.id)

                    try:
                        result = original_run(self, *args, **kwargs)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        raise
        else:
            # No context, run normally
            return original_run(self, *args, **kwargs)

    task_cls.run = traced_run
    return task_cls


class PropagationMiddleware:
    """Middleware for automatic context propagation in web frameworks."""

    def __init__(self, propagator: ContextPropagator = None):
        self.propagator = propagator or HTTPPropagator()

    async def __call__(self, request, call_next):
        """Process request with context propagation."""
        # Extract context from request headers
        headers = dict(request.headers)
        extracted_context = self.propagator.extract(headers)

        if extracted_context:
            with propagated_context(extracted_context):
                response = await call_next(request)

                # Inject context into response headers
                response_headers = dict(response.headers)
                self.propagator.inject(response_headers)

                return response
        else:
            return await call_next(request)


# Utility functions for common propagation scenarios
def propagate_to_thread(func: Callable) -> Callable:
    """Propagate context to thread execution."""

    def wrapper(*args, **kwargs):
        # Capture current context
        ctx = get_current()

        def run_with_context():
            token = attach(ctx)
            try:
                return func(*args, **kwargs)
            finally:
                detach(token)

        return run_with_context()

    return wrapper


def propagate_to_process(func: Callable) -> Callable:
    """Propagate context to process execution."""

    def wrapper(*args, **kwargs):
        # Serialize current context
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            ctx = span.get_span_context()
            context_data = {
                "trace_id": format(ctx.trace_id, "032x"),
                "span_id": format(ctx.span_id, "016x"),
                "trace_flags": format(ctx.trace_flags, "02x"),
                "baggage": dict(baggage.get_all()),
            }

            # Add context to kwargs
            kwargs["_trace_context"] = pickle.dumps(context_data)

        return func(*args, **kwargs)

    return wrapper


__all__ = [
    "PropagatedContext",
    "ContextPropagator",
    "HTTPPropagator",
    "RedisPropagator",
    "CeleryPropagator",
    "MessageQueuePropagator",
    "ActorPropagator",
    "propagated_context",
    "trace_async_task",
    "trace_celery_task",
    "PropagationMiddleware",
    "propagate_to_thread",
    "propagate_to_process",
]
