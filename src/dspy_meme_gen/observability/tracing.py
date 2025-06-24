"""
Distributed tracing with OpenTelemetry for the actor system.
Provides comprehensive tracing across actor boundaries with correlation IDs and span context propagation.
"""

import asyncio
import functools
import json
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import structlog
from opentelemetry import baggage, context, trace
from opentelemetry.context import attach, detach
from opentelemetry.propagate import extract, inject
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from ..actors.core import Actor, Message
from .telemetry import SpanAttributes, get_tracer

# Type definitions
F = TypeVar("F", bound=Callable[..., Any])

logger = structlog.get_logger(__name__)


class ActorTracingContext:
    """Context for tracing actor interactions with correlation IDs."""

    def __init__(
        self,
        actor_name: str,
        operation: str,
        correlation_id: Optional[str] = None,
        parent_span_context: Optional[Dict[str, str]] = None,
    ):
        self.actor_name = actor_name
        self.operation = operation
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.parent_span_context = parent_span_context or {}
        self._span = None
        self._context_token = None

    def __enter__(self):
        """Enter tracing context."""
        tracer = get_tracer()

        # Extract parent context if provided
        if self.parent_span_context:
            parent_ctx = extract(self.parent_span_context)
            ctx_token = attach(parent_ctx)
            self._context_token = ctx_token

        # Start new span
        self._span = tracer.start_span(
            f"{self.actor_name}.{self.operation}",
            kind=SpanKind.INTERNAL,
            attributes={
                SpanAttributes.AGENT_NAME: self.actor_name,
                SpanAttributes.AGENT_OPERATION: self.operation,
                "correlation_id": self.correlation_id,
                "actor.type": "dspy_actor",
            },
        )

        # Set baggage for correlation
        baggage.set_baggage("correlation_id", self.correlation_id)
        baggage.set_baggage("actor_name", self.actor_name)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit tracing context."""
        if exc_type:
            if self._span:
                self._span.record_exception(exc_val)
                self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))

        if self._span:
            self._span.end()

        if self._context_token:
            detach(self._context_token)

    def add_attribute(self, key: str, value: Any):
        """Add attribute to current span."""
        if self._span:
            self._span.set_attribute(key, str(value))

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to current span."""
        if self._span:
            self._span.add_event(name, attributes or {})

    def get_trace_context(self) -> Dict[str, str]:
        """Get trace context for propagation."""
        headers = {}
        inject(headers)
        return headers


class MessageTracer:
    """Tracer for actor messages with automatic span creation."""

    @staticmethod
    def trace_message_send(
        sender: str, receiver: str, message: Message, correlation_id: Optional[str] = None
    ) -> Dict[str, str]:
        """Trace message send operation and return context for propagation."""
        tracer = get_tracer()
        correlation_id = correlation_id or str(uuid.uuid4())

        with tracer.start_as_current_span(
            f"message.send",
            kind=SpanKind.PRODUCER,
            attributes={
                "message.sender": sender,
                "message.receiver": receiver,
                "message.type": type(message).__name__,
                "correlation_id": correlation_id,
                SpanAttributes.AGENT_OPERATION: "send_message",
            },
        ) as span:
            # Add message-specific attributes
            if hasattr(message, "id"):
                span.set_attribute("message.id", str(message.id))

            # Set baggage for correlation
            baggage.set_baggage("correlation_id", correlation_id)
            baggage.set_baggage("sender", sender)

            # Create propagation context
            headers = {}
            inject(headers)

            return headers

    @staticmethod
    def trace_message_receive(
        receiver: str, message: Message, trace_context: Optional[Dict[str, str]] = None
    ) -> ActorTracingContext:
        """Trace message receive operation."""
        correlation_id = None

        # Extract correlation ID from context
        if trace_context:
            parent_ctx = extract(trace_context)
            with context.use_context(parent_ctx):
                correlation_id = baggage.get_baggage("correlation_id")

        return ActorTracingContext(
            actor_name=receiver,
            operation="receive_message",
            correlation_id=correlation_id,
            parent_span_context=trace_context,
        )

    @staticmethod
    def trace_message_processing(
        actor_name: str, message: Message, trace_context: Optional[Dict[str, str]] = None
    ) -> ActorTracingContext:
        """Trace message processing operation."""
        correlation_id = None

        # Extract correlation ID from context
        if trace_context:
            parent_ctx = extract(trace_context)
            with context.use_context(parent_ctx):
                correlation_id = baggage.get_baggage("correlation_id")

        return ActorTracingContext(
            actor_name=actor_name,
            operation=f"process_{type(message).__name__}",
            correlation_id=correlation_id,
            parent_span_context=trace_context,
        )


class ActorSpanManager:
    """Manager for actor-specific spans with lifecycle tracking."""

    def __init__(self, actor_name: str):
        self.actor_name = actor_name
        self._active_spans: Dict[str, Any] = {}
        self._tracer = get_tracer()

    def start_span(
        self,
        operation: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        parent_context: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start a new span and return its ID."""
        span_id = str(uuid.uuid4())

        # Extract parent context if provided
        ctx_token = None
        if parent_context:
            parent_ctx = extract(parent_context)
            ctx_token = attach(parent_ctx)

        # Create span attributes
        span_attributes = {
            SpanAttributes.AGENT_NAME: self.actor_name,
            SpanAttributes.AGENT_OPERATION: operation,
            "span.id": span_id,
            **(attributes or {}),
        }

        # Start span
        span = self._tracer.start_span(
            f"{self.actor_name}.{operation}", kind=kind, attributes=span_attributes
        )

        # Store span info
        self._active_spans[span_id] = {
            "span": span,
            "context_token": ctx_token,
            "start_time": time.time(),
            "operation": operation,
        }

        logger.info("Started span", actor=self.actor_name, operation=operation, span_id=span_id)

        return span_id

    def add_span_attribute(self, span_id: str, key: str, value: Any):
        """Add attribute to active span."""
        if span_id in self._active_spans:
            span_info = self._active_spans[span_id]
            span_info["span"].set_attribute(key, str(value))

    def add_span_event(
        self, span_id: str, event_name: str, attributes: Optional[Dict[str, Any]] = None
    ):
        """Add event to active span."""
        if span_id in self._active_spans:
            span_info = self._active_spans[span_id]
            span_info["span"].add_event(event_name, attributes or {})

    def record_exception(self, span_id: str, exception: Exception):
        """Record exception on span."""
        if span_id in self._active_spans:
            span_info = self._active_spans[span_id]
            span_info["span"].record_exception(exception)
            span_info["span"].set_status(Status(StatusCode.ERROR, str(exception)))

    def end_span(self, span_id: str, success: bool = True):
        """End an active span."""
        if span_id not in self._active_spans:
            logger.warning("Attempted to end unknown span", span_id=span_id)
            return

        span_info = self._active_spans[span_id]
        span = span_info["span"]

        # Calculate duration
        duration = time.time() - span_info["start_time"]
        span.set_attribute(SpanAttributes.DURATION_MS, duration * 1000)

        # Set status
        if not success:
            span.set_status(Status(StatusCode.ERROR, "Operation failed"))

        # End span
        span.end()

        # Clean up context
        if span_info["context_token"]:
            detach(span_info["context_token"])

        # Remove from active spans
        del self._active_spans[span_id]

        logger.info(
            "Ended span",
            actor=self.actor_name,
            operation=span_info["operation"],
            span_id=span_id,
            duration_ms=duration * 1000,
            success=success,
        )

    def get_current_trace_context(self) -> Dict[str, str]:
        """Get current trace context for propagation."""
        headers = {}
        inject(headers)
        return headers

    def cleanup_expired_spans(self, max_age_seconds: int = 300):
        """Clean up spans that have been active too long."""
        current_time = time.time()
        expired_spans = [
            span_id
            for span_id, info in self._active_spans.items()
            if current_time - info["start_time"] > max_age_seconds
        ]

        for span_id in expired_spans:
            logger.warning("Cleaning up expired span", actor=self.actor_name, span_id=span_id)
            self.end_span(span_id, success=False)


def trace_actor_method(
    operation: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    capture_args: bool = False,
    capture_result: bool = False,
):
    """Decorator for tracing actor methods."""

    def decorator(func: F) -> F:
        op_name = operation or func.__name__

        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            actor_name = getattr(self, "name", self.__class__.__name__)

            with ActorTracingContext(actor_name, op_name) as ctx:
                # Capture arguments if requested
                if capture_args:
                    ctx.add_attribute("method.args", str(args)[:1000])
                    ctx.add_attribute("method.kwargs", str(kwargs)[:1000])

                try:
                    result = await func(self, *args, **kwargs)

                    # Capture result if requested
                    if capture_result:
                        ctx.add_attribute("method.result", str(result)[:1000])

                    return result

                except Exception as e:
                    ctx.add_attribute(SpanAttributes.ERROR_TYPE, type(e).__name__)
                    ctx.add_attribute(SpanAttributes.ERROR_MESSAGE, str(e))
                    raise

        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            actor_name = getattr(self, "name", self.__class__.__name__)

            with ActorTracingContext(actor_name, op_name) as ctx:
                # Capture arguments if requested
                if capture_args:
                    ctx.add_attribute("method.args", str(args)[:1000])
                    ctx.add_attribute("method.kwargs", str(kwargs)[:1000])

                try:
                    result = func(self, *args, **kwargs)

                    # Capture result if requested
                    if capture_result:
                        ctx.add_attribute("method.result", str(result)[:1000])

                    return result

                except Exception as e:
                    ctx.add_attribute(SpanAttributes.ERROR_TYPE, type(e).__name__)
                    ctx.add_attribute(SpanAttributes.ERROR_MESSAGE, str(e))
                    raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


class PipelineTracer:
    """Tracer for multi-stage pipelines with stage tracking."""

    def __init__(self, pipeline_name: str, correlation_id: Optional[str] = None):
        self.pipeline_name = pipeline_name
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.stages: List[Dict[str, Any]] = []
        self._current_stage: Optional[str] = None
        self._pipeline_span = None
        self._stage_span = None
        self._tracer = get_tracer()

    def start_pipeline(self, attributes: Optional[Dict[str, Any]] = None):
        """Start pipeline tracing."""
        pipeline_attributes = {
            "pipeline.name": self.pipeline_name,
            "pipeline.correlation_id": self.correlation_id,
            SpanAttributes.PIPELINE_STAGE: "pipeline_start",
            **(attributes or {}),
        }

        self._pipeline_span = self._tracer.start_span(
            f"pipeline.{self.pipeline_name}", kind=SpanKind.INTERNAL, attributes=pipeline_attributes
        )

        # Set baggage for correlation
        baggage.set_baggage("pipeline_correlation_id", self.correlation_id)
        baggage.set_baggage("pipeline_name", self.pipeline_name)

        logger.info(
            "Started pipeline", pipeline=self.pipeline_name, correlation_id=self.correlation_id
        )

    @asynccontextmanager
    async def trace_stage(self, stage_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing pipeline stages."""
        if not self._pipeline_span:
            raise RuntimeError("Pipeline not started")

        stage_attributes = {
            "pipeline.name": self.pipeline_name,
            "pipeline.correlation_id": self.correlation_id,
            SpanAttributes.PIPELINE_STAGE: stage_name,
            "stage.order": len(self.stages),
            **(attributes or {}),
        }

        # Start stage span
        stage_span = self._tracer.start_span(
            f"pipeline.{self.pipeline_name}.{stage_name}",
            kind=SpanKind.INTERNAL,
            attributes=stage_attributes,
        )

        stage_info = {
            "name": stage_name,
            "start_time": time.time(),
            "span": stage_span,
            "success": True,
        }

        self._current_stage = stage_name
        self.stages.append(stage_info)

        try:
            yield stage_span

        except Exception as e:
            stage_info["success"] = False
            stage_span.record_exception(e)
            stage_span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

        finally:
            # Calculate stage duration
            duration = time.time() - stage_info["start_time"]
            stage_span.set_attribute(SpanAttributes.DURATION_MS, duration * 1000)
            stage_span.end()

            logger.info(
                "Completed pipeline stage",
                pipeline=self.pipeline_name,
                stage=stage_name,
                duration_ms=duration * 1000,
                success=stage_info["success"],
            )

            self._current_stage = None

    def add_pipeline_attribute(self, key: str, value: Any):
        """Add attribute to pipeline span."""
        if self._pipeline_span:
            self._pipeline_span.set_attribute(key, str(value))

    def add_pipeline_event(self, event_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to pipeline span."""
        if self._pipeline_span:
            self._pipeline_span.add_event(
                event_name,
                {
                    "pipeline.correlation_id": self.correlation_id,
                    "pipeline.current_stage": self._current_stage or "none",
                    **(attributes or {}),
                },
            )

    def end_pipeline(self, success: bool = True):
        """End pipeline tracing."""
        if not self._pipeline_span:
            return

        # Add pipeline summary
        total_stages = len(self.stages)
        successful_stages = sum(1 for stage in self.stages if stage["success"])
        total_duration = sum(time.time() - stage["start_time"] for stage in self.stages)

        self._pipeline_span.set_attribute("pipeline.total_stages", total_stages)
        self._pipeline_span.set_attribute("pipeline.successful_stages", successful_stages)
        self._pipeline_span.set_attribute("pipeline.total_duration_ms", total_duration * 1000)

        if not success:
            self._pipeline_span.set_status(Status(StatusCode.ERROR, "Pipeline failed"))

        self._pipeline_span.end()

        logger.info(
            "Ended pipeline",
            pipeline=self.pipeline_name,
            correlation_id=self.correlation_id,
            total_stages=total_stages,
            successful_stages=successful_stages,
            success=success,
        )

    def get_trace_context(self) -> Dict[str, str]:
        """Get current trace context for propagation."""
        headers = {}
        inject(headers)
        return headers


class DistributedTracer:
    """Main interface for distributed tracing in the actor system."""

    def __init__(self):
        self._actor_managers: Dict[str, ActorSpanManager] = {}
        self._propagator = TraceContextTextMapPropagator()

    def get_actor_manager(self, actor_name: str) -> ActorSpanManager:
        """Get or create span manager for actor."""
        if actor_name not in self._actor_managers:
            self._actor_managers[actor_name] = ActorSpanManager(actor_name)
        return self._actor_managers[actor_name]

    def create_pipeline_tracer(
        self, pipeline_name: str, correlation_id: Optional[str] = None
    ) -> PipelineTracer:
        """Create a new pipeline tracer."""
        return PipelineTracer(pipeline_name, correlation_id)

    def inject_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject current trace context into headers."""
        inject(headers)
        return headers

    def extract_context(self, headers: Dict[str, str]) -> Optional[context.Context]:
        """Extract trace context from headers."""
        return extract(headers)

    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID from baggage."""
        return baggage.get_baggage("correlation_id")

    def cleanup_expired_spans(self, max_age_seconds: int = 300):
        """Clean up expired spans across all actors."""
        for manager in self._actor_managers.values():
            manager.cleanup_expired_spans(max_age_seconds)


# Global distributed tracer instance
distributed_tracer = DistributedTracer()

# Export public API
__all__ = [
    "ActorTracingContext",
    "MessageTracer",
    "ActorSpanManager",
    "PipelineTracer",
    "DistributedTracer",
    "distributed_tracer",
    "trace_actor_method",
]
