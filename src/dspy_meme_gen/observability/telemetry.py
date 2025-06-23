"""
OpenTelemetry instrumentation for distributed tracing and observability.
Provides comprehensive tracing, metrics, and logging correlation for the MemesPy system.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Callable, TypeVar, List, Union
from functools import wraps
from contextlib import contextmanager
import asyncio
from datetime import datetime
import json

from opentelemetry import trace, metrics, baggage
from opentelemetry.context import attach, detach, get_current
from opentelemetry.trace import (
    Tracer, SpanKind, Status, StatusCode, Link, SpanContext,
    get_tracer_provider, set_tracer_provider
)
from opentelemetry.metrics import (
    Meter, Counter, Histogram, UpDownCounter, ObservableGauge,
    get_meter_provider, set_meter_provider
)
from opentelemetry.baggage import set_baggage, get_baggage
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor, ConsoleSpanExporter, SpanExporter
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader, ConsoleMetricExporter
)
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.propagate import set_global_textmap, extract, inject
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.sdk.trace.sampling import (
    TraceIdRatioBased, ParentBased, AlwaysOn, AlwaysOff
)

from structlog import get_logger
from prometheus_client import REGISTRY

from ..config.config import settings

# Type definitions
F = TypeVar('F', bound=Callable[..., Any])

# Logger
logger = get_logger(__name__)

# Global tracer and meter instances
_tracer: Optional[Tracer] = None
_meter: Optional[Meter] = None

# Span attribute keys
class SpanAttributes:
    """Standard span attributes for the MemesPy system."""
    # Service attributes
    MEME_ID = "meme.id"
    MEME_TOPIC = "meme.topic"
    MEME_FORMAT = "meme.format"
    MEME_QUALITY_SCORE = "meme.quality_score"
    
    # Agent attributes
    AGENT_NAME = "agent.name"
    AGENT_TYPE = "agent.type"
    AGENT_VERSION = "agent.version"
    AGENT_OPERATION = "agent.operation"
    
    # Pipeline attributes
    PIPELINE_STAGE = "pipeline.stage"
    PIPELINE_ITERATION = "pipeline.iteration"
    REFINEMENT_COUNT = "refinement.count"
    
    # Cache attributes
    CACHE_HIT = "cache.hit"
    CACHE_KEY = "cache.key"
    CACHE_TYPE = "cache.type"
    
    # Database attributes
    DB_OPERATION = "db.operation"
    DB_TABLE = "db.table"
    DB_ROWS_AFFECTED = "db.rows_affected"
    
    # Error attributes
    ERROR_TYPE = "error.type"
    ERROR_MESSAGE = "error.message"
    ERROR_STACK = "error.stack"
    
    # Performance attributes
    DURATION_MS = "duration.ms"
    MEMORY_USED_MB = "memory.used_mb"
    CPU_PERCENT = "cpu.percent"

class MetricNames:
    """Standard metric names for the MemesPy system."""
    # Request metrics
    REQUEST_DURATION = "meme.request.duration"
    REQUEST_COUNT = "meme.request.count"
    REQUEST_SIZE = "meme.request.size"
    
    # Pipeline metrics
    PIPELINE_DURATION = "meme.pipeline.duration"
    PIPELINE_STAGE_DURATION = "meme.pipeline.stage.duration"
    QUALITY_SCORE_DISTRIBUTION = "meme.quality.score.distribution"
    REFINEMENT_ITERATIONS = "meme.refinement.iterations"
    
    # Agent metrics
    AGENT_EXECUTION_DURATION = "meme.agent.execution.duration"
    AGENT_SUCCESS_RATE = "meme.agent.success.rate"
    AGENT_ERROR_COUNT = "meme.agent.error.count"
    
    # Cache metrics
    CACHE_HIT_RATE = "meme.cache.hit.rate"
    CACHE_MISS_COUNT = "meme.cache.miss.count"
    CACHE_EVICTION_COUNT = "meme.cache.eviction.count"
    
    # Resource metrics
    MEMORY_USAGE = "meme.resource.memory.usage"
    CPU_USAGE = "meme.resource.cpu.usage"
    CONNECTION_POOL_SIZE = "meme.db.connection.pool.size"
    CONNECTION_POOL_ACTIVE = "meme.db.connection.pool.active"

def init_telemetry(
    service_name: str = "memespy",
    service_version: str = settings.app_version,
    otlp_endpoint: Optional[str] = None,
    jaeger_endpoint: Optional[str] = None,
    prometheus_port: int = 9090,
    sampling_rate: float = 1.0,
    enable_console_export: bool = False
) -> None:
    """
    Initialize OpenTelemetry instrumentation for the MemesPy system.
    
    Args:
        service_name: Name of the service
        service_version: Version of the service
        otlp_endpoint: OTLP collector endpoint
        jaeger_endpoint: Jaeger collector endpoint
        prometheus_port: Port for Prometheus metrics
        sampling_rate: Trace sampling rate (0.0 to 1.0)
        enable_console_export: Enable console exporters for debugging
    """
    global _tracer, _meter
    
    # Create resource
    resource = Resource(attributes={
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "service.environment": settings.app_env,
        "service.instance.id": os.environ.get("HOSTNAME", "local"),
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
    })
    
    # Initialize trace provider
    trace_provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(root=TraceIdRatioBased(sampling_rate))
    )
    
    # Add span processors
    if enable_console_export:
        trace_provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )
    
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True  # Use secure=True in production
        )
        trace_provider.add_span_processor(
            BatchSpanProcessor(otlp_exporter)
        )
    
    if jaeger_endpoint:
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_endpoint.split(":")[0],
            agent_port=int(jaeger_endpoint.split(":")[1]) if ":" in jaeger_endpoint else 6831,
            udp_split_oversized_batches=True,
        )
        trace_provider.add_span_processor(
            BatchSpanProcessor(jaeger_exporter)
        )
    
    set_tracer_provider(trace_provider)
    _tracer = trace.get_tracer(__name__, service_version)
    
    # Initialize metrics provider
    metric_readers = []
    
    if enable_console_export:
        metric_readers.append(
            PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=10000
            )
        )
    
    if otlp_endpoint:
        otlp_metric_exporter = OTLPMetricExporter(
            endpoint=otlp_endpoint,
            insecure=True
        )
        metric_readers.append(
            PeriodicExportingMetricReader(
                otlp_metric_exporter,
                export_interval_millis=10000
            )
        )
    
    # Add Prometheus reader
    prometheus_reader = PrometheusMetricReader(
        target_info=resource.attributes,
        registry=REGISTRY
    )
    metric_readers.append(prometheus_reader)
    
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=metric_readers
    )
    set_meter_provider(meter_provider)
    _meter = metrics.get_meter(__name__, service_version)
    
    # Set up propagator
    set_global_textmap(TraceContextTextMapPropagator())
    
    # Instrument libraries
    FastAPIInstrumentor.instrument(tracer_provider=trace_provider)
    SQLAlchemyInstrumentor().instrument(tracer_provider=trace_provider)
    RedisInstrumentor().instrument(tracer_provider=trace_provider)
    HTTPXClientInstrumentor().instrument(tracer_provider=trace_provider)
    LoggingInstrumentor().instrument(set_logging_format=True)
    
    logger.info(
        "OpenTelemetry initialized",
        service_name=service_name,
        service_version=service_version,
        sampling_rate=sampling_rate
    )

def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer(__name__)
    return _tracer

def get_meter() -> Meter:
    """Get the global meter instance."""
    global _meter
    if _meter is None:
        _meter = metrics.get_meter(__name__)
    return _meter

@contextmanager
def trace_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
    links: Optional[List[Link]] = None
):
    """
    Context manager for creating a trace span.
    
    Args:
        name: Span name
        kind: Span kind (INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER)
        attributes: Initial span attributes
        links: Links to other spans
        
    Yields:
        The created span
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        name,
        kind=kind,
        attributes=attributes,
        links=links
    ) as span:
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            # Add performance metrics
            if hasattr(span, "_start_time"):
                duration = time.time() - span._start_time
                span.set_attribute(SpanAttributes.DURATION_MS, duration * 1000)

def trace_method(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    Decorator for tracing methods.
    
    Args:
        name: Optional span name (defaults to function name)
        kind: Span kind
        attributes: Additional span attributes
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with trace_span(span_name, kind, attributes) as span:
                # Add function arguments as attributes
                span.set_attribute("function.args", str(args))
                span.set_attribute("function.kwargs", str(kwargs))
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.result", str(result)[:1000])
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace_span(span_name, kind, attributes) as span:
                # Add function arguments as attributes
                span.set_attribute("function.args", str(args))
                span.set_attribute("function.kwargs", str(kwargs))
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.result", str(result)[:1000])
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

class TracingContext:
    """Context manager for maintaining trace context across async boundaries."""
    
    def __init__(self, trace_id: Optional[str] = None, span_id: Optional[str] = None):
        self.trace_id = trace_id
        self.span_id = span_id
        self._token = None
    
    def __enter__(self):
        """Attach trace context."""
        if self.trace_id and self.span_id:
            # Create context from IDs
            span_context = SpanContext(
                trace_id=int(self.trace_id, 16),
                span_id=int(self.span_id, 16),
                is_remote=True
            )
            ctx = trace.set_span_in_context(
                trace.NonRecordingSpan(span_context)
            )
            self._token = attach(ctx)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Detach trace context."""
        if self._token:
            detach(self._token)
    
    @staticmethod
    def get_current_trace_context() -> Dict[str, str]:
        """Get current trace context as dictionary."""
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            ctx = span.get_span_context()
            return {
                "trace_id": format(ctx.trace_id, '032x'),
                "span_id": format(ctx.span_id, '016x'),
                "trace_flags": format(ctx.trace_flags, '02x')
            }
        return {}
    
    @staticmethod
    def inject_to_headers(headers: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into HTTP headers."""
        inject(headers)
        return headers
    
    @staticmethod
    def extract_from_headers(headers: Dict[str, str]) -> TracingContext:
        """Extract trace context from HTTP headers."""
        ctx = extract(headers)
        span = trace.get_current_span(ctx)
        if span and span.get_span_context().is_valid:
            span_ctx = span.get_span_context()
            return TracingContext(
                trace_id=format(span_ctx.trace_id, '032x'),
                span_id=format(span_ctx.span_id, '016x')
            )
        return TracingContext()

class MetricsCollector:
    """Collector for application metrics with cardinality control."""
    
    def __init__(self):
        meter = get_meter()
        
        # Create metrics with controlled cardinality
        self._request_duration = meter.create_histogram(
            MetricNames.REQUEST_DURATION,
            unit="ms",
            description="Request duration in milliseconds"
        )
        
        self._request_counter = meter.create_counter(
            MetricNames.REQUEST_COUNT,
            unit="requests",
            description="Total number of requests"
        )
        
        self._pipeline_duration = meter.create_histogram(
            MetricNames.PIPELINE_DURATION,
            unit="ms",
            description="Pipeline execution duration"
        )
        
        self._quality_score = meter.create_histogram(
            MetricNames.QUALITY_SCORE_DISTRIBUTION,
            unit="score",
            description="Distribution of meme quality scores"
        )
        
        self._agent_duration = meter.create_histogram(
            MetricNames.AGENT_EXECUTION_DURATION,
            unit="ms",
            description="Agent execution duration"
        )
        
        self._cache_hits = meter.create_counter(
            MetricNames.CACHE_HIT_RATE,
            unit="hits",
            description="Cache hit count"
        )
        
        self._cache_misses = meter.create_counter(
            MetricNames.CACHE_MISS_COUNT,
            unit="misses",
            description="Cache miss count"
        )
        
        # Resource metrics with callbacks
        meter.create_observable_gauge(
            MetricNames.MEMORY_USAGE,
            callbacks=[self._get_memory_usage],
            unit="MB",
            description="Memory usage in MB"
        )
        
        meter.create_observable_gauge(
            MetricNames.CPU_USAGE,
            callbacks=[self._get_cpu_usage],
            unit="percent",
            description="CPU usage percentage"
        )
    
    def record_request(
        self,
        duration_ms: float,
        method: str,
        endpoint: str,
        status_code: int
    ):
        """Record HTTP request metrics with controlled cardinality."""
        # Normalize endpoint to reduce cardinality
        normalized_endpoint = self._normalize_endpoint(endpoint)
        
        attributes = {
            "method": method,
            "endpoint": normalized_endpoint,
            "status_class": f"{status_code // 100}xx"
        }
        
        self._request_duration.record(duration_ms, attributes)
        self._request_counter.add(1, attributes)
    
    def record_pipeline_execution(
        self,
        duration_ms: float,
        stage: str,
        success: bool
    ):
        """Record pipeline execution metrics."""
        attributes = {
            "stage": stage,
            "success": str(success)
        }
        self._pipeline_duration.record(duration_ms, attributes)
    
    def record_quality_score(self, score: float, format_type: str):
        """Record meme quality score."""
        # Bucket score to reduce cardinality
        score_bucket = round(score, 1)
        attributes = {
            "format": format_type,
            "score_bucket": str(score_bucket)
        }
        self._quality_score.record(score, attributes)
    
    def record_agent_execution(
        self,
        agent_name: str,
        duration_ms: float,
        success: bool
    ):
        """Record agent execution metrics."""
        attributes = {
            "agent": agent_name,
            "success": str(success)
        }
        self._agent_duration.record(duration_ms, attributes)
    
    def record_cache_access(self, hit: bool, cache_type: str):
        """Record cache access metrics."""
        attributes = {"cache_type": cache_type}
        if hit:
            self._cache_hits.add(1, attributes)
        else:
            self._cache_misses.add(1, attributes)
    
    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        """Normalize endpoint to reduce cardinality."""
        # Replace UUIDs with placeholder
        import re
        endpoint = re.sub(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '{id}',
            endpoint
        )
        # Replace numeric IDs
        endpoint = re.sub(r'/\d+', '/{id}', endpoint)
        return endpoint
    
    @staticmethod
    def _get_memory_usage(options):
        """Callback for memory usage metric."""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        yield metrics.Observation(memory_mb, {})
    
    @staticmethod
    def _get_cpu_usage(options):
        """Callback for CPU usage metric."""
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        yield metrics.Observation(cpu_percent, {})

# Global metrics collector instance
metrics_collector = MetricsCollector()

class LogCorrelation:
    """Utilities for correlating logs with traces."""
    
    @staticmethod
    def inject_trace_context(record: logging.LogRecord) -> None:
        """Inject trace context into log record."""
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            ctx = span.get_span_context()
            record.trace_id = format(ctx.trace_id, '032x')
            record.span_id = format(ctx.span_id, '016x')
            record.trace_flags = format(ctx.trace_flags, '02x')
        else:
            record.trace_id = "0" * 32
            record.span_id = "0" * 16
            record.trace_flags = "00"
    
    @staticmethod
    def get_trace_context_for_logging() -> Dict[str, str]:
        """Get trace context for structured logging."""
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            ctx = span.get_span_context()
            return {
                "trace_id": format(ctx.trace_id, '032x'),
                "span_id": format(ctx.span_id, '016x'),
                "trace_flags": format(ctx.trace_flags, '02x')
            }
        return {}
    
    @staticmethod
    def configure_logging():
        """Configure logging to include trace context."""
        class TraceContextFilter(logging.Filter):
            def filter(self, record):
                LogCorrelation.inject_trace_context(record)
                return True
        
        # Add filter to all handlers
        for handler in logging.root.handlers:
            handler.addFilter(TraceContextFilter())
        
        # Update format to include trace context
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[trace_id=%(trace_id)s span_id=%(span_id)s] - %(message)s'
        )
        for handler in logging.root.handlers:
            handler.setFormatter(formatter)

# Performance profiling integration
class PerformanceProfiler:
    """Integration with performance profiling tools."""
    
    @staticmethod
    def profile_span(span: Span) -> None:
        """Add performance profiling data to span."""
        try:
            import psutil
            import resource
            
            # CPU stats
            cpu_times = psutil.cpu_times()
            span.set_attribute("cpu.user_time", cpu_times.user)
            span.set_attribute("cpu.system_time", cpu_times.system)
            
            # Memory stats
            memory = psutil.virtual_memory()
            span.set_attribute("memory.used_mb", memory.used / 1024 / 1024)
            span.set_attribute("memory.percent", memory.percent)
            
            # Process stats
            process = psutil.Process()
            span.set_attribute("process.cpu_percent", process.cpu_percent())
            span.set_attribute("process.memory_mb", process.memory_info().rss / 1024 / 1024)
            span.set_attribute("process.num_threads", process.num_threads())
            
            # Resource usage
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            span.set_attribute("resource.user_time", rusage.ru_utime)
            span.set_attribute("resource.system_time", rusage.ru_stime)
            span.set_attribute("resource.max_rss", rusage.ru_maxrss)
            
        except Exception as e:
            logger.warning("Failed to collect performance data", error=str(e))

# Distributed debugging utilities
class DistributedDebugger:
    """Utilities for distributed debugging with OpenTelemetry."""
    
    @staticmethod
    def add_debug_info(span: Span, **kwargs) -> None:
        """Add debug information to span."""
        for key, value in kwargs.items():
            span.set_attribute(f"debug.{key}", str(value))
    
    @staticmethod
    def create_debug_snapshot(span: Span) -> Dict[str, Any]:
        """Create a debug snapshot of current state."""
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "trace_id": format(span.get_span_context().trace_id, '032x'),
            "span_id": format(span.get_span_context().span_id, '016x'),
            "attributes": dict(span.attributes) if hasattr(span, 'attributes') else {},
            "events": [],
            "links": []
        }
        
        # Add current baggage
        baggage_items = baggage.get_all()
        if baggage_items:
            snapshot["baggage"] = dict(baggage_items)
        
        return snapshot
    
    @staticmethod
    def emit_debug_event(span: Span, event_name: str, attributes: Dict[str, Any]) -> None:
        """Emit a debug event on the span."""
        span.add_event(
            name=f"debug.{event_name}",
            attributes={f"debug.{k}": str(v) for k, v in attributes.items()}
        )

# Export all public APIs
__all__ = [
    "init_telemetry",
    "get_tracer",
    "get_meter",
    "trace_span",
    "trace_method",
    "TracingContext",
    "MetricsCollector",
    "metrics_collector",
    "LogCorrelation",
    "PerformanceProfiler",
    "DistributedDebugger",
    "SpanAttributes",
    "MetricNames"
]