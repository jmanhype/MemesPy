"""Observability module for metrics, tracing, and telemetry."""

import logging

logger = logging.getLogger(__name__)

# Try to import observability components, fall back to no-op if dependencies missing
try:
    from .metrics import (
        register_metrics,
        record_meme_generation_metric,
        record_generation_time,
        record_error_metric,
        get_metrics_registry
    )
    METRICS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Metrics not available: {e}")
    METRICS_AVAILABLE = False
    # No-op functions
    def register_metrics(): pass
    def record_meme_generation_metric(*args, **kwargs): pass
    def record_generation_time(*args, **kwargs): pass
    def record_error_metric(*args, **kwargs): pass
    def get_metrics_registry(): return None

try:
    from .tracing import (
        get_tracer,
        create_span,
        trace_meme_generation,
        trace_verification,
        trace_scoring
    )
    TRACING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Tracing not available: {e}")
    TRACING_AVAILABLE = False
    # No-op functions
    def get_tracer(*args, **kwargs): return None
    def create_span(*args, **kwargs): return None
    def trace_meme_generation(*args, **kwargs): pass
    def trace_verification(*args, **kwargs): pass
    def trace_scoring(*args, **kwargs): pass

try:
    from .telemetry import (
        initialize_telemetry,
        shutdown_telemetry,
        TelemetryConfig
    )
    TELEMETRY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Telemetry not available: {e}")
    TELEMETRY_AVAILABLE = False
    # No-op functions
    def initialize_telemetry(*args, **kwargs): pass
    def shutdown_telemetry(*args, **kwargs): pass
    class TelemetryConfig: pass

# Spans and propagation are simpler
try:
    from .spans import (
        SpanContext,
        create_child_span,
        add_span_attributes,
        set_span_status
    )
except ImportError:
    class SpanContext: pass
    def create_child_span(*args, **kwargs): return None
    def add_span_attributes(*args, **kwargs): pass
    def set_span_status(*args, **kwargs): pass

try:
    from .propagation import (
        extract_trace_context,
        inject_trace_context,
        get_current_trace_id,
        get_current_span_id
    )
except ImportError:
    def extract_trace_context(*args, **kwargs): return {}
    def inject_trace_context(*args, **kwargs): pass
    def get_current_trace_id(): return None
    def get_current_span_id(): return None

__all__ = [
    # Metrics
    "register_metrics",
    "record_meme_generation_metric", 
    "record_generation_time",
    "record_error_metric",
    "get_metrics_registry",
    
    # Tracing
    "get_tracer",
    "create_span",
    "trace_meme_generation",
    "trace_verification", 
    "trace_scoring",
    
    # Telemetry
    "initialize_telemetry",
    "shutdown_telemetry",
    "TelemetryConfig",
    
    # Spans
    "SpanContext",
    "create_child_span",
    "add_span_attributes",
    "set_span_status",
    
    # Propagation
    "extract_trace_context",
    "inject_trace_context", 
    "get_current_trace_id",
    "get_current_span_id",
]