"""
Comprehensive span definitions for all MemesPy operations.
Provides standardized span creation and management for consistent tracing.
"""

from typing import Dict, Any, Optional, List, Callable, TypeVar
from functools import wraps
from contextlib import contextmanager
import time
import inspect

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from structlog import get_logger

from .telemetry import SpanAttributes, get_tracer, PerformanceProfiler, trace_span

logger = get_logger(__name__)

F = TypeVar('F', bound=Callable[..., Any])

class SpanDefinitions:
    """Standardized span definitions for MemesPy operations."""
    
    # API Layer Spans
    API_REQUEST = "api.request"
    API_RESPONSE = "api.response"
    API_MIDDLEWARE = "api.middleware"
    API_ERROR_HANDLER = "api.error_handler"
    
    # Pipeline Spans
    PIPELINE_EXECUTE = "pipeline.execute"
    PIPELINE_STAGE = "pipeline.stage"
    PIPELINE_ROUTER = "pipeline.router"
    PIPELINE_VERIFICATION = "pipeline.verification"
    PIPELINE_GENERATION = "pipeline.generation"
    PIPELINE_REFINEMENT = "pipeline.refinement"
    
    # Agent Spans
    AGENT_EXECUTE = "agent.execute"
    AGENT_ROUTER = "agent.router"
    AGENT_TREND_SCANNER = "agent.trend_scanner"
    AGENT_FORMAT_SELECTOR = "agent.format_selector"
    AGENT_PROMPT_GENERATOR = "agent.prompt_generator"
    AGENT_IMAGE_RENDERER = "agent.image_renderer"
    AGENT_FACTUALITY = "agent.factuality"
    AGENT_APPROPRIATENESS = "agent.appropriateness"
    AGENT_INSTRUCTION_FOLLOWING = "agent.instruction_following"
    AGENT_SCORER = "agent.scorer"
    AGENT_REFINEMENT = "agent.refinement"
    
    # Service Layer Spans
    SERVICE_MEME_CREATE = "service.meme.create"
    SERVICE_MEME_GET = "service.meme.get"
    SERVICE_MEME_LIST = "service.meme.list"
    SERVICE_TREND_ANALYZE = "service.trend.analyze"
    SERVICE_FORMAT_SELECT = "service.format.select"
    
    # Database Spans
    DB_QUERY = "db.query"
    DB_TRANSACTION = "db.transaction"
    DB_CONNECTION = "db.connection"
    DB_MIGRATION = "db.migration"
    
    # Cache Spans
    CACHE_GET = "cache.get"
    CACHE_SET = "cache.set"
    CACHE_DELETE = "cache.delete"
    CACHE_CLEAR = "cache.clear"
    
    # External Service Spans
    EXTERNAL_OPENAI = "external.openai"
    EXTERNAL_IMAGE_API = "external.image_api"
    EXTERNAL_TREND_API = "external.trend_api"
    
    # Background Task Spans
    TASK_CLEANUP = "task.cleanup"
    TASK_METRICS_EXPORT = "task.metrics_export"
    TASK_CACHE_WARM = "task.cache_warm"

class SpanManager:
    """Manager for creating and configuring spans with consistent attributes."""
    
    @staticmethod
    def create_span(
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[List[trace.Link]] = None
    ) -> Span:
        """Create a span with standard attributes."""
        tracer = get_tracer()
        span = tracer.start_span(
            name=name,
            kind=kind,
            attributes=attributes,
            links=links
        )
        
        # Add standard attributes
        span.set_attribute("service.layer", SpanManager._get_layer_from_name(name))
        span.set_attribute("span.type", name.split(".")[0])
        
        return span
    
    @staticmethod
    def _get_layer_from_name(span_name: str) -> str:
        """Determine service layer from span name."""
        prefix = span_name.split(".")[0]
        layer_map = {
            "api": "presentation",
            "service": "business",
            "agent": "ai",
            "pipeline": "orchestration",
            "db": "data",
            "cache": "data",
            "external": "integration",
            "task": "background"
        }
        return layer_map.get(prefix, "unknown")

@contextmanager
def api_span(
    operation: str,
    method: str,
    path: str,
    headers: Optional[Dict[str, str]] = None
):
    """Create span for API operations."""
    with trace_span(
        SpanDefinitions.API_REQUEST,
        kind=SpanKind.SERVER,
        attributes={
            "http.method": method,
            "http.path": path,
            "http.operation": operation,
            "http.headers": str(headers) if headers else None
        }
    ) as span:
        yield span

@contextmanager
def pipeline_span(
    stage: str,
    topic: str,
    format: Optional[str] = None,
    iteration: Optional[int] = None
):
    """Create span for pipeline operations."""
    attributes = {
        SpanAttributes.PIPELINE_STAGE: stage,
        SpanAttributes.MEME_TOPIC: topic
    }
    
    if format:
        attributes[SpanAttributes.MEME_FORMAT] = format
    if iteration is not None:
        attributes[SpanAttributes.PIPELINE_ITERATION] = iteration
    
    with trace_span(
        f"{SpanDefinitions.PIPELINE_STAGE}.{stage}",
        attributes=attributes
    ) as span:
        yield span

@contextmanager
def agent_span(
    agent_name: str,
    operation: str,
    input_data: Optional[Dict[str, Any]] = None
):
    """Create span for agent operations."""
    attributes = {
        SpanAttributes.AGENT_NAME: agent_name,
        SpanAttributes.AGENT_OPERATION: operation
    }
    
    if input_data:
        attributes["agent.input"] = str(input_data)[:1000]  # Limit size
    
    with trace_span(
        f"{SpanDefinitions.AGENT_EXECUTE}.{agent_name}",
        attributes=attributes
    ) as span:
        yield span

@contextmanager
def db_span(
    operation: str,
    table: str,
    query: Optional[str] = None
):
    """Create span for database operations."""
    attributes = {
        SpanAttributes.DB_OPERATION: operation,
        SpanAttributes.DB_TABLE: table,
        "db.system": "postgresql"
    }
    
    if query:
        attributes["db.statement"] = query[:500]  # Limit query size
    
    with trace_span(
        f"{SpanDefinitions.DB_QUERY}.{operation}",
        kind=SpanKind.CLIENT,
        attributes=attributes
    ) as span:
        yield span

@contextmanager
def cache_span(
    operation: str,
    cache_type: str,
    key: str,
    hit: Optional[bool] = None
):
    """Create span for cache operations."""
    attributes = {
        SpanAttributes.CACHE_TYPE: cache_type,
        SpanAttributes.CACHE_KEY: key[:100],  # Limit key size
        "cache.operation": operation
    }
    
    if hit is not None:
        attributes[SpanAttributes.CACHE_HIT] = hit
    
    with trace_span(
        f"{SpanDefinitions.CACHE_GET}.{operation}",
        kind=SpanKind.CLIENT,
        attributes=attributes
    ) as span:
        yield span

@contextmanager
def external_service_span(
    service: str,
    operation: str,
    url: Optional[str] = None
):
    """Create span for external service calls."""
    attributes = {
        "external.service": service,
        "external.operation": operation
    }
    
    if url:
        attributes["http.url"] = url
    
    with trace_span(
        f"external.{service}.{operation}",
        kind=SpanKind.CLIENT,
        attributes=attributes
    ) as span:
        yield span

def trace_api_endpoint(
    operation: str,
    tags: Optional[List[str]] = None
) -> Callable[[F], F]:
    """Decorator for tracing API endpoints."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            with api_span(
                operation=operation,
                method=request.method,
                path=str(request.url.path),
                headers=dict(request.headers)
            ) as span:
                # Add tags
                if tags:
                    span.set_attribute("api.tags", tags)
                
                # Add request info
                span.set_attribute("http.client_ip", request.client.host)
                span.set_attribute("http.user_agent", request.headers.get("user-agent"))
                
                try:
                    response = await func(request, *args, **kwargs)
                    span.set_attribute("http.status_code", response.status_code)
                    return response
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator

def trace_pipeline_stage(
    stage: str
) -> Callable[[F], F]:
    """Decorator for tracing pipeline stages."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract context from arguments
            topic = kwargs.get("topic", "unknown")
            format = kwargs.get("format")
            
            with pipeline_span(stage, topic, format) as span:
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Add result attributes
                    if isinstance(result, dict):
                        if "score" in result:
                            span.set_attribute(SpanAttributes.MEME_QUALITY_SCORE, result["score"])
                        if "meme_id" in result:
                            span.set_attribute(SpanAttributes.MEME_ID, result["meme_id"])
                    
                    duration = (time.time() - start_time) * 1000
                    span.set_attribute(SpanAttributes.DURATION_MS, duration)
                    
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator

def trace_agent(
    agent_name: str
) -> Callable[[F], F]:
    """Decorator for tracing agent execution."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get operation from function name
            operation = func.__name__
            
            # Extract input data from arguments
            input_data = {}
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            for param_name, param_value in bound_args.arguments.items():
                if param_name != "self":
                    input_data[param_name] = str(param_value)[:100]
            
            with agent_span(agent_name, operation, input_data) as span:
                # Add performance profiling
                PerformanceProfiler.profile_span(span)
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Add result info
                    if result:
                        span.set_attribute("agent.result_type", type(result).__name__)
                        if isinstance(result, dict) and "success" in result:
                            span.set_attribute("agent.success", result["success"])
                    
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute(SpanAttributes.ERROR_TYPE, type(e).__name__)
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get operation from function name
            operation = func.__name__
            
            # Extract input data from arguments
            input_data = {}
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            for param_name, param_value in bound_args.arguments.items():
                if param_name != "self":
                    input_data[param_name] = str(param_value)[:100]
            
            with agent_span(agent_name, operation, input_data) as span:
                # Add performance profiling
                PerformanceProfiler.profile_span(span)
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Add result info
                    if result:
                        span.set_attribute("agent.result_type", type(result).__name__)
                        if isinstance(result, dict) and "success" in result:
                            span.set_attribute("agent.success", result["success"])
                    
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute(SpanAttributes.ERROR_TYPE, type(e).__name__)
                    raise
        
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    return decorator

def trace_database_operation(
    operation: str
) -> Callable[[F], F]:
    """Decorator for tracing database operations."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract table name from function arguments or name
            table = kwargs.get("table", "unknown")
            if table == "unknown" and args and hasattr(args[0], "__tablename__"):
                table = args[0].__tablename__
            
            # Extract query if available
            query = kwargs.get("query")
            
            with db_span(operation, table, query) as span:
                try:
                    result = await func(*args, **kwargs)
                    
                    # Add result metrics
                    if hasattr(result, "rowcount"):
                        span.set_attribute(SpanAttributes.DB_ROWS_AFFECTED, result.rowcount)
                    
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator

def trace_cache_operation(
    operation: str,
    cache_type: str = "redis"
) -> Callable[[F], F]:
    """Decorator for tracing cache operations."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract key from arguments
            key = str(args[0]) if args else kwargs.get("key", "unknown")
            
            with cache_span(operation, cache_type, key) as span:
                try:
                    result = await func(*args, **kwargs)
                    
                    # Determine cache hit/miss
                    if operation == "get":
                        hit = result is not None
                        span.set_attribute(SpanAttributes.CACHE_HIT, hit)
                    
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator

def trace_external_call(
    service: str
) -> Callable[[F], F]:
    """Decorator for tracing external service calls."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            operation = func.__name__
            url = kwargs.get("url")
            
            with external_service_span(service, operation, url) as span:
                try:
                    result = await func(*args, **kwargs)
                    
                    # Add response info if available
                    if hasattr(result, "status_code"):
                        span.set_attribute("http.status_code", result.status_code)
                    
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator

def trace_actor_message(
    message_type: str
) -> Callable[[F], F]:
    """Decorator for tracing actor message handling."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(self, message, *args, **kwargs):
            actor_name = getattr(self, 'name', self.__class__.__name__)
            
            with trace_span(
                f"actor.message.{message_type}",
                kind=SpanKind.CONSUMER,
                attributes={
                    SpanAttributes.AGENT_NAME: actor_name,
                    SpanAttributes.AGENT_OPERATION: f"handle_{message_type}",
                    "message.type": message_type,
                    "message.id": getattr(message, 'id', 'unknown'),
                    "actor.message_queue_size": getattr(self, '_mailbox_size', 0)
                }
            ) as span:
                # Add message correlation if available
                if hasattr(message, 'correlation_id'):
                    span.set_attribute("message.correlation_id", message.correlation_id)
                
                # Add message priority if available
                if hasattr(message, 'priority'):
                    span.set_attribute("message.priority", message.priority)
                
                try:
                    result = await func(self, message, *args, **kwargs)
                    
                    # Add processing result info
                    if isinstance(result, dict):
                        if "success" in result:
                            span.set_attribute("message.processing.success", result["success"])
                        if "response_type" in result:
                            span.set_attribute("message.response_type", result["response_type"])
                    
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute(SpanAttributes.ERROR_TYPE, type(e).__name__)
                    span.set_attribute("actor.message_processing_failed", True)
                    raise
        
        @wraps(func)
        def sync_wrapper(self, message, *args, **kwargs):
            actor_name = getattr(self, 'name', self.__class__.__name__)
            
            with trace_span(
                f"actor.message.{message_type}",
                kind=SpanKind.CONSUMER,
                attributes={
                    SpanAttributes.AGENT_NAME: actor_name,
                    SpanAttributes.AGENT_OPERATION: f"handle_{message_type}",
                    "message.type": message_type,
                    "message.id": getattr(message, 'id', 'unknown'),
                    "actor.message_queue_size": getattr(self, '_mailbox_size', 0)
                }
            ) as span:
                # Add message correlation if available
                if hasattr(message, 'correlation_id'):
                    span.set_attribute("message.correlation_id", message.correlation_id)
                
                # Add message priority if available
                if hasattr(message, 'priority'):
                    span.set_attribute("message.priority", message.priority)
                
                try:
                    result = func(self, message, *args, **kwargs)
                    
                    # Add processing result info
                    if isinstance(result, dict):
                        if "success" in result:
                            span.set_attribute("message.processing.success", result["success"])
                        if "response_type" in result:
                            span.set_attribute("message.response_type", result["response_type"])
                    
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute(SpanAttributes.ERROR_TYPE, type(e).__name__)
                    span.set_attribute("actor.message_processing_failed", True)
                    raise
        
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    return decorator

def trace_actor_lifecycle(
    lifecycle_event: str
) -> Callable[[F], F]:
    """Decorator for tracing actor lifecycle events."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            actor_name = getattr(self, 'name', self.__class__.__name__)
            
            with trace_span(
                f"actor.lifecycle.{lifecycle_event}",
                attributes={
                    SpanAttributes.AGENT_NAME: actor_name,
                    SpanAttributes.AGENT_OPERATION: lifecycle_event,
                    "actor.lifecycle_event": lifecycle_event,
                    "actor.state": getattr(self, '_state', 'unknown')
                }
            ) as span:
                try:
                    result = await func(self, *args, **kwargs)
                    
                    # Add lifecycle result info
                    span.set_attribute("actor.lifecycle_success", True)
                    if hasattr(self, '_state'):
                        span.set_attribute("actor.new_state", self._state)
                    
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute("actor.lifecycle_success", False)
                    raise
        
        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            actor_name = getattr(self, 'name', self.__class__.__name__)
            
            with trace_span(
                f"actor.lifecycle.{lifecycle_event}",
                attributes={
                    SpanAttributes.AGENT_NAME: actor_name,
                    SpanAttributes.AGENT_OPERATION: lifecycle_event,
                    "actor.lifecycle_event": lifecycle_event,
                    "actor.state": getattr(self, '_state', 'unknown')
                }
            ) as span:
                try:
                    result = func(self, *args, **kwargs)
                    
                    # Add lifecycle result info
                    span.set_attribute("actor.lifecycle_success", True)
                    if hasattr(self, '_state'):
                        span.set_attribute("actor.new_state", self._state)
                    
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute("actor.lifecycle_success", False)
                    raise
        
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    return decorator

@contextmanager
def actor_supervision_span(
    supervisor_name: str,
    supervised_actor: str,
    supervision_action: str
):
    """Create span for actor supervision operations."""
    with trace_span(
        f"actor.supervision.{supervision_action}",
        attributes={
            "supervisor.name": supervisor_name,
            "supervised.actor": supervised_actor,
            "supervision.action": supervision_action,
            SpanAttributes.AGENT_OPERATION: f"supervise_{supervision_action}"
        }
    ) as span:
        yield span

def trace_meme_generation_pipeline(
    stage: str
) -> Callable[[F], F]:
    """Decorator specifically for meme generation pipeline stages."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract meme context from arguments
            meme_id = kwargs.get("meme_id", "unknown")
            topic = kwargs.get("topic", "unknown")
            format_type = kwargs.get("format", "unknown")
            
            with trace_span(
                f"meme.pipeline.{stage}",
                attributes={
                    SpanAttributes.PIPELINE_STAGE: stage,
                    SpanAttributes.MEME_ID: meme_id,
                    SpanAttributes.MEME_TOPIC: topic,
                    SpanAttributes.MEME_FORMAT: format_type,
                    "pipeline.type": "meme_generation"
                }
            ) as span:
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Add stage-specific metrics
                    duration = (time.time() - start_time) * 1000
                    span.set_attribute(SpanAttributes.DURATION_MS, duration)
                    
                    # Add result metrics based on stage
                    if isinstance(result, dict):
                        if stage == "scoring" and "score" in result:
                            span.set_attribute(SpanAttributes.MEME_QUALITY_SCORE, result["score"])
                        elif stage == "refinement" and "iterations" in result:
                            span.set_attribute(SpanAttributes.REFINEMENT_COUNT, result["iterations"])
                        elif "success" in result:
                            span.set_attribute(f"stage.{stage}.success", result["success"])
                    
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute(f"stage.{stage}.failed", True)
                    raise
        
        return wrapper
    return decorator

# Export all span-related utilities
__all__ = [
    "SpanDefinitions",
    "SpanManager",
    "api_span",
    "pipeline_span",
    "agent_span",
    "db_span",
    "cache_span",
    "external_service_span",
    "actor_supervision_span",
    "trace_api_endpoint",
    "trace_pipeline_stage",
    "trace_agent",
    "trace_database_operation",
    "trace_cache_operation",
    "trace_external_call",
    "trace_actor_message",
    "trace_actor_lifecycle",
    "trace_meme_generation_pipeline"
]