"""Prometheus metrics and monitoring utilities for the DSPy Meme Generation pipeline."""

from typing import Dict, Any, Optional
import time
import asyncio
import psutil
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    REGISTRY,
    start_http_server
)
from fastapi import Request
from structlog import get_logger

logger = get_logger()

# Generation Pipeline Metrics
GENERATION_TIME = Histogram(
    "meme_generation_seconds",
    "Time spent generating memes",
    ["status", "template_type"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

GENERATION_TOTAL = Counter(
    "meme_generation_total",
    "Total number of meme generation attempts",
    ["status"]
)

MEME_QUALITY_SCORE = Histogram(
    "meme_quality_score",
    "Distribution of meme quality scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Agent Performance Metrics
AGENT_EXECUTION_TIME = Histogram(
    "agent_execution_seconds",
    "Time spent in each agent",
    ["agent_name"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

AGENT_SUCCESS_RATE = Counter(
    "agent_success_total",
    "Successful agent operations",
    ["agent_name"]
)

AGENT_FAILURES = Counter(
    "agent_failures_total",
    "Failed agent operations",
    ["agent_name", "error_type"]
)

# External Service Metrics
EXTERNAL_SERVICE_LATENCY = Histogram(
    "external_service_latency_seconds",
    "External service request latency",
    ["service_name", "operation"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

EXTERNAL_SERVICE_ERRORS = Counter(
    "external_service_errors_total",
    "External service errors",
    ["service_name", "error_type"]
)

# Cache Performance Metrics
CACHE_OPERATIONS = Counter(
    "cache_operations_total",
    "Cache operations (hits/misses)",
    ["operation", "cache_type"]
)

CACHE_LATENCY = Histogram(
    "cache_latency_seconds",
    "Cache operation latency",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
)

# Database Metrics
DB_OPERATION_LATENCY = Histogram(
    "db_operation_latency_seconds",
    "Database operation latency",
    ["operation", "table"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

DB_POOL_SIZE = Gauge(
    "db_pool_size",
    "Current database connection pool size"
)

DB_POOL_AVAILABLE = Gauge(
    "db_pool_available_connections",
    "Available connections in the pool"
)

# Resource Usage Metrics
RESOURCE_USAGE = Gauge(
    "resource_usage",
    "System resource usage",
    ["resource_type"]
)

# Request Metrics
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path", "status"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)

# System Info
SYSTEM_INFO = Info(
    "dspy_meme_gen",
    "DSPy Meme Generator system information"
)

class MetricsMiddleware:
    """Middleware for collecting HTTP request metrics."""
    
    async def __call__(self, request: Request, call_next):
        """Process the request and collect metrics.
        
        Args:
            request: The incoming HTTP request
            call_next: The next middleware in the chain
            
        Returns:
            The HTTP response
        """
        start_time = time.time()
        method = request.method
        path = request.url.path
        
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as e:
            status = 500
            raise e
        finally:
            duration = time.time() - start_time
            REQUEST_LATENCY.labels(
                method=method,
                path=path,
                status=status
            ).observe(duration)
            REQUEST_COUNT.labels(
                method=method,
                path=path,
                status=status
            ).inc()
            
        return response

class MetricsCollector:
    """Collector for application-specific metrics."""
    
    @staticmethod
    async def track_meme_generation(
        template_type: str,
        start_time: float,
        status: str = "success"
    ) -> None:
        """Track meme generation metrics.
        
        Args:
            template_type: Type of meme template used
            start_time: Generation start timestamp
            status: Generation status (success/error)
        """
        duration = time.time() - start_time
        GENERATION_TIME.labels(
            status=status,
            template_type=template_type
        ).observe(duration)
        GENERATION_TOTAL.labels(status=status).inc()
    
    @staticmethod
    async def track_agent_execution(
        agent_name: str,
        start_time: float,
        success: bool = True,
        error_type: Optional[str] = None
    ) -> None:
        """Track agent execution metrics.
        
        Args:
            agent_name: Name of the agent
            start_time: Execution start timestamp
            success: Whether execution was successful
            error_type: Type of error if execution failed
        """
        duration = time.time() - start_time
        AGENT_EXECUTION_TIME.labels(agent_name=agent_name).observe(duration)
        
        if success:
            AGENT_SUCCESS_RATE.labels(agent_name=agent_name).inc()
        else:
            AGENT_FAILURES.labels(
                agent_name=agent_name,
                error_type=error_type or "unknown"
            ).inc()
    
    @staticmethod
    async def track_external_service(
        service_name: str,
        operation: str,
        start_time: float,
        success: bool = True,
        error_type: Optional[str] = None
    ) -> None:
        """Track external service metrics.
        
        Args:
            service_name: Name of the external service
            operation: Operation being performed
            start_time: Operation start timestamp
            success: Whether operation was successful
            error_type: Type of error if operation failed
        """
        duration = time.time() - start_time
        EXTERNAL_SERVICE_LATENCY.labels(
            service_name=service_name,
            operation=operation
        ).observe(duration)
        
        if not success:
            EXTERNAL_SERVICE_ERRORS.labels(
                service_name=service_name,
                error_type=error_type or "unknown"
            ).inc()
    
    @staticmethod
    async def track_cache_operation(
        operation: str,
        cache_type: str,
        start_time: float
    ) -> None:
        """Track cache operation metrics.
        
        Args:
            operation: Type of cache operation (hit/miss)
            cache_type: Type of cache being accessed
            start_time: Operation start timestamp
        """
        duration = time.time() - start_time
        CACHE_OPERATIONS.labels(
            operation=operation,
            cache_type=cache_type
        ).inc()
        CACHE_LATENCY.labels(operation=operation).observe(duration)
    
    @staticmethod
    async def track_db_operation(
        operation: str,
        table: str,
        start_time: float
    ) -> None:
        """Track database operation metrics.
        
        Args:
            operation: Type of database operation
            table: Database table being accessed
            start_time: Operation start timestamp
        """
        duration = time.time() - start_time
        DB_OPERATION_LATENCY.labels(
            operation=operation,
            table=table
        ).observe(duration)

async def start_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus metrics server.
    
    Args:
        port: Port to run the metrics server on
    """
    try:
        start_http_server(port)
        logger.info("Started metrics server", port=port)
    except Exception as e:
        logger.error("Failed to start metrics server", error=str(e))
        raise

async def track_resource_usage() -> None:
    """Track system resource usage metrics."""
    while True:
        try:
            RESOURCE_USAGE.labels("cpu").set(psutil.cpu_percent())
            RESOURCE_USAGE.labels("memory").set(psutil.virtual_memory().percent)
            RESOURCE_USAGE.labels("disk").set(psutil.disk_usage("/").percent)
            
            # Update system info
            SYSTEM_INFO.info({
                "python_version": psutil.python_version(),
                "platform": psutil.platform(),
                "cpu_count": str(psutil.cpu_count()),
                "total_memory": str(psutil.virtual_memory().total)
            })
            
            await asyncio.sleep(60)
        except Exception as e:
            logger.error("Failed to track resource usage", error=str(e))
            await asyncio.sleep(60)  # Retry after error

def setup_metrics(app: Any) -> None:
    """Set up metrics collection for the application.
    
    Args:
        app: The FastAPI application instance
    """
    # Add metrics middleware
    app.add_middleware(MetricsMiddleware)
    
    # Start metrics server
    asyncio.create_task(start_metrics_server())
    
    # Start resource usage tracking
    asyncio.create_task(track_resource_usage())
    
    logger.info("Metrics collection configured") 