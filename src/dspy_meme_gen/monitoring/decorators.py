"""Decorators for tracking metrics in the DSPy Meme Generation pipeline."""

import time
import functools
from typing import Any, Callable, Optional, Type, TypeVar, cast
from .metrics import MetricsCollector

F = TypeVar('F', bound=Callable[..., Any])

def track_agent_metrics(agent_name: Optional[str] = None) -> Callable[[F], F]:
    """Decorator to track agent execution metrics.
    
    Args:
        agent_name: Name of the agent to track. If not provided, uses the class name.
        
    Returns:
        A decorator function that tracks agent metrics.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get agent name from class if not provided
            _agent_name = agent_name
            if _agent_name is None and args and hasattr(args[0], "__class__"):
                _agent_name = args[0].__class__.__name__
                
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                await MetricsCollector.track_agent_execution(
                    agent_name=_agent_name or "unknown",
                    start_time=start_time,
                    success=True
                )
                return result
            except Exception as e:
                await MetricsCollector.track_agent_execution(
                    agent_name=_agent_name or "unknown",
                    start_time=start_time,
                    success=False,
                    error_type=e.__class__.__name__
                )
                raise
                
        return cast(F, wrapper)
    return decorator

def track_external_service(service_name: str, operation: str) -> Callable[[F], F]:
    """Decorator to track external service metrics.
    
    Args:
        service_name: Name of the external service
        operation: Operation being performed
        
    Returns:
        A decorator function that tracks external service metrics.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                await MetricsCollector.track_external_service(
                    service_name=service_name,
                    operation=operation,
                    start_time=start_time,
                    success=True
                )
                return result
            except Exception as e:
                await MetricsCollector.track_external_service(
                    service_name=service_name,
                    operation=operation,
                    start_time=start_time,
                    success=False,
                    error_type=e.__class__.__name__
                )
                raise
                
        return cast(F, wrapper)
    return decorator

def track_cache_operation(cache_type: str) -> Callable[[F], F]:
    """Decorator to track cache operation metrics.
    
    Args:
        cache_type: Type of cache being accessed
        
    Returns:
        A decorator function that tracks cache metrics.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                # Determine if it was a hit or miss based on result
                operation = "hit" if result is not None else "miss"
                await MetricsCollector.track_cache_operation(
                    operation=operation,
                    cache_type=cache_type,
                    start_time=start_time
                )
                return result
            except Exception as e:
                await MetricsCollector.track_cache_operation(
                    operation="error",
                    cache_type=cache_type,
                    start_time=start_time
                )
                raise
                
        return cast(F, wrapper)
    return decorator

def track_db_operation(operation: str, table: str) -> Callable[[F], F]:
    """Decorator to track database operation metrics.
    
    Args:
        operation: Type of database operation
        table: Database table being accessed
        
    Returns:
        A decorator function that tracks database metrics.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                await MetricsCollector.track_db_operation(
                    operation=operation,
                    table=table,
                    start_time=start_time
                )
                return result
            except Exception as e:
                await MetricsCollector.track_db_operation(
                    operation=f"{operation}_error",
                    table=table,
                    start_time=start_time
                )
                raise
                
        return cast(F, wrapper)
    return decorator

def track_meme_generation(template_type: str) -> Callable[[F], F]:
    """Decorator to track meme generation metrics.
    
    Args:
        template_type: Type of meme template used
        
    Returns:
        A decorator function that tracks meme generation metrics.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                await MetricsCollector.track_meme_generation(
                    template_type=template_type,
                    start_time=start_time,
                    status="success"
                )
                return result
            except Exception as e:
                await MetricsCollector.track_meme_generation(
                    template_type=template_type,
                    start_time=start_time,
                    status="error"
                )
                raise
                
        return cast(F, wrapper)
    return decorator 