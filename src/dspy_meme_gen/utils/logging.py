"""Logging utilities."""

import json
import logging
import os
import sys
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast

import structlog
from structlog.types import EventDict, Processor

# Type variables for decorators
F = TypeVar('F', bound=Callable[..., Any])


def add_timestamp(
    _: logging.Logger,
    __: str,
    event_dict: EventDict
) -> EventDict:
    """Add ISO-format timestamp to log event."""
    event_dict["timestamp"] = datetime.utcnow().isoformat()
    return event_dict


def add_log_level(
    _: logging.Logger,
    method_name: str,
    event_dict: EventDict
) -> EventDict:
    """Add log level to event dict."""
    event_dict["level"] = method_name
    return event_dict


def add_caller_info(
    _: logging.Logger,
    __: str,
    event_dict: EventDict
) -> EventDict:
    """Add caller information to log event."""
    frame = sys._getframe()
    while frame:
        module = frame.f_globals.get("__name__", "")
        if module.startswith("dspy_meme_gen"):
            event_dict.update({
                "module": module,
                "function": frame.f_code.co_name,
                "line": frame.f_lineno
            })
            break
        frame = frame.f_back
    return event_dict


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: Optional[str] = None
) -> None:
    """
    Set up structured logging.
    
    Args:
        level: Log level
        json_format: Whether to output logs in JSON format
        log_file: Optional file to write logs to
    """
    # Set up processors
    processors: list[Processor] = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        add_caller_info,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set up standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


def log_function_call(func: F) -> F:
    """
    Decorator to log function calls with arguments and return values.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    logger = get_logger(func.__module__)
    
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        # Create context
        context = {
            "function": func.__name__,
            "args": str(args),
            "kwargs": str(kwargs)
        }
        
        logger.info("function_call", **context)
        try:
            result = await func(*args, **kwargs)
            logger.info(
                "function_return",
                function=func.__name__,
                result=str(result)
            )
            return result
        except Exception as e:
            logger.error(
                "function_error",
                function=func.__name__,
                error=str(e),
                exc_info=True
            )
            raise
            
    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        # Create context
        context = {
            "function": func.__name__,
            "args": str(args),
            "kwargs": str(kwargs)
        }
        
        logger.info("function_call", **context)
        try:
            result = func(*args, **kwargs)
            logger.info(
                "function_return",
                function=func.__name__,
                result=str(result)
            )
            return result
        except Exception as e:
            logger.error(
                "function_error",
                function=func.__name__,
                error=str(e),
                exc_info=True
            )
            raise
            
    return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper)


def log_performance(func: F) -> F:
    """
    Decorator to log function performance metrics.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    logger = get_logger(func.__module__)
    
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = datetime.now()
        try:
            result = await func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                "function_performance",
                function=func.__name__,
                duration_seconds=duration
            )
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                "function_error_performance",
                function=func.__name__,
                duration_seconds=duration,
                error=str(e)
            )
            raise
            
    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                "function_performance",
                function=func.__name__,
                duration_seconds=duration
            )
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                "function_error_performance",
                function=func.__name__,
                duration_seconds=duration,
                error=str(e)
            )
            raise
            
    return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper)


# Set up logging with default configuration
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_format=os.getenv("LOG_JSON", "true").lower() == "true",
    log_file=os.getenv("LOG_FILE")
) 