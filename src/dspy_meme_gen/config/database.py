"""Database configuration and utilities."""

import os
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Type variables for retry decorator
F = TypeVar('F', bound=Callable[..., Any])


class DatabaseConfig:
    """Database configuration settings."""
    
    def __init__(
        self,
        url: Optional[str] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 1800,
        echo: bool = False
    ) -> None:
        """
        Initialize database configuration.
        
        Args:
            url: Database URL (defaults to DATABASE_URL env var)
            pool_size: Size of the connection pool
            max_overflow: Maximum number of connections to allow over pool_size
            pool_timeout: Seconds to wait before giving up on getting a connection
            pool_recycle: Seconds after which to recycle a connection
            echo: Whether to log SQL statements
        """
        self.url = url or os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/db")
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo
        
    def get_engine(self) -> AsyncEngine:
        """
        Create database engine with current configuration.
        
        Returns:
            AsyncEngine instance
        """
        url = make_url(self.url)
        
        # Create engine with connection pooling
        engine = create_async_engine(
            url,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            echo=self.echo
        )
        
        return engine


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1,
    max_wait: float = 10
) -> Callable[[F], F]:
    """
    Decorator to retry database operations with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries in seconds
        max_wait: Maximum wait time between retries in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=min_wait, max=max_wait),
            retry=lambda e: isinstance(e, SQLAlchemyError),
            before=lambda retry_state: logger.warning(
                f"Retrying {func.__name__} (attempt {retry_state.attempt_number})"
            )
        )
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except SQLAlchemyError as e:
                logger.error(f"Database error in {func.__name__}: {str(e)}")
                raise
                
        return cast(F, wrapper)
    return decorator


# Global database configuration instance
config = DatabaseConfig() 