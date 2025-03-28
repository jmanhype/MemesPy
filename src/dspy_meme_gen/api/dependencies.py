"""API dependencies."""

import logging
from typing import Optional, Any
import redis.asyncio as redis
from fastapi import Depends

from ..config.config import settings

# Configure logger
logger = logging.getLogger(__name__)

# Redis connection
redis_client: Optional[redis.Redis] = None

async def get_cache():
    """
    Get Redis cache connection.
    
    Returns:
        Redis: Redis client connection
    """
    global redis_client
    
    # Return existing connection if available
    if redis_client is not None:
        return redis_client
    
    # Create connection if Redis URL is configured
    if settings.redis_url:
        try:
            logger.info(f"Connecting to Redis at {settings.redis_url}")
            redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            return redis_client
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
    
    # Return dummy Redis client if no connection
    return DummyRedis()


async def close_connections():
    """Close Redis connections."""
    global redis_client
    if redis_client is not None:
        await redis_client.close()
        redis_client = None
        logger.info("Redis connection closed")


class DummyRedis:
    """
    Dummy Redis implementation for when Redis is not available.
    
    Implements simple cache interface with local memory storage.
    """
    
    def __init__(self):
        """Initialize the dummy cache."""
        self.cache = {}
        logger.warning("Using dummy Redis implementation (no persistence)")
    
    async def get(self, key: str) -> Optional[str]:
        """Get a value from cache."""
        return self.cache.get(key)
    
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set a value in cache."""
        self.cache[key] = value
        return True
    
    async def delete(self, key: str) -> int:
        """Delete a value from cache."""
        if key in self.cache:
            del self.cache[key]
            return 1
        return 0
    
    async def close(self) -> None:
        """Close connection (dummy operation)."""
        self.cache = {} 