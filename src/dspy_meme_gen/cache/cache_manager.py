"""Cache management for the DSPy Meme Generator."""
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta

from redis.asyncio import Redis, from_url

from ..config.config import get_settings

class CacheManager:
    """Manages caching operations using Redis."""

    def __init__(self, redis_client: Redis) -> None:
        """
        Initialize the cache manager.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client

    async def get(self, key: str) -> Optional[str]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found, None otherwise
        """
        return await self.redis.get(key)

    async def set(
        self,
        key: str,
        value: Union[str, bytes, int, float],
        expire: Optional[int] = None
    ) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            expire: Optional expiration time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.redis.set(key, value, ex=expire)
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.redis.delete(key)
            return True
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        return await self.redis.exists(key) > 0

    async def set_hash(self, key: str, mapping: Dict[str, Any]) -> bool:
        """
        Set a hash in the cache.
        
        Args:
            key: Cache key
            mapping: Dictionary to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.redis.hset(key, mapping=mapping)
            return True
        except Exception:
            return False

    async def get_hash(self, key: str) -> Dict[str, Any]:
        """
        Get a hash from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Dictionary of hash values
        """
        result = await self.redis.hgetall(key)
        return result if result else {}

    async def clear(self) -> bool:
        """
        Clear all cached data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.redis.flushdb()
            return True
        except Exception:
            return False

# Global cache manager instance
cache_manager: Optional[CacheManager] = None

def init_cache_manager(redis_client: Redis) -> None:
    """
    Initialize the global cache manager.
    
    Args:
        redis_client: Redis client instance
    """
    global cache_manager
    cache_manager = CacheManager(redis_client)

def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.
    
    Returns:
        CacheManager instance
        
    Raises:
        RuntimeError: If cache manager is not initialized
    """
    if cache_manager is None:
        raise RuntimeError("Cache manager not initialized")
    return cache_manager

async def get_async_cache() -> Redis:
    """
    Get an async Redis client instance.
    
    Returns:
        Redis: Initialized Redis client
        
    Raises:
        Exception: If connection fails
    """
    settings = get_settings()
    redis_client = from_url(
        str(settings.cache_url),
        encoding="utf-8",
        decode_responses=True,
        max_connections=settings.cache_max_connections
    )
    
    # Test connection
    await redis_client.ping()
    
    # Initialize cache manager if needed
    if cache_manager is None:
        init_cache_manager(redis_client)
        
    return redis_client 