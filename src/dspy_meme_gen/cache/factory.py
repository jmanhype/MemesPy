"""Cache factory for managing different cache implementations."""
from enum import Enum
from typing import Optional, Type

from .base import BaseCache
from .redis import RedisCache
from ..utils.config import AppConfig


class CacheType(Enum):
    """Supported cache types."""
    REDIS = "redis"
    # Add more cache types as needed


class CacheFactory:
    """Factory for creating cache instances."""
    
    _instances: dict[str, BaseCache] = {}
    _config: Optional[AppConfig] = None
    
    @classmethod
    def initialize(cls, config: AppConfig) -> None:
        """Initialize the cache factory with configuration.
        
        Args:
            config: Application configuration
        """
        cls._config = config
    
    @classmethod
    def get_cache(
        cls,
        cache_type: CacheType = CacheType.REDIS,
        namespace: Optional[str] = None
    ) -> BaseCache:
        """Get a cache instance.
        
        Args:
            cache_type: Type of cache to create
            namespace: Optional namespace for the cache
            
        Returns:
            BaseCache: Cache instance
            
        Raises:
            ValueError: If factory not initialized or invalid cache type
        """
        if not cls._config:
            raise ValueError("Cache factory not initialized")
        
        # Create cache key
        cache_key = f"{cache_type.value}:{namespace}" if namespace else cache_type.value
        
        # Return existing instance if available
        if cache_key in cls._instances:
            return cls._instances[cache_key]
        
        # Create new instance
        cache_class = cls._get_cache_class(cache_type)
        cache_config = cls._get_cache_config(cache_type)
        
        if namespace:
            cache_config["namespace"] = namespace
        
        cache = cache_class(**cache_config)
        cls._instances[cache_key] = cache
        
        return cache
    
    @classmethod
    def clear_instances(cls) -> None:
        """Clear all cache instances."""
        cls._instances.clear()
    
    @classmethod
    def _get_cache_class(cls, cache_type: CacheType) -> Type[BaseCache]:
        """Get the cache class for a cache type.
        
        Args:
            cache_type: Type of cache
            
        Returns:
            Type[BaseCache]: Cache class
            
        Raises:
            ValueError: If invalid cache type
        """
        if cache_type == CacheType.REDIS:
            return RedisCache
        raise ValueError(f"Invalid cache type: {cache_type}")
    
    @classmethod
    def _get_cache_config(cls, cache_type: CacheType) -> dict:
        """Get configuration for a cache type.
        
        Args:
            cache_type: Type of cache
            
        Returns:
            dict: Cache configuration
            
        Raises:
            ValueError: If invalid cache type
        """
        assert cls._config is not None
        
        if cache_type == CacheType.REDIS:
            return {
                "host": cls._config.redis.host,
                "port": cls._config.redis.port,
                "db": cls._config.redis.db,
                "password": cls._config.redis.password,
                "pool_size": cls._config.redis.pool_size,
                "ttl": cls._config.redis.ttl,
                "prefix": cls._config.redis.prefix,
            }
        raise ValueError(f"Invalid cache type: {cache_type}") 