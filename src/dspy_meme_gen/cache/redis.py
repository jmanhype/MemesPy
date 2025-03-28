"""Redis cache implementation."""
import json
from typing import Any, Optional, Union

import redis
from redis.client import Pipeline
from redis.connection import ConnectionPool

from .base import BaseCache


class RedisCache(BaseCache):
    """Redis cache implementation."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        pool_size: int = 10,
        ttl: int = 3600,
        prefix: str = "dspy_meme_gen",
        namespace: Optional[str] = None
    ) -> None:
        """Initialize Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            pool_size: Connection pool size
            ttl: Default TTL in seconds
            prefix: Key prefix
            namespace: Optional namespace
        """
        self.pool = ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=pool_size,
            decode_responses=True
        )
        self.client = redis.Redis(connection_pool=self.pool)
        self.default_ttl = ttl
        self.prefix = prefix
        self.namespace = namespace
    
    def _make_key(self, key: str) -> str:
        """Create a prefixed key.
        
        Args:
            key: Original key
            
        Returns:
            str: Prefixed key
        """
        parts = [self.prefix]
        if self.namespace:
            parts.append(self.namespace)
        parts.append(key)
        return ":".join(parts)
    
    def _serialize(self, value: Any) -> str:
        """Serialize a value to JSON.
        
        Args:
            value: Value to serialize
            
        Returns:
            str: JSON string
            
        Raises:
            ValueError: If value cannot be serialized
        """
        try:
            return json.dumps(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot serialize value: {e}")
    
    def _deserialize(self, value: Optional[str]) -> Any:
        """Deserialize a JSON string.
        
        Args:
            value: JSON string
            
        Returns:
            Any: Deserialized value
            
        Raises:
            ValueError: If value cannot be deserialized
        """
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Cannot deserialize value: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None if not found
            
        Raises:
            ValueError: If value cannot be deserialized
        """
        value = self.client.get(self._make_key(key))
        return self._deserialize(value)
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds
            
        Returns:
            bool: True if successful
            
        Raises:
            ValueError: If value cannot be serialized
        """
        key = self._make_key(key)
        value = self._serialize(value)
        ttl = ttl if ttl is not None else self.default_ttl
        return bool(self.client.set(key, value, ex=ttl))
    
    def delete(self, key: str) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful
        """
        return bool(self.client.delete(self._make_key(key)))
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if key exists
        """
        return bool(self.client.exists(self._make_key(key)))
    
    def ttl(self, key: str) -> Optional[int]:
        """Get the TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[int]: TTL in seconds or None if key doesn't exist
        """
        ttl = self.client.ttl(self._make_key(key))
        return ttl if ttl > 0 else None
    
    def incr(self, key: str, amount: int = 1) -> int:
        """Increment a value in the cache.
        
        Args:
            key: Cache key
            amount: Amount to increment by
            
        Returns:
            int: New value
            
        Raises:
            ValueError: If value is not an integer
        """
        try:
            if amount >= 0:
                return self.client.incrby(self._make_key(key), amount)
            else:
                return self.client.decrby(self._make_key(key), abs(amount))
        except redis.ResponseError:
            raise ValueError("Value is not an integer")
    
    def decr(self, key: str, amount: int = 1) -> int:
        """Decrement a value in the cache.
        
        Args:
            key: Cache key
            amount: Amount to decrement by
            
        Returns:
            int: New value
            
        Raises:
            ValueError: If value is not an integer
        """
        return self.incr(key, -amount)
    
    def clear_namespace(self, namespace: str) -> bool:
        """Clear all keys in a namespace.
        
        Args:
            namespace: Cache namespace
            
        Returns:
            bool: True if successful
        """
        pattern = f"{self.prefix}:{namespace}:*"
        keys = self.client.keys(pattern)
        if keys:
            return bool(self.client.delete(*keys))
        return True
    
    def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from the cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            dict: Dictionary of key-value pairs (only for found keys)
            
        Raises:
            ValueError: If any value cannot be deserialized
        """
        prefixed_keys = [self._make_key(key) for key in keys]
        values = self.client.mget(prefixed_keys)
        result = {}
        
        for key, value in zip(keys, values):
            if value is not None:
                result[key] = self._deserialize(value)
        
        return result
    
    def set_many(
        self,
        mapping: dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values in the cache.
        
        Args:
            mapping: Dictionary of key-value pairs
            ttl: Optional TTL in seconds
            
        Returns:
            bool: True if all successful
            
        Raises:
            ValueError: If any value cannot be serialized
        """
        ttl = ttl if ttl is not None else self.default_ttl
        pipe: Pipeline = self.client.pipeline()
        
        try:
            for key, value in mapping.items():
                key = self._make_key(key)
                value = self._serialize(value)
                pipe.set(key, value, ex=ttl)
            
            results = pipe.execute()
            return all(results)
        except redis.RedisError:
            return False
    
    def delete_many(self, keys: list[str]) -> bool:
        """Delete multiple values from the cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            bool: True if all successful
        """
        prefixed_keys = [self._make_key(key) for key in keys]
        return bool(self.client.delete(*prefixed_keys))
    
    def get_or_set(
        self,
        key: str,
        default: Any,
        ttl: Optional[int] = None
    ) -> Any:
        """Get a value from the cache or set it if not found.
        
        Args:
            key: Cache key
            default: Default value or callable
            ttl: Optional TTL in seconds
            
        Returns:
            Any: Cached value or default
            
        Raises:
            ValueError: If value cannot be serialized or deserialized
        """
        value = self.get(key)
        if value is None:
            value = default() if callable(default) else default
            self.set(key, value, ttl)
        return value
    
    def touch(self, key: str, ttl: Optional[int] = None) -> bool:
        """Update the TTL for a key.
        
        Args:
            key: Cache key
            ttl: Optional new TTL in seconds
            
        Returns:
            bool: True if successful
        """
        ttl = ttl if ttl is not None else self.default_ttl
        return bool(self.client.expire(self._make_key(key), ttl))
    
    def close(self) -> None:
        """Close the cache connection."""
        self.client.close()
        self.pool.disconnect() 