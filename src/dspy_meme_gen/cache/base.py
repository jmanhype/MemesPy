"""Base cache interface."""
from abc import ABC, abstractmethod
from typing import Any, Optional, Union


class BaseCache(ABC):
    """Base class for cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None if not found
        """
        pass
    
    @abstractmethod
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
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if key exists
        """
        pass
    
    @abstractmethod
    def ttl(self, key: str) -> Optional[int]:
        """Get the TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[int]: TTL in seconds or None if key doesn't exist
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def clear_namespace(self, namespace: str) -> bool:
        """Clear all keys in a namespace.
        
        Args:
            namespace: Cache namespace
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from the cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            dict: Dictionary of key-value pairs (only for found keys)
        """
        pass
    
    @abstractmethod
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
        """
        pass
    
    @abstractmethod
    def delete_many(self, keys: list[str]) -> bool:
        """Delete multiple values from the cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            bool: True if all successful
        """
        pass
    
    @abstractmethod
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
        """
        pass
    
    @abstractmethod
    def touch(self, key: str, ttl: Optional[int] = None) -> bool:
        """Update the TTL for a key.
        
        Args:
            key: Cache key
            ttl: Optional new TTL in seconds
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the cache connection."""
        pass
    
    def __enter__(self) -> "BaseCache":
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self.close() 