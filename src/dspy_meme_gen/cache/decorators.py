"""Cache decorators."""
import functools
import hashlib
import inspect
from typing import Any, Callable, Optional, TypeVar, Union, cast

from .factory import CacheFactory, CacheType


T = TypeVar("T", bound=Callable[..., Any])


def cached(
    namespace: Optional[str] = None,
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None,
    cache_type: CacheType = CacheType.REDIS,
    include_args: bool = True,
    include_kwargs: bool = True
) -> Callable[[T], T]:
    """Cache function results.
    
    Args:
        namespace: Cache namespace
        ttl: Cache TTL in seconds
        key_prefix: Cache key prefix
        cache_type: Type of cache to use
        include_args: Include positional arguments in cache key
        include_kwargs: Include keyword arguments in cache key
        
    Returns:
        Callable: Decorated function
        
    Example:
        >>> @cached(namespace="memes", ttl=3600)
        ... def get_meme(meme_id: int) -> dict:
        ...     # Expensive operation
        ...     return {"id": meme_id, "title": "Example"}
    """
    def decorator(func: T) -> T:
        # Get function signature for better cache keys
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get cache instance
            cache = CacheFactory.get_cache(cache_type, namespace)
            
            # Build cache key
            key_parts = []
            
            # Add prefix if provided, otherwise use function name
            prefix = key_prefix or func.__name__
            key_parts.append(prefix)
            
            if include_args and args:
                # Add positional arguments
                bound_args = sig.bind_partial(*args).arguments
                for param_name, value in bound_args.items():
                    key_parts.append(f"{param_name}={_hash_value(value)}")
            
            if include_kwargs and kwargs:
                # Add keyword arguments (sorted for consistency)
                for key, value in sorted(kwargs.items()):
                    key_parts.append(f"{key}={_hash_value(value)}")
            
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        return cast(T, wrapper)
    
    return decorator


def _hash_value(value: Any) -> str:
    """Create a hash for a cache key value.
    
    Args:
        value: Value to hash
        
    Returns:
        str: Hash of the value
    """
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    
    # For other types, create a stable hash
    try:
        # Try to convert to string first
        value_str = str(value)
    except Exception:
        # If that fails, use repr
        value_str = repr(value)
    
    # Create MD5 hash (good enough for cache keys)
    return hashlib.md5(value_str.encode()).hexdigest()


def invalidate_cache(
    func: Callable[..., Any],
    namespace: Optional[str] = None,
    key_prefix: Optional[str] = None,
    cache_type: CacheType = CacheType.REDIS,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None
) -> None:
    """Invalidate cached results for a function.
    
    Args:
        func: Function to invalidate cache for
        namespace: Cache namespace
        key_prefix: Cache key prefix
        cache_type: Type of cache to use
        args: Optional positional arguments to include in cache key
        kwargs: Optional keyword arguments to include in cache key
        
    Example:
        >>> @cached(namespace="memes")
        ... def get_meme(meme_id: int) -> dict:
        ...     return {"id": meme_id}
        ...
        >>> # Invalidate all cached results
        >>> invalidate_cache(get_meme, namespace="memes")
        >>> # Invalidate specific result
        >>> invalidate_cache(get_meme, namespace="memes", args=(42,))
    """
    # Get cache instance
    cache = CacheFactory.get_cache(cache_type, namespace)
    
    if args is None and kwargs is None:
        # No specific arguments provided, clear entire namespace
        cache.clear_namespace(namespace or func.__name__)
        return
    
    # Build cache key for specific arguments
    sig = inspect.signature(func)
    key_parts = []
    
    # Add prefix if provided, otherwise use function name
    prefix = key_prefix or func.__name__
    key_parts.append(prefix)
    
    if args:
        # Add positional arguments
        bound_args = sig.bind_partial(*args).arguments
        for param_name, value in bound_args.items():
            key_parts.append(f"{param_name}={_hash_value(value)}")
    
    if kwargs:
        # Add keyword arguments (sorted for consistency)
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}={_hash_value(value)}")
    
    cache_key = ":".join(key_parts)
    cache.delete(cache_key) 