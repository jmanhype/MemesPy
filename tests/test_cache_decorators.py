"""Tests for cache decorators."""
from typing import Any, Dict, Optional
import time

import pytest

from dspy_meme_gen.cache.decorators import cached, invalidate_cache
from dspy_meme_gen.cache.factory import CacheFactory, CacheType
from dspy_meme_gen.utils.config import AppConfig


@pytest.fixture(autouse=True)
def setup_cache() -> None:
    """Set up cache factory for tests."""
    config = AppConfig(
        database={
            "url": "sqlite:///test.db",
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
            "echo": False,
        },
        cloudinary={
            "cloud_name": "test",
            "api_key": "test",
            "api_secret": "test",
            "secure": True,
            "folder": "test",
        },
        openai={
            "api_key": "test",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 150,
        },
        logging={
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": None,
            "rotate": True,
            "max_size": "10MB",
            "backup_count": 5,
        },
        redis={
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": None,
            "pool_size": 10,
            "ttl": 3600,
            "prefix": "test",
        },
        debug=True,
        testing=True,
        secret_key="test",
        allowed_formats=["JPEG", "PNG", "GIF"],
        max_file_size=10 * 1024 * 1024,
    )
    CacheFactory.initialize(config)
    yield
    CacheFactory.clear_instances()


def test_cached_basic() -> None:
    """Test basic caching functionality."""
    call_count = 0
    
    @cached(namespace="test")
    def example_function(x: int, y: int) -> int:
        nonlocal call_count
        call_count += 1
        return x + y
    
    # First call should execute the function
    result1 = example_function(1, 2)
    assert result1 == 3
    assert call_count == 1
    
    # Second call should use cached result
    result2 = example_function(1, 2)
    assert result2 == 3
    assert call_count == 1
    
    # Different arguments should execute the function again
    result3 = example_function(2, 3)
    assert result3 == 5
    assert call_count == 2


def test_cached_ttl() -> None:
    """Test cache TTL."""
    call_count = 0
    
    @cached(namespace="test", ttl=1)
    def example_function() -> int:
        nonlocal call_count
        call_count += 1
        return 42
    
    # First call should execute the function
    result1 = example_function()
    assert result1 == 42
    assert call_count == 1
    
    # Second call should use cached result
    result2 = example_function()
    assert result2 == 42
    assert call_count == 1
    
    # Wait for TTL to expire
    time.sleep(1.1)
    
    # Third call should execute the function again
    result3 = example_function()
    assert result3 == 42
    assert call_count == 2


def test_cached_key_prefix() -> None:
    """Test cache key prefix."""
    call_count = 0
    
    @cached(namespace="test", key_prefix="custom")
    def example_function() -> int:
        nonlocal call_count
        call_count += 1
        return 42
    
    # Call with custom prefix
    result = example_function()
    assert result == 42
    
    # Verify key in cache
    cache = CacheFactory.get_cache(CacheType.REDIS, "test")
    assert cache.exists("custom")


def test_cached_complex_args() -> None:
    """Test caching with complex arguments."""
    call_count = 0
    
    @cached(namespace="test")
    def example_function(
        x: int,
        y: Optional[str] = None,
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {
            "x": x,
            "y": y,
            "args": args,
            "kwargs": kwargs
        }
    
    # Call with various argument types
    result1 = example_function(
        1,
        "test",
        42,
        True,
        key1="value1",
        key2={"nested": "value"}
    )
    assert call_count == 1
    
    # Same call should use cache
    result2 = example_function(
        1,
        "test",
        42,
        True,
        key1="value1",
        key2={"nested": "value"}
    )
    assert result2 == result1
    assert call_count == 1
    
    # Different argument order for kwargs should still use cache
    result3 = example_function(
        1,
        "test",
        42,
        True,
        key2={"nested": "value"},
        key1="value1"
    )
    assert result3 == result1
    assert call_count == 1


def test_cached_exclude_args() -> None:
    """Test caching with excluded arguments."""
    call_count = 0
    
    @cached(namespace="test", include_args=False)
    def example_function(x: int, y: int) -> int:
        nonlocal call_count
        call_count += 1
        return x + y
    
    # Different arguments should use same cache key
    result1 = example_function(1, 2)
    assert result1 == 3
    assert call_count == 1
    
    result2 = example_function(3, 4)
    assert result2 == 3  # Should return cached result
    assert call_count == 1


def test_cached_exclude_kwargs() -> None:
    """Test caching with excluded keyword arguments."""
    call_count = 0
    
    @cached(namespace="test", include_kwargs=False)
    def example_function(x: int, **kwargs: Any) -> int:
        nonlocal call_count
        call_count += 1
        return x + sum(kwargs.values())
    
    # Different kwargs should use same cache key
    result1 = example_function(1, y=2)
    assert result1 == 3
    assert call_count == 1
    
    result2 = example_function(1, y=3)
    assert result2 == 3  # Should return cached result
    assert call_count == 1


def test_invalidate_cache() -> None:
    """Test cache invalidation."""
    call_count = 0
    
    @cached(namespace="test")
    def example_function(x: int, y: int = 0) -> int:
        nonlocal call_count
        call_count += 1
        return x + y
    
    # Call function and verify caching
    result1 = example_function(1, 2)
    assert result1 == 3
    assert call_count == 1
    
    result2 = example_function(1, 2)
    assert result2 == 3
    assert call_count == 1
    
    # Invalidate specific cache entry
    invalidate_cache(example_function, namespace="test", args=(1, 2))
    
    # Call should execute function again
    result3 = example_function(1, 2)
    assert result3 == 3
    assert call_count == 2
    
    # Other cache entries should remain
    example_function(2, 3)
    assert call_count == 3
    
    example_function(2, 3)  # Should use cache
    assert call_count == 3
    
    # Invalidate entire namespace
    invalidate_cache(example_function, namespace="test")
    
    # All calls should execute function again
    example_function(1, 2)
    assert call_count == 4
    
    example_function(2, 3)
    assert call_count == 5


def test_cached_error_handling() -> None:
    """Test caching with function errors."""
    call_count = 0
    
    @cached(namespace="test")
    def example_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        if x < 0:
            raise ValueError("Negative value")
        return x
    
    # Successful call should be cached
    result1 = example_function(1)
    assert result1 == 1
    assert call_count == 1
    
    result2 = example_function(1)
    assert result2 == 1
    assert call_count == 1
    
    # Error should not be cached
    with pytest.raises(ValueError):
        example_function(-1)
    assert call_count == 2
    
    with pytest.raises(ValueError):
        example_function(-1)
    assert call_count == 3 