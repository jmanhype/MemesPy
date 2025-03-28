"""Tests for Redis cache implementation."""
from typing import Any, Generator
import json
import pytest
from redis import Redis

from dspy_meme_gen.cache.redis import RedisCache


@pytest.fixture
def redis_cache() -> Generator[RedisCache, None, None]:
    """Fixture for Redis cache instance.
    
    Yields:
        RedisCache: Redis cache instance
    """
    cache = RedisCache(
        host="localhost",
        port=6379,
        db=0,
        prefix="test",
        namespace="test"
    )
    yield cache
    # Clean up after test
    pattern = f"{cache.prefix}:{cache.namespace}:*"
    keys = cache.client.keys(pattern)
    if keys:
        cache.client.delete(*keys)
    cache.close()


def test_make_key(redis_cache: RedisCache) -> None:
    """Test key prefixing.
    
    Args:
        redis_cache: Redis cache instance
    """
    key = redis_cache._make_key("test_key")
    assert key == "test:test:test_key"


def test_serialize_deserialize(redis_cache: RedisCache) -> None:
    """Test value serialization and deserialization.
    
    Args:
        redis_cache: Redis cache instance
    """
    data = {
        "string": "test",
        "int": 42,
        "float": 3.14,
        "list": [1, 2, 3],
        "dict": {"key": "value"}
    }
    
    # Test serialization
    serialized = redis_cache._serialize(data)
    assert isinstance(serialized, str)
    assert json.loads(serialized) == data
    
    # Test deserialization
    deserialized = redis_cache._deserialize(serialized)
    assert deserialized == data
    
    # Test None handling
    assert redis_cache._deserialize(None) is None
    
    # Test invalid JSON
    with pytest.raises(ValueError):
        redis_cache._deserialize("invalid json")


def test_get_set(redis_cache: RedisCache) -> None:
    """Test get and set operations.
    
    Args:
        redis_cache: Redis cache instance
    """
    # Test setting and getting a value
    assert redis_cache.set("test_key", "test_value")
    assert redis_cache.get("test_key") == "test_value"
    
    # Test setting with TTL
    assert redis_cache.set("test_key_ttl", "test_value", ttl=1)
    assert redis_cache.get("test_key_ttl") == "test_value"
    assert redis_cache.ttl("test_key_ttl") == 1
    
    # Test getting non-existent key
    assert redis_cache.get("non_existent") is None


def test_delete(redis_cache: RedisCache) -> None:
    """Test delete operation.
    
    Args:
        redis_cache: Redis cache instance
    """
    # Set a value and verify it exists
    redis_cache.set("test_key", "test_value")
    assert redis_cache.exists("test_key")
    
    # Delete the value and verify it's gone
    assert redis_cache.delete("test_key")
    assert not redis_cache.exists("test_key")
    assert redis_cache.get("test_key") is None
    
    # Test deleting non-existent key
    assert not redis_cache.delete("non_existent")


def test_exists_ttl(redis_cache: RedisCache) -> None:
    """Test exists and TTL operations.
    
    Args:
        redis_cache: Redis cache instance
    """
    # Test exists
    assert not redis_cache.exists("test_key")
    redis_cache.set("test_key", "test_value")
    assert redis_cache.exists("test_key")
    
    # Test TTL
    redis_cache.set("test_key_ttl", "test_value", ttl=10)
    assert redis_cache.ttl("test_key_ttl") == 10
    assert redis_cache.ttl("non_existent") is None


def test_incr_decr(redis_cache: RedisCache) -> None:
    """Test increment and decrement operations.
    
    Args:
        redis_cache: Redis cache instance
    """
    # Test increment
    assert redis_cache.incr("counter") == 1
    assert redis_cache.incr("counter", 2) == 3
    
    # Test decrement
    assert redis_cache.decr("counter") == 2
    assert redis_cache.decr("counter", 2) == 0
    
    # Test increment/decrement with non-integer value
    redis_cache.set("string", "not_an_int")
    with pytest.raises(ValueError):
        redis_cache.incr("string")
    with pytest.raises(ValueError):
        redis_cache.decr("string")


def test_clear_namespace(redis_cache: RedisCache) -> None:
    """Test namespace clearing.
    
    Args:
        redis_cache: Redis cache instance
    """
    # Set multiple values
    redis_cache.set("key1", "value1")
    redis_cache.set("key2", "value2")
    
    # Clear namespace and verify
    assert redis_cache.clear_namespace("test")
    assert not redis_cache.exists("key1")
    assert not redis_cache.exists("key2")
    
    # Test clearing empty namespace
    assert redis_cache.clear_namespace("non_existent")


def test_get_many_set_many(redis_cache: RedisCache) -> None:
    """Test bulk get and set operations.
    
    Args:
        redis_cache: Redis cache instance
    """
    # Test set_many
    data = {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3"
    }
    assert redis_cache.set_many(data)
    
    # Test get_many
    result = redis_cache.get_many(["key1", "key2", "key3", "non_existent"])
    assert result == {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3"
    }
    
    # Test delete_many
    assert redis_cache.delete_many(["key1", "key2"])
    result = redis_cache.get_many(["key1", "key2", "key3"])
    assert result == {"key3": "value3"}


def test_get_or_set(redis_cache: RedisCache) -> None:
    """Test get_or_set operation.
    
    Args:
        redis_cache: Redis cache instance
    """
    # Test with non-existent key
    value = redis_cache.get_or_set("test_key", "default_value")
    assert value == "default_value"
    assert redis_cache.get("test_key") == "default_value"
    
    # Test with existing key
    value = redis_cache.get_or_set("test_key", "new_value")
    assert value == "default_value"
    
    # Test with callable default
    def default_callable() -> str:
        return "callable_value"
    
    value = redis_cache.get_or_set("new_key", default_callable)
    assert value == "callable_value"


def test_touch(redis_cache: RedisCache) -> None:
    """Test touch operation.
    
    Args:
        redis_cache: Redis cache instance
    """
    # Set a value with TTL
    redis_cache.set("test_key", "test_value", ttl=10)
    assert redis_cache.ttl("test_key") == 10
    
    # Update TTL
    assert redis_cache.touch("test_key", 20)
    assert redis_cache.ttl("test_key") == 20
    
    # Test touching non-existent key
    assert not redis_cache.touch("non_existent")


def test_context_manager(redis_cache: RedisCache) -> None:
    """Test context manager interface.
    
    Args:
        redis_cache: Redis cache instance
    """
    with redis_cache as cache:
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
    
    # Verify connection is closed
    with pytest.raises(Exception):
        redis_cache.client.ping() 