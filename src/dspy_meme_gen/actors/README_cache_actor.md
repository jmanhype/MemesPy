# Redis Cache Actor

The Redis Cache Actor provides high-performance caching with graceful degradation and production-ready features.

## Features

- **Async Redis Operations**: Non-blocking Redis operations using aioredis
- **Connection Pooling**: Automatic connection pooling with reconnection
- **Serialization Support**: JSON and Pickle serialization options
- **TTL Management**: Configurable time-to-live for cache entries
- **Cache-Aside Pattern**: Implements the cache-aside pattern for easy integration
- **Graceful Degradation**: Falls back to in-memory cache when Redis is unavailable
- **Circuit Breaker**: Fault tolerance with automatic recovery
- **Metrics Integration**: Comprehensive metrics collection
- **Health Checks**: Built-in health check endpoints

## Usage

### Basic Setup

```python
from src.dspy_meme_gen.actors import CacheActor, SerializationType
from src.dspy_meme_gen.actors.core import ActorSystem
from src.dspy_meme_gen.actors.messages import CacheGetRequest, CacheSetRequest

# Create actor system
system = ActorSystem("my_system")
await system.start()

# Create cache actor
cache_actor = CacheActor(
    name="my_cache",
    redis_url="redis://localhost:6379",
    serialization=SerializationType.JSON,
    enable_fallback=True,
    default_ttl=3600  # 1 hour
)

# Register with system
cache_ref = await system.register_actor(cache_actor)
```

### Setting Values

```python
# Set a value with default TTL
set_request = CacheSetRequest(
    key="user:123:profile",
    value={"name": "John", "email": "john@example.com"},
    ttl=1800  # 30 minutes
)

response = await cache_ref.ask(set_request)
print(f"Success: {response.success}")
```

### Getting Values

```python
# Get a value
get_request = CacheGetRequest(key="user:123:profile")
response = await cache_ref.ask(get_request)

if response.found:
    print(f"Value: {response.value}")
else:
    print("Key not found")
```

### Health Checks

```python
# Check cache health
ping_response = await cache_ref.ask({"type": "ping"})
print(f"Healthy: {ping_response['healthy']}")
print(f"Stats: {ping_response['stats']}")
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "cache_actor" | Actor name |
| `redis_url` | str | from settings | Redis connection URL |
| `pool_size` | int | 10 | Connection pool size |
| `max_connections` | int | 50 | Maximum connections |
| `default_ttl` | int | 3600 | Default TTL in seconds |
| `serialization` | SerializationType | JSON | Serialization method |
| `enable_fallback` | bool | True | Enable in-memory fallback |
| `circuit_breaker_threshold` | int | 5 | Failures before circuit opens |
| `circuit_breaker_timeout` | int | 30 | Circuit breaker timeout (seconds) |

## Serialization Types

- **JSON**: Fast, human-readable, supports basic Python types
- **Pickle**: Supports all Python objects but less secure

```python
# JSON serialization (recommended)
cache_actor = CacheActor(serialization=SerializationType.JSON)

# Pickle serialization (for complex objects)
cache_actor = CacheActor(serialization=SerializationType.PICKLE)
```

## Error Handling

The cache actor implements graceful error handling:

1. **Circuit Breaker**: Opens after repeated failures, prevents cascade failures
2. **Fallback Cache**: In-memory cache when Redis is unavailable
3. **Automatic Reconnection**: Attempts to reconnect when circuit breaker resets
4. **Metrics**: Tracks errors, hits, misses, and performance

## Integration with Existing Services

### Cache-Aside Pattern

```python
class MemeService:
    def __init__(self, cache_ref):
        self.cache_ref = cache_ref
    
    async def get_meme(self, meme_id: str):
        # 1. Check cache first
        cache_key = f"meme:{meme_id}"
        get_request = CacheGetRequest(key=cache_key)
        
        try:
            response = await self.cache_ref.ask(get_request)
            if response.found:
                return response.value  # Cache hit
        except Exception:
            pass  # Cache error, proceed to database
        
        # 2. Cache miss, get from database
        meme = await self.database.get_meme(meme_id)
        
        # 3. Cache the result
        if meme:
            set_request = CacheSetRequest(
                key=cache_key,
                value=meme,
                ttl=3600
            )
            try:
                await self.cache_ref.ask(set_request)
            except Exception:
                pass  # Cache error is not critical
        
        return meme
```

## Monitoring and Metrics

The cache actor provides comprehensive metrics:

- **Hit Rate**: Percentage of cache hits
- **Miss Count**: Number of cache misses
- **Error Count**: Number of cache errors
- **Request Duration**: Time taken for cache operations
- **Circuit Breaker Status**: Whether circuit breaker is open/closed
- **Fallback Cache Size**: Number of items in fallback cache

Access metrics via health check:

```python
ping_response = await cache_ref.ask({"type": "ping"})
stats = ping_response.get("stats", {})

print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Total requests: {stats['total_requests']}")
print(f"Errors: {stats['errors']}")
```

## Performance Considerations

1. **Connection Pooling**: Reuses connections for better performance
2. **Serialization**: JSON is faster than Pickle for simple data
3. **TTL Management**: Set appropriate TTL to balance freshness and performance
4. **Batch Operations**: Consider batching for multiple operations
5. **Key Design**: Use consistent, hierarchical key patterns

## Best Practices

1. **Key Naming**: Use consistent, hierarchical patterns like `service:type:id`
2. **TTL Strategy**: Set TTL based on data freshness requirements
3. **Error Handling**: Always handle cache errors gracefully
4. **Monitoring**: Monitor cache hit rates and error rates
5. **Security**: Use JSON serialization for untrusted data
6. **Capacity Planning**: Monitor Redis memory usage

## Troubleshooting

### Common Issues

1. **Connection Errors**: Check Redis URL and network connectivity
2. **Serialization Errors**: Ensure data is serializable with chosen method
3. **Memory Issues**: Monitor Redis memory usage and set appropriate TTL
4. **Circuit Breaker**: Check error logs when circuit breaker opens

### Debugging

Enable debug logging to see cache operations:

```python
import logging
logging.getLogger("actor.cache_actor").setLevel(logging.DEBUG)
```

### Health Checks

Regular health checks help identify issues:

```python
async def check_cache_health():
    try:
        response = await cache_ref.ask({"type": "ping"}, timeout=1000)
        return response.get("healthy", False)
    except Exception:
        return False
```