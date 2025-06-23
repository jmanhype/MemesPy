# OpenAI Connection Pool Actor

A production-ready connection pool implementation for OpenAI API calls with comprehensive resilience patterns.

## Features

### Core Resilience Patterns
- **Rate Limiting**: Token bucket algorithm with configurable burst capacity (default: 10 req/sec)
- **Circuit Breaker**: Automatic failure detection and recovery (default: 5 failures = 30s break)
- **Exponential Backoff**: Retry with jitter to prevent thundering herd (max 3 attempts)
- **Connection Pooling**: Efficient HTTP connection reuse (default: 100 max connections)

### Observability & Monitoring
- **OpenTelemetry Integration**: Full tracing and metrics collection
- **Comprehensive Metrics**: Request duration, error rates, connection pool stats
- **Structured Logging**: Contextual logging with correlation IDs
- **Performance Profiling**: CPU, memory, and resource usage tracking

### Production Features
- **Async/Await**: Full async support throughout
- **Context Manager**: Easy resource management
- **Graceful Shutdown**: Proper cleanup of connections and resources
- **Error Handling**: Comprehensive error categorization and recovery

## Usage

### Basic Usage
```python
from dspy_meme_gen.actors import create_openai_pool

# Create a pool with default settings
async with create_openai_pool(api_key="your-key") as pool:
    response = await pool.create_chat_completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
```

### Advanced Configuration
```python
from dspy_meme_gen.actors import (
    OpenAIConnectionPool,
    RateLimitConfig,
    CircuitBreakerConfig,
    RetryConfig,
    ConnectionPoolConfig
)

pool = OpenAIConnectionPool(
    api_key="your-key",
    rate_limit_config=RateLimitConfig(
        requests_per_second=20,
        burst_size=50
    ),
    circuit_breaker_config=CircuitBreakerConfig(
        failure_threshold=10,
        recovery_timeout=60
    ),
    retry_config=RetryConfig(
        max_attempts=5,
        base_delay=2.0,
        max_delay=120.0
    ),
    pool_config=ConnectionPoolConfig(
        max_connections=200,
        max_connections_per_host=50
    )
)
```

### Available Methods

#### Chat Completions
```python
response = await pool.create_chat_completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=100
)
```

#### Image Generation  
```python
response = await pool.create_image(
    prompt="A beautiful sunset",
    model="dall-e-3",
    size="1024x1024",
    quality="hd"
)
```

#### Embeddings
```python
response = await pool.create_embedding(
    model="text-embedding-ada-002",
    input="Text to embed"
)
```

#### Generic API Calls
```python
response = await pool.call_api(
    endpoint="/chat/completions",
    method="POST",
    data={"model": "gpt-3.5-turbo", "messages": [...]}
)
```

## Architecture

### Circuit Breaker States
- **CLOSED**: Normal operation, all requests allowed
- **OPEN**: Failing state, requests immediately rejected
- **HALF_OPEN**: Testing recovery, limited requests allowed

### Rate Limiting Algorithm
Uses token bucket algorithm:
- Tokens refilled at configured rate
- Burst capacity allows temporary spikes
- Graceful waiting when tokens exhausted

### Retry Strategy
Exponential backoff with jitter:
- Base delay increases exponentially per attempt
- Random jitter prevents thundering herd
- Maximum delay cap prevents excessive waits

## Metrics

### Request Metrics
- `openai.request.duration` - Request latency histogram
- `openai.request.count` - Total request counter
- `openai.request.errors` - Error counter by type

### Resilience Metrics
- `openai.circuit_breaker.state_changes` - State transition counter
- `openai.rate_limiter.throttled` - Throttled request counter
- `openai.request.retries` - Retry attempt counter

### Resource Metrics
- `openai.connections.active` - Active connection gauge
- `openai.rate_limiter.tokens` - Available token gauge

## Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your-api-key
```

### Default Settings
- Rate limit: 10 requests/second
- Circuit breaker: 5 failures trigger 30s break
- Max retries: 3 attempts with exponential backoff
- Connection pool: 100 max connections, 30 per host
- Timeout: 60s total, 10s connect, 30s read

## Error Handling

### Exception Types
- `ExternalServiceError`: API failures after retries
- `CircuitBreakerOpen`: Circuit breaker rejecting requests
- `aiohttp.ClientError`: HTTP-level errors

### Automatic Recovery
- Circuit breaker automatically attempts recovery
- Rate limiter gracefully handles bursts
- Retry logic handles transient failures
- Connection pool manages connection lifecycle

## Integration

### With DSPy Agents
The pool can be integrated with existing DSPy agents:

```python
class MyAgent(dspy.Module):
    def __init__(self, pool):
        self.pool = pool
        
    async def forward(self, query):
        response = await self.pool.create_chat_completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}]
        )
        return response
```

### With FastAPI
```python
from fastapi import FastAPI, Depends

app = FastAPI()
pool = create_openai_pool()

@app.on_event("startup")
async def startup():
    await pool.start()

@app.on_event("shutdown") 
async def shutdown():
    await pool.stop()

@app.get("/chat")
async def chat(query: str):
    return await pool.create_chat_completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query}]
    )
```

## Files Created

- `src/dspy_meme_gen/actors/openai_pool.py` - Main implementation (643 lines)
- `src/dspy_meme_gen/actors/__init__.py` - Module exports

## Implementation Details

### Classes
- `OpenAIConnectionPool` - Main pool class
- `CircuitBreaker` - Circuit breaker implementation  
- `RateLimiter` - Token bucket rate limiter
- Configuration classes for all components

### Key Features
- 44 methods and classes total
- Full async/await support
- Comprehensive error handling
- Production-ready logging and metrics
- Type hints throughout
- Extensive documentation

This implementation provides a solid foundation for the actor-based architecture as specified in the incremental plan, delivering immediate value through improved resilience and observability.