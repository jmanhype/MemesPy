# Incremental Actor-Based Architecture Implementation Plan

## Executive Summary

This plan outlines pragmatic, incremental improvements to add actor-based patterns to MemesPy without breaking existing functionality. Focus is on quick wins that provide immediate value.

## Current State Analysis

### Strengths
- **Privacy System**: Comprehensive GDPR-compliant implementation
- **Observability**: Full OpenTelemetry integration
- **DSPy Integration**: Clean agent abstractions
- **CQRS Foundation**: Event sourcing structure ready

### Issues
- **Sequential Processing**: No parallelization in meme pipeline
- **No Resilience**: Missing circuit breakers, retries, bulkheading
- **Resource Inefficiency**: Single instances, no pooling
- **In-Memory State**: Data lost on restart

## Quick Wins (1-2 Days Each)

### 1. Async Connection Pool for OpenAI (Day 1)
```python
# src/dspy_meme_gen/actors/openai_pool.py
class OpenAIConnectionPool:
    - Rate limiting (10 req/sec default)
    - Circuit breaker (5 failures = 30s break)
    - Exponential backoff retry
    - Connection reuse
    - Metrics integration
```

### 2. Redis Cache Actor (Day 1)
```python
# src/dspy_meme_gen/actors/cache_actor.py
class CacheActor:
    - Async Redis operations
    - Connection pooling
    - Automatic serialization
    - TTL management
    - Cache-aside pattern
```

### 3. Background Task Worker (Day 2)
```python
# src/dspy_meme_gen/actors/task_worker.py
class TaskWorker:
    - Async task processing
    - Progress tracking
    - Result caching
    - Error handling
    - Graceful shutdown
```

### 4. Parallel Verification Pipeline (Day 2)
```python
# src/dspy_meme_gen/services/parallel_meme_service.py
async def create_meme_parallel():
    - Concurrent verification stages
    - Parallel API calls
    - Result aggregation
    - Partial failure handling
```

## Phase 1: Basic Actor Infrastructure (Week 1)

### Base Actor Implementation
```python
# src/dspy_meme_gen/actors/base.py
class Actor:
    - Message inbox (asyncio.Queue)
    - State management
    - Error boundaries
    - Lifecycle hooks
    - Telemetry integration
```

### Simple Supervisor
```python
# src/dspy_meme_gen/actors/supervisor.py
class SimpleSupervisor:
    - One-for-one restart strategy
    - Health checks
    - Restart limits
    - Error escalation
```

### Message Types
```python
# src/dspy_meme_gen/actors/messages.py
- Request/Response patterns
- Events
- System messages (Ping, Restart, Shutdown)
```

## Phase 2: Service Migration (Week 2)

### MemeGeneratorActor
- Port existing meme_generator
- Add state management
- Implement message handlers
- Add supervision

### VerificationActor Pool
- Create worker pool
- Load balancing
- Parallel processing
- Result aggregation

### CacheManagerActor
- Centralized cache management
- Invalidation strategies
- Metrics collection

## Phase 3: Advanced Features (Week 3)

### Event Sourcing Integration
- Connect to existing CQRS
- Event persistence
- State replay
- Snapshots

### Distributed State
- Redis-backed state store
- State replication
- Consistency guarantees

### Advanced Supervision
- One-for-all strategy
- Rest-for-one strategy
- Dynamic child management

## Implementation Priority

### Immediate (This Week)
1. **OpenAI Connection Pool** - Biggest bottleneck
2. **Redis Cache Actor** - Quick performance win
3. **Parallel Verification** - 3x speedup
4. **Background Workers** - Better UX

### Next Week
1. Base Actor framework
2. Simple supervision
3. Message bus
4. Service migration

### Future
1. Full supervision trees
2. Distributed state
3. Advanced patterns

## Success Metrics

### Performance
- Meme generation: 5s → 2s (60% reduction)
- Concurrent requests: 10 → 100 (10x increase)
- Cache hit rate: 0% → 70%

### Reliability
- Error recovery: Manual → Automatic
- Uptime: 99% → 99.9%
- Partial failures: Total failure → Graceful degradation

### Resource Usage
- CPU: Better utilization via concurrency
- Memory: Pooling reduces overhead
- Network: Connection reuse

## Migration Strategy

### 1. Parallel Implementation
- Keep existing code working
- Add new actor-based endpoints
- A/B test performance
- Gradual migration

### 2. Feature Flags
```python
if settings.use_actor_pool:
    return await actor_pool.generate_meme(topic)
else:
    return await meme_generator.generate_meme(topic)
```

### 3. Monitoring
- Track both implementations
- Compare metrics
- Identify issues early
- Rollback capability

## Code Structure

```
src/dspy_meme_gen/
├── actors/
│   ├── __init__.py
│   ├── base.py           # Base actor class
│   ├── supervisor.py     # Supervision strategies
│   ├── messages.py       # Message types
│   ├── openai_pool.py   # OpenAI connection pool
│   ├── cache_actor.py   # Redis cache actor
│   └── task_worker.py   # Background task worker
├── services/
│   ├── parallel_meme_service.py  # Parallel implementation
│   └── actor_meme_service.py     # Actor-based service
```

## Example: OpenAI Pool Implementation

```python
import asyncio
from typing import Optional, Dict, Any
import time
import aiohttp
from ..observability.telemetry import trace_method, metrics_collector

class OpenAIConnectionPool:
    def __init__(self, 
                 max_connections: int = 10,
                 rate_limit: int = 10,  # requests per second
                 circuit_breaker_threshold: int = 5):
        self.semaphore = asyncio.Semaphore(max_connections)
        self.rate_limiter = RateLimiter(rate_limit)
        self.circuit_breaker = CircuitBreaker(circuit_breaker_threshold)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def start(self):
        self.session = aiohttp.ClientSession()
        
    async def stop(self):
        if self.session:
            await self.session.close()
    
    @trace_method()
    async def call_api(self, 
                      endpoint: str, 
                      data: Dict[str, Any],
                      retry_count: int = 3) -> Dict[str, Any]:
        async with self.semaphore:
            await self.rate_limiter.acquire()
            
            for attempt in range(retry_count):
                try:
                    if not self.circuit_breaker.is_open():
                        response = await self._make_request(endpoint, data)
                        self.circuit_breaker.record_success()
                        return response
                    else:
                        raise CircuitBreakerOpen()
                        
                except Exception as e:
                    self.circuit_breaker.record_failure()
                    if attempt == retry_count - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

## Conclusion

This incremental approach allows us to:
1. Start seeing benefits immediately
2. Learn and adjust as we go
3. Maintain system stability
4. Build team expertise gradually

The focus on quick wins ensures we deliver value while building toward the complete actor-based architecture described in the design document.