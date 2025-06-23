"""Redis cache actor for high-performance caching with graceful degradation."""

import asyncio
import json
import pickle
import time
from typing import Any, Dict, Optional, Union
from enum import Enum
import logging

import aioredis
from aioredis.exceptions import ConnectionError, TimeoutError, RedisError

from .core import Actor, Message, Request, Response
from .messages import CacheGetRequest, CacheGetResponse, CacheSetRequest, CacheSetResponse
from ..config.config import settings
from ..observability.telemetry import trace_method, metrics_collector, SpanAttributes


class SerializationType(Enum):
    """Supported serialization types."""
    JSON = "json"
    PICKLE = "pickle"


class CacheActor(Actor):
    """
    Redis cache actor with production-ready features:
    - Async Redis operations using aioredis
    - Connection pooling with automatic reconnection
    - Automatic serialization/deserialization (JSON/Pickle)
    - TTL management with configurable defaults
    - Cache-aside pattern implementation
    - Graceful degradation if Redis is unavailable
    - Comprehensive metrics integration
    - Circuit breaker for fault tolerance
    """
    
    def __init__(
        self,
        name: str = "cache_actor",
        redis_url: Optional[str] = None,
        pool_size: int = 10,
        max_connections: int = 50,
        default_ttl: int = 3600,  # 1 hour
        serialization: SerializationType = SerializationType.JSON,
        enable_fallback: bool = True,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 30
    ):
        super().__init__(name)
        
        # Configuration
        self.redis_url = redis_url or settings.redis_url or "redis://localhost:6379"
        self.pool_size = pool_size
        self.max_connections = max_connections
        self.default_ttl = default_ttl
        self.serialization = serialization
        self.enable_fallback = enable_fallback
        
        # Connection management
        self.pool: Optional[aioredis.ConnectionPool] = None
        self.redis: Optional[aioredis.Redis] = None
        self.connected = False
        
        # Circuit breaker for fault tolerance
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.failure_count = 0
        self.circuit_open_until: Optional[float] = None
        
        # Fallback in-memory cache (with size limit)
        self.fallback_cache: Dict[str, tuple[Any, Optional[float]]] = {}
        self.max_fallback_size = 1000
        
        # Metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_errors = 0
        self.total_requests = 0
        
    async def on_start(self) -> None:
        """Initialize Redis connection pool on actor start."""
        await self._connect()
        
    async def on_stop(self) -> None:
        """Clean up Redis connections on actor stop."""
        await self._disconnect()
        
    async def on_error(self, error: Exception) -> None:
        """Handle errors with circuit breaker logic."""
        self.logger.error(f"Cache actor error: {error}")
        self.cache_errors += 1
        
        if isinstance(error, (ConnectionError, TimeoutError, RedisError)):
            self.failure_count += 1
            if self.failure_count >= self.circuit_breaker_threshold:
                self._open_circuit_breaker()
    
    async def _connect(self) -> None:
        """Establish Redis connection with pool."""
        try:
            self.pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                decode_responses=False,  # We handle encoding/decoding ourselves
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={
                    1: 1,  # TCP_KEEPIDLE
                    2: 1,  # TCP_KEEPINTVL
                    3: 5,  # TCP_KEEPCNT
                }
            )
            
            self.redis = aioredis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.redis.ping()
            self.connected = True
            self.failure_count = 0
            self.logger.info(f"Redis connection established: {self.redis_url}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self.connected = False
            if not self.enable_fallback:
                raise
                
    async def _disconnect(self) -> None:
        """Close Redis connections."""
        if self.pool:
            await self.pool.disconnect()
            self.pool = None
            self.redis = None
            self.connected = False
            
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.circuit_open_until is None:
            return False
        
        if time.time() >= self.circuit_open_until:
            # Try to close circuit
            self.circuit_open_until = None
            self.failure_count = 0
            self.logger.info("Circuit breaker closed, retrying Redis connection")
            asyncio.create_task(self._connect())
            return False
            
        return True
        
    def _open_circuit_breaker(self) -> None:
        """Open circuit breaker after threshold failures."""
        self.circuit_open_until = time.time() + self.circuit_breaker_timeout
        self.connected = False
        self.logger.warning(
            f"Circuit breaker opened for {self.circuit_breaker_timeout}s "
            f"after {self.failure_count} failures"
        )
        
    def _serialize(self, value: Any) -> bytes:
        """Serialize value based on configured serialization type."""
        if self.serialization == SerializationType.JSON:
            return json.dumps(value).encode('utf-8')
        else:  # PICKLE
            return pickle.dumps(value)
            
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data based on configured serialization type."""
        if self.serialization == SerializationType.JSON:
            return json.loads(data.decode('utf-8'))
        else:  # PICKLE
            return pickle.loads(data)
    
    @trace_method()
    async def handle_cachegetrequest(self, message: CacheGetRequest) -> CacheGetResponse:
        """Handle cache get request with fallback logic."""
        self.total_requests += 1
        start_time = time.time()
        
        try:
            # Check circuit breaker
            if self._is_circuit_open():
                raise ConnectionError("Circuit breaker is open")
                
            # Check Redis availability
            if not self.connected or not self.redis:
                raise ConnectionError("Redis not connected")
                
            # Get from Redis
            data = await self.redis.get(message.key)
            
            if data is not None:
                value = self._deserialize(data)
                self.cache_hits += 1
                
                # Record metrics
                metrics_collector.record_cache_access(hit=True, cache_type="redis")
                
                return CacheGetResponse(
                    request_id=message.id,
                    status='success',
                    found=True,
                    value=value
                )
            else:
                self.cache_misses += 1
                
                # Record metrics
                metrics_collector.record_cache_access(hit=False, cache_type="redis")
                
                return CacheGetResponse(
                    request_id=message.id,
                    status='success',
                    found=False,
                    value=None
                )
                
        except Exception as e:
            self.logger.error(f"Redis get error for key {message.key}: {e}")
            await self.on_error(e)
            
            # Try fallback cache if enabled
            if self.enable_fallback:
                if message.key in self.fallback_cache:
                    value, expiry = self.fallback_cache[message.key]
                    
                    # Check expiry
                    if expiry is None or time.time() < expiry:
                        self.cache_hits += 1
                        metrics_collector.record_cache_access(hit=True, cache_type="fallback")
                        
                        return CacheGetResponse(
                            request_id=message.id,
                            status='success',
                            found=True,
                            value=value
                        )
                    else:
                        # Expired, remove from fallback
                        del self.fallback_cache[message.key]
                        
                self.cache_misses += 1
                metrics_collector.record_cache_access(hit=False, cache_type="fallback")
                
                return CacheGetResponse(
                    request_id=message.id,
                    status='success',
                    found=False,
                    value=None
                )
            else:
                # No fallback, return error
                return CacheGetResponse(
                    request_id=message.id,
                    status='error',
                    error=str(e),
                    found=False,
                    value=None
                )
        finally:
            # Record request duration
            duration_ms = (time.time() - start_time) * 1000
            metrics_collector.record_agent_execution(
                agent_name="cache_actor",
                duration_ms=duration_ms,
                success=True
            )
    
    @trace_method()
    async def handle_cachesetrequest(self, message: CacheSetRequest) -> CacheSetResponse:
        """Handle cache set request with fallback logic."""
        self.total_requests += 1
        start_time = time.time()
        ttl = message.ttl or self.default_ttl
        
        try:
            # Serialize value
            data = self._serialize(message.value)
            
            # Check circuit breaker
            if self._is_circuit_open():
                raise ConnectionError("Circuit breaker is open")
                
            # Check Redis availability
            if not self.connected or not self.redis:
                raise ConnectionError("Redis not connected")
                
            # Set in Redis with TTL
            if ttl > 0:
                await self.redis.setex(message.key, ttl, data)
            else:
                await self.redis.set(message.key, data)
                
            # Also update fallback cache
            if self.enable_fallback:
                expiry = time.time() + ttl if ttl > 0 else None
                self._update_fallback_cache(message.key, message.value, expiry)
                
            return CacheSetResponse(
                request_id=message.id,
                status='success',
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Redis set error for key {message.key}: {e}")
            await self.on_error(e)
            
            # Try to at least save to fallback cache
            if self.enable_fallback:
                try:
                    expiry = time.time() + ttl if ttl > 0 else None
                    self._update_fallback_cache(message.key, message.value, expiry)
                    
                    return CacheSetResponse(
                        request_id=message.id,
                        status='success',
                        success=True
                    )
                except Exception as fallback_error:
                    self.logger.error(f"Fallback cache error: {fallback_error}")
                    
            return CacheSetResponse(
                request_id=message.id,
                status='error',
                error=str(e),
                success=False
            )
        finally:
            # Record request duration
            duration_ms = (time.time() - start_time) * 1000
            metrics_collector.record_agent_execution(
                agent_name="cache_actor",
                duration_ms=duration_ms,
                success=True
            )
    
    def _update_fallback_cache(self, key: str, value: Any, expiry: Optional[float]) -> None:
        """Update fallback cache with size limit enforcement."""
        # Add to cache
        self.fallback_cache[key] = (value, expiry)
        
        # Enforce size limit with LRU eviction
        if len(self.fallback_cache) > self.max_fallback_size:
            # Remove oldest entries (simple FIFO for now)
            # In production, consider using OrderedDict or lru_cache
            oldest_keys = list(self.fallback_cache.keys())[:len(self.fallback_cache) - self.max_fallback_size]
            for old_key in oldest_keys:
                del self.fallback_cache[old_key]
                # Note: Add cache eviction metric if needed
                self.logger.debug(f"Evicted cache key: {old_key}")
    
    async def handle_ping(self, message: Message) -> None:
        """Health check handler."""
        # Test Redis connection
        healthy = False
        if self.connected and self.redis and not self._is_circuit_open():
            try:
                await self.redis.ping()
                healthy = True
            except Exception:
                pass
                
        return {
            "healthy": healthy,
            "connected": self.connected,
            "circuit_breaker_open": self._is_circuit_open(),
            "stats": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "errors": self.cache_errors,
                "total_requests": self.total_requests,
                "hit_rate": self.cache_hits / max(1, self.total_requests),
                "fallback_cache_size": len(self.fallback_cache)
            }
        }
    
    async def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        if not self.connected or not self.redis:
            return 0
            
        try:
            # Find all matching keys
            keys = []
            async for key in self.redis.scan_iter(pattern):
                keys.append(key)
                
            # Delete in batches
            if keys:
                await self.redis.delete(*keys)
                
            # Also clean fallback cache
            if self.enable_fallback:
                import fnmatch
                fallback_keys = [k for k in self.fallback_cache.keys() 
                                if fnmatch.fnmatch(k, pattern)]
                for key in fallback_keys:
                    del self.fallback_cache[key]
                    
            return len(keys)
            
        except Exception as e:
            self.logger.error(f"Error invalidating cache: {e}")
            return 0
    
    async def clear_all(self) -> bool:
        """Clear all cache entries."""
        try:
            if self.connected and self.redis:
                await self.redis.flushdb()
                
            self.fallback_cache.clear()
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False