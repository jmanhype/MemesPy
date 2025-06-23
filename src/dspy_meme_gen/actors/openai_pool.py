"""
OpenAI Connection Pool Actor

Production-ready connection pool for OpenAI API calls with:
- Rate limiting (configurable, default 10 req/sec)
- Circuit breaker pattern (5 failures = 30s break)
- Exponential backoff retry with jitter
- Connection reuse via session pooling
- Comprehensive metrics and observability
- Async/await patterns throughout
- Proper error handling and recovery
"""

import asyncio
import time
import random
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from contextlib import asynccontextmanager

import aiohttp
from structlog import get_logger

from ..observability.telemetry import (
    trace_method,
    trace_span,
    get_meter,
    SpanAttributes,
    MetricNames,
    metrics_collector,
    SpanKind
)
from ..exceptions import ExternalServiceError
from ..config.config import settings


logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 30  # seconds
    half_open_requests: int = 3


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: int = 10
    burst_size: int = 20


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1


@dataclass
class ConnectionPoolConfig:
    """Configuration for the connection pool."""
    max_connections: int = 100
    max_connections_per_host: int = 30
    keepalive_timeout: int = 30
    timeout: aiohttp.ClientTimeout = field(
        default_factory=lambda: aiohttp.ClientTimeout(
            total=60,
            connect=10,
            sock_read=30
        )
    )


class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_requests = 0
        self._lock = asyncio.Lock()
        
        # Metrics
        meter = get_meter()
        self.state_changes = meter.create_counter(
            "openai.circuit_breaker.state_changes",
            unit="changes",
            description="Circuit breaker state changes"
        )
        self.rejections = meter.create_counter(
            "openai.circuit_breaker.rejections",
            unit="requests",
            description="Requests rejected by circuit breaker"
        )
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.rejections.add(1, {"reason": "circuit_open"})
                    raise ExternalServiceError("Circuit breaker is OPEN")
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_requests >= self.config.half_open_requests:
                    self.rejections.add(1, {"reason": "half_open_limit"})
                    raise ExternalServiceError("Circuit breaker is testing recovery")
                self.half_open_requests += 1
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_requests >= self.config.half_open_requests:
                    self._transition_to_closed()
            self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            elif self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return False
        
        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure > timedelta(seconds=self.config.recovery_timeout)
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        logger.info("Circuit breaker transitioning to CLOSED")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_requests = 0
        self.state_changes.add(1, {"from": "half_open", "to": "closed"})
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        logger.warning("Circuit breaker transitioning to OPEN")
        self.state = CircuitState.OPEN
        self.half_open_requests = 0
        self.state_changes.add(1, {"from": self.state.value, "to": "open"})
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        logger.info("Circuit breaker transitioning to HALF_OPEN")
        self.state = CircuitState.HALF_OPEN
        self.half_open_requests = 0
        self.failure_count = 0
        self.state_changes.add(1, {"from": "open", "to": "half_open"})


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.burst_size
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        
        # Metrics
        meter = get_meter()
        self.throttled_requests = meter.create_counter(
            "openai.rate_limiter.throttled",
            unit="requests",
            description="Requests throttled by rate limiter"
        )
        self.tokens_available = meter.create_observable_gauge(
            "openai.rate_limiter.tokens",
            callbacks=[self._observe_tokens],
            unit="tokens",
            description="Available rate limit tokens"
        )
    
    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, waiting if necessary. Returns wait time."""
        start_time = time.monotonic()
        
        async with self._lock:
            await self._refill()
            
            while self.tokens < tokens:
                wait_time = tokens / self.config.requests_per_second
                await asyncio.sleep(wait_time)
                await self._refill()
                self.throttled_requests.add(1)
            
            self.tokens -= tokens
            
        return time.monotonic() - start_time
    
    async def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        
        new_tokens = elapsed * self.config.requests_per_second
        self.tokens = min(self.tokens + new_tokens, self.config.burst_size)
        self.last_refill = now
    
    def _observe_tokens(self, options):
        """Callback for token gauge metric."""
        from opentelemetry.metrics import Observation
        yield Observation(self.tokens, {})


class OpenAIConnectionPool:
    """
    Production-ready connection pool for OpenAI API calls.
    
    Features:
    - Connection pooling with session reuse
    - Rate limiting with token bucket algorithm
    - Circuit breaker for fault tolerance
    - Exponential backoff retry with jitter
    - Comprehensive metrics and tracing
    - Async/await patterns throughout
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        rate_limit_config: Optional[RateLimitConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        pool_config: Optional[ConnectionPoolConfig] = None
    ):
        """Initialize the connection pool."""
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.base_url = base_url
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config or CircuitBreakerConfig())
        self.retry_config = retry_config or RetryConfig()
        self.pool_config = pool_config or ConnectionPoolConfig()
        
        # Connection pool
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        
        # Metrics
        self._init_metrics()
        
        # State
        self._started = False
        self._lock = asyncio.Lock()
        
        logger.info(
            "OpenAI connection pool initialized",
            rate_limit=self.rate_limiter.config.requests_per_second,
            circuit_breaker_threshold=self.circuit_breaker.config.failure_threshold
        )
    
    def _init_metrics(self):
        """Initialize metrics collectors."""
        meter = get_meter()
        
        self.request_duration = meter.create_histogram(
            "openai.request.duration",
            unit="ms",
            description="OpenAI API request duration"
        )
        
        self.request_count = meter.create_counter(
            "openai.request.count",
            unit="requests",
            description="Total OpenAI API requests"
        )
        
        self.error_count = meter.create_counter(
            "openai.request.errors",
            unit="errors",
            description="OpenAI API request errors"
        )
        
        self.retry_count = meter.create_counter(
            "openai.request.retries",
            unit="retries",
            description="OpenAI API request retries"
        )
        
        self.active_connections = meter.create_up_down_counter(
            "openai.connections.active",
            unit="connections",
            description="Active OpenAI API connections"
        )
    
    async def start(self):
        """Start the connection pool."""
        async with self._lock:
            if self._started:
                return
            
            self._connector = aiohttp.TCPConnector(
                limit=self.pool_config.max_connections,
                limit_per_host=self.pool_config.max_connections_per_host,
                keepalive_timeout=self.pool_config.keepalive_timeout,
                force_close=False
            )
            
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=self.pool_config.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": f"MemesPy/{settings.app_version}"
                }
            )
            
            self._started = True
            logger.info("OpenAI connection pool started")
    
    async def stop(self):
        """Stop the connection pool and clean up resources."""
        async with self._lock:
            if not self._started:
                return
            
            if self._session:
                await self._session.close()
                self._session = None
            
            if self._connector:
                await self._connector.close()
                self._connector = None
            
            self._started = False
            logger.info("OpenAI connection pool stopped")
    
    @asynccontextmanager
    async def _ensure_started(self):
        """Ensure the pool is started."""
        if not self._started:
            await self.start()
        yield
    
    @trace_method(kind=SpanKind.CLIENT)
    async def call_api(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make an API call to OpenAI with full resilience features.
        
        Args:
            endpoint: API endpoint (e.g., "/chat/completions")
            method: HTTP method
            data: Request body
            params: Query parameters
            headers: Additional headers
            **kwargs: Additional arguments passed to the request
            
        Returns:
            Response data as dictionary
            
        Raises:
            ExternalServiceError: On API failures after retries
        """
        async with self._ensure_started():
            # Acquire rate limit token
            wait_time = await self.rate_limiter.acquire()
            if wait_time > 0:
                logger.debug(f"Rate limited, waited {wait_time:.2f}s")
            
            # Execute with circuit breaker
            return await self.circuit_breaker.call(
                self._execute_with_retry,
                endpoint, method, data, params, headers, **kwargs
            )
    
    async def _execute_with_retry(
        self,
        endpoint: str,
        method: str,
        data: Optional[Dict[str, Any]],
        params: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute request with retry logic."""
        url = f"{self.base_url}{endpoint}"
        
        last_error = None
        for attempt in range(self.retry_config.max_attempts):
            try:
                return await self._make_request(
                    url, method, data, params, headers, **kwargs
                )
            except aiohttp.ClientError as e:
                last_error = e
                if attempt < self.retry_config.max_attempts - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        f"Request failed, retrying in {delay:.2f}s",
                        attempt=attempt + 1,
                        error=str(e)
                    )
                    self.retry_count.add(1, {"endpoint": endpoint, "attempt": attempt + 1})
                    await asyncio.sleep(delay)
                else:
                    raise ExternalServiceError(f"OpenAI API request failed: {e}") from e
            except Exception as e:
                # Don't retry on non-client errors
                raise ExternalServiceError(f"Unexpected error: {e}") from e
        
        # Should never reach here
        raise ExternalServiceError(f"Max retries exceeded: {last_error}")
    
    async def _make_request(
        self,
        url: str,
        method: str,
        data: Optional[Dict[str, Any]],
        params: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Make a single HTTP request."""
        start_time = time.time()
        
        # Merge headers
        request_headers = dict(self._session.headers)
        if headers:
            request_headers.update(headers)
        
        # Track active connections
        self.active_connections.add(1)
        
        try:
            async with self._session.request(
                method,
                url,
                json=data,
                params=params,
                headers=request_headers,
                **kwargs
            ) as response:
                # Record metrics
                duration_ms = (time.time() - start_time) * 1000
                self.request_duration.record(
                    duration_ms,
                    {"endpoint": url.split("/")[-1], "status": response.status}
                )
                self.request_count.add(
                    1,
                    {"endpoint": url.split("/")[-1], "status": response.status}
                )
                
                # Check for errors
                if response.status >= 400:
                    error_data = await response.json()
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                    self.error_count.add(
                        1,
                        {"endpoint": url.split("/")[-1], "status": response.status}
                    )
                    
                    # Log rate limit headers
                    if response.status == 429:
                        logger.warning(
                            "Rate limited by OpenAI",
                            retry_after=response.headers.get("Retry-After"),
                            limit=response.headers.get("X-RateLimit-Limit"),
                            remaining=response.headers.get("X-RateLimit-Remaining"),
                            reset=response.headers.get("X-RateLimit-Reset")
                        )
                    
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=error_msg,
                        headers=response.headers
                    )
                
                # Parse response
                return await response.json()
                
        finally:
            self.active_connections.add(-1)
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        base_delay = self.retry_config.base_delay
        delay = min(
            base_delay * (self.retry_config.exponential_base ** attempt),
            self.retry_config.max_delay
        )
        
        # Add jitter
        jitter_range = delay * self.retry_config.jitter
        jitter = random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay + jitter)
    
    # Convenience methods for common OpenAI endpoints
    
    @trace_method(name="openai.chat.completions")
    async def create_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a chat completion."""
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        
        return await self.call_api("/chat/completions", data=data)
    
    @trace_method(name="openai.embeddings")
    async def create_embedding(
        self,
        model: str,
        input: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Create embeddings."""
        data = {
            "model": model,
            "input": input,
            **kwargs
        }
        
        return await self.call_api("/embeddings", data=data)
    
    @trace_method(name="openai.images.generate")
    async def create_image(
        self,
        prompt: str,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate an image."""
        data = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": n,
            **kwargs
        }
        
        return await self.call_api("/images/generations", data=data)
    
    @trace_method(name="openai.models.list")
    async def list_models(self) -> Dict[str, Any]:
        """List available models."""
        return await self.call_api("/models", method="GET")
    
    # Context manager support
    
    async def __aenter__(self):
        """Enter context manager."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        await self.stop()


# Factory function for easy creation
def create_openai_pool(
    api_key: Optional[str] = None,
    rate_limit: int = 10,
    circuit_breaker_threshold: int = 5,
    max_connections: int = 100
) -> OpenAIConnectionPool:
    """
    Factory function to create an OpenAI connection pool with common settings.
    
    Args:
        api_key: OpenAI API key (defaults to environment)
        rate_limit: Requests per second limit
        circuit_breaker_threshold: Number of failures before opening circuit
        max_connections: Maximum number of connections in pool
        
    Returns:
        Configured OpenAIConnectionPool instance
    """
    return OpenAIConnectionPool(
        api_key=api_key,
        rate_limit_config=RateLimitConfig(requests_per_second=rate_limit),
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=circuit_breaker_threshold),
        pool_config=ConnectionPoolConfig(max_connections=max_connections)
    )