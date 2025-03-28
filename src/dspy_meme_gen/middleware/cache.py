"""Cache middleware for API responses."""
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import urlencode

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from dspy_meme_gen.cache.factory import CacheFactory, CacheType
from dspy_meme_gen.utils.config import AppConfig


class CacheMiddleware(BaseHTTPMiddleware):
    """Middleware for caching API responses."""

    def __init__(
        self,
        app: ASGIApp,
        config: AppConfig,
        namespace: str = "api",
        ttl: Optional[int] = None,
        exclude_paths: Optional[list[str]] = None,
        exclude_methods: Optional[list[str]] = None,
    ) -> None:
        """Initialize the cache middleware.
        
        Args:
            app: The ASGI application.
            config: Application configuration.
            namespace: Cache namespace.
            ttl: Cache TTL in seconds.
            exclude_paths: List of paths to exclude from caching.
            exclude_methods: List of HTTP methods to exclude from caching.
        """
        super().__init__(app)
        self.namespace = namespace
        self.ttl = ttl
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]
        self.exclude_methods = exclude_methods or ["POST", "PUT", "DELETE", "PATCH"]
        self.cache = CacheFactory.get_cache(CacheType.REDIS, namespace)

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process the request and cache the response if applicable.
        
        Args:
            request: The incoming request.
            call_next: The next request handler.
            
        Returns:
            The response, either from cache or from the handler.
        """
        # Skip caching for excluded paths and methods
        if (
            request.url.path in self.exclude_paths
            or request.method in self.exclude_methods
        ):
            return await call_next(request)

        # Generate cache key from request
        cache_key = self._make_cache_key(request)

        # Try to get response from cache
        cached_response = self.cache.get(cache_key)
        if cached_response is not None:
            return JSONResponse(
                content=cached_response["content"],
                status_code=cached_response["status_code"],
                headers=cached_response["headers"],
            )

        # Get response from handler
        response = await call_next(request)

        # Cache the response if it's successful
        if 200 <= response.status_code < 400:
            try:
                response_body = await self._get_response_body(response)
                cache_data = {
                    "content": response_body,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                }
                self.cache.set(cache_key, cache_data, ttl=self.ttl)
            except Exception:
                # Log error but don't fail the request
                pass

        return response

    def _make_cache_key(self, request: Request) -> str:
        """Generate a cache key from the request.
        
        Args:
            request: The incoming request.
            
        Returns:
            A unique cache key for the request.
        """
        # Include method and path in key
        key_parts = [request.method, request.url.path]

        # Add query parameters if present
        if request.query_params:
            key_parts.append(urlencode(sorted(request.query_params.items())))

        # Add request body for GET requests with body
        if request.method == "GET" and request.headers.get("content-length", "0") != "0":
            body = request.body()
            if body:
                key_parts.append(str(body))

        return ":".join(key_parts)

    async def _get_response_body(self, response: Response) -> Any:
        """Extract the response body.
        
        Args:
            response: The response object.
            
        Returns:
            The response body content.
        """
        if isinstance(response, JSONResponse):
            return response.body
        return await response.body()


def cache_response(
    namespace: str = "api",
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None,
    include_args: bool = True,
    include_kwargs: bool = True,
) -> Callable:
    """Decorator for caching FastAPI endpoint responses.
    
    Args:
        namespace: Cache namespace.
        ttl: Cache TTL in seconds.
        key_prefix: Optional prefix for cache keys.
        include_args: Whether to include positional args in cache key.
        include_kwargs: Whether to include keyword args in cache key.
        
    Returns:
        A decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get cache instance
            cache = CacheFactory.get_cache(CacheType.REDIS, namespace)

            # Build cache key
            key_parts = [key_prefix or func.__name__]
            if include_args and args:
                key_parts.extend(str(arg) for arg in args)
            if include_kwargs and kwargs:
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Call function and cache result
            result = await func(*args, **kwargs)
            if result is not None:
                cache.set(cache_key, result, ttl=ttl)

            return result

        return wrapper

    return decorator 