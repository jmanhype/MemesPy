"""Tests for cache middleware."""
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request, Response
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from dspy_meme_gen.cache.factory import CacheFactory, CacheType
from dspy_meme_gen.middleware.cache import CacheMiddleware, cache_response
from dspy_meme_gen.utils.config import AppConfig


@pytest.fixture
def app_config() -> AppConfig:
    """Create test app configuration."""
    return AppConfig(
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


@pytest.fixture
def test_app(app_config: AppConfig) -> FastAPI:
    """Create test FastAPI application."""
    app = FastAPI()
    CacheFactory.initialize(app_config)
    app.add_middleware(CacheMiddleware, config=app_config)
    return app


@pytest.fixture
def test_client(test_app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(test_app)


def test_cache_middleware_excluded_paths(test_app: FastAPI, test_client: TestClient) -> None:
    """Test that excluded paths bypass cache."""
    call_count = 0

    @test_app.get("/health")
    async def health() -> Dict[str, str]:
        nonlocal call_count
        call_count += 1
        return {"status": "ok"}

    # First call
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert call_count == 1

    # Second call should not use cache
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert call_count == 2


def test_cache_middleware_excluded_methods(test_app: FastAPI, test_client: TestClient) -> None:
    """Test that excluded methods bypass cache."""
    call_count = 0

    @test_app.post("/test")
    async def test_post() -> Dict[str, int]:
        nonlocal call_count
        call_count += 1
        return {"count": call_count}

    # First call
    response = test_client.post("/test")
    assert response.status_code == 200
    assert response.json() == {"count": 1}
    assert call_count == 1

    # Second call should not use cache
    response = test_client.post("/test")
    assert response.status_code == 200
    assert response.json() == {"count": 2}
    assert call_count == 2


def test_cache_middleware_successful_get(test_app: FastAPI, test_client: TestClient) -> None:
    """Test caching of successful GET requests."""
    call_count = 0

    @test_app.get("/test")
    async def test_get() -> Dict[str, int]:
        nonlocal call_count
        call_count += 1
        return {"count": call_count}

    # First call
    response = test_client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"count": 1}
    assert call_count == 1

    # Second call should use cache
    response = test_client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"count": 1}
    assert call_count == 1


def test_cache_middleware_query_params(test_app: FastAPI, test_client: TestClient) -> None:
    """Test caching with query parameters."""
    call_count = 0

    @test_app.get("/test")
    async def test_get(param: str) -> Dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {"param": param, "count": call_count}

    # First call with param=a
    response = test_client.get("/test?param=a")
    assert response.status_code == 200
    assert response.json() == {"param": "a", "count": 1}
    assert call_count == 1

    # Second call with same param should use cache
    response = test_client.get("/test?param=a")
    assert response.status_code == 200
    assert response.json() == {"param": "a", "count": 1}
    assert call_count == 1

    # Call with different param should not use cache
    response = test_client.get("/test?param=b")
    assert response.status_code == 200
    assert response.json() == {"param": "b", "count": 2}
    assert call_count == 2


def test_cache_middleware_error_response(test_app: FastAPI, test_client: TestClient) -> None:
    """Test that error responses are not cached."""
    call_count = 0

    @test_app.get("/test")
    async def test_get() -> Dict[str, int]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return JSONResponse(
                status_code=500,
                content={"error": "Server error"}
            )
        return {"count": call_count}

    # First call returns error
    response = test_client.get("/test")
    assert response.status_code == 500
    assert response.json() == {"error": "Server error"}
    assert call_count == 1

    # Second call should not use cache
    response = test_client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"count": 2}
    assert call_count == 2


def test_cache_response_decorator(test_app: FastAPI, test_client: TestClient) -> None:
    """Test the cache_response decorator."""
    call_count = 0

    @test_app.get("/test/{id}")
    @cache_response(namespace="test", ttl=60)
    async def test_get(id: int, param: Optional[str] = None) -> Dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {"id": id, "param": param, "count": call_count}

    # First call
    response = test_client.get("/test/1?param=a")
    assert response.status_code == 200
    assert response.json() == {"id": 1, "param": "a", "count": 1}
    assert call_count == 1

    # Second call with same params should use cache
    response = test_client.get("/test/1?param=a")
    assert response.status_code == 200
    assert response.json() == {"id": 1, "param": "a", "count": 1}
    assert call_count == 1

    # Call with different params should not use cache
    response = test_client.get("/test/2?param=b")
    assert response.status_code == 200
    assert response.json() == {"id": 2, "param": "b", "count": 2}
    assert call_count == 2


def test_cache_response_exclude_args(test_app: FastAPI, test_client: TestClient) -> None:
    """Test cache_response with excluded arguments."""
    call_count = 0

    @test_app.get("/test/{id}")
    @cache_response(namespace="test", include_args=False)
    async def test_get(id: int) -> Dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {"id": id, "count": call_count}

    # First call
    response = test_client.get("/test/1")
    assert response.status_code == 200
    assert response.json() == {"id": 1, "count": 1}
    assert call_count == 1

    # Call with different id should use same cache
    response = test_client.get("/test/2")
    assert response.status_code == 200
    assert response.json() == {"id": 1, "count": 1}  # Should return cached result
    assert call_count == 1


def test_cache_response_exclude_kwargs(test_app: FastAPI, test_client: TestClient) -> None:
    """Test cache_response with excluded keyword arguments."""
    call_count = 0

    @test_app.get("/test")
    @cache_response(namespace="test", include_kwargs=False)
    async def test_get(param: Optional[str] = None) -> Dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {"param": param, "count": call_count}

    # First call
    response = test_client.get("/test?param=a")
    assert response.status_code == 200
    assert response.json() == {"param": "a", "count": 1}
    assert call_count == 1

    # Call with different param should use same cache
    response = test_client.get("/test?param=b")
    assert response.status_code == 200
    assert response.json() == {"param": "a", "count": 1}  # Should return cached result
    assert call_count == 1 