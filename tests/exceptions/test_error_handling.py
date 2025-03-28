"""Tests for error handling system."""

import asyncio
from typing import Any, Dict, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

import pytest
from sqlalchemy.exc import SQLAlchemyError
from aioredis.exceptions import RedisError
from fastapi import HTTPException, FastAPI
from httpx import AsyncClient

from dspy_meme_gen.exceptions.base import DSPyMemeError, ErrorCode, MemeGenerationError
from dspy_meme_gen.exceptions.specific import (
    APIConnectionError,
    CacheConnectionError,
    DatabaseQueryError,
    ContentValidationError
)
from dspy_meme_gen.middleware.error_handler import handle_exceptions, error_to_dict


@handle_exceptions()
async def async_raise_error(error: Exception) -> None:
    """Helper function to raise an error in async context."""
    raise error


@handle_exceptions()
def sync_raise_error(error: Exception) -> None:
    """Helper function to raise an error in sync context."""
    raise error


@pytest.mark.asyncio
async def test_handle_sqlalchemy_error() -> None:
    """Test handling of SQLAlchemy errors."""
    original_error = SQLAlchemyError("Database connection failed")
    
    with pytest.raises(DatabaseQueryError) as exc_info:
        await async_raise_error(original_error)
    
    error = exc_info.value
    assert isinstance(error, DatabaseQueryError)
    assert str(error) == "Database connection failed"
    assert error.code == ErrorCode.DB_QUERY_ERROR
    assert error.original_error == original_error


@pytest.mark.asyncio
async def test_handle_redis_error() -> None:
    """Test handling of Redis errors."""
    original_error = RedisError("Cache connection failed")
    
    with pytest.raises(CacheConnectionError) as exc_info:
        await async_raise_error(original_error)
    
    error = exc_info.value
    assert isinstance(error, CacheConnectionError)
    assert str(error) == "Cache connection failed"
    assert error.code == ErrorCode.CACHE_CONNECTION_ERROR
    assert error.original_error == original_error


def test_handle_sync_error() -> None:
    """Test handling of errors in synchronous context."""
    original_error = ConnectionError("API connection failed")
    
    with pytest.raises(APIConnectionError) as exc_info:
        sync_raise_error(original_error)
    
    error = exc_info.value
    assert isinstance(error, APIConnectionError)
    assert str(error) == "API connection failed"
    assert error.code == ErrorCode.API_CONNECTION_ERROR
    assert error.original_error == original_error


@pytest.mark.asyncio
async def test_handle_custom_error_map() -> None:
    """Test handling errors with custom error mapping."""
    class CustomError(Exception):
        pass
    
    @handle_exceptions({CustomError: ContentValidationError})
    async def raise_custom_error() -> None:
        raise CustomError("Custom validation error")
    
    with pytest.raises(ContentValidationError) as exc_info:
        await raise_custom_error()
    
    error = exc_info.value
    assert isinstance(error, ContentValidationError)
    assert str(error) == "Custom validation error"
    assert error.code == ErrorCode.CONTENT_VALIDATION_ERROR


@pytest.mark.asyncio
async def test_handle_unknown_error() -> None:
    """Test handling of unknown error types."""
    class UnknownError(Exception):
        pass
    
    original_error = UnknownError("Unknown error occurred")
    
    with pytest.raises(DSPyMemeError) as exc_info:
        await async_raise_error(original_error)
    
    error = exc_info.value
    assert isinstance(error, DSPyMemeError)
    assert str(error) == "Unknown error occurred"
    assert error.code == ErrorCode.UNKNOWN_ERROR
    assert error.original_error == original_error


def test_error_to_dict_dspy_error() -> None:
    """Test converting DSPyMemeError to dictionary."""
    error = ContentValidationError(
        "Invalid content",
        details={"field": "text", "reason": "too long"}
    )
    
    error_dict = error_to_dict(error)
    
    assert error_dict["code"] == ErrorCode.CONTENT_VALIDATION_ERROR.value
    assert error_dict["message"] == "Invalid content"
    assert error_dict["type"] == "ContentValidationError"
    assert error_dict["details"] == {"field": "text", "reason": "too long"}


def test_error_to_dict_standard_error() -> None:
    """Test converting standard exception to dictionary."""
    error = ValueError("Invalid value")
    
    error_dict = error_to_dict(error)
    
    assert error_dict["code"] == ErrorCode.UNKNOWN_ERROR.value
    assert error_dict["message"] == "Invalid value"
    assert error_dict["type"] == "ValueError"
    assert "traceback" in error_dict


@pytest.mark.asyncio
async def test_error_logging(caplog: "LogCaptureFixture") -> None:
    """Test error logging."""
    with caplog.at_level("ERROR"):
        error = MemeGenerationError("Test error")
        assert "Test error" in str(error)
        assert any("Test error" in record.message for record in caplog.records)


@pytest.fixture
def app() -> FastAPI:
    """Create a test FastAPI application."""
    app = FastAPI()
    
    @app.get("/test-error")
    @handle_exceptions
    async def test_endpoint():
        raise MemeGenerationError("Test error")
    
    return app


@pytest.fixture
async def client(app: FastAPI) -> AsyncClient:
    """Create a test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


async def test_error_handling(client: AsyncClient) -> None:
    """Test error handling middleware."""
    response = await client.get("/test-error")
    assert response.status_code == 500
    assert "error" in response.json()
    assert "Test error" in response.json()["error"]

    caplog.set_level("ERROR")
    error_message = "Test error"
    assert error_message in caplog.text 