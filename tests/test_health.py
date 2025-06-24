"""Tests for health check functionality."""

from typing import Dict, Any, cast, TYPE_CHECKING
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncEngine
import aioredis
from prometheus_client import REGISTRY
from dspy_meme_gen.health.checks import HealthCheck
from fastapi import FastAPI
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_db_engine(mocker: "MockerFixture") -> AsyncEngine:
    """Create a mock database engine.

    Args:
        mocker: pytest-mock fixture

    Returns:
        Mock AsyncEngine instance
    """
    mock_engine = AsyncMock(spec=AsyncEngine)
    mock_pool = MagicMock()
    mock_pool.size.return_value = 5
    mock_pool.checkedout.return_value = 2
    mock_pool.overflow.return_value = 0
    mock_engine.pool = mock_pool
    return cast(AsyncEngine, mock_engine)


@pytest.fixture
def mock_redis_client(mocker: "MockerFixture") -> aioredis.Redis:
    """Create a mock Redis client.

    Args:
        mocker: pytest-mock fixture

    Returns:
        Mock Redis client instance
    """
    mock_redis = AsyncMock(spec=aioredis.Redis)
    mock_redis.info = AsyncMock(
        return_value={
            "redis_version": "6.2.6",
            "used_memory_human": "1.00M",
            "connected_clients": 1,
        }
    )
    mock_redis.ping = AsyncMock(return_value=True)
    return cast(aioredis.Redis, mock_redis)


@pytest.fixture
def health_check(mock_db_engine: AsyncEngine, mock_redis_client: aioredis.Redis) -> HealthCheck:
    """Create a HealthCheck instance with mock dependencies.

    Args:
        mock_db_engine: Mock database engine
        mock_redis_client: Mock Redis client

    Returns:
        HealthCheck instance
    """
    return HealthCheck(mock_db_engine, mock_redis_client)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client.

    Args:
        app: FastAPI application.

    Returns:
        TestClient: Test client.
    """
    return TestClient(app)


@pytest.mark.asyncio
async def test_check_health_all_healthy(
    health_check: HealthCheck,
    mock_db_engine: AsyncEngine,
    mock_redis_client: aioredis.Redis,
    mocker: "MockerFixture",
) -> None:
    """Test health check when all components are healthy.

    Args:
        health_check: HealthCheck instance
        mock_db_engine: Mock database engine
        mock_redis_client: Mock Redis client
        mocker: pytest-mock fixture
    """
    # Mock system checks
    mocker.patch("psutil.cpu_percent", return_value=50.0)
    mocker.patch(
        "psutil.virtual_memory",
        return_value=MagicMock(total=16000000000, available=8000000000, percent=50.0),
    )
    mocker.patch(
        "psutil.disk_usage",
        return_value=MagicMock(total=100000000000, free=50000000000, percent=50.0),
    )

    # Perform health check
    is_healthy, details = await health_check.check_health()

    assert is_healthy is True
    assert details["status"] == "healthy"
    assert all(component in details for component in ["database", "redis", "system", "metrics"])
    assert details["database"]["status"] == "connected"
    assert details["redis"]["status"] == "connected"
    assert details["system"]["status"] == "healthy"
    assert details["metrics"]["status"] == "healthy"


@pytest.mark.asyncio
async def test_check_health_database_unhealthy(
    health_check: HealthCheck, mock_db_engine: AsyncEngine, mocker: "MockerFixture"
) -> None:
    """Test health check when database is unhealthy.

    Args:
        health_check: HealthCheck instance
        mock_db_engine: Mock database engine
        mocker: pytest-mock fixture
    """
    # Make database check fail
    mock_db_engine.connect.side_effect = Exception("Database connection failed")

    # Mock system checks
    mocker.patch("psutil.cpu_percent", return_value=50.0)
    mocker.patch(
        "psutil.virtual_memory",
        return_value=MagicMock(total=16000000000, available=8000000000, percent=50.0),
    )
    mocker.patch(
        "psutil.disk_usage",
        return_value=MagicMock(total=100000000000, free=50000000000, percent=50.0),
    )

    # Perform health check
    is_healthy, details = await health_check.check_health()

    assert is_healthy is False
    assert details["status"] == "unhealthy"
    assert details["database"]["status"] == "error"
    assert "Database connection failed" in details["database"]["message"]


@pytest.mark.asyncio
async def test_check_health_redis_unhealthy(
    health_check: HealthCheck, mock_redis_client: aioredis.Redis, mocker: "MockerFixture"
) -> None:
    """Test health check when Redis is unhealthy.

    Args:
        health_check: HealthCheck instance
        mock_redis_client: Mock Redis client
        mocker: pytest-mock fixture
    """
    # Make Redis check fail
    mock_redis_client.ping.side_effect = Exception("Redis connection failed")
    mock_redis_client.info.side_effect = Exception("Redis connection failed")

    # Mock system checks
    mocker.patch("psutil.cpu_percent", return_value=50.0)
    mocker.patch(
        "psutil.virtual_memory",
        return_value=MagicMock(total=16000000000, available=8000000000, percent=50.0),
    )
    mocker.patch(
        "psutil.disk_usage",
        return_value=MagicMock(total=100000000000, free=50000000000, percent=50.0),
    )

    # Perform health check
    is_healthy, details = await health_check.check_health()

    assert is_healthy is False
    assert details["status"] == "unhealthy"
    assert details["redis"]["status"] == "error"
    assert "Redis connection failed" in details["redis"]["message"]


@pytest.mark.asyncio
async def test_check_health_system_warning(
    health_check: HealthCheck, mocker: "MockerFixture"
) -> None:
    """Test health check when system resources are near capacity.

    Args:
        health_check: HealthCheck instance
        mocker: pytest-mock fixture
    """
    # Mock high system resource usage
    mocker.patch("psutil.cpu_percent", return_value=95.0)
    mocker.patch(
        "psutil.virtual_memory",
        return_value=MagicMock(total=16000000000, available=1000000000, percent=95.0),
    )
    mocker.patch(
        "psutil.disk_usage",
        return_value=MagicMock(total=100000000000, free=5000000000, percent=95.0),
    )

    # Perform health check
    is_healthy, details = await health_check.check_health()

    assert is_healthy is False
    assert details["status"] == "unhealthy"
    assert details["system"]["status"] == "warning"
    assert "95" in details["system"]["cpu_usage"]
    assert "95" in details["system"]["memory_usage"]
    assert "95" in details["system"]["disk_usage"]


@pytest.mark.asyncio
async def test_check_health_metrics_unhealthy(
    health_check: HealthCheck, mocker: "MockerFixture"
) -> None:
    """Test health check when metrics collection fails.

    Args:
        health_check: HealthCheck instance
        mocker: pytest-mock fixture
    """
    # Mock metrics collection failure
    mocker.patch.object(REGISTRY, "collect", side_effect=Exception("Metrics collection failed"))

    # Mock system checks
    mocker.patch("psutil.cpu_percent", return_value=50.0)
    mocker.patch(
        "psutil.virtual_memory",
        return_value=MagicMock(total=16000000000, available=8000000000, percent=50.0),
    )
    mocker.patch(
        "psutil.disk_usage",
        return_value=MagicMock(total=100000000000, free=50000000000, percent=50.0),
    )

    # Perform health check
    is_healthy, details = await health_check.check_health()

    assert is_healthy is False
    assert details["status"] == "unhealthy"
    assert details["metrics"]["status"] == "error"
    assert "Metrics collection failed" in details["metrics"]["message"]


def test_health_endpoint(client: TestClient) -> None:
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_liveness_endpoint(client: TestClient) -> None:
    """Test liveness probe endpoint."""
    response = client.get("/health/liveness")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"


def test_readiness_endpoint(client: TestClient) -> None:
    """Test readiness probe endpoint."""
    response = client.get("/health/readiness")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
