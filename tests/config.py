"""Test configuration and settings."""

from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestConfig:
    """Test configuration settings."""

    # Database settings
    TEST_DATABASE_URL: str = "postgresql+asyncpg://test:test@localhost:5432/dspy_test"
    TEST_DATABASE_POOL_SIZE: int = 5
    TEST_DATABASE_MAX_OVERFLOW: int = 10

    # Cache settings
    TEST_REDIS_URL: str = "redis://localhost:6379/1"
    TEST_CACHE_TTL: int = 300

    # Load test settings
    LOAD_TEST_CONCURRENT_USERS: int = 50
    LOAD_TEST_DURATION_SECONDS: int = 60
    LOAD_TEST_SPAWN_RATE: int = 5  # Users per second

    # Performance test thresholds
    MAX_QUERY_TIME_MS: int = 100
    MAX_CACHE_TIME_MS: int = 10
    MAX_RESPONSE_TIME_MS: int = 200

    # Mock data settings
    MOCK_DATA_DIR: Path = Path(__file__).parent / "mock_data"

    @classmethod
    def get_test_settings(cls) -> Dict[str, Any]:
        """Get test settings as dictionary.

        Returns:
            Dict[str, Any]: Test settings
        """
        return {field: getattr(cls, field) for field in cls.__annotations__}


# Global test configuration instance
test_config = TestConfig()
