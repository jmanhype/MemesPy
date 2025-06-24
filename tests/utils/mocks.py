"""Mock implementations for external services."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json
import os
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from tests.config import test_config


@dataclass
class MockResponse:
    """Mock HTTP response."""

    status_code: int
    content: bytes
    headers: Dict[str, str]

    def json(self) -> Dict[str, Any]:
        """Get JSON response.

        Returns:
            Dict[str, Any]: JSON response data
        """
        return json.loads(self.content)


class MockOpenAI:
    """Mock OpenAI API client."""

    def __init__(self, api_key: str = "mock_key"):
        """Initialize mock OpenAI client.

        Args:
            api_key: API key (not used in mock)
        """
        self.api_key = api_key
        self.mock_responses: Dict[str, Any] = {
            "image_generation": {"data": [{"url": "https://mock-image.com/1.jpg"}]},
            "text_completion": {"choices": [{"text": "Mock completion text"}]},
        }

    async def create_image(
        self, prompt: str, n: int = 1, size: str = "1024x1024"
    ) -> Dict[str, Any]:
        """Mock image generation.

        Args:
            prompt: Image generation prompt
            n: Number of images to generate
            size: Image size

        Returns:
            Dict[str, Any]: Mock response data
        """
        return self.mock_responses["image_generation"]

    async def create_completion(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Mock text completion.

        Args:
            prompt: Text completion prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict[str, Any]: Mock response data
        """
        return self.mock_responses["text_completion"]


class MockCloudinary:
    """Mock Cloudinary client."""

    def __init__(self):
        """Initialize mock Cloudinary client."""
        self.uploaded_files: List[Dict[str, Any]] = []

    def upload(self, file: Any, folder: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Mock file upload.

        Args:
            file: File to upload
            folder: Upload folder
            **kwargs: Additional upload options

        Returns:
            Dict[str, Any]: Mock upload response
        """
        mock_response = {
            "secure_url": "https://mock-cloudinary.com/image.jpg",
            "public_id": "mock_public_id",
            "version": "1234567890",
            "format": "jpg",
            "width": 1024,
            "height": 1024,
        }
        self.uploaded_files.append(mock_response)
        return mock_response


class MockRedis:
    """Mock Redis client."""

    def __init__(self):
        """Initialize mock Redis client."""
        self.data: Dict[str, bytes] = {}
        self.ttl: Dict[str, int] = {}

    async def get(self, key: str) -> Optional[bytes]:
        """Mock get operation.

        Args:
            key: Key to get

        Returns:
            Optional[bytes]: Value if exists
        """
        return self.data.get(key)

    async def set(self, key: str, value: bytes, ex: Optional[int] = None) -> bool:
        """Mock set operation.

        Args:
            key: Key to set
            value: Value to set
            ex: Expiration time in seconds

        Returns:
            bool: Success status
        """
        self.data[key] = value
        if ex is not None:
            self.ttl[key] = ex
        return True

    async def delete(self, key: str) -> int:
        """Mock delete operation.

        Args:
            key: Key to delete

        Returns:
            int: Number of keys deleted
        """
        if key in self.data:
            del self.data[key]
            if key in self.ttl:
                del self.ttl[key]
            return 1
        return 0


@pytest.fixture
def mock_openai(mocker: MockerFixture) -> MockOpenAI:
    """Fixture for mock OpenAI client.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        MockOpenAI: Mock OpenAI client
    """
    mock_client = MockOpenAI()
    mocker.patch("openai.AsyncClient", return_value=mock_client)
    return mock_client


@pytest.fixture
def mock_cloudinary(mocker: MockerFixture) -> MockCloudinary:
    """Fixture for mock Cloudinary client.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        MockCloudinary: Mock Cloudinary client
    """
    mock_client = MockCloudinary()
    mocker.patch("cloudinary.uploader", mock_client)
    return mock_client


@pytest.fixture
async def mock_redis(mocker: MockerFixture) -> MockRedis:
    """Fixture for mock Redis client.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        MockRedis: Mock Redis client
    """
    mock_client = MockRedis()
    mocker.patch("aioredis.Redis", return_value=mock_client)
    return mock_client


@pytest.fixture
def mock_data_dir() -> Path:
    """Fixture for mock data directory.

    Returns:
        Path: Mock data directory path
    """
    data_dir = Path(test_config.MOCK_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
