"""Pytest configuration and fixtures."""

import os
import sys
import pytest
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {"choices": [{"message": {"content": "Test meme content"}}]}


@pytest.fixture
def sample_meme_request():
    """Sample meme generation request."""
    return {
        "topic": "testing",
        "format": "drake",
        "context": "When tests finally pass",
        "style": "humor",
    }
