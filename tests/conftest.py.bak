"""Test configuration and shared fixtures."""

from typing import Dict, Any, Generator, List, Optional, Union
import os
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from _pytest.fixtures import FixtureRequest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from pytest_mock import MockerFixture
from pytest_mock.plugin import MockerFixture
import dspy
from dspy import BaseLM
import json

class MockLM(dspy.BaseLM):
    """Mock language model for testing."""
    
    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        """Initialize mock LM."""
        super().__init__()
        self.model = model
        
    def basic_request(self, prompt: str, **kwargs: Any) -> str:
        """Handle basic completion requests."""
        # Return a properly formatted response based on the prompt
        if "score" in prompt.lower() or "evaluate" in prompt.lower():
            return {
                "reasoning": "This meme effectively combines humor with relatability. The caption is clever and the image choice supports the joke well.",
                "humor_score": 0.85,
                "clarity_score": 0.90,
                "creativity_score": 0.75,
                "shareability_score": 0.80,
                "feedback": "Strong humor and clarity, could be slightly more creative."
            }
        elif "route" in prompt.lower():
            return {
                "topic": "programming",
                "format": "drake",
                "verification_needs": {
                    "factuality": True,
                    "appropriateness": True
                },
                "constraints": {
                    "aspect_ratio": "1:1",
                    "style": "modern"
                },
                "generation_approach": "standard"
            }
        else:
            return {
                "response": "Mock response for: " + prompt,
                "confidence": 0.9
            }
            
    def __call__(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """Handle both chat and completion requests."""
        if messages:
            # Handle chat format
            last_message = messages[-1]["content"]
            return self.basic_request(last_message)
        else:
            # Handle completion format
            prompt = kwargs.get("prompt", "")
            return self.basic_request(prompt)

@pytest.fixture(autouse=True)
def mock_lm(monkeypatch: MonkeyPatch) -> MockLM:
    """Provide a mock language model for testing."""
    mock = MockLM()
    monkeypatch.setattr(dspy.settings, "lm", mock)
    return mock

@pytest.fixture(autouse=True)
def env_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up test environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "test-cloud")
    monkeypatch.setenv("CLOUDINARY_API_KEY", "test-key")
    monkeypatch.setenv("CLOUDINARY_API_SECRET", "test-secret")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FILE", "tests/test.log")

@pytest.fixture
def mock_openai_response(mocker: MockerFixture) -> Dict[str, Any]:
    """Mock OpenAI API response."""
    return {
        "data": [
            {
                "url": "https://test-image-url.com/image.jpg",
                "revised_prompt": "Test prompt"
            }
        ]
    }

@pytest.fixture
def mock_cloudinary_response(mocker: MockerFixture) -> Dict[str, Any]:
    """Mock Cloudinary upload response."""
    return {
        "secure_url": "https://test-cloudinary-url.com/image.jpg",
        "public_id": "test_image_id",
        "width": 1024,
        "height": 1024
    }

@pytest.fixture
def sample_user_request() -> str:
    """Sample user request for testing."""
    return "Create a meme about Python programming using the Drake format in minimalist style"

@pytest.fixture
def sample_route_result() -> Dict[str, Any]:
    """Sample routing result for testing."""
    return {
        "topic": "Python programming",
        "format": "Drake",
        "verification_needs": {
            "factuality": False,
            "appropriateness": True
        },
        "constraints": {
            "style": "minimalist"
        },
        "generation_approach": "standard"
    }

@pytest.fixture
def sample_prompt_result() -> Dict[str, Any]:
    """Sample prompt generation result for testing."""
    return {
        "caption": "When you use print() for debugging\nWhen you use a proper debugger",
        "image_prompt": "Split image in Drake meme format. Top: A frustrated programmer surrounded by print statements. Bottom: A confident programmer using a modern IDE debugger. Minimalist style.",
        "reasoning": "This meme contrasts the common beginner approach of using print statements for debugging with the more professional approach of using proper debugging tools."
    }

@pytest.fixture
def sample_image_result() -> Dict[str, Any]:
    """Sample image generation result for testing."""
    return {
        "image_url": "https://test-cdn.com/meme.jpg",
        "cloudinary_id": "memes/test123",
        "width": 1024,
        "height": 1024
    } 