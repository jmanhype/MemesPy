"""Pytest configuration and fixtures."""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
import dspy

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"
# Remove the test-key setting - we'll mock DSPy instead
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def mock_dspy_config(monkeypatch):
    """Mock DSPy configuration to prevent real API calls."""
    # Create a mock LM (Language Model) object
    mock_lm = Mock()
    mock_lm.request = Mock(return_value=["mocked response"])
    
    # Mock dspy.configure to do nothing
    monkeypatch.setattr("dspy.configure", Mock())
    
    # Mock dspy.settings.configure to do nothing
    monkeypatch.setattr("dspy.settings.configure", Mock())
    
    # Set the mock LM in dspy.settings
    monkeypatch.setattr("dspy.settings.lm", mock_lm)
    
    # Mock common DSPy components
    mock_chain_of_thought = Mock()
    mock_chain_of_thought.return_value = Mock()
    monkeypatch.setattr("dspy.ChainOfThought", mock_chain_of_thought)
    
    mock_predict = Mock()
    mock_predict.return_value = Mock()
    monkeypatch.setattr("dspy.Predict", mock_predict)
    
    return {
        "lm": mock_lm,
        "chain_of_thought": mock_chain_of_thought,
        "predict": mock_predict
    }


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
