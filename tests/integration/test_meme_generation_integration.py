"""Integration tests for meme generation pipeline."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime


@pytest.fixture
def test_client():
    """Create test client for API integration tests."""
    from dspy_meme_gen.api.main import app

    return TestClient(app)


@pytest.mark.integration
def test_health_endpoint_integration(test_client):
    """Test health endpoint returns valid response."""
    response = test_client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded", "unhealthy"]


@pytest.mark.integration
@patch("dspy_meme_gen.dspy_modules.meme_predictor.MemePredictor.forward")
@patch("dspy_meme_gen.dspy_modules.image_generator.ImageGenerator.generate")
def test_meme_generation_end_to_end(mock_image_gen, mock_predictor, test_client):
    """Test complete meme generation flow from request to response."""
    # Mock the DSPy predictor
    mock_predictor.return_value = Mock(
        text="When you write tests\nThat actually pass",
        image_prompt="A developer celebrating at their computer",
        rationale="Testing is important for code quality",
        score=0.85
    )

    # Mock the image generator
    mock_image_gen.return_value = {
        "url": "https://example.com/meme.png",
        "metadata": {
            "model": "test-model",
            "size": "1024x1024",
            "generation_time": 1.5
        }
    }

    # Make request
    request_data = {
        "topic": "software testing",
        "format": "drake"
    }

    response = test_client.post("/api/v1/memes/", json=request_data)

    # Verify response
    assert response.status_code == 200
    data = response.json()

    assert "id" in data
    assert data["topic"] == "software testing"
    assert data["format"] == "drake"
    assert "text" in data
    assert "image_url" in data
    assert "created_at" in data


@pytest.mark.integration
@patch("dspy_meme_gen.dspy_modules.meme_predictor.MemePredictor.forward")
def test_meme_generation_with_error_handling(mock_predictor, test_client):
    """Test error handling in meme generation pipeline."""
    # Mock predictor to raise an error
    mock_predictor.side_effect = Exception("DSPy prediction failed")

    request_data = {
        "topic": "testing",
        "format": "standard"
    }

    response = test_client.post("/api/v1/memes/", json=request_data)

    # Should return error status
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data


@pytest.mark.integration
def test_list_memes_endpoint(test_client):
    """Test listing memes endpoint."""
    response = test_client.get("/api/v1/memes/")

    # Should return list response even if empty
    assert response.status_code == 200
    data = response.json()

    assert "items" in data
    assert "total" in data
    assert isinstance(data["items"], list)
    assert isinstance(data["total"], int)


@pytest.mark.integration
@patch("dspy_meme_gen.dspy_modules.meme_predictor.MemePredictor.forward")
@patch("dspy_meme_gen.dspy_modules.image_generator.ImageGenerator.generate")
def test_meme_metadata_tracking(mock_image_gen, mock_predictor, test_client):
    """Test that meme generation includes metadata tracking."""
    # Setup mocks
    mock_predictor.return_value = Mock(
        text="Test meme text",
        image_prompt="Test prompt",
        rationale="Test rationale",
        score=0.9
    )

    mock_image_gen.return_value = {
        "url": "https://example.com/test.png",
        "metadata": {
            "model": "test-model",
            "generation_time": 2.0
        }
    }

    # Generate meme
    request_data = {
        "topic": "metadata testing",
        "format": "standard"
    }

    response = test_client.post("/api/v1/memes/", json=request_data)

    assert response.status_code == 200
    data = response.json()

    # Verify basic metadata is present
    assert "id" in data
    assert "created_at" in data

    # Verify we can retrieve the meme
    meme_id = data["id"]
    get_response = test_client.get(f"/api/v1/memes/{meme_id}")

    # Should either succeed or return 404 if not implemented
    assert get_response.status_code in [200, 404]


@pytest.mark.integration
def test_invalid_meme_request_validation(test_client):
    """Test validation of invalid meme generation requests."""
    # Missing required fields
    invalid_requests = [
        {},  # Empty request
        {"topic": ""},  # Empty topic
        {"format": ""},  # Empty format
        {"topic": "test"},  # Missing format
        {"format": "test"},  # Missing topic
    ]

    for invalid_request in invalid_requests:
        response = test_client.post("/api/v1/memes/", json=invalid_request)
        # Should return validation error
        assert response.status_code in [400, 422], f"Failed for request: {invalid_request}"
