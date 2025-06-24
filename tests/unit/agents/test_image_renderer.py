"""Tests for the image rendering agent."""

from typing import Dict, Any
import pytest
from io import BytesIO
from PIL import Image
import json
import requests
from dspy_meme_gen.agents.image_renderer import ImageRenderingAgent


@pytest.fixture
def mock_openai_response(mocker):
    """Mock OpenAI API response."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"url": "https://example.com/image.jpg"}]}
    return mock_response


@pytest.fixture
def mock_image_response(mocker):
    """Mock image download response."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.content = b"mock_image_data"
    return mock_response


@pytest.fixture
def mock_cloudinary_response(mocker):
    """Mock Cloudinary upload response."""
    return {
        "secure_url": "https://cloudinary.com/image.jpg",
        "public_id": "meme_123",
        "width": 1024,
        "height": 1024,
    }


@pytest.fixture
def mock_image():
    """Create a mock image."""
    img = Image.new("RGB", (100, 100), color="white")
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture
def sample_prompt_result():
    """Sample prompt generation result."""
    return {
        "image_prompt": "A funny cat meme",
        "caption": "When the code works on the first try",
        "reasoning": "This meme captures the rare joy of code working immediately",
    }


def test_image_generation_success(
    mocker,
    mock_openai_response,
    mock_image_response,
    mock_cloudinary_response,
    sample_prompt_result,
):
    """Test successful image generation."""
    # Mock requests
    mocker.patch("requests.post", return_value=mock_openai_response)
    mocker.patch("requests.get", return_value=mock_image_response)
    mocker.patch("cloudinary.uploader.upload", return_value=mock_cloudinary_response)

    # Create agent and generate image
    agent = ImageRenderingAgent(api_key="test_key")
    result = agent.forward(
        image_prompt=sample_prompt_result["image_prompt"], caption=sample_prompt_result["caption"]
    )

    # Verify result
    assert result["image_url"] == mock_cloudinary_response["secure_url"]
    assert result["cloudinary_id"] == mock_cloudinary_response["public_id"]
    assert result["width"] == mock_cloudinary_response["width"]
    assert result["height"] == mock_cloudinary_response["height"]


def test_image_generation_with_text_overlay(
    mocker,
    mock_openai_response,
    mock_image_response,
    mock_cloudinary_response,
    sample_prompt_result,
    mock_image,
):
    """Test image generation with text overlay."""
    # Mock requests and image processing
    mocker.patch("requests.post", return_value=mock_openai_response)
    mocker.patch("requests.get", return_value=mock_image_response)
    mocker.patch("cloudinary.uploader.upload", return_value=mock_cloudinary_response)
    mocker.patch("PIL.Image.open", return_value=Image.new("RGB", (100, 100)))

    # Create agent and generate image with text overlay
    agent = ImageRenderingAgent(api_key="test_key")
    result = agent.forward(
        image_prompt=sample_prompt_result["image_prompt"],
        caption=sample_prompt_result["caption"],
        format_details={"requirements": ["text_overlay"]},
    )

    # Verify result
    assert result["image_url"] == mock_cloudinary_response["secure_url"]
    assert result["width"] == mock_cloudinary_response["width"]
    assert result["height"] == mock_cloudinary_response["height"]


def test_openai_api_error_handling(mocker, mock_openai_response, sample_prompt_result):
    """Test error handling for OpenAI API failures."""
    # Mock failed API response
    mock_openai_response.status_code = 400
    mock_openai_response.text = "API Error"
    mocker.patch("requests.post", return_value=mock_openai_response)

    agent = ImageRenderingAgent(api_key="test_key")

    with pytest.raises(RuntimeError) as exc_info:
        agent.forward(image_prompt=sample_prompt_result["image_prompt"])

    assert "OpenAI API error" in str(exc_info.value)


def test_cloudinary_upload_error_handling(
    mocker, mock_openai_response, mock_image_response, sample_prompt_result
):
    """Test error handling for Cloudinary upload failures."""
    # Mock successful OpenAI response but failed Cloudinary upload
    mocker.patch("requests.post", return_value=mock_openai_response)
    mocker.patch("requests.get", return_value=mock_image_response)
    mocker.patch("cloudinary.uploader.upload", side_effect=Exception("Upload failed"))

    agent = ImageRenderingAgent(api_key="test_key")

    with pytest.raises(RuntimeError) as exc_info:
        agent.forward(image_prompt=sample_prompt_result["image_prompt"])

    assert "Failed to upload image" in str(exc_info.value)


@pytest.mark.parametrize(
    "service,size", [("openai", "1024x1024"), ("stability", "512x512"), ("midjourney", "256x256")]
)
def test_different_image_services(
    mocker, mock_image_response, mock_cloudinary_response, sample_prompt_result, service, size
):
    """Test support for different image services."""
    # Mock API responses
    mock_service_response = mocker.Mock()
    mock_service_response.status_code = 200
    if service == "openai":
        mock_service_response.json.return_value = {
            "data": [{"url": "https://example.com/image.jpg"}]
        }
    elif service == "stability":
        mock_service_response.artifacts = [mocker.Mock(binary=b"mock_image_data")]
    else:  # midjourney
        mock_service_response.json.return_value = {"image_url": "https://example.com/image.jpg"}

    mocker.patch("requests.post", return_value=mock_service_response)
    mocker.patch("requests.get", return_value=mock_image_response)
    mocker.patch("cloudinary.uploader.upload", return_value=mock_cloudinary_response)

    # For Stability AI
    if service == "stability":
        stability_mock = mocker.Mock()
        stability_mock.generate.return_value = mock_service_response
        mocker.patch("stability_sdk.client.StabilityInference", return_value=stability_mock)

    agent = ImageRenderingAgent(api_key="test_key", image_service=service)
    result = agent.forward(image_prompt=sample_prompt_result["image_prompt"])

    assert result["image_url"] == mock_cloudinary_response["secure_url"]
    assert result["width"] == mock_cloudinary_response["width"]
    assert result["height"] == mock_cloudinary_response["height"]
