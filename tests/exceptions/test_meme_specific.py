"""Tests for meme-specific exceptions."""

from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

import pytest
from fastapi import HTTPException

from dspy_meme_gen.exceptions.base import ErrorCode
from dspy_meme_gen.exceptions.meme_specific import (
    RouterError,
    RouteNotFoundError,
    InvalidRouteError,
    MemeGenerationError,
    PromptGenerationError,
    ImageGenerationError,
    TrendAnalysisError,
    FormatSelectionError,
    ContentVerificationError,
    ScoringError,
    MemeTemplateNotFoundError,
    InvalidMemeFormatError,
)


@pytest.fixture
def sample_route() -> Dict[str, Any]:
    """Fixture providing a sample route configuration."""
    return {
        "topic": "programming",
        "format": "drake",
        "style": "minimal",
        "constraints": {"aspect_ratio": "1:1"},
    }


@pytest.fixture
def sample_format_details() -> Dict[str, Any]:
    """Fixture providing sample format details."""
    return {
        "name": "drake",
        "template_url": "https://example.com/drake.jpg",
        "text_positions": ["top", "bottom"],
        "aspect_ratio": "1:1",
    }


def test_router_error() -> None:
    """Test RouterError initialization and details."""
    route_info = {"path": "meme/generate", "method": "POST"}
    error = RouterError("Failed to route request", route_info=route_info)

    assert str(error) == "Failed to route request"
    assert error.code == ErrorCode.AGENT_EXECUTION_ERROR
    assert error.details["route_info"] == route_info


def test_route_not_found_error(sample_route: Dict[str, Any]) -> None:
    """Test RouteNotFoundError initialization and details."""
    error = RouteNotFoundError("No suitable route found for request", request=sample_route)

    assert str(error) == "No suitable route found for request"
    assert error.code == ErrorCode.AGENT_VALIDATION_ERROR
    assert error.details["route_info"]["request"] == sample_route


def test_invalid_route_error(sample_route: Dict[str, Any]) -> None:
    """Test InvalidRouteError initialization and details."""
    reason = "Unsupported meme format"
    error = InvalidRouteError("Invalid route configuration", route=sample_route, reason=reason)

    assert str(error) == "Invalid route configuration"
    assert error.code == ErrorCode.AGENT_VALIDATION_ERROR
    assert error.details["route_info"]["route"] == sample_route
    assert error.details["route_info"]["reason"] == reason


def test_meme_generation_error() -> None:
    """Test MemeGenerationError initialization and details."""
    generation_info = {"attempt": 1, "status": "failed"}
    error = MemeGenerationError("Failed to generate meme", generation_info=generation_info)

    assert str(error) == "Failed to generate meme"
    assert error.code == ErrorCode.CONTENT_GENERATION_ERROR
    assert error.details["generation_info"] == generation_info


def test_prompt_generation_error(sample_format_details: Dict[str, Any]) -> None:
    """Test PromptGenerationError initialization and details."""
    error = PromptGenerationError(
        "Failed to generate prompt", topic="programming", format_details=sample_format_details
    )

    assert str(error) == "Failed to generate prompt"
    assert error.code == ErrorCode.CONTENT_GENERATION_ERROR
    assert error.details["generation_info"]["topic"] == "programming"
    assert error.details["generation_info"]["format_details"] == sample_format_details


def test_image_generation_error() -> None:
    """Test ImageGenerationError initialization and details."""
    error = ImageGenerationError(
        "Failed to generate image", prompt="A programmer debugging code", service="dall-e"
    )

    assert str(error) == "Failed to generate image"
    assert error.code == ErrorCode.CONTENT_GENERATION_ERROR
    assert error.details["generation_info"]["prompt"] == "A programmer debugging code"
    assert error.details["generation_info"]["service"] == "dall-e"


def test_trend_analysis_error() -> None:
    """Test TrendAnalysisError initialization and details."""
    sources = ["twitter", "reddit"]
    error = TrendAnalysisError(
        "Failed to analyze trends", sources=sources, query="programming memes"
    )

    assert str(error) == "Failed to analyze trends"
    assert error.code == ErrorCode.AGENT_EXECUTION_ERROR
    assert error.details["sources"] == sources
    assert error.details["query"] == "programming memes"


def test_format_selection_error() -> None:
    """Test FormatSelectionError initialization and details."""
    constraints = {"style": "minimal", "aspect_ratio": "1:1"}
    error = FormatSelectionError(
        "Failed to select format", topic="programming", constraints=constraints
    )

    assert str(error) == "Failed to select format"
    assert error.code == ErrorCode.AGENT_EXECUTION_ERROR
    assert error.details["topic"] == "programming"
    assert error.details["constraints"] == constraints


def test_content_verification_error() -> None:
    """Test ContentVerificationError initialization and details."""
    issues = ["Contains inappropriate language", "References sensitive topics"]
    error = ContentVerificationError(
        "Content verification failed",
        content_type="meme",
        verification_type="appropriateness",
        issues=issues,
    )

    assert str(error) == "Content verification failed"
    assert error.code == ErrorCode.CONTENT_VALIDATION_ERROR
    assert error.details["content_type"] == "meme"
    assert error.details["verification_type"] == "appropriateness"
    assert error.details["issues"] == issues


def test_scoring_error() -> None:
    """Test ScoringError initialization and details."""
    criteria = ["humor", "relevance", "originality"]
    error = ScoringError("Failed to score meme", meme_id="123", criteria=criteria)

    assert str(error) == "Failed to score meme"
    assert error.code == ErrorCode.AGENT_EXECUTION_ERROR
    assert error.details["meme_id"] == "123"
    assert error.details["criteria"] == criteria


@pytest.mark.asyncio
async def test_meme_generation_error():
    """Test MemeGenerationError handling."""
    with pytest.raises(MemeGenerationError) as exc_info:
        raise MemeGenerationError("Failed to generate meme")
    assert str(exc_info.value) == "Failed to generate meme"


@pytest.mark.asyncio
async def test_template_not_found_error():
    """Test MemeTemplateNotFoundError handling."""
    with pytest.raises(MemeTemplateNotFoundError) as exc_info:
        raise MemeTemplateNotFoundError("Template not found")
    assert str(exc_info.value) == "Template not found"


@pytest.mark.asyncio
async def test_invalid_format_error():
    """Test InvalidMemeFormatError handling."""
    with pytest.raises(InvalidMemeFormatError) as exc_info:
        raise InvalidMemeFormatError("Invalid meme format")
    assert str(exc_info.value) == "Invalid meme format"
