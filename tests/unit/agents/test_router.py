"""Unit tests for the Router Agent."""

from typing import Dict, Any
import pytest
from pytest_mock import MockerFixture
from dspy_meme_gen.agents.router import RouterAgent, RouteResult


def test_router_initialization() -> None:
    """Test Router Agent initialization."""
    router = RouterAgent()
    assert router is not None
    assert hasattr(router, "router")


@pytest.fixture
def sample_user_request() -> str:
    """Sample user request for testing."""
    return "Create a meme about Python programming using the Drake format in minimalist style"

@pytest.fixture
def sample_route_result() -> Dict[str, Any]:
    """Sample routing result."""
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

def test_router_forward_success(
    mocker: MockerFixture,
    sample_user_request: str,
    sample_route_result: Dict[str, Any]
) -> None:
    """
    Test successful routing of a user request.
    
    Args:
        mocker: Pytest mocker fixture
        sample_user_request: Sample user request
        sample_route_result: Sample routing result
    """
    # Mock the DSPy predictor response
    mock_predictor = mocker.MagicMock()
    mock_predictor.topic = sample_route_result["topic"]
    mock_predictor.format = sample_route_result["format"]
    mock_predictor.verification_needs = {
        "factuality": False,
        "appropriateness": True
    }
    mock_predictor.constraints = sample_route_result["constraints"]
    mock_predictor.generation_approach = sample_route_result["generation_approach"]
    
    router = RouterAgent()
    router.router = mocker.MagicMock(return_value=mock_predictor)
    
    result = router(sample_user_request)
    
    assert isinstance(result, dict)
    assert result["topic"] == sample_route_result["topic"]
    assert result["format"] == sample_route_result["format"]
    assert result["verification_needs"]["factuality"] == False
    assert result["verification_needs"]["appropriateness"] == True
    assert result["constraints"] == sample_route_result["constraints"]
    assert result["generation_approach"] == sample_route_result["generation_approach"]


def test_router_forward_minimal_request(mocker: MockerFixture) -> None:
    """
    Test routing with minimal user request.
    
    Args:
        mocker: Pytest mocker fixture
    """
    minimal_request = "Create a meme about cats"
    
    # Mock the DSPy predictor response
    mock_predictor = mocker.MagicMock()
    mock_predictor.topic = "cats"
    mock_predictor.format = None
    mock_predictor.verification_needs = {
        "factuality": False,
        "appropriateness": True
    }
    mock_predictor.constraints = {}
    mock_predictor.generation_approach = "standard"
    
    router = RouterAgent()
    router.router = mocker.MagicMock(return_value=mock_predictor)
    
    result = router(minimal_request)
    
    assert isinstance(result, dict)
    assert result["topic"] == "cats"
    assert result["format"] is None
    assert result["verification_needs"]["factuality"] == False
    assert result["verification_needs"]["appropriateness"] == True
    assert result["constraints"] == {}
    assert result["generation_approach"] == "standard"


def test_router_forward_error_handling(mocker: MockerFixture) -> None:
    """
    Test error handling in router forward pass.
    
    Args:
        mocker: Pytest mocker fixture
    """
    router = RouterAgent()
    router.router = mocker.MagicMock(side_effect=Exception("Test error"))
    
    with pytest.raises(RuntimeError) as exc_info:
        router("Create a meme")
    
    assert "Failed to process routing request" in str(exc_info.value)


@pytest.mark.parametrize("request_text,expected_format", [
    ("Create a meme using the Drake format", "Drake"),
    ("Make a distracted boyfriend meme", "Distracted Boyfriend"),
    ("Generate a meme about coding", None),
])
def test_router_format_detection(
    mocker: MockerFixture,
    request_text: str,
    expected_format: str
) -> None:
    """
    Test meme format detection from various requests.
    
    Args:
        mocker: Pytest mocker fixture
        request_text: Input request text
        expected_format: Expected meme format
    """
    # Mock the DSPy predictor response
    mock_predictor = mocker.MagicMock()
    mock_predictor.topic = "test topic"
    mock_predictor.format = expected_format
    mock_predictor.generation_approach = "standard"
    
    router = RouterAgent()
    router.router = mocker.MagicMock(return_value=mock_predictor)
    
    result = router(request_text)
    
    assert isinstance(result, dict)
    assert result["format"] == expected_format 