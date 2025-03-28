"""Tests for the AppropriatenessAgent."""

from typing import Dict, Any, List, TYPE_CHECKING
import pytest
from dspy_meme_gen.agents.appropriateness import AppropriatenessAgent, ContentFlag
from dspy_meme_gen.models.content_guidelines import GuidelineRepository, SeverityLevel

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_guidelines() -> Dict[str, Dict[str, Any]]:
    """
    Create mock guidelines data.
    
    Returns:
        Dictionary containing mock guidelines
    """
    return {
        "language": {
            "profanity": {
                "severity": "high",
                "description": "Use of explicit or offensive language"
            },
            "tone": {
                "severity": "medium",
                "description": "Overly aggressive tone"
            }
        },
        "cultural": {
            "stereotypes": {
                "severity": "high",
                "description": "Use of cultural stereotypes"
            }
        }
    }


@pytest.fixture
def mock_guideline_repository(mocker: "MockerFixture", mock_guidelines: Dict[str, Dict[str, Any]]) -> GuidelineRepository:
    """
    Create a mock GuidelineRepository.
    
    Args:
        mocker: PyTest mocker fixture
        mock_guidelines: Mock guidelines data
        
    Returns:
        Mocked GuidelineRepository instance
    """
    repository = mocker.Mock(spec=GuidelineRepository)
    repository.get_all_guidelines.return_value = mock_guidelines
    return repository


@pytest.fixture
def appropriateness_agent(mock_guideline_repository: GuidelineRepository) -> AppropriatenessAgent:
    """
    Create an instance of AppropriatenessAgent for testing.
    
    Args:
        mock_guideline_repository: Mocked repository instance
        
    Returns:
        AppropriatenessAgent instance
    """
    return AppropriatenessAgent(mock_guideline_repository)


@pytest.fixture
def mock_content_analysis() -> Dict[str, Any]:
    """
    Create mock content analysis results.
    
    Returns:
        Dictionary containing mock analysis data
    """
    return {
        "issues": [
            {
                "category": "language",
                "severity": "medium",
                "description": "Potentially offensive language",
                "elements": ["caption"],
                "suggestion": "Use more neutral language"
            },
            {
                "category": "cultural",
                "severity": "high",
                "description": "Cultural stereotype present",
                "elements": ["image_concept"],
                "suggestion": "Remove stereotypical elements"
            }
        ]
    }


@pytest.fixture
def mock_audience_analysis() -> Dict[str, Any]:
    """
    Create mock audience analysis results.
    
    Returns:
        Dictionary containing mock audience data
    """
    return {
        "suitable_audiences": ["general", "teens", "young_adults"],
        "impact_assessment": {
            "positive": ["humor_appreciation"],
            "negative": ["cultural_sensitivity"]
        }
    }


def test_initialization(appropriateness_agent: AppropriatenessAgent) -> None:
    """
    Test proper initialization of AppropriatenessAgent.
    
    Args:
        appropriateness_agent: Fixture providing AppropriatenessAgent instance
    """
    assert appropriateness_agent.content_analyzer is not None
    assert appropriateness_agent.audience_analyzer is not None
    assert appropriateness_agent.guideline_repository is not None


def test_evaluate_appropriateness_empty_flags(appropriateness_agent: AppropriatenessAgent) -> None:
    """
    Test appropriateness evaluation with no flags.
    
    Args:
        appropriateness_agent: Fixture providing AppropriatenessAgent instance
    """
    flags: List[ContentFlag] = []
    result = appropriateness_agent._evaluate_appropriateness(flags, strict_mode=False)
    assert result is True


def test_evaluate_appropriateness_strict_mode(appropriateness_agent: AppropriatenessAgent) -> None:
    """
    Test appropriateness evaluation in strict mode.
    
    Args:
        appropriateness_agent: Fixture providing AppropriatenessAgent instance
    """
    flags: List[ContentFlag] = [
        {
            "category": "language",
            "severity": "medium",
            "description": "Test flag",
            "affected_elements": ["test"],
            "suggestion": None
        }
    ]
    
    # Should pass with one medium flag
    result = appropriateness_agent._evaluate_appropriateness(flags, strict_mode=True)
    assert result is True
    
    # Should fail with high severity flag
    flags.append({
        "category": "content",
        "severity": "high",
        "description": "Test flag",
        "affected_elements": ["test"],
        "suggestion": None
    })
    result = appropriateness_agent._evaluate_appropriateness(flags, strict_mode=True)
    assert result is False


def test_determine_rating(appropriateness_agent: AppropriatenessAgent) -> None:
    """
    Test content rating determination.
    
    Args:
        appropriateness_agent: Fixture providing AppropriatenessAgent instance
    """
    # Test empty flags
    assert appropriateness_agent._determine_rating([]) == "safe"
    
    # Test with low severity flag
    flags: List[ContentFlag] = [
        {
            "category": "professional",
            "severity": "low",
            "description": "Test flag",
            "affected_elements": ["test"],
            "suggestion": None
        }
    ]
    assert appropriateness_agent._determine_rating(flags) == "safe"
    
    # Test with medium severity flag
    flags.append({
        "category": "language",
        "severity": "medium",
        "description": "Test flag",
        "affected_elements": ["test"],
        "suggestion": None
    })
    assert appropriateness_agent._determine_rating(flags) == "questionable"
    
    # Test with high severity flag
    flags.append({
        "category": "cultural",
        "severity": "high",
        "description": "Test flag",
        "affected_elements": ["test"],
        "suggestion": None
    })
    assert appropriateness_agent._determine_rating(flags) == "inappropriate"


def test_forward_integration(
    appropriateness_agent: AppropriatenessAgent,
    mocker: "MockerFixture",
    mock_content_analysis: Dict[str, Any],
    mock_audience_analysis: Dict[str, Any],
    mock_guidelines: Dict[str, Dict[str, Any]]
) -> None:
    """
    Test full integration of forward method.
    
    Args:
        appropriateness_agent: Fixture providing AppropriatenessAgent instance
        mocker: PyTest mocker fixture
        mock_content_analysis: Fixture providing mock content analysis
        mock_audience_analysis: Fixture providing mock audience analysis
        mock_guidelines: Fixture providing mock guidelines
    """
    # Mock the analyzer responses
    mocker.patch.object(
        appropriateness_agent.content_analyzer,
        "__call__",
        return_value=mock_content_analysis
    )
    mocker.patch.object(
        appropriateness_agent.audience_analyzer,
        "__call__",
        return_value=mock_audience_analysis
    )
    
    # Test concept
    concept = {
        "topic": "test_topic",
        "caption": "test caption",
        "image_concept": "test image concept"
    }
    
    result = appropriateness_agent.forward(concept)
    
    assert isinstance(result, dict)
    assert "is_appropriate" in result
    assert "flags" in result
    assert "overall_rating" in result
    assert "target_audience" in result
    assert "suggested_modifications" in result
    
    # Verify flags were processed correctly
    assert len(result["flags"]) == len(mock_content_analysis["issues"])
    assert result["overall_rating"] == "inappropriate"  # Due to high severity flag
    assert result["target_audience"] == mock_audience_analysis["suitable_audiences"]
    
    # Verify guidelines were fetched from repository
    appropriateness_agent.guideline_repository.get_all_guidelines.assert_called_once()


def test_forward_error_handling(
    appropriateness_agent: AppropriatenessAgent,
    mocker: "MockerFixture",
    caplog: "LogCaptureFixture"
) -> None:
    """
    Test error handling in forward method.
    
    Args:
        appropriateness_agent: Fixture providing AppropriatenessAgent instance
        mocker: PyTest mocker fixture
        caplog: Fixture for capturing log output
    """
    # Mock content_analyzer to raise an exception
    mocker.patch.object(
        appropriateness_agent.content_analyzer,
        "__call__",
        side_effect=Exception("Test error")
    )
    
    concept = {"topic": "test_topic"}
    
    with pytest.raises(RuntimeError) as exc_info:
        appropriateness_agent.forward(concept)
    
    assert "Failed to screen content" in str(exc_info.value)
    assert "Test error" in str(exc_info.value)
    assert "Error in appropriateness screening" in caplog.text 