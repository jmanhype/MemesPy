"""Unit tests for the Factuality Agent."""

from typing import Dict, Any, List
import pytest
from unittest.mock import Mock, patch
from dspy_meme_gen.agents.factuality import FactualityAgent, FactCheck, FactualityResult


@pytest.fixture
def mock_claim_extractor() -> Mock:
    """Fixture for mocked claim extractor."""
    mock = Mock()
    mock.return_value.claims = [
        Mock(text="Python is the most popular programming language"),
        Mock(text="JavaScript was created in 10 days"),
    ]
    return mock


@pytest.fixture
def mock_fact_checker() -> Mock:
    """Fixture for mocked fact checker."""
    mock = Mock()
    mock.return_value = Mock(is_factual=True, confidence=0.85, evidence=["Source 1", "Source 2"])
    return mock


@pytest.fixture
def mock_correction_generator() -> Mock:
    """Fixture for mocked correction generator."""
    mock = Mock()
    mock.return_value = Mock(correction="JavaScript was created in about two weeks")
    return mock


@pytest.fixture
def factuality_agent(
    mock_claim_extractor: Mock, mock_fact_checker: Mock, mock_correction_generator: Mock
) -> FactualityAgent:
    """Fixture for FactualityAgent with mocked predictors."""
    agent = FactualityAgent()
    agent.claim_extractor = mock_claim_extractor
    agent.fact_checker = mock_fact_checker
    agent.correction_generator = mock_correction_generator
    return agent


def test_factuality_agent_initialization() -> None:
    """Test FactualityAgent initialization."""
    agent = FactualityAgent()

    assert agent is not None
    assert hasattr(agent, "claim_extractor")
    assert hasattr(agent, "fact_checker")
    assert hasattr(agent, "correction_generator")


def test_successful_verification(factuality_agent: FactualityAgent) -> None:
    """Test successful verification of factual claims."""
    concept = {
        "topic": "Programming Languages",
        "caption": "Python vs JavaScript",
        "format": "comparison",
    }

    result = factuality_agent.forward(concept=concept, topic="programming")

    assert isinstance(result, dict)
    assert result["is_factual"] is True
    assert result["confidence"] > 0.0
    assert len(result["checks"]) == 2
    assert all(check["is_factual"] for check in result["checks"])
    assert "All claims verified as factual" in result["overall_assessment"]


def test_verification_with_corrections(
    factuality_agent: FactualityAgent, mock_fact_checker: Mock
) -> None:
    """Test verification that generates corrections."""
    # Modify fact checker to return false for one claim
    original_return = mock_fact_checker.return_value
    mock_fact_checker.side_effect = [
        original_return,
        Mock(is_factual=False, confidence=0.9, evidence=["Source 3"]),
    ]

    concept = {
        "topic": "Programming History",
        "caption": "JavaScript Creation Myth",
        "format": "fact",
    }

    result = factuality_agent.forward(concept=concept, topic="programming")

    assert not result["is_factual"]  # Should be False because one claim is false
    assert len(result["suggested_corrections"]) == 1
    assert "JavaScript was created in 10 days" in result["suggested_corrections"]
    assert "significant inaccuracies" in result["overall_assessment"]


@pytest.mark.parametrize("strict_mode,expected_threshold", [(True, 90), (False, 75)])
def test_strict_mode_verification(
    factuality_agent: FactualityAgent,
    mock_fact_checker: Mock,
    strict_mode: bool,
    expected_threshold: int,
) -> None:
    """Test verification with different strict mode settings."""
    # Set up fact checker to return mixed results
    mock_fact_checker.side_effect = [
        Mock(is_factual=True, confidence=0.95, evidence=["Source 1"]),
        Mock(is_factual=True, confidence=0.85, evidence=["Source 2"]),
    ]

    concept = {"topic": "Programming", "caption": "Language Facts", "format": "list"}

    result = factuality_agent.forward(concept=concept, topic="programming", strict_mode=strict_mode)

    assert result["is_factual"]  # Both claims are true
    assert f"{expected_threshold}" in result["overall_assessment"]


def test_empty_claims_handling(
    factuality_agent: FactualityAgent, mock_claim_extractor: Mock
) -> None:
    """Test handling of concepts with no factual claims."""
    mock_claim_extractor.return_value.claims = []

    concept = {"topic": "Memes", "caption": "Just for fun", "format": "joke"}

    result = factuality_agent.forward(concept=concept, topic="humor")

    assert result["is_factual"]  # Should be True when no claims to verify
    assert result["confidence"] == 1.0
    assert len(result["checks"]) == 0
    assert "No factual claims identified" in result["overall_assessment"]


def test_error_handling(factuality_agent: FactualityAgent) -> None:
    """Test error handling in verification process."""
    # Make claim extractor raise an exception
    factuality_agent.claim_extractor.side_effect = Exception("API Error")

    concept = {"topic": "Error Test", "caption": "Test error handling", "format": "test"}

    with pytest.raises(RuntimeError) as exc_info:
        factuality_agent.forward(concept=concept, topic="error")

    assert "Failed to verify factuality" in str(exc_info.value)


def test_assessment_generation(factuality_agent: FactualityAgent) -> None:
    """Test the assessment generation with various scenarios."""
    # Create some test fact checks
    checks: List[FactCheck] = [
        {
            "claim": "Claim 1",
            "is_factual": True,
            "confidence": 0.9,
            "evidence": ["Evidence 1"],
            "correction": None,
        },
        {
            "claim": "Claim 2",
            "is_factual": False,
            "confidence": 0.8,
            "evidence": ["Evidence 2"],
            "correction": "Correction 2",
        },
    ]

    assessment = factuality_agent._generate_assessment(checks, strict_mode=False)
    assert "50.0% accurate" in assessment

    # Test with strict mode
    strict_assessment = factuality_agent._generate_assessment(checks, strict_mode=True)
    assert "significant inaccuracies" in strict_assessment


@pytest.mark.parametrize(
    "concept,expected_claims",
    [
        ({"topic": "Python", "caption": "Python is fast"}, ["Python is fast"]),
        (
            {"topic": "JavaScript", "caption": "JS was created quickly"},
            ["JavaScript was created in 10 days"],
        ),
    ],
)
def test_claim_extraction(
    factuality_agent: FactualityAgent, concept: Dict[str, Any], expected_claims: List[str]
) -> None:
    """Test claim extraction from different concepts."""
    result = factuality_agent.forward(concept=concept, topic=concept["topic"])

    assert len(result["checks"]) == len(expected_claims)
    actual_claims = [check["claim"] for check in result["checks"]]
    assert all(claim in actual_claims for claim in expected_claims)
