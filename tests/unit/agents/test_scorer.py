"""Tests for the scoring agent."""

from typing import TYPE_CHECKING, Dict, Any, List

import pytest
from dspy_meme_gen.agents.scorer import ScoringAgent, MemeScore

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_scorer_response(mocker: "MockerFixture") -> Dict[str, float]:
    """Mock the scorer response."""
    return mocker.MagicMock(
        humor_score=0.8, clarity_score=0.7, creativity_score=0.9, shareability_score=0.6
    )


@pytest.fixture
def mock_meme_candidates() -> List[Dict[str, Any]]:
    """Create mock meme candidates for testing."""
    return [
        {"caption": "Test caption 1", "image_url": "http://example.com/image1.jpg"},
        {"caption": "Test caption 2", "image_url": "http://example.com/image2.jpg"},
    ]


@pytest.fixture
def scoring_agent(mocker: "MockerFixture", mock_scorer_response: Dict[str, float]) -> ScoringAgent:
    """Create a scoring agent with mocked scorer."""
    agent = ScoringAgent()
    mocker.patch.object(agent.scorer, "__call__", return_value=mock_scorer_response)
    return agent


def test_meme_score_creation() -> None:
    """Test creating a MemeScore object."""
    score = MemeScore(
        humor_score=0.8,
        clarity_score=0.7,
        creativity_score=0.9,
        shareability_score=0.6,
        final_score=0.75,
        feedback="Test feedback",
    )

    assert score.humor_score == 0.8
    assert score.clarity_score == 0.7
    assert score.creativity_score == 0.9
    assert score.shareability_score == 0.6
    assert score.final_score == 0.75
    assert score.feedback == "Test feedback"


def test_scoring_agent_initialization() -> None:
    """Test initializing the scoring agent."""
    agent = ScoringAgent()
    assert agent.scorer is not None


def test_scoring_memes(
    scoring_agent: ScoringAgent, mock_meme_candidates: List[Dict[str, Any]]
) -> None:
    """Test scoring meme candidates."""
    scored_memes = scoring_agent.forward(mock_meme_candidates)

    assert len(scored_memes) == 2
    for meme in scored_memes:
        assert "score" in meme
        assert isinstance(meme["score"], MemeScore)
        assert meme["score"].humor_score == 0.8
        assert meme["score"].clarity_score == 0.7
        assert meme["score"].creativity_score == 0.9
        assert meme["score"].shareability_score == 0.6
        # Expected weighted score: (0.8 * 0.35) + (0.7 * 0.25) + (0.9 * 0.20) + (0.6 * 0.20) = 0.755
        assert pytest.approx(meme["score"].final_score, 0.001) == 0.755


def test_scoring_with_factuality_check(
    scoring_agent: ScoringAgent, mock_meme_candidates: List[Dict[str, Any]]
) -> None:
    """Test scoring with factuality check."""
    factuality_check = {"is_factual": False, "factual_issues": ["Minor issue"]}
    scored_memes = scoring_agent.forward(mock_meme_candidates, factuality_check=factuality_check)

    for meme in scored_memes:
        # Expected score should be reduced by factuality factor (0.7)
        assert pytest.approx(meme["score"].final_score, 0.001) == 0.755 * 0.7


def test_scoring_with_instruction_check(
    scoring_agent: ScoringAgent, mock_meme_candidates: List[Dict[str, Any]]
) -> None:
    """Test scoring with instruction check."""
    instruction_check = {"constraints_met": False, "violations": ["Minor violation"]}
    scored_memes = scoring_agent.forward(mock_meme_candidates, instruction_check=instruction_check)

    for meme in scored_memes:
        # Expected score should be reduced by instruction factor (0.8)
        assert pytest.approx(meme["score"].final_score, 0.001) == 0.755 * 0.8


def test_scoring_with_appropriateness_check(
    scoring_agent: ScoringAgent, mock_meme_candidates: List[Dict[str, Any]]
) -> None:
    """Test scoring with appropriateness check."""
    appropriateness_check = {"is_appropriate": False}
    scored_memes = scoring_agent.forward(
        mock_meme_candidates, appropriateness_check=appropriateness_check
    )

    for meme in scored_memes:
        # Expected score should be reduced to 0 by appropriateness factor
        assert meme["score"].final_score == 0.0


def test_scoring_with_all_checks(
    scoring_agent: ScoringAgent, mock_meme_candidates: List[Dict[str, Any]]
) -> None:
    """Test scoring with all checks applied."""
    factuality_check = {"is_factual": False, "factual_issues": ["Minor issue"]}
    instruction_check = {"constraints_met": False, "violations": ["Minor violation"]}
    appropriateness_check = {"is_appropriate": True}

    scored_memes = scoring_agent.forward(
        mock_meme_candidates,
        factuality_check=factuality_check,
        instruction_check=instruction_check,
        appropriateness_check=appropriateness_check,
    )

    for meme in scored_memes:
        # Expected score should be reduced by both factuality (0.7) and instruction (0.8) factors
        assert pytest.approx(meme["score"].final_score, 0.001) == 0.755 * 0.7 * 0.8


def test_memes_sorted_by_score(scoring_agent: ScoringAgent, mocker: "MockerFixture") -> None:
    """Test that memes are sorted by score descending."""
    # Create mock responses with different scores
    mock_responses = [
        mocker.MagicMock(
            humor_score=0.5, clarity_score=0.5, creativity_score=0.5, shareability_score=0.5
        ),
        mocker.MagicMock(
            humor_score=0.9, clarity_score=0.9, creativity_score=0.9, shareability_score=0.9
        ),
    ]

    mocker.patch.object(scoring_agent.scorer, "__call__", side_effect=mock_responses)

    meme_candidates = [
        {"caption": "Low score", "image_url": "http://example.com/low.jpg"},
        {"caption": "High score", "image_url": "http://example.com/high.jpg"},
    ]

    scored_memes = scoring_agent.forward(meme_candidates)

    assert len(scored_memes) == 2
    assert scored_memes[0]["caption"] == "High score"
    assert scored_memes[1]["caption"] == "Low score"
    assert scored_memes[0]["score"].final_score > scored_memes[1]["score"].final_score
