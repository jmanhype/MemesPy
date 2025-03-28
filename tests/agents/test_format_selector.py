"""Tests for the format selection agent."""
from typing import TYPE_CHECKING, Dict, Any, List
import pytest
from sqlalchemy import insert

from dspy_meme_gen.agents.format_selector import FormatSelectionAgent, FormatSelectionResult
from dspy_meme_gen.models.meme import MemeTemplate

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture
    from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture
async def mock_templates(db_session: "AsyncSession") -> List[Dict[str, Any]]:
    """Create mock meme templates in the database."""
    templates = [
        {
            "name": "Drake",
            "description": "Two-panel format showing rejection and approval",
            "example_url": "http://example.com/drake.jpg",
            "format_type": "comparison",
            "structure": {
                "panels": 2,
                "text_positions": ["left", "right"],
                "aspect_ratio": "1:1",
                "style": "cartoon"
            },
            "popularity_score": 0.9,
            "tags": ["comparison", "choice", "preference"]
        },
        {
            "name": "Distracted Boyfriend",
            "description": "Three-role format showing divided attention",
            "example_url": "http://example.com/distracted.jpg",
            "format_type": "situation",
            "structure": {
                "panels": 1,
                "text_positions": ["left", "center", "right"],
                "aspect_ratio": "16:9",
                "style": "photo"
            },
            "popularity_score": 0.8,
            "tags": ["relationships", "choice", "distraction"]
        }
    ]
    
    # Insert templates into database
    for template in templates:
        stmt = insert(MemeTemplate).values(**template)
        await db_session.execute(stmt)
    await db_session.commit()
    
    return templates


@pytest.fixture
def mock_format_matcher_response(mocker: "MockerFixture") -> Dict[str, Any]:
    """Mock the format matcher response."""
    return mocker.MagicMock(
        format_characteristics={
            "panels": 0.8,
            "text_positions": 0.6,
            "style": 0.7
        },
        topic_requirements={
            "comparison": 0.9,
            "choice": 0.7
        },
        rationale="Selected format matches topic requirements"
    )


@pytest.fixture
async def format_selector(
    db_session: "AsyncSession",
    mock_format_matcher_response: Dict[str, Any],
    mocker: "MockerFixture"
) -> FormatSelectionAgent:
    """Create a format selection agent with mocked components."""
    agent = FormatSelectionAgent(db_session)
    mocker.patch.object(agent.format_matcher, "__call__", return_value=mock_format_matcher_response)
    return agent


@pytest.mark.asyncio
async def test_get_available_formats(
    format_selector: FormatSelectionAgent,
    mock_templates: List[Dict[str, Any]]
) -> None:
    """Test retrieving available formats from database."""
    formats = await format_selector.get_available_formats()
    
    assert len(formats) == 2
    assert formats[0]["name"] == "Drake"  # Should be first due to higher popularity
    assert formats[1]["name"] == "Distracted Boyfriend"


@pytest.mark.asyncio
async def test_format_selection_basic(format_selector: FormatSelectionAgent) -> None:
    """Test basic format selection without constraints."""
    result = await format_selector.forward("comparison between two options")
    
    assert isinstance(result, FormatSelectionResult)
    assert result.selected_format == "Drake"
    assert result.confidence_score > 0.7
    assert "rationale" in result.__dict__


@pytest.mark.asyncio
async def test_format_selection_with_constraints(format_selector: FormatSelectionAgent) -> None:
    """Test format selection with specific constraints."""
    constraints = {
        "aspect_ratio": "16:9",
        "style": "photo"
    }
    
    result = await format_selector.forward(
        "person being distracted",
        constraints=constraints
    )
    
    assert result.selected_format == "Distracted Boyfriend"
    assert result.format_details["structure"]["aspect_ratio"] == "16:9"
    assert result.format_details["structure"]["style"] == "photo"


@pytest.mark.asyncio
async def test_format_selection_with_trends(format_selector: FormatSelectionAgent) -> None:
    """Test format selection with trending information."""
    trends = {
        "trending_formats": ["Drake"],
        "trending_topics": ["technology", "AI"]
    }
    
    result = await format_selector.forward(
        "AI vs traditional methods",
        trends=trends
    )
    
    assert result.selected_format == "Drake"
    assert result.confidence_score > 0.7


@pytest.mark.asyncio
async def test_format_selection_no_matching_constraints(format_selector: FormatSelectionAgent) -> None:
    """Test format selection when no format matches constraints."""
    constraints = {
        "aspect_ratio": "9:16",  # No format has this ratio
        "style": "pixel_art"
    }
    
    result = await format_selector.forward(
        "random topic",
        constraints=constraints
    )
    
    # Should fall back to most popular format
    assert result.selected_format == "Drake"
    assert result.confidence_score == 0.5
    assert "No format perfectly matches constraints" in result.rationale


@pytest.mark.asyncio
async def test_meets_constraints() -> None:
    """Test the constraint checking logic."""
    format_dict = {
        "aspect_ratio": "1:1",
        "style": "cartoon",
        "text_positions": ["left", "right"]
    }
    
    # Test matching constraints
    matching_constraints = {
        "aspect_ratio": "1:1",
        "style": "cartoon"
    }
    assert FormatSelectionAgent._meets_constraints(None, format_dict, matching_constraints)
    
    # Test non-matching constraints
    non_matching_constraints = {
        "aspect_ratio": "16:9",
        "style": "photo"
    }
    assert not FormatSelectionAgent._meets_constraints(None, format_dict, non_matching_constraints)


@pytest.mark.asyncio
async def test_calculate_match_score() -> None:
    """Test the format match score calculation."""
    format_dict = {
        "tags": ["comparison", "choice"],
        "panels": 2,
        "text_positions": ["left", "right"],
        "style": "cartoon",
        "popularity_score": 0.9
    }
    
    format_characteristics = {
        "panels": 0.8,
        "text_positions": 0.6,
        "style": 0.7
    }
    
    topic_requirements = {
        "comparison": 0.9,
        "choice": 0.7
    }
    
    score = FormatSelectionAgent._calculate_match_score(
        None,
        format_dict,
        format_characteristics,
        topic_requirements
    )
    
    assert 0 <= score <= 1
    assert score > 0.7  # Should have a high score given the matching characteristics 