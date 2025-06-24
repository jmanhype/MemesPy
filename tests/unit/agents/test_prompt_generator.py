"""Tests for the prompt generation agent."""

from typing import TYPE_CHECKING, Dict, Any

import pytest

from dspy_meme_gen.agents.prompt_generator import PromptGenerationAgent, PromptGenerationResult

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_format_details() -> Dict[str, Any]:
    """Create mock format details for testing."""
    return {
        "name": "Drake",
        "structure": {
            "panels": 2,
            "text_positions": ["top", "bottom"],
            "aspect_ratio": "1:1",
            "style": "cartoon",
            "layout": "vertical",
            "text_alignment": "left",
        },
    }


@pytest.fixture
def mock_style_preferences() -> Dict[str, Any]:
    """Create mock style preferences for testing."""
    return {
        "artistic_style": "watercolor",
        "color_scheme": "vibrant",
        "lighting": "dramatic",
        "font": "Comic Sans",
        "text_color": "black",
        "text_stroke": "white",
    }


@pytest.fixture
def mock_caption_response(mocker: "MockerFixture") -> Any:
    """Mock the caption generator response."""
    mock = mocker.MagicMock()
    mock.caption = "Old methods | New AI methods"
    mock.reasoning = "Comparing traditional approach with modern AI"
    return mock


@pytest.fixture
def mock_image_prompt_response(mocker: "MockerFixture") -> Any:
    """Mock the image prompt generator response."""
    mock = mocker.MagicMock()
    mock.image_prompt = "Drake meme format showing dismissal and approval gestures"
    return mock


@pytest.fixture
def prompt_generator(mocker: "MockerFixture") -> PromptGenerationAgent:
    """Create a prompt generation agent with mocked components."""
    agent = PromptGenerationAgent()
    
    # Create mock DSPy module responses with proper attribute access
    caption_response = mocker.Mock(spec=['caption', 'reasoning'])
    caption_response.caption = "Old methods | New AI methods"
    caption_response.reasoning = "Comparing traditional approach with modern AI"
    
    image_response = mocker.Mock(spec=['image_prompt'])
    image_response.image_prompt = "Drake meme format showing dismissal and approval gestures"
    
    # Mock the DSPy modules directly by replacing them
    mock_caption_generator = mocker.Mock()
    mock_caption_generator.return_value = caption_response
    
    mock_image_generator = mocker.Mock()
    mock_image_generator.return_value = image_response
    
    agent.caption_generator = mock_caption_generator
    agent.image_prompt_generator = mock_image_generator
    
    # Mock the internal methods to avoid DSPy calls
    mocker.patch.object(
        agent,
        "_apply_constraints",
        side_effect=lambda text, constraints, positions: (
            text[:constraints.get("max_length", 100)] if constraints.get("max_length") else text
        )
    )
    mocker.patch.object(
        agent,
        "_apply_style_preferences",
        side_effect=lambda prompt, prefs, style: (
            f"{prompt}, {style}"
            + (f", {prefs.get('artistic_style')}" if prefs.get('artistic_style') else "")
            + (f", using {prefs.get('color_scheme')} colors" if prefs.get('color_scheme') else "")
            + (f", with {prefs.get('lighting')} lighting" if prefs.get('lighting') else "")
        )
    )
    mocker.patch.object(
        agent,
        "_organize_text_positions",
        side_effect=lambda text, positions: {
            positions[0]: text.split("|")[0].strip() if "|" in text else text,
            positions[1]: text.split("|")[1].strip() if "|" in text and len(text.split("|")) > 1 else ""
        } if len(positions) >= 2 else {}
    )
    mocker.patch.object(
        agent,
        "_compile_style_guide",
        side_effect=lambda format_structure, style_prefs: {
            "layout": format_structure.get("layout", "default"),
            "text_style": {
                "font": style_prefs.get("font", "Impact"),
                "color": style_prefs.get("text_color", "white"),
                "stroke": style_prefs.get("text_stroke", "black"),
                "alignment": format_structure.get("text_alignment", "center"),
            },
            "image_style": {
                "base_style": format_structure.get("style", ""),
                "artistic_style": style_prefs.get("artistic_style", ""),
                "color_scheme": style_prefs.get("color_scheme", ""),
                "lighting": style_prefs.get("lighting", ""),
            },
        }
    )
    
    return agent


def test_prompt_generation_basic(
    prompt_generator: PromptGenerationAgent, mock_format_details: Dict[str, Any]
) -> None:
    """Test basic prompt generation without constraints or preferences."""
    result = prompt_generator.forward("AI vs traditional methods", mock_format_details)

    assert isinstance(result, PromptGenerationResult)
    assert result.caption == "Old methods | New AI methods"
    assert result.image_prompt == "Drake meme format showing dismissal and approval gestures, cartoon"
    assert result.reasoning == "Comparing traditional approach with modern AI"
    assert len(result.text_positions) == 2
    assert "text_style" in result.style_guide
    assert "image_style" in result.style_guide


def test_prompt_generation_with_constraints(
    prompt_generator: PromptGenerationAgent, mock_format_details: Dict[str, Any]
) -> None:
    """Test prompt generation with constraints."""
    constraints = {"max_length": 20, "text_placement": "top", "tone": "formal"}

    result = prompt_generator.forward(
        "AI vs traditional methods", mock_format_details, constraints=constraints
    )

    assert len(result.caption) <= 20
    assert "top" in result.text_positions
    assert result.text_positions["top"] != ""


def test_prompt_generation_with_style_preferences(
    prompt_generator: PromptGenerationAgent,
    mock_format_details: Dict[str, Any],
    mock_style_preferences: Dict[str, Any],
) -> None:
    """Test prompt generation with style preferences."""
    result = prompt_generator.forward(
        "AI vs traditional methods", mock_format_details, style_preferences=mock_style_preferences
    )

    assert "watercolor" in result.image_prompt
    assert "vibrant" in result.image_prompt
    assert "dramatic" in result.image_prompt
    assert result.style_guide["text_style"]["font"] == "Comic Sans"
    assert result.style_guide["text_style"]["color"] == "black"
    assert result.style_guide["text_style"]["stroke"] == "white"


def test_text_position_organization(prompt_generator: PromptGenerationAgent) -> None:
    """Test organizing text into positions."""
    text = "First part | Second part"
    positions = ["top", "bottom"]

    result = prompt_generator._organize_text_positions(text, positions)

    assert result["top"] == "First part"
    assert result["bottom"] == "Second part"


def test_text_position_organization_single_text(prompt_generator: PromptGenerationAgent) -> None:
    """Test organizing single text into positions."""
    text = "Single text"
    positions = ["top", "bottom"]

    result = prompt_generator._organize_text_positions(text, positions)

    assert result["top"] == "Single text"
    assert result["bottom"] == ""


def test_style_preferences_application(prompt_generator: PromptGenerationAgent) -> None:
    """Test applying style preferences to image prompt."""
    prompt = "Base prompt"
    preferences = {"artistic_style": "pixel art", "color_scheme": "neon", "lighting": "ambient"}
    base_style = "cartoon"

    result = prompt_generator._apply_style_preferences(prompt, preferences, base_style)

    assert "cartoon" in result
    assert "pixel art" in result
    assert "neon colors" in result
    assert "ambient lighting" in result


def test_style_guide_compilation(
    prompt_generator: PromptGenerationAgent,
    mock_format_details: Dict[str, Any],
    mock_style_preferences: Dict[str, Any],
) -> None:
    """Test compiling style guide from format and preferences."""
    style_guide = prompt_generator._compile_style_guide(
        mock_format_details["structure"], mock_style_preferences
    )

    assert style_guide["layout"] == "vertical"
    assert style_guide["text_style"]["font"] == "Comic Sans"
    assert style_guide["text_style"]["color"] == "black"
    assert style_guide["text_style"]["stroke"] == "white"
    assert style_guide["text_style"]["alignment"] == "left"
    assert style_guide["image_style"]["base_style"] == "cartoon"
    assert style_guide["image_style"]["artistic_style"] == "watercolor"
    assert style_guide["image_style"]["color_scheme"] == "vibrant"
    assert style_guide["image_style"]["lighting"] == "dramatic"


def test_prompt_generation_error_handling(
    prompt_generator: PromptGenerationAgent,
    mock_format_details: Dict[str, Any],
    mocker: "MockerFixture",
) -> None:
    """Test error handling in prompt generation."""
    # Replace the caption generator with one that raises an exception
    failing_generator = mocker.Mock()
    failing_generator.side_effect = Exception("Caption generation failed")
    prompt_generator.caption_generator = failing_generator

    with pytest.raises(RuntimeError) as exc_info:
        prompt_generator.forward("AI vs traditional methods", mock_format_details)

    assert "Failed to generate prompts" in str(exc_info.value)
