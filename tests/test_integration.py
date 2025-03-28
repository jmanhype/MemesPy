"""Integration tests for the meme generation pipeline."""
from typing import Any, Dict, List, Optional
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.dspy_meme_gen.models.repository import MemeTemplateRepository
from src.dspy_meme_gen.models.meme import MemeTemplate
from src.dspy_meme_gen.pipeline import MemeGenerationPipeline
from src.dspy_meme_gen.agents.router import RouterAgent
from src.dspy_meme_gen.agents.trend_scanner import TrendScanningAgent
from src.dspy_meme_gen.agents.format_selector import FormatSelectionAgent
from src.dspy_meme_gen.agents.prompt_generator import PromptGenerationAgent
from src.dspy_meme_gen.agents.image_renderer import ImageRenderingAgent
from src.dspy_meme_gen.agents.verification import (
    FactualityAgent,
    InstructionFollowingAgent,
    AppropriatenessAgent
)
from src.dspy_meme_gen.agents.scoring import ScoringAgent
from src.dspy_meme_gen.agents.refinement import RefinementLoopAgent

from tests.utils.mocks import (
    MockOpenAI,
    MockCloudinary,
    MockRedis
)

@pytest.fixture
async def pipeline(
    load_test_db: AsyncSession,
    mock_openai: MockOpenAI,
    mock_cloudinary: MockCloudinary,
    mock_redis: MockRedis
) -> MemeGenerationPipeline:
    """Fixture for meme generation pipeline.
    
    Args:
        load_test_db: Database session with performance monitoring
        mock_openai: Mock OpenAI client
        mock_cloudinary: Mock Cloudinary client
        mock_redis: Mock Redis client
        
    Returns:
        MemeGenerationPipeline: Configured pipeline instance
    """
    # Initialize pipeline components
    router = RouterAgent()
    trend_scanner = TrendScanningAgent()
    format_selector = FormatSelectionAgent()
    prompt_generator = PromptGenerationAgent()
    image_renderer = ImageRenderingAgent()
    factuality_agent = FactualityAgent()
    instruction_agent = InstructionFollowingAgent()
    appropriateness_agent = AppropriatenessAgent()
    scorer = ScoringAgent()
    refinement_agent = RefinementLoopAgent()
    
    # Create pipeline instance
    pipeline = MemeGenerationPipeline(
        router=router,
        trend_scanner=trend_scanner,
        format_selector=format_selector,
        prompt_generator=prompt_generator,
        image_renderer=image_renderer,
        factuality_agent=factuality_agent,
        instruction_agent=instruction_agent,
        appropriateness_agent=appropriateness_agent,
        scorer=scorer,
        refinement_agent=refinement_agent
    )
    
    return pipeline

@pytest.mark.asyncio
async def test_basic_meme_generation(
    pipeline: MemeGenerationPipeline,
    load_test_db: AsyncSession
) -> None:
    """Test basic meme generation flow.
    
    Args:
        pipeline: Meme generation pipeline
        load_test_db: Database session with performance monitoring
    """
    # Create test template
    repository = MemeTemplateRepository(load_test_db)
    template = MemeTemplate(
        name="test_template",
        description="Test template",
        format_type="image",
        structure={"text": ["top", "bottom"]},
        popularity_score=1.0
    )
    await repository.create(template)
    
    # Generate meme
    result = await pipeline.generate(
        topic="python programming",
        style="minimalist"
    )
    
    # Verify result structure
    assert result["status"] == "success"
    assert "meme" in result
    assert "template" in result["meme"]
    assert "image_url" in result["meme"]
    assert "caption" in result["meme"]
    assert "alternatives" in result

@pytest.mark.asyncio
async def test_meme_generation_with_constraints(
    pipeline: MemeGenerationPipeline,
    load_test_db: AsyncSession
) -> None:
    """Test meme generation with specific constraints.
    
    Args:
        pipeline: Meme generation pipeline
        load_test_db: Database session with performance monitoring
    """
    # Create test template
    repository = MemeTemplateRepository(load_test_db)
    template = MemeTemplate(
        name="test_template",
        description="Test template",
        format_type="image",
        structure={"text": ["top", "bottom"]},
        popularity_score=1.0
    )
    await repository.create(template)
    
    # Generate meme with constraints
    result = await pipeline.generate(
        topic="python programming",
        style="minimalist",
        constraints={
            "aspect_ratio": "1:1",
            "max_text_length": 50,
            "style_requirements": ["clean", "modern"]
        }
    )
    
    # Verify constraints were followed
    assert result["status"] == "success"
    assert result["meme"]["image_url"].endswith(".jpg")
    assert len(result["meme"]["caption"]) <= 50

@pytest.mark.asyncio
async def test_meme_generation_with_verification(
    pipeline: MemeGenerationPipeline,
    load_test_db: AsyncSession
) -> None:
    """Test meme generation with content verification.
    
    Args:
        pipeline: Meme generation pipeline
        load_test_db: Database session with performance monitoring
    """
    # Create test template
    repository = MemeTemplateRepository(load_test_db)
    template = MemeTemplate(
        name="test_template",
        description="Test template",
        format_type="image",
        structure={"text": ["top", "bottom"]},
        popularity_score=1.0
    )
    await repository.create(template)
    
    # Generate meme with verification
    result = await pipeline.generate(
        topic="python programming",
        style="minimalist",
        verification_needs={
            "factuality": True,
            "instructions": True,
            "appropriateness": True
        }
    )
    
    # Verify content checks were performed
    assert result["status"] == "success"
    assert result["verification_results"]["is_factual"]
    assert result["verification_results"]["constraints_met"]
    assert result["verification_results"]["is_appropriate"]

@pytest.mark.asyncio
async def test_meme_generation_with_trends(
    pipeline: MemeGenerationPipeline,
    load_test_db: AsyncSession
) -> None:
    """Test meme generation using trend data.
    
    Args:
        pipeline: Meme generation pipeline
        load_test_db: Database session with performance monitoring
    """
    # Create test template
    repository = MemeTemplateRepository(load_test_db)
    template = MemeTemplate(
        name="test_template",
        description="Test template",
        format_type="image",
        structure={"text": ["top", "bottom"]},
        popularity_score=1.0
    )
    await repository.create(template)
    
    # Generate meme with trend analysis
    result = await pipeline.generate(
        topic="python programming",
        style="minimalist",
        use_trends=True
    )
    
    # Verify trend data was incorporated
    assert result["status"] == "success"
    assert "trend_analysis" in result
    assert "relevance_score" in result["trend_analysis"]
    assert result["trend_analysis"]["relevance_score"] > 0.5

@pytest.mark.asyncio
async def test_meme_generation_with_refinement(
    pipeline: MemeGenerationPipeline,
    load_test_db: AsyncSession
) -> None:
    """Test meme generation with quality refinement.
    
    Args:
        pipeline: Meme generation pipeline
        load_test_db: Database session with performance monitoring
    """
    # Create test template
    repository = MemeTemplateRepository(load_test_db)
    template = MemeTemplate(
        name="test_template",
        description="Test template",
        format_type="image",
        structure={"text": ["top", "bottom"]},
        popularity_score=1.0
    )
    await repository.create(template)
    
    # Generate meme with refinement
    result = await pipeline.generate(
        topic="python programming",
        style="minimalist",
        quality_threshold=0.8
    )
    
    # Verify refinement improved quality
    assert result["status"] == "success"
    assert result["meme"]["final_score"] >= 0.8
    assert "refinement_history" in result
    assert len(result["refinement_history"]) > 0

@pytest.mark.asyncio
async def test_error_handling(
    pipeline: MemeGenerationPipeline,
    load_test_db: AsyncSession
) -> None:
    """Test error handling in the pipeline.
    
    Args:
        pipeline: Meme generation pipeline
        load_test_db: Database session with performance monitoring
    """
    # Test with invalid topic
    result = await pipeline.generate(
        topic="",  # Empty topic should fail
        style="minimalist"
    )
    assert result["status"] == "error"
    assert "Invalid topic" in result["error"]
    
    # Test with invalid style
    result = await pipeline.generate(
        topic="python programming",
        style="nonexistent_style"
    )
    assert result["status"] == "error"
    assert "Invalid style" in result["error"]
    
    # Test with inappropriate content
    result = await pipeline.generate(
        topic="inappropriate content",
        style="minimalist"
    )
    assert result["status"] == "rejected"
    assert "Content deemed inappropriate" in result["reason"] 