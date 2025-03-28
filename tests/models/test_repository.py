"""Tests for repository implementations."""
from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from dspy_meme_gen.exceptions.database import EntityNotFoundError, InvalidDataError
from dspy_meme_gen.models.base import Base
from dspy_meme_gen.models.meme import (
    GeneratedMeme,
    MemeTemplate,
    MemeTrendAssociation,
    TrendingTopic,
    UserFeedback,
)
from dspy_meme_gen.models.repository import (
    GeneratedMemeRepository,
    MemeTemplateRepository,
    TrendingTopicRepository,
    UserFeedbackRepository,
)

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

@pytest.fixture(scope="function")
async def async_engine():
    """Create a test async database engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=True
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine

@pytest.fixture(scope="function")
async def async_session(async_engine) -> AsyncSession:
    """Create a test async database session."""
    async_session = sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    async with async_session() as session:
        yield session

@pytest.fixture(scope="function")
async def meme_template_repo() -> MemeTemplateRepository:
    """Create a meme template repository."""
    return MemeTemplateRepository()

@pytest.fixture(scope="function")
async def generated_meme_repo() -> GeneratedMemeRepository:
    """Create a generated meme repository."""
    return GeneratedMemeRepository()

@pytest.fixture(scope="function")
async def trending_topic_repo() -> TrendingTopicRepository:
    """Create a trending topic repository."""
    return TrendingTopicRepository()

@pytest.fixture(scope="function")
async def user_feedback_repo() -> UserFeedbackRepository:
    """Create a user feedback repository."""
    return UserFeedbackRepository()

@pytest.mark.asyncio
async def test_create_meme_template(
    async_session: AsyncSession,
    meme_template_repo: MemeTemplateRepository
):
    """Test creating a meme template."""
    template_data = {
        "name": "Test Template",
        "description": "A test template",
        "format_type": "image",
        "example_url": "http://example.com/test.jpg",
        "structure": {"text_positions": ["top", "bottom"]},
        "popularity_score": 0.8
    }
    
    template = await meme_template_repo.create(async_session, template_data)
    assert template.name == "Test Template"
    assert template.format_type == "image"
    assert template.popularity_score == 0.8

@pytest.mark.asyncio
async def test_get_meme_template(
    async_session: AsyncSession,
    meme_template_repo: MemeTemplateRepository
):
    """Test getting a meme template."""
    # Create template
    template_data = {
        "name": "Test Template",
        "format_type": "image"
    }
    template = await meme_template_repo.create(async_session, template_data)
    
    # Get template
    retrieved = await meme_template_repo.get(async_session, template.id)
    assert retrieved is not None
    assert retrieved.name == "Test Template"

@pytest.mark.asyncio
async def test_get_meme_template_404(
    async_session: AsyncSession,
    meme_template_repo: MemeTemplateRepository
):
    """Test getting a non-existent meme template."""
    with pytest.raises(EntityNotFoundError):
        await meme_template_repo.get_or_404(async_session, 999)

@pytest.mark.asyncio
async def test_list_meme_templates(
    async_session: AsyncSession,
    meme_template_repo: MemeTemplateRepository
):
    """Test listing meme templates."""
    # Create templates
    templates = [
        {"name": f"Template {i}", "format_type": "image"}
        for i in range(3)
    ]
    for template_data in templates:
        await meme_template_repo.create(async_session, template_data)
    
    # List templates
    templates = await meme_template_repo.list(async_session, limit=10)
    assert len(templates) == 3

@pytest.mark.asyncio
async def test_update_meme_template(
    async_session: AsyncSession,
    meme_template_repo: MemeTemplateRepository
):
    """Test updating a meme template."""
    # Create template
    template = await meme_template_repo.create(
        async_session,
        {"name": "Test Template", "format_type": "image"}
    )
    
    # Update template
    updated = await meme_template_repo.update(
        async_session,
        template.id,
        {"name": "Updated Template"}
    )
    assert updated.name == "Updated Template"

@pytest.mark.asyncio
async def test_delete_meme_template(
    async_session: AsyncSession,
    meme_template_repo: MemeTemplateRepository
):
    """Test deleting a meme template."""
    # Create template
    template = await meme_template_repo.create(
        async_session,
        {"name": "Test Template", "format_type": "image"}
    )
    
    # Delete template
    await meme_template_repo.delete(async_session, template.id)
    
    # Verify deletion
    with pytest.raises(EntityNotFoundError):
        await meme_template_repo.get_or_404(async_session, template.id)

@pytest.mark.asyncio
async def test_create_generated_meme(
    async_session: AsyncSession,
    generated_meme_repo: GeneratedMemeRepository,
    meme_template_repo: MemeTemplateRepository
):
    """Test creating a generated meme."""
    # Create template
    template = await meme_template_repo.create(
        async_session,
        {"name": "Test Template", "format_type": "image"}
    )
    
    # Create meme
    meme_data = {
        "template_id": template.id,
        "topic": "test topic",
        "caption": "Test caption",
        "image_url": "http://example.com/meme.jpg",
        "score": 0.9
    }
    meme = await generated_meme_repo.create(async_session, meme_data)
    assert meme.topic == "test topic"
    assert meme.score == 0.9

@pytest.mark.asyncio
async def test_create_trending_topic(
    async_session: AsyncSession,
    trending_topic_repo: TrendingTopicRepository
):
    """Test creating a trending topic."""
    topic_data = {
        "topic": "Test Topic",
        "source": "twitter",
        "relevance_score": 0.7
    }
    topic = await trending_topic_repo.create(async_session, topic_data)
    assert topic.topic == "Test Topic"
    assert topic.source == "twitter"

@pytest.mark.asyncio
async def test_create_user_feedback(
    async_session: AsyncSession,
    user_feedback_repo: UserFeedbackRepository,
    generated_meme_repo: GeneratedMemeRepository,
    meme_template_repo: MemeTemplateRepository
):
    """Test creating user feedback."""
    # Create template and meme
    template = await meme_template_repo.create(
        async_session,
        {"name": "Test Template", "format_type": "image"}
    )
    meme = await generated_meme_repo.create(
        async_session,
        {
            "template_id": template.id,
            "topic": "test topic",
            "image_url": "http://example.com/meme.jpg"
        }
    )
    
    # Create feedback
    feedback_data = {
        "meme_id": meme.id,
        "rating": 5,
        "comment": "Great meme!",
        "feedback_type": "rating"
    }
    feedback = await user_feedback_repo.create(async_session, feedback_data)
    assert feedback.rating == 5
    assert feedback.comment == "Great meme!"

@pytest.mark.asyncio
async def test_list_by_format_type(
    async_session: AsyncSession,
    meme_template_repo: MemeTemplateRepository
):
    """Test listing templates by format type."""
    # Create templates
    for i in range(3):
        await meme_template_repo.create(
            async_session,
            {"name": f"Template {i}", "format_type": "image"}
        )
    await meme_template_repo.create(
        async_session,
        {"name": "Video Template", "format_type": "video"}
    )
    
    # List image templates
    templates = await meme_template_repo.list_by_format_type(
        async_session,
        "image"
    )
    assert len(templates) == 3
    assert all(t.format_type == "image" for t in templates)

@pytest.mark.asyncio
async def test_list_by_template(
    async_session: AsyncSession,
    generated_meme_repo: GeneratedMemeRepository,
    meme_template_repo: MemeTemplateRepository
):
    """Test listing memes by template."""
    # Create template
    template = await meme_template_repo.create(
        async_session,
        {"name": "Test Template", "format_type": "image"}
    )
    
    # Create memes
    for i in range(3):
        await generated_meme_repo.create(
            async_session,
            {
                "template_id": template.id,
                "topic": f"topic {i}",
                "image_url": "http://example.com/meme.jpg"
            }
        )
    
    # List memes
    memes = await generated_meme_repo.list_by_template(
        async_session,
        template.id
    )
    assert len(memes) == 3
    assert all(m.template_id == template.id for m in memes)

@pytest.mark.asyncio
async def test_list_by_source(
    async_session: AsyncSession,
    trending_topic_repo: TrendingTopicRepository
):
    """Test listing topics by source."""
    # Create topics
    for i in range(3):
        await trending_topic_repo.create(
            async_session,
            {
                "topic": f"Topic {i}",
                "source": "twitter",
                "relevance_score": 0.7
            }
        )
    await trending_topic_repo.create(
        async_session,
        {
            "topic": "Reddit Topic",
            "source": "reddit",
            "relevance_score": 0.7
        }
    )
    
    # List twitter topics
    topics = await trending_topic_repo.list_by_source(
        async_session,
        "twitter"
    )
    assert len(topics) == 3
    assert all(t.source == "twitter" for t in topics)

@pytest.mark.asyncio
async def test_list_by_meme(
    async_session: AsyncSession,
    user_feedback_repo: UserFeedbackRepository,
    generated_meme_repo: GeneratedMemeRepository,
    meme_template_repo: MemeTemplateRepository
):
    """Test listing feedback by meme."""
    # Create template and meme
    template = await meme_template_repo.create(
        async_session,
        {"name": "Test Template", "format_type": "image"}
    )
    meme = await generated_meme_repo.create(
        async_session,
        {
            "template_id": template.id,
            "topic": "test topic",
            "image_url": "http://example.com/meme.jpg"
        }
    )
    
    # Create feedback
    for i in range(3):
        await user_feedback_repo.create(
            async_session,
            {
                "meme_id": meme.id,
                "rating": i + 3,
                "comment": f"Feedback {i}",
                "feedback_type": "rating"
            }
        )
    
    # List feedback
    feedback = await user_feedback_repo.list_by_meme(
        async_session,
        meme.id
    )
    assert len(feedback) == 3
    assert all(f.meme_id == meme.id for f in feedback) 