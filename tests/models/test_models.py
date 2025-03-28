"""Tests for database models."""
from datetime import datetime
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from dspy_meme_gen.models.base import Base
from dspy_meme_gen.models.meme import (
    GeneratedMeme,
    MemeTemplate,
    MemeTrendAssociation,
    TrendingTopic,
    UserFeedback,
)

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

@pytest.fixture(scope="function")
def db_engine():
    """Create a test database engine."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine

@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create a test database session."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture(scope="function")
def meme_template(db_session: Session) -> MemeTemplate:
    """Create a test meme template."""
    template = MemeTemplate(
        name="Test Template",
        description="A test template",
        format_type="image",
        example_url="http://example.com/test.jpg",
        structure={"text_positions": ["top", "bottom"]},
        popularity_score=0.8
    )
    db_session.add(template)
    db_session.commit()
    return template

@pytest.fixture(scope="function")
def generated_meme(db_session: Session, meme_template: MemeTemplate) -> GeneratedMeme:
    """Create a test generated meme."""
    meme = GeneratedMeme(
        template_id=meme_template.id,
        topic="test topic",
        caption="Test caption",
        image_prompt="A test image prompt",
        image_url="http://example.com/meme.jpg",
        score=0.9,
        user_request="Generate a test meme",
        feedback={"rating": 5}
    )
    db_session.add(meme)
    db_session.commit()
    return meme

@pytest.fixture(scope="function")
def trending_topic(db_session: Session) -> TrendingTopic:
    """Create a test trending topic."""
    topic = TrendingTopic(
        topic="Test Topic",
        source="twitter",
        relevance_score=0.7,
        metadata={"hashtags": ["test"]}
    )
    db_session.add(topic)
    db_session.commit()
    return topic

@pytest.fixture(scope="function")
def user_feedback(db_session: Session, generated_meme: GeneratedMeme) -> UserFeedback:
    """Create a test user feedback."""
    feedback = UserFeedback(
        meme_id=generated_meme.id,
        rating=5,
        comment="Great meme!",
        feedback_type="rating",
        metadata={"source": "web"}
    )
    db_session.add(feedback)
    db_session.commit()
    return feedback

def test_meme_template_creation(meme_template: MemeTemplate):
    """Test creating a meme template."""
    assert meme_template.name == "Test Template"
    assert meme_template.format_type == "image"
    assert meme_template.popularity_score == 0.8
    assert isinstance(meme_template.created_at, datetime)

def test_generated_meme_creation(generated_meme: GeneratedMeme, meme_template: MemeTemplate):
    """Test creating a generated meme."""
    assert generated_meme.template_id == meme_template.id
    assert generated_meme.topic == "test topic"
    assert generated_meme.score == 0.9
    assert isinstance(generated_meme.created_at, datetime)

def test_trending_topic_creation(trending_topic: TrendingTopic):
    """Test creating a trending topic."""
    assert trending_topic.topic == "Test Topic"
    assert trending_topic.source == "twitter"
    assert trending_topic.relevance_score == 0.7
    assert isinstance(trending_topic.created_at, datetime)

def test_user_feedback_creation(user_feedback: UserFeedback, generated_meme: GeneratedMeme):
    """Test creating user feedback."""
    assert user_feedback.meme_id == generated_meme.id
    assert user_feedback.rating == 5
    assert user_feedback.feedback_type == "rating"
    assert isinstance(user_feedback.created_at, datetime)

def test_meme_template_relationships(
    db_session: Session,
    meme_template: MemeTemplate,
    generated_meme: GeneratedMeme
):
    """Test meme template relationships."""
    assert len(meme_template.generated_memes) == 1
    assert meme_template.generated_memes[0].id == generated_meme.id

def test_generated_meme_relationships(
    db_session: Session,
    generated_meme: GeneratedMeme,
    meme_template: MemeTemplate,
    user_feedback: UserFeedback
):
    """Test generated meme relationships."""
    assert generated_meme.template.id == meme_template.id
    assert len(generated_meme.feedback) == 1
    assert generated_meme.feedback[0].id == user_feedback.id

def test_trending_topic_relationships(
    db_session: Session,
    trending_topic: TrendingTopic,
    generated_meme: GeneratedMeme
):
    """Test trending topic relationships."""
    # Create association
    association = MemeTrendAssociation(
        meme_id=generated_meme.id,
        trend_id=trending_topic.id,
        association_strength=0.8
    )
    db_session.add(association)
    db_session.commit()

    assert len(trending_topic.memes) == 1
    assert trending_topic.memes[0].id == generated_meme.id

def test_user_feedback_relationships(
    db_session: Session,
    user_feedback: UserFeedback,
    generated_meme: GeneratedMeme
):
    """Test user feedback relationships."""
    assert user_feedback.meme.id == generated_meme.id

def test_meme_template_to_dict(meme_template: MemeTemplate):
    """Test converting meme template to dictionary."""
    data = meme_template.to_dict()
    assert data["name"] == "Test Template"
    assert data["format_type"] == "image"
    assert data["popularity_score"] == 0.8

def test_generated_meme_to_dict(generated_meme: GeneratedMeme):
    """Test converting generated meme to dictionary."""
    data = generated_meme.to_dict()
    assert data["topic"] == "test topic"
    assert data["score"] == 0.9
    assert data["image_url"] == "http://example.com/meme.jpg"

def test_trending_topic_to_dict(trending_topic: TrendingTopic):
    """Test converting trending topic to dictionary."""
    data = trending_topic.to_dict()
    assert data["topic"] == "Test Topic"
    assert data["source"] == "twitter"
    assert data["relevance_score"] == 0.7

def test_user_feedback_to_dict(user_feedback: UserFeedback):
    """Test converting user feedback to dictionary."""
    data = user_feedback.to_dict()
    assert data["rating"] == 5
    assert data["feedback_type"] == "rating"
    assert data["comment"] == "Great meme!"

def test_meme_template_from_dict():
    """Test creating meme template from dictionary."""
    data = {
        "name": "New Template",
        "format_type": "video",
        "popularity_score": 0.5
    }
    template = MemeTemplate.from_dict(data)
    assert template.name == "New Template"
    assert template.format_type == "video"
    assert template.popularity_score == 0.5

def test_generated_meme_from_dict():
    """Test creating generated meme from dictionary."""
    data = {
        "topic": "new topic",
        "score": 0.7,
        "image_url": "http://example.com/new.jpg"
    }
    meme = GeneratedMeme.from_dict(data)
    assert meme.topic == "new topic"
    assert meme.score == 0.7
    assert meme.image_url == "http://example.com/new.jpg"

def test_trending_topic_from_dict():
    """Test creating trending topic from dictionary."""
    data = {
        "topic": "New Topic",
        "source": "reddit",
        "relevance_score": 0.6
    }
    topic = TrendingTopic.from_dict(data)
    assert topic.topic == "New Topic"
    assert topic.source == "reddit"
    assert topic.relevance_score == 0.6

def test_user_feedback_from_dict():
    """Test creating user feedback from dictionary."""
    data = {
        "rating": 4,
        "feedback_type": "comment",
        "comment": "Nice!"
    }
    feedback = UserFeedback.from_dict(data)
    assert feedback.rating == 4
    assert feedback.feedback_type == "comment"
    assert feedback.comment == "Nice!" 