"""Tests for database layer functionality."""
from typing import AsyncGenerator, Generator
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from dspy_meme_gen.models.connection import DatabaseConnectionManager, db_manager
from dspy_meme_gen.models.meme import MemeTemplate
from dspy_meme_gen.models.repository import MemeTemplateRepository
from dspy_meme_gen.exceptions.database import DatabaseConnectionError, EntityNotFoundError

@pytest.fixture
def db_session(db_manager: DatabaseConnectionManager) -> Generator[Session, None, None]:
    """Fixture for database session.
    
    Args:
        db_manager: Database connection manager
        
    Yields:
        Session: Database session
    """
    with db_manager.get_session() as session:
        yield session

@pytest.fixture
async def async_db_session(db_manager: DatabaseConnectionManager) -> AsyncGenerator[AsyncSession, None]:
    """Fixture for async database session.
    
    Args:
        db_manager: Database connection manager
        
    Yields:
        AsyncSession: Async database session
    """
    async with db_manager.get_async_session() as session:
        yield session

@pytest.fixture
def meme_template_repository() -> MemeTemplateRepository:
    """Fixture for meme template repository.
    
    Returns:
        MemeTemplateRepository: Repository instance
    """
    return MemeTemplateRepository()

@pytest.mark.asyncio
async def test_connection_manager_initialization():
    """Test database connection manager initialization."""
    assert db_manager._sync_engine is None
    assert db_manager._async_engine is None
    
    # Access engines to initialize them
    _ = db_manager.sync_engine
    _ = db_manager.async_engine
    
    assert db_manager._sync_engine is not None
    assert db_manager._async_engine is not None

@pytest.mark.asyncio
async def test_connection_health_check():
    """Test database connection health check."""
    # Should not raise an exception
    assert await db_manager.check_connection() is True

@pytest.mark.asyncio
async def test_crud_operations(
    async_db_session: AsyncSession,
    meme_template_repository: MemeTemplateRepository
):
    """Test CRUD operations with query optimization.
    
    Args:
        async_db_session: Async database session
        meme_template_repository: Meme template repository
    """
    # Create test data
    template_data = {
        "name": "Test Template",
        "description": "A test template",
        "format_type": "image",
        "example_url": "https://example.com/test.jpg",
        "structure": {"text_positions": ["top", "bottom"]},
    }
    
    # Test create
    template = await meme_template_repository.create(async_db_session, template_data)
    assert template.name == template_data["name"]
    
    # Test get with optimization
    retrieved = await meme_template_repository.get(
        async_db_session,
        template.id,
        select_fields=["name", "format_type"],
        cache_key=f"template:{template.id}",
        cache_ttl=300,
    )
    assert retrieved is not None
    assert retrieved.name == template_data["name"]
    
    # Test list with optimization
    templates = await meme_template_repository.list(
        async_db_session,
        select_fields=["name", "format_type"],
        join_loads=["trends"],
        cache_key="templates:all",
        cache_ttl=300,
    )
    assert len(templates) > 0
    assert any(t.name == template_data["name"] for t in templates)
    
    # Test update
    update_data = {"description": "Updated description"}
    updated = await meme_template_repository.update(
        async_db_session,
        template.id,
        update_data
    )
    assert updated.description == update_data["description"]
    
    # Test delete
    await meme_template_repository.delete(async_db_session, template.id)
    
    # Verify deletion
    with pytest.raises(EntityNotFoundError):
        await meme_template_repository.get_or_404(async_db_session, template.id)

@pytest.mark.asyncio
async def test_query_optimization(
    async_db_session: AsyncSession,
    meme_template_repository: MemeTemplateRepository
):
    """Test query optimization features.
    
    Args:
        async_db_session: Async database session
        meme_template_repository: Meme template repository
    """
    # Create test data
    template_data = {
        "name": "Optimization Test Template",
        "description": "A template for testing query optimization",
        "format_type": "image",
        "example_url": "https://example.com/test.jpg",
        "structure": {"text_positions": ["top", "bottom"]},
    }
    
    template = await meme_template_repository.create(async_db_session, template_data)
    
    # Test selective field loading
    retrieved = await meme_template_repository.get(
        async_db_session,
        template.id,
        select_fields=["name", "format_type"]
    )
    assert retrieved is not None
    assert retrieved.name == template_data["name"]
    assert retrieved.format_type == template_data["format_type"]
    
    # Test relationship loading
    templates = await meme_template_repository.list(
        async_db_session,
        join_loads=["trends"],
        select_loads=["feedback"]
    )
    assert len(templates) > 0
    
    # Test caching
    cached = await meme_template_repository.get(
        async_db_session,
        template.id,
        cache_key=f"template:{template.id}",
        cache_ttl=300
    )
    assert cached is not None
    assert cached.name == template_data["name"]
    
    # Clean up
    await meme_template_repository.delete(async_db_session, template.id)

@pytest.mark.asyncio
async def test_error_handling(
    async_db_session: AsyncSession,
    meme_template_repository: MemeTemplateRepository
):
    """Test error handling in database operations.
    
    Args:
        async_db_session: Async database session
        meme_template_repository: Meme template repository
    """
    # Test invalid data
    with pytest.raises(EntityNotFoundError):
        await meme_template_repository.get_or_404(async_db_session, 99999)
    
    # Test invalid field in update
    template_data = {
        "name": "Error Test Template",
        "description": "A template for testing error handling",
        "format_type": "image",
        "example_url": "https://example.com/test.jpg",
        "structure": {"text_positions": ["top", "bottom"]},
    }
    
    template = await meme_template_repository.create(async_db_session, template_data)
    
    with pytest.raises(Exception):
        await meme_template_repository.update(
            async_db_session,
            template.id,
            {"invalid_field": "value"}
        )
    
    # Clean up
    await meme_template_repository.delete(async_db_session, template.id) 