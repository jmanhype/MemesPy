"""Repository pattern implementation for database operations."""

from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from dspy_meme_gen.exceptions.database import (
    DatabaseError,
    EntityNotFoundError,
    InvalidDataError,
)
from dspy_meme_gen.models.base import Base
from dspy_meme_gen.models.meme import (
    GeneratedMeme,
    MemeTemplate,
    MemeTrendAssociation,
    TrendingTopic,
    UserFeedback,
)
from dspy_meme_gen.models.query_optimizer import QueryOptimizer

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository class with common CRUD operations.

    Args:
        model: SQLAlchemy model class
    """

    def __init__(self, model: Type[ModelType]):
        self.model = model
        self.query_optimizer = QueryOptimizer()

    async def create(self, db: AsyncSession, data: Dict[str, Any]) -> ModelType:
        """Create a new entity.

        Args:
            db: Database session
            data: Entity data

        Returns:
            ModelType: Created entity

        Raises:
            InvalidDataError: If data is invalid
            DatabaseError: If database operation fails
        """
        try:
            entity = self.model(**data)
            db.add(entity)
            await db.commit()
            await db.refresh(entity)
            return entity
        except ValueError as e:
            raise InvalidDataError(str(e))
        except Exception as e:
            raise DatabaseError(f"Failed to create {self.model.__name__}: {str(e)}")

    async def get(
        self,
        db: AsyncSession,
        id: Union[int, str],
        *,
        select_fields: Optional[List[str]] = None,
        join_loads: Optional[List[str]] = None,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
    ) -> Optional[ModelType]:
        """Get entity by ID with optimization options.

        Args:
            db: Database session
            id: Entity ID
            select_fields: Optional list of fields to select
            join_loads: Optional list of relationships to join load
            cache_key: Optional cache key
            cache_ttl: Optional cache TTL in seconds

        Returns:
            Optional[ModelType]: Found entity or None

        Raises:
            DatabaseError: If database operation fails
        """
        try:
            query = select(self.model).filter(self.model.id == id)

            # Apply optimizations
            query = self.query_optimizer.optimize_query(
                query,
                self.model,
                select_fields=select_fields,
                join_loads=join_loads,
            )

            # Execute with caching
            result = await self.query_optimizer.execute_optimized(
                db,
                query,
                cache_key=cache_key,
                cache_ttl=cache_ttl,
            )

            return result[0] if result else None
        except Exception as e:
            raise DatabaseError(f"Failed to get {self.model.__name__}: {str(e)}")

    async def get_or_404(self, db: AsyncSession, id: Union[int, str]) -> ModelType:
        """Get entity by ID or raise 404.

        Args:
            db: Database session
            id: Entity ID

        Returns:
            ModelType: Found entity

        Raises:
            EntityNotFoundError: If entity not found
            DatabaseError: If database operation fails
        """
        entity = await self.get(db, id)
        if not entity:
            raise EntityNotFoundError(f"{self.model.__name__} with id {id} not found")
        return entity

    async def list(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
        select_fields: Optional[List[str]] = None,
        join_loads: Optional[List[str]] = None,
        select_loads: Optional[List[str]] = None,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
        **filters: Any,
    ) -> List[ModelType]:
        """List entities with optimization options.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            select_fields: Optional list of fields to select
            join_loads: Optional list of relationships to join load
            select_loads: Optional list of relationships to select load
            cache_key: Optional cache key
            cache_ttl: Optional cache TTL in seconds
            **filters: Filter conditions

        Returns:
            List[ModelType]: List of entities

        Raises:
            DatabaseError: If database operation fails
        """
        try:
            # Build base query
            query = select(self.model)

            # Apply filters
            for field, value in filters.items():
                if hasattr(self.model, field):
                    query = query.filter(getattr(self.model, field) == value)

            # Apply pagination
            query = query.offset(skip).limit(limit)

            # Apply optimizations
            query = self.query_optimizer.optimize_query(
                query,
                self.model,
                select_fields=select_fields,
                join_loads=join_loads,
                select_loads=select_loads,
            )

            # Execute with caching
            return await self.query_optimizer.execute_optimized(
                db,
                query,
                cache_key=cache_key,
                cache_ttl=cache_ttl,
            )
        except Exception as e:
            raise DatabaseError(f"Failed to list {self.model.__name__}: {str(e)}")

    async def update(
        self, db: AsyncSession, id: Union[int, str], data: Dict[str, Any]
    ) -> ModelType:
        """Update entity by ID.

        Args:
            db: Database session
            id: Entity ID
            data: Update data

        Returns:
            ModelType: Updated entity

        Raises:
            EntityNotFoundError: If entity not found
            InvalidDataError: If data is invalid
            DatabaseError: If database operation fails
        """
        try:
            entity = await self.get_or_404(db, id)
            for field, value in data.items():
                if hasattr(entity, field):
                    setattr(entity, field, value)
            await db.commit()
            await db.refresh(entity)
            return entity
        except EntityNotFoundError:
            raise
        except ValueError as e:
            raise InvalidDataError(str(e))
        except Exception as e:
            raise DatabaseError(f"Failed to update {self.model.__name__}: {str(e)}")

    async def delete(self, db: AsyncSession, id: Union[int, str]) -> None:
        """Delete entity by ID.

        Args:
            db: Database session
            id: Entity ID

        Raises:
            EntityNotFoundError: If entity not found
            DatabaseError: If database operation fails
        """
        try:
            entity = await self.get_or_404(db, id)
            await db.delete(entity)
            await db.commit()
        except EntityNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to delete {self.model.__name__}: {str(e)}")


class MemeTemplateRepository(BaseRepository[MemeTemplate]):
    """Repository for meme template operations."""

    def __init__(self):
        super().__init__(MemeTemplate)

    async def get_by_name(self, db: AsyncSession, name: str) -> Optional[MemeTemplate]:
        """Get template by name.

        Args:
            db: Database session
            name: Template name

        Returns:
            Optional[MemeTemplate]: Found template or None
        """
        try:
            result = await db.execute(select(self.model).filter(self.model.name == name))
            return result.scalar_one_or_none()
        except Exception as e:
            raise DatabaseError(f"Failed to get template by name: {str(e)}")

    async def list_by_format_type(
        self, db: AsyncSession, format_type: str, *, skip: int = 0, limit: int = 100
    ) -> List[MemeTemplate]:
        """List templates by format type.

        Args:
            db: Database session
            format_type: Format type to filter by
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List[MemeTemplate]: List of templates
        """
        try:
            query = (
                select(self.model)
                .filter(self.model.format_type == format_type)
                .offset(skip)
                .limit(limit)
            )
            result = await db.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise DatabaseError(f"Failed to list templates by format type: {str(e)}")


class GeneratedMemeRepository(BaseRepository[GeneratedMeme]):
    """Repository for generated meme operations."""

    def __init__(self):
        super().__init__(GeneratedMeme)

    async def list_by_template(
        self, db: AsyncSession, template_id: int, *, skip: int = 0, limit: int = 100
    ) -> List[GeneratedMeme]:
        """List memes by template.

        Args:
            db: Database session
            template_id: Template ID to filter by
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List[GeneratedMeme]: List of memes
        """
        try:
            query = (
                select(self.model)
                .filter(self.model.template_id == template_id)
                .offset(skip)
                .limit(limit)
            )
            result = await db.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise DatabaseError(f"Failed to list memes by template: {str(e)}")


class TrendingTopicRepository(BaseRepository[TrendingTopic]):
    """Repository for trending topic operations."""

    def __init__(self):
        super().__init__(TrendingTopic)

    async def list_by_source(
        self, db: AsyncSession, source: str, *, skip: int = 0, limit: int = 100
    ) -> List[TrendingTopic]:
        """List topics by source.

        Args:
            db: Database session
            source: Source to filter by
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List[TrendingTopic]: List of topics
        """
        try:
            query = select(self.model).filter(self.model.source == source).offset(skip).limit(limit)
            result = await db.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise DatabaseError(f"Failed to list topics by source: {str(e)}")


class UserFeedbackRepository(BaseRepository[UserFeedback]):
    """Repository for user feedback operations."""

    def __init__(self):
        super().__init__(UserFeedback)

    async def list_by_meme(
        self, db: AsyncSession, meme_id: int, *, skip: int = 0, limit: int = 100
    ) -> List[UserFeedback]:
        """List feedback by meme.

        Args:
            db: Database session
            meme_id: Meme ID to filter by
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List[UserFeedback]: List of feedback
        """
        try:
            query = (
                select(self.model).filter(self.model.meme_id == meme_id).offset(skip).limit(limit)
            )
            result = await db.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise DatabaseError(f"Failed to list feedback by meme: {str(e)}")


# Create repository instances
meme_template_repo = MemeTemplateRepository()
generated_meme_repo = GeneratedMemeRepository()
trending_topic_repo = TrendingTopicRepository()
user_feedback_repo = UserFeedbackRepository()
