"""Query optimization utilities for database operations."""

from typing import Any, List, Optional, Tuple, Type, TypeVar

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import (
    joinedload,
    selectinload,
    contains_eager,
    load_only,
    noload,
    Query,
)
from sqlalchemy.sql import Select

from dspy_meme_gen.models.base import Base
from dspy_meme_gen.cache.cache_manager import cache_manager

ModelType = TypeVar("ModelType", bound=Base)


class QueryOptimizer:
    """Query optimization utilities for database operations."""

    @staticmethod
    def optimize_query(query: Select, model: Type[ModelType], **options: Any) -> Select:
        """Optimize a query based on provided options.

        Args:
            query: Base query to optimize
            model: Model class
            **options: Optimization options
                - select_fields: List of fields to select
                - join_loads: List of relationships to join load
                - select_loads: List of relationships to select load
                - count_only: Return count only
                - group_by: Fields to group by

        Returns:
            Select: Optimized query
        """
        # Select specific fields if specified
        if select_fields := options.get("select_fields"):
            columns = [getattr(model, field) for field in select_fields]
            query = query.options(load_only(*columns))

        # Join load relationships
        if join_loads := options.get("join_loads"):
            for relationship in join_loads:
                query = query.options(joinedload(getattr(model, relationship)))

        # Select load relationships (N+1 optimization)
        if select_loads := options.get("select_loads"):
            for relationship in select_loads:
                query = query.options(selectinload(getattr(model, relationship)))

        # Count only optimization
        if options.get("count_only"):
            query = select(func.count()).select_from(query.subquery())

        # Group by
        if group_by := options.get("group_by"):
            query = query.group_by(*[getattr(model, field) for field in group_by])

        return query

    @staticmethod
    async def execute_optimized(
        db: AsyncSession,
        query: Select,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
    ) -> Any:
        """Execute an optimized query with optional caching.

        Args:
            db: Database session
            query: Query to execute
            cache_key: Optional cache key
            cache_ttl: Optional cache TTL in seconds

        Returns:
            Any: Query results
        """
        # Try to get from cache first
        if cache_key:
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

        # Execute query
        result = await db.execute(query)
        data = result.scalars().all()

        # Cache result if needed
        if cache_key and cache_ttl:
            await cache_manager.set(cache_key, data, ttl=cache_ttl)

        return data

    @staticmethod
    def paginate(
        query: Select,
        page: int = 1,
        per_page: int = 20,
    ) -> Tuple[Select, int]:
        """Add pagination to a query.

        Args:
            query: Query to paginate
            page: Page number (1-based)
            per_page: Items per page

        Returns:
            Tuple[Select, int]: Paginated query and total count
        """
        # Calculate offset
        offset = (page - 1) * per_page

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())

        # Add pagination
        query = query.offset(offset).limit(per_page)

        return query, count_query
