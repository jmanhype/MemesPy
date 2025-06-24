"""Database session management."""

from typing import AsyncGenerator, Generator

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from ..models.connection import db_manager


def get_db() -> Generator[Session, None, None]:
    """Get a database session.

    Yields:
        Session: Database session

    Example:
        ```python
        with get_db() as db:
            db.query(User).all()
        ```
    """
    with db_manager.get_session() as session:
        yield session


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session.

    Yields:
        AsyncSession: Async database session

    Example:
        ```python
        async with get_async_db() as db:
            result = await db.execute(select(User))
            users = result.scalars().all()
        ```
    """
    async with db_manager.get_async_session() as session:
        yield session


def init_db() -> None:
    """Initialize database schema."""
    db_manager.init_db()


async def init_async_db() -> None:
    """Initialize database schema asynchronously."""
    await db_manager.init_db()
