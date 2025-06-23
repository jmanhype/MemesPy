"""Database connection module."""

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from ..config.config import settings

# Create database engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base model class
Base = declarative_base()

def get_session() -> Generator[Session, None, None]:
    """
    Get a database session dependency.
    
    Yields:
        Session: SQLAlchemy database session
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_async_engine():
    """
    Get async engine for privacy tasks.
    
    Returns:
        AsyncEngine: SQLAlchemy async engine
    """
    # Convert sync database URL to async
    if settings.database_url.startswith("sqlite"):
        async_url = settings.database_url.replace("sqlite://", "sqlite+aiosqlite://")
    elif settings.database_url.startswith("postgresql"):
        async_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
    else:
        # Default to sync engine wrapped in async
        async_url = settings.database_url
    
    return create_async_engine(
        async_url,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False} if async_url.startswith("sqlite") else {}
    )


async def get_db():
    """
    Get async database session dependency.
    
    Yields:
        AsyncSession: SQLAlchemy async database session
    """
    async_engine = get_async_engine()
    async_session_maker = sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close() 