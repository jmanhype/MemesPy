"""Database connection management."""
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Optional

from sqlalchemy import create_engine, event, exc
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from ..config.config import get_settings
from ..exceptions.database import DatabaseConnectionError
from ..models.base import Base

settings = get_settings()

class DatabaseConnectionManager:
    """Database connection manager.
    
    Handles connection pooling, lifecycle, and health checks.
    """

    def __init__(self):
        """Initialize connection manager."""
        self._sync_engine: Optional[Engine] = None
        self._async_engine: Optional[AsyncEngine] = None
        self._sync_session_factory: Optional[sessionmaker] = None
        self._async_session_factory: Optional[async_sessionmaker] = None

    @property
    def sync_engine(self) -> Engine:
        """Get synchronous engine, creating if needed.
        
        Returns:
            Engine: SQLAlchemy engine
        """
        if not self._sync_engine:
            self._sync_engine = create_engine(
                str(settings.database.DATABASE_URL),
                poolclass=QueuePool,
                pool_size=settings.database.DATABASE_POOL_SIZE,
                max_overflow=settings.database.DATABASE_MAX_OVERFLOW,
                pool_timeout=settings.database.DATABASE_POOL_TIMEOUT,
                pool_recycle=settings.database.DATABASE_POOL_RECYCLE,
                pool_pre_ping=True,  # Enable connection health checks
                echo=settings.database.DATABASE_ECHO,
            )
            
            # Set up engine event listeners
            event.listen(self._sync_engine, "connect", self._on_connect)
            event.listen(self._sync_engine, "checkout", self._on_checkout)
            
        return self._sync_engine

    @property
    def async_engine(self) -> AsyncEngine:
        """Get asynchronous engine, creating if needed.
        
        Returns:
            AsyncEngine: SQLAlchemy async engine
        """
        if not self._async_engine:
            self._async_engine = create_async_engine(
                str(settings.database.DATABASE_URL).replace(
                    "postgresql://",
                    "postgresql+asyncpg://"
                ),
                pool_size=settings.database.DATABASE_POOL_SIZE,
                max_overflow=settings.database.DATABASE_MAX_OVERFLOW,
                pool_timeout=settings.database.DATABASE_POOL_TIMEOUT,
                pool_recycle=settings.database.DATABASE_POOL_RECYCLE,
                pool_pre_ping=True,  # Enable connection health checks
                echo=settings.database.DATABASE_ECHO,
            )
        return self._async_engine

    @property
    def sync_session_factory(self) -> sessionmaker:
        """Get synchronous session factory, creating if needed.
        
        Returns:
            sessionmaker: SQLAlchemy session factory
        """
        if not self._sync_session_factory:
            self._sync_session_factory = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.sync_engine,
            )
        return self._sync_session_factory

    @property
    def async_session_factory(self) -> async_sessionmaker:
        """Get asynchronous session factory, creating if needed.
        
        Returns:
            async_sessionmaker: SQLAlchemy async session factory
        """
        if not self._async_session_factory:
            self._async_session_factory = async_sessionmaker(
                self.async_engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
            )
        return self._async_session_factory

    def _on_connect(self, dbapi_connection, connection_record):
        """Handle new connections.
        
        Args:
            dbapi_connection: DBAPI connection
            connection_record: Connection pool record
        """
        # Set session parameters
        cursor = dbapi_connection.cursor()
        cursor.execute("SET timezone TO 'UTC'")
        cursor.close()

    def _on_checkout(self, dbapi_connection, connection_record, connection_proxy):
        """Handle connection checkout from pool.
        
        Args:
            dbapi_connection: DBAPI connection
            connection_record: Connection pool record
            connection_proxy: Connection proxy
            
        Raises:
            DatabaseConnectionError: If connection is invalid
        """
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        except exc.DBAPIError:
            raise DatabaseConnectionError("Database connection is invalid")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a synchronous database session.
        
        Yields:
            Session: Database session
            
        Example:
            ```python
            with db_manager.get_session() as session:
                session.query(User).all()
            ```
        """
        session = self.sync_session_factory()
        try:
            yield session
        finally:
            session.close()

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an asynchronous database session.
        
        Yields:
            AsyncSession: Database session
            
        Example:
            ```python
            async with db_manager.get_async_session() as session:
                result = await session.execute(select(User))
                users = result.scalars().all()
            ```
        """
        session = self.async_session_factory()
        try:
            yield session
        finally:
            await session.close()

    async def init_db(self) -> None:
        """Initialize database schema."""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def check_connection(self) -> bool:
        """Check database connection health.
        
        Returns:
            bool: True if connection is healthy
            
        Raises:
            DatabaseConnectionError: If connection check fails
        """
        try:
            async with self.get_async_session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            raise DatabaseConnectionError(f"Database connection check failed: {str(e)}")

# Global connection manager instance
db_manager = DatabaseConnectionManager() 