"""Tests for database CLI commands."""

import os
from typing import AsyncGenerator, Generator, TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from typer.testing import CliRunner

from dspy_meme_gen.cli.db import app, get_alembic_config
from dspy_meme_gen.models.base import Base


@pytest.fixture
def mock_env_vars(monkeypatch: "MonkeyPatch") -> None:
    """Set up mock environment variables."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test_db")


@pytest.fixture
async def test_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    engine = create_async_engine(
        "postgresql+asyncpg://user:pass@localhost/test_db",
        echo=True
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        
    # Clean up
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


def test_get_alembic_config() -> None:
    """Test getting Alembic configuration."""
    # Test with default URL
    config = get_alembic_config()
    assert config is not None
    assert config.get_main_option("script_location") == "src/dspy_meme_gen/models/migrations"
    
    # Test with custom URL
    custom_url = "postgresql://custom@localhost/custom_db"
    config = get_alembic_config(custom_url)
    assert config.get_main_option("sqlalchemy.url") == custom_url


@pytest.mark.asyncio
async def test_upgrade_command(
    mock_env_vars: None,
    mocker: "MockerFixture",
    capsys: "CaptureFixture[str]"
) -> None:
    """Test database upgrade command."""
    # Mock Alembic command
    mock_upgrade = mocker.patch("alembic.command.upgrade")
    
    # Run upgrade command
    result = app.command()(upgrade)("head")
    
    # Check that upgrade was called
    mock_upgrade.assert_called_once()
    
    # Check output
    captured = capsys.readouterr()
    assert "Upgraded database to revision head" in captured.out


@pytest.mark.asyncio
async def test_downgrade_command(
    mock_env_vars: None,
    mocker: "MockerFixture",
    capsys: "CaptureFixture[str]"
) -> None:
    """Test database downgrade command."""
    # Mock Alembic command
    mock_downgrade = mocker.patch("alembic.command.downgrade")
    
    # Run downgrade command
    result = app.command()(downgrade)("base")
    
    # Check that downgrade was called
    mock_downgrade.assert_called_once()
    
    # Check output
    captured = capsys.readouterr()
    assert "Downgraded database to revision base" in captured.out


@pytest.mark.asyncio
async def test_seed_command(
    mock_env_vars: None,
    mocker: "MockerFixture",
    capsys: "CaptureFixture[str]"
) -> None:
    """Test database seed command."""
    # Mock seed_guidelines function
    mock_seed = mocker.patch(
        "dspy_meme_gen.models.seed_data.content_guidelines.seed_guidelines",
        new_callable=AsyncMock
    )
    
    # Run seed command
    result = app.command()(seed)()
    
    # Check that seed was called
    mock_seed.assert_called_once()
    
    # Check output
    captured = capsys.readouterr()
    assert "Successfully seeded database with content guidelines" in captured.out


@pytest.mark.asyncio
async def test_seed_command_error(
    mock_env_vars: None,
    mocker: "MockerFixture",
    capsys: "CaptureFixture[str]"
) -> None:
    """Test database seed command with error."""
    # Mock seed_guidelines function to raise an error
    mock_seed = mocker.patch(
        "dspy_meme_gen.models.seed_data.content_guidelines.seed_guidelines",
        new_callable=AsyncMock,
        side_effect=Exception("Test error")
    )
    
    # Run seed command and check for error
    with pytest.raises(SystemExit):
        result = app.command()(seed)()
    
    # Check output
    captured = capsys.readouterr()
    assert "Error seeding database: Test error" in captured.out


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


def test_init_db(cli_runner: CliRunner, mock_db_engine):
    """Test database initialization command."""
    result = cli_runner.invoke(cli_app, ["init-db"])
    assert result.exit_code == 0
    assert "Database initialized successfully" in result.stdout


def test_migrate_db(cli_runner: CliRunner, mock_db_engine):
    """Test database migration command."""
    result = cli_runner.invoke(cli_app, ["migrate"])
    assert result.exit_code == 0
    assert "Database migrated successfully" in result.stdout


def test_seed_db(cli_runner: CliRunner, mock_db_engine):
    """Test database seeding command."""
    result = cli_runner.invoke(cli_app, ["seed"])
    assert result.exit_code == 0
    assert "Database seeded successfully" in result.stdout 