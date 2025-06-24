"""Database CLI commands."""

import os
from typing import Optional
from pathlib import Path

import typer
from alembic.config import Config
from alembic import command

from ..models.connection import DatabaseConnectionManager
from ..config.config import get_settings

app = typer.Typer(help="Database management commands")


def get_alembic_config(script_location: Optional[str] = None, dsn: Optional[str] = None) -> Config:
    """
    Get Alembic configuration.

    Args:
        script_location: Optional path to migration scripts
        dsn: Optional database connection string

    Returns:
        Config: Alembic configuration
    """
    config = Config()

    if script_location:
        config.set_main_option("script_location", script_location)
    else:
        config.set_main_option("script_location", str(Path(__file__).parent.parent / "migrations"))

    if dsn:
        config.set_main_option("sqlalchemy.url", dsn)
    else:
        config.set_main_option("sqlalchemy.url", get_settings().DATABASE_URL)

    return config


@app.command()
def init(directory: str = typer.Option(None, help="Directory for migration scripts")) -> None:
    """Initialize Alembic migrations."""
    config = get_alembic_config(script_location=directory)
    command.init(config, directory)


@app.command()
def migrate(
    message: str = typer.Option(..., prompt=True, help="Migration message"),
    autogenerate: bool = typer.Option(True, help="Auto-generate migration from models"),
) -> None:
    """Create a new migration."""
    config = get_alembic_config()
    command.revision(config, message, autogenerate=autogenerate)


@app.command()
def upgrade(revision: str = typer.Option("head", help="Target revision")) -> None:
    """Upgrade database to a later version."""
    config = get_alembic_config()
    command.upgrade(config, revision)


@app.command()
def downgrade(revision: str = typer.Option("-1", help="Target revision")) -> None:
    """Revert database to a previous version."""
    config = get_alembic_config()
    command.downgrade(config, revision)


@app.command()
def current() -> None:
    """Show current revision."""
    config = get_alembic_config()
    command.current(config)


@app.command()
def history() -> None:
    """Show migration history."""
    config = get_alembic_config()
    command.history(config)


@app.command()
def check() -> None:
    """Check database connection."""
    settings = get_settings()
    db = DatabaseConnectionManager(settings.DATABASE_URL)

    try:
        db.check_connection()
        typer.echo("Database connection successful!")
    except Exception as e:
        typer.echo(f"Database connection failed: {str(e)}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
