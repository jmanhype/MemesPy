"""Application configuration module."""

import os
from typing import List, Optional, Any, Literal

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings using Pydantic BaseSettings.

    Automatically reads from environment variables.
    """

    # Application settings
    app_name: str = "DSPy Meme Generator"
    app_version: str = "0.1.0"
    app_env: str = "development"
    log_level: str = "INFO"

    # API settings
    api_prefix: str = "/api"
    api_docs_url: str = "/docs"

    # Database settings
    database_url: str = "sqlite:///./meme_generator.db"

    # CORS settings
    cors_origins: List[str] = ["*"]

    # Redis settings
    redis_url: Optional[str] = None
    cache_ttl: int = 3600  # 1 hour

    # OpenAI settings
    openai_api_key: Optional[str] = None

    # DSPy settings
    dspy_model: str = "gpt-3.5-turbo-0125"

    # Image generation settings
    image_provider: Literal[
        "placeholder", "dalle", "gpt4o", "gpt-image-1", "gpt-image", "openai"
    ] = "placeholder"

    # Privacy settings
    privacy_policy_version: str = "1.0.0"
    pseudonym_salt: str = Field(default_factory=lambda: os.urandom(32).hex())
    encryption_key: str = Field(default_factory=lambda: os.urandom(32).hex())
    admin_api_key: Optional[str] = None

    # CQRS Event System settings
    event_store_connection_string: Optional[str] = None
    event_bus_buffer_size: int = 10000
    enable_event_sourcing: bool = True
    enable_projections: bool = True
    enable_actor_system: bool = True
    actor_system_name: str = "meme_gen_actors"
    projection_rebuild_on_startup: bool = False

    # Actor system timeouts (in milliseconds)
    actor_ask_timeout: int = 30000
    actor_startup_timeout: int = 60000
    actor_shutdown_timeout: int = 30000

    # Validation
    @validator("database_url", pre=True)
    def validate_database_url(cls, v: Any) -> Any:
        """Validate and normalize database URL."""
        if not v:
            return "sqlite:///./meme_generator.db"
        return v

    @validator("image_provider", pre=True)
    def validate_image_provider(cls, v: str, values: dict) -> str:
        """
        Validate and set appropriate image provider based on available API keys.

        If OpenAI API key is available and no provider is explicitly set,
        defaults to 'gpt-image-1' now that organization is verified.
        """
        if not v or v == "placeholder":
            # Auto-upgrade to gpt-image-1 if we have an OpenAI API key
            if values.get("openai_api_key"):
                return "gpt-image-1"

        # Handle legacy or alternative provider names
        if v == "imogen" or v == "gpt-image" or v == "openai":
            return "gpt-image-1"

        return v

    @validator("event_store_connection_string", pre=True)
    def validate_event_store_connection(cls, v: Any, values: dict) -> str:
        """
        Set event store connection string to database URL if not specified.
        """
        if not v:
            # Use the same database for event store if not specified
            database_url = values.get("database_url", "sqlite:///./meme_generator.db")
            # Convert to async if it's SQLite
            if database_url.startswith("sqlite:///"):
                return database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
            return database_url
        return v

    # Updated config for Pydantic v2
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # This allows extra fields to be ignored
    )


# Create settings instance
settings = Settings()
