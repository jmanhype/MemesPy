"""Configuration utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: str = Field(..., description="Database connection URL")
    pool_size: int = Field(5, description="Connection pool size")
    max_overflow: int = Field(10, description="Maximum number of connections to overflow")
    pool_timeout: int = Field(30, description="Pool timeout in seconds")
    echo: bool = Field(False, description="Enable SQL query logging")


class CloudinaryConfig(BaseModel):
    """Cloudinary configuration."""

    cloud_name: str = Field(..., description="Cloudinary cloud name")
    api_key: str = Field(..., description="Cloudinary API key")
    api_secret: str = Field(..., description="Cloudinary API secret")
    secure: bool = Field(True, description="Use HTTPS")
    folder: str = Field("memes", description="Default folder for uploads")


class OpenAIConfig(BaseModel):
    """OpenAI configuration."""

    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field("gpt-4", description="Default model to use")
    temperature: float = Field(0.7, description="Default temperature")
    max_tokens: int = Field(150, description="Default max tokens")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field("INFO", description="Logging level")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format"
    )
    file: Optional[str] = Field(None, description="Log file path")
    rotate: bool = Field(True, description="Enable log rotation")
    max_size: str = Field("10MB", description="Maximum log file size")
    backup_count: int = Field(5, description="Number of backup files to keep")


class RedisConfig(BaseModel):
    """Redis configuration."""

    host: str = Field("localhost", description="Redis host")
    port: int = Field(6379, description="Redis port")
    db: int = Field(0, description="Redis database number")
    password: Optional[str] = Field(None, description="Redis password")
    pool_size: int = Field(10, description="Connection pool size")
    ttl: int = Field(3600, description="Default TTL in seconds")
    prefix: str = Field("dspy_meme_gen", description="Key prefix")


class AppConfig(BaseModel):
    """Application configuration."""

    database: DatabaseConfig
    cloudinary: CloudinaryConfig
    openai: OpenAIConfig
    logging: LoggingConfig
    redis: RedisConfig
    debug: bool = Field(False, description="Enable debug mode")
    testing: bool = Field(False, description="Enable testing mode")
    secret_key: str = Field(..., description="Application secret key")
    allowed_formats: list[str] = Field(["JPEG", "PNG", "GIF"], description="Allowed image formats")
    max_file_size: int = Field(10 * 1024 * 1024, description="Maximum file size in bytes")  # 10MB


def load_config(
    config_path: Optional[Union[str, Path]] = None, env_prefix: str = "DSPY_MEME_"
) -> AppConfig:
    """Load application configuration.

    Configuration is loaded from:
    1. Default values
    2. Configuration file (if provided)
    3. Environment variables (overrides file)

    Args:
        config_path: Path to YAML configuration file
        env_prefix: Prefix for environment variables

    Returns:
        AppConfig: Application configuration

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If configuration is invalid
    """
    # Start with empty config
    config_dict: Dict[str, Any] = {}

    # Load from file if provided
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            file_config = yaml.safe_load(f)
            if file_config:
                config_dict.update(file_config)

    # Load from environment variables
    env_config = {
        "database": {
            "url": os.getenv(f"{env_prefix}DATABASE_URL"),
            "pool_size": int(os.getenv(f"{env_prefix}DATABASE_POOL_SIZE", "5")),
            "max_overflow": int(os.getenv(f"{env_prefix}DATABASE_MAX_OVERFLOW", "10")),
            "pool_timeout": int(os.getenv(f"{env_prefix}DATABASE_POOL_TIMEOUT", "30")),
            "echo": os.getenv(f"{env_prefix}DATABASE_ECHO", "").lower() == "true",
        },
        "cloudinary": {
            "cloud_name": os.getenv(f"{env_prefix}CLOUDINARY_CLOUD_NAME"),
            "api_key": os.getenv(f"{env_prefix}CLOUDINARY_API_KEY"),
            "api_secret": os.getenv(f"{env_prefix}CLOUDINARY_API_SECRET"),
            "secure": os.getenv(f"{env_prefix}CLOUDINARY_SECURE", "true").lower() == "true",
            "folder": os.getenv(f"{env_prefix}CLOUDINARY_FOLDER", "memes"),
        },
        "openai": {
            "api_key": os.getenv(f"{env_prefix}OPENAI_API_KEY"),
            "model": os.getenv(f"{env_prefix}OPENAI_MODEL", "gpt-4"),
            "temperature": float(os.getenv(f"{env_prefix}OPENAI_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv(f"{env_prefix}OPENAI_MAX_TOKENS", "150")),
        },
        "logging": {
            "level": os.getenv(f"{env_prefix}LOG_LEVEL", "INFO"),
            "format": os.getenv(
                f"{env_prefix}LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
            "file": os.getenv(f"{env_prefix}LOG_FILE"),
            "rotate": os.getenv(f"{env_prefix}LOG_ROTATE", "true").lower() == "true",
            "max_size": os.getenv(f"{env_prefix}LOG_MAX_SIZE", "10MB"),
            "backup_count": int(os.getenv(f"{env_prefix}LOG_BACKUP_COUNT", "5")),
        },
        "redis": {
            "host": os.getenv(f"{env_prefix}REDIS_HOST", "localhost"),
            "port": int(os.getenv(f"{env_prefix}REDIS_PORT", "6379")),
            "db": int(os.getenv(f"{env_prefix}REDIS_DB", "0")),
            "password": os.getenv(f"{env_prefix}REDIS_PASSWORD"),
            "pool_size": int(os.getenv(f"{env_prefix}REDIS_POOL_SIZE", "10")),
            "ttl": int(os.getenv(f"{env_prefix}REDIS_TTL", "3600")),
            "prefix": os.getenv(f"{env_prefix}REDIS_PREFIX", "dspy_meme_gen"),
        },
        "debug": os.getenv(f"{env_prefix}DEBUG", "").lower() == "true",
        "testing": os.getenv(f"{env_prefix}TESTING", "").lower() == "true",
        "secret_key": os.getenv(f"{env_prefix}SECRET_KEY"),
        "allowed_formats": os.getenv(f"{env_prefix}ALLOWED_FORMATS", "JPEG,PNG,GIF").split(","),
        "max_file_size": int(os.getenv(f"{env_prefix}MAX_FILE_SIZE", str(10 * 1024 * 1024))),
    }

    # Update config dict with environment variables (only if set)
    for section, values in env_config.items():
        if section not in config_dict:
            config_dict[section] = {}
        for key, value in values.items():
            if value is not None:  # Only update if environment variable was set
                config_dict[section][key] = value

    # Create and validate config
    try:
        return AppConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")


def get_config(
    config_path: Optional[Union[str, Path]] = None, env_prefix: str = "DSPY_MEME_"
) -> AppConfig:
    """Get application configuration (singleton).

    Args:
        config_path: Path to YAML configuration file
        env_prefix: Prefix for environment variables

    Returns:
        AppConfig: Application configuration
    """
    if not hasattr(get_config, "_config"):
        get_config._config = load_config(config_path, env_prefix)
    return get_config._config
