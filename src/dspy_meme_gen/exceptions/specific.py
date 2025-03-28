"""Specific exception classes for DSPy Meme Generator."""

from typing import Any, Optional

from .base import (
    AgentError,
    CacheError,
    ConfigurationError,
    ContentError,
    DatabaseError,
    ErrorCode,
    ExternalServiceError
)


# Database Exceptions
class DatabaseConnectionError(DatabaseError):
    """Raised when unable to connect to the database."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.DB_CONNECTION_ERROR, **kwargs)


class DatabaseQueryError(DatabaseError):
    """Raised when a database query fails."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.DB_QUERY_ERROR, **kwargs)


class DatabaseIntegrityError(DatabaseError):
    """Raised when a database constraint is violated."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.DB_INTEGRITY_ERROR, **kwargs)


class DatabaseMigrationError(DatabaseError):
    """Raised when a database migration fails."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.DB_MIGRATION_ERROR, **kwargs)


# Cache Exceptions
class CacheConnectionError(CacheError):
    """Raised when unable to connect to the cache."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.CACHE_CONNECTION_ERROR, **kwargs)


class CacheSetError(CacheError):
    """Raised when unable to set a value in cache."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.CACHE_SET_ERROR, **kwargs)


class CacheGetError(CacheError):
    """Raised when unable to get a value from cache."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.CACHE_GET_ERROR, **kwargs)


# Agent Exceptions
class AgentInitializationError(AgentError):
    """Raised when an agent fails to initialize."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.AGENT_INITIALIZATION_ERROR, **kwargs)


class AgentExecutionError(AgentError):
    """Raised when an agent fails during execution."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.AGENT_EXECUTION_ERROR, **kwargs)


class AgentValidationError(AgentError):
    """Raised when agent input/output validation fails."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.AGENT_VALIDATION_ERROR, **kwargs)


class AgentTimeoutError(AgentError):
    """Raised when an agent operation times out."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.AGENT_TIMEOUT_ERROR, **kwargs)


# Content Exceptions
class ContentValidationError(ContentError):
    """Raised when content validation fails."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.CONTENT_VALIDATION_ERROR, **kwargs)


class ContentGenerationError(ContentError):
    """Raised when content generation fails."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.CONTENT_GENERATION_ERROR, **kwargs)


class ContentInappropriateError(ContentError):
    """Raised when content is deemed inappropriate."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.CONTENT_INAPPROPRIATE_ERROR, **kwargs)


class ContentFactualError(ContentError):
    """Raised when content contains factual errors."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.CONTENT_FACTUAL_ERROR, **kwargs)


# External Service Exceptions
class APIConnectionError(ExternalServiceError):
    """Raised when unable to connect to an external API."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.API_CONNECTION_ERROR, **kwargs)


class APIRateLimitError(ExternalServiceError):
    """Raised when API rate limit is exceeded."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.API_RATE_LIMIT_ERROR, **kwargs)


class APIAuthenticationError(ExternalServiceError):
    """Raised when API authentication fails."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.API_AUTHENTICATION_ERROR, **kwargs)


class APIResponseError(ExternalServiceError):
    """Raised when API returns an error response."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.API_RESPONSE_ERROR, **kwargs)


# Configuration Exceptions
class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.CONFIG_VALIDATION_ERROR, **kwargs)


class ConfigMissingError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.CONFIG_MISSING_ERROR, **kwargs)


class ConfigTypeError(ConfigurationError):
    """Raised when configuration has incorrect type."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, ErrorCode.CONFIG_TYPE_ERROR, **kwargs) 