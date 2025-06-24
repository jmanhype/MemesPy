"""Base exceptions and error codes for the DSPy Meme Generator."""

from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    """Error codes for different types of errors."""

    # Database Errors (1000-1999)
    DB_CONNECTION_ERROR = "DSPY-1000"
    DB_QUERY_ERROR = "DSPY-1001"
    DB_INTEGRITY_ERROR = "DSPY-1002"
    DB_MIGRATION_ERROR = "DSPY-1003"
    DB_TIMEOUT_ERROR = "DSPY-1004"

    # Cache Errors (2000-2999)
    CACHE_CONNECTION_ERROR = "DSPY-2000"
    CACHE_SET_ERROR = "DSPY-2001"
    CACHE_GET_ERROR = "DSPY-2002"
    CACHE_INVALIDATION_ERROR = "DSPY-2003"

    # Agent Errors (3000-3999)
    AGENT_INITIALIZATION_ERROR = "DSPY-3000"
    AGENT_EXECUTION_ERROR = "DSPY-3001"
    AGENT_VALIDATION_ERROR = "DSPY-3002"
    AGENT_TIMEOUT_ERROR = "DSPY-3003"

    # Content Errors (4000-4999)
    CONTENT_VALIDATION_ERROR = "DSPY-4000"
    CONTENT_GENERATION_ERROR = "DSPY-4001"
    CONTENT_INAPPROPRIATE_ERROR = "DSPY-4002"
    CONTENT_FACTUAL_ERROR = "DSPY-4003"

    # External Service Errors (5000-5999)
    API_CONNECTION_ERROR = "DSPY-5000"
    API_RATE_LIMIT_ERROR = "DSPY-5001"
    API_AUTHENTICATION_ERROR = "DSPY-5002"
    API_RESPONSE_ERROR = "DSPY-5003"

    # Configuration Errors (6000-6999)
    CONFIG_VALIDATION_ERROR = "DSPY-6000"
    CONFIG_MISSING_ERROR = "DSPY-6001"
    CONFIG_TYPE_ERROR = "DSPY-6002"

    # General Errors (9000-9999)
    UNKNOWN_ERROR = "DSPY-9000"
    NOT_IMPLEMENTED_ERROR = "DSPY-9001"
    VALIDATION_ERROR = "DSPY-9002"
    TIMEOUT_ERROR = "DSPY-9003"


class DSPyMemeError(Exception):
    """Base exception class for DSPy Meme Generator."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize base exception.

        Args:
            message: Error message
            code: Error code
            details: Additional error details
            original_error: Original exception if this is a wrapped exception
        """
        self.message = message
        self.code = code
        self.details = details or {}
        self.original_error = original_error
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary representation.

        Returns:
            Dictionary containing error details
        """
        error_dict = {
            "code": self.code.value,
            "message": self.message,
            "type": self.__class__.__name__,
        }

        if self.details:
            error_dict["details"] = self.details

        if self.original_error:
            error_dict["original_error"] = str(self.original_error)

        return error_dict


class DatabaseError(DSPyMemeError):
    """Base class for database-related errors."""

    def __init__(
        self, message: str, code: ErrorCode = ErrorCode.DB_QUERY_ERROR, **kwargs: Any
    ) -> None:
        """Initialize database error."""
        super().__init__(message, code, **kwargs)


class CacheError(DSPyMemeError):
    """Base class for cache-related errors."""

    def __init__(
        self, message: str, code: ErrorCode = ErrorCode.CACHE_CONNECTION_ERROR, **kwargs: Any
    ) -> None:
        """Initialize cache error."""
        super().__init__(message, code, **kwargs)


class AgentError(DSPyMemeError):
    """Base class for agent-related errors."""

    def __init__(
        self, message: str, code: ErrorCode = ErrorCode.AGENT_EXECUTION_ERROR, **kwargs: Any
    ) -> None:
        """Initialize agent error."""
        super().__init__(message, code, **kwargs)


class ContentError(DSPyMemeError):
    """Base class for content-related errors."""

    def __init__(
        self, message: str, code: ErrorCode = ErrorCode.CONTENT_VALIDATION_ERROR, **kwargs: Any
    ) -> None:
        """Initialize content error."""
        super().__init__(message, code, **kwargs)


class ExternalServiceError(DSPyMemeError):
    """Base class for external service-related errors."""

    def __init__(
        self, message: str, code: ErrorCode = ErrorCode.API_CONNECTION_ERROR, **kwargs: Any
    ) -> None:
        """Initialize external service error."""
        super().__init__(message, code, **kwargs)


class ConfigurationError(DSPyMemeError):
    """Base class for configuration-related errors."""

    def __init__(
        self, message: str, code: ErrorCode = ErrorCode.CONFIG_VALIDATION_ERROR, **kwargs: Any
    ) -> None:
        """Initialize configuration error."""
        super().__init__(message, code, **kwargs)


class MemeGenerationError(DSPyMemeError):
    """Base exception class for meme generation errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.CONTENT_GENERATION_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize the MemeGenerationError.

        Args:
            message: Error message
            code: Error code
            details: Optional additional error details
            original_error: Optional original exception
        """
        super().__init__(message, code=code, details=details, original_error=original_error)
