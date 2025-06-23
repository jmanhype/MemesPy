"""Custom exceptions for the meme generation service."""


class DSPyMemeError(Exception):
    """Base exception for all meme generation errors."""
    pass


class ValidationError(DSPyMemeError):
    """Raised when input validation fails."""
    pass


class GenerationError(DSPyMemeError):
    """Raised when meme generation fails."""
    pass


class ContentError(DSPyMemeError):
    """Raised when content violates policies."""
    pass


class ExternalServiceError(DSPyMemeError):
    """Raised when external service calls fail."""
    pass


class PrivacyError(DSPyMemeError):
    """Raised when privacy-related operations fail."""
    pass


class ConsentError(PrivacyError):
    """Raised when required consent is missing or invalid."""
    pass


class DataRetentionError(PrivacyError):
    """Raised when data retention policies are violated."""
    pass


class AnonymizationError(PrivacyError):
    """Raised when data anonymization fails."""
    pass