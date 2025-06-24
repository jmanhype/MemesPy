"""DSPy-specific exceptions for meme generation."""

from typing import Any, Dict, Optional, List

from .base import AgentError, ContentError, ErrorCode


class RouterError(AgentError):
    """Base class for router-related errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.AGENT_EXECUTION_ERROR,
        route_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize router error.

        Args:
            message: Error message
            code: Error code
            route_info: Information about the routing attempt
            **kwargs: Additional keyword arguments
        """
        details = kwargs.get("details", {})
        if route_info:
            details["route_info"] = route_info
        kwargs["details"] = details
        super().__init__(message, code, **kwargs)


class RouteNotFoundError(RouterError):
    """Raised when no suitable route is found for a request."""

    def __init__(self, message: str, request: Dict[str, Any], **kwargs: Any) -> None:
        """Initialize route not found error."""
        route_info = {"request": request}
        super().__init__(message, ErrorCode.AGENT_VALIDATION_ERROR, route_info=route_info, **kwargs)


class InvalidRouteError(RouterError):
    """Raised when a route is invalid or cannot be executed."""

    def __init__(self, message: str, route: Dict[str, Any], reason: str, **kwargs: Any) -> None:
        """Initialize invalid route error."""
        route_info = {"route": route, "reason": reason}
        super().__init__(message, ErrorCode.AGENT_VALIDATION_ERROR, route_info=route_info, **kwargs)


class MemeGenerationError(ContentError):
    """Base class for meme generation errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.CONTENT_GENERATION_ERROR,
        generation_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize meme generation error.

        Args:
            message: Error message
            code: Error code
            generation_info: Information about the generation attempt
            **kwargs: Additional keyword arguments
        """
        details = kwargs.get("details", {})
        if generation_info:
            details["generation_info"] = generation_info
        kwargs["details"] = details
        super().__init__(message, code, **kwargs)


class PromptGenerationError(MemeGenerationError):
    """Raised when prompt generation fails."""

    def __init__(
        self, message: str, topic: str, format_details: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Initialize prompt generation error."""
        generation_info = {"topic": topic, "format_details": format_details}
        super().__init__(
            message, ErrorCode.CONTENT_GENERATION_ERROR, generation_info=generation_info, **kwargs
        )


class ImageGenerationError(MemeGenerationError):
    """Raised when image generation fails."""

    def __init__(self, message: str, prompt: str, service: str, **kwargs: Any) -> None:
        """Initialize image generation error."""
        generation_info = {"prompt": prompt, "service": service}
        super().__init__(
            message, ErrorCode.CONTENT_GENERATION_ERROR, generation_info=generation_info, **kwargs
        )


class TrendAnalysisError(AgentError):
    """Raised when trend analysis fails."""

    def __init__(
        self, message: str, sources: List[str], query: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Initialize trend analysis error."""
        details = {"sources": sources, "query": query}
        super().__init__(message, ErrorCode.AGENT_EXECUTION_ERROR, details=details, **kwargs)


class FormatSelectionError(AgentError):
    """Raised when format selection fails."""

    def __init__(
        self, message: str, topic: str, constraints: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Initialize format selection error."""
        details = {"topic": topic, "constraints": constraints}
        super().__init__(message, ErrorCode.AGENT_EXECUTION_ERROR, details=details, **kwargs)


class ContentVerificationError(ContentError):
    """Raised when content verification fails."""

    def __init__(
        self,
        message: str,
        content_type: str,
        verification_type: str,
        issues: List[str],
        **kwargs: Any,
    ) -> None:
        """Initialize content verification error."""
        details = {
            "content_type": content_type,
            "verification_type": verification_type,
            "issues": issues,
        }
        super().__init__(message, ErrorCode.CONTENT_VALIDATION_ERROR, details=details, **kwargs)


class ScoringError(AgentError):
    """Raised when meme scoring fails."""

    def __init__(self, message: str, meme_id: str, criteria: List[str], **kwargs: Any) -> None:
        """Initialize scoring error."""
        details = {"meme_id": meme_id, "criteria": criteria}
        super().__init__(message, ErrorCode.AGENT_EXECUTION_ERROR, details=details, **kwargs)


class MemeTemplateNotFoundError(MemeGenerationError):
    """Raised when a requested meme template cannot be found."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the MemeTemplateNotFoundError.

        Args:
            message: Error message
            details: Optional additional error details
        """
        super().__init__(message, details=details)


class InvalidMemeFormatError(MemeGenerationError):
    """Raised when a meme format is invalid or unsupported."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the InvalidMemeFormatError.

        Args:
            message: Error message
            details: Optional additional error details
        """
        super().__init__(message, details=details)


class MemeGenerationFailedError(MemeGenerationError):
    """Raised when meme generation fails for any reason."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the MemeGenerationFailedError.

        Args:
            message: Error message
            details: Optional additional error details
        """
        super().__init__(message, details=details)


class RefinementError(MemeGenerationError):
    """Raised when meme refinement fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the RefinementError.

        Args:
            message: Error message
            details: Optional additional error details
        """
        super().__init__(message, details=details)


class PipelineError(MemeGenerationError):
    """Raised when there is an error in the meme generation pipeline."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the PipelineError.

        Args:
            message: Error message
            details: Optional additional error details
        """
        super().__init__(message, details=details)
