"""Error handling middleware for DSPy Meme Generator."""

import asyncio
import functools
import traceback
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast, Union

from sqlalchemy.exc import SQLAlchemyError
from redis.exceptions import RedisError
import dspy
from dspy import Prediction

from ..exceptions.base import DSPyMemeError, ErrorCode, MemeGenerationError
from ..exceptions.specific import (
    APIConnectionError,
    CacheConnectionError,
    DatabaseConnectionError,
    DatabaseQueryError
)
from ..exceptions.meme_specific import (
    RouterError,
    MemeGenerationFailedError,
    PromptGenerationError,
    ImageGenerationError,
    TrendAnalysisError,
    FormatSelectionError,
    ContentVerificationError,
    ScoringError,
    MemeTemplateNotFoundError,
    InvalidMemeFormatError
)
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Type variables
F = TypeVar('F', bound=Callable[..., Any])


def handle_exceptions(
    error_map: Optional[Dict[Type[Exception], Type[MemeGenerationError]]] = None
) -> Callable:
    """
    A decorator that handles exceptions in DSPy meme generation functions.
    
    Args:
        error_map: Optional mapping of exception types to MemeGenerationError types.
        
    Returns:
        Callable: The decorated function.
    """
    default_error_map = {
        ValueError: InvalidMemeFormatError,
        KeyError: MemeTemplateNotFoundError,
        dspy.Prediction: MemeGenerationError,
    }
    
    if error_map:
        default_error_map.update(error_map)
        
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_type = type(e)
                if error_type in default_error_map:
                    error_class = default_error_map[error_type]
                    raise error_class(str(e))
                else:
                    logger.error(f"Unhandled error in meme generation: {str(e)}")
                    raise MemeGenerationError(f"Unexpected error: {str(e)}")
        return wrapper
    return decorator


def error_to_dict(error: Exception) -> Dict[str, Any]:
    """
    Convert any exception to a dictionary representation.
    
    Args:
        error: Exception to convert
        
    Returns:
        Dictionary containing error details
    """
    if isinstance(error, DSPyMemeError):
        return error.to_dict()
    
    return {
        "code": ErrorCode.UNKNOWN_ERROR.value,
        "message": str(error),
        "type": error.__class__.__name__,
        "traceback": traceback.format_exc()
    } 