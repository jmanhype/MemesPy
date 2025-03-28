"""Error handler middleware for API errors."""

import logging
from typing import Dict, Any, Optional, Type
from fastapi import HTTPException
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

# Configure logger
logger = logging.getLogger(__name__)

def error_to_dict(exc: Exception, default_message: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert an exception to a dictionary for the API response.
    
    Args:
        exc: The exception that occurred
        default_message: Optional default message to use if the exception has no message
        
    Returns:
        Dictionary with error details
    """
    error_type = exc.__class__.__name__
    error_message = str(exc) if str(exc) else default_message
    
    # For HTTP exceptions, use the status code and detail
    if isinstance(exc, HTTPException):
        return {
            "detail": exc.detail,
            "type": error_type,
            "status_code": exc.status_code
        }
    
    # For Redis errors
    if isinstance(exc, RedisError):
        logger.error(f"Redis error: {error_message}")
        return {
            "detail": "Cache service unavailable",
            "type": error_type
        }
    
    # For SQLAlchemy errors
    if isinstance(exc, SQLAlchemyError):
        logger.error(f"Database error: {error_message}")
        return {
            "detail": "Database service unavailable",
            "type": error_type
        }
    
    # For all other errors
    return {
        "detail": error_message or "An unexpected error occurred",
        "type": error_type
    } 