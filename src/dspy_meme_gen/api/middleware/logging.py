"""Logging middleware for API requests."""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logger
logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging API requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request, log request and response details.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            The response from the next middleware or route handler
        """
        start_time = time.time()
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)

        # Log request
        logger.debug(f"Request: {method} {path} {query_params}")

        # Process request
        try:
            response = await call_next(request)
            process_time = round((time.time() - start_time) * 1000, 2)

            # Log response
            logger.debug(
                f"Response: {method} {path} Status: {response.status_code} "
                f"Time: {process_time}ms"
            )

            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)
            return response

        except Exception as e:
            process_time = round((time.time() - start_time) * 1000, 2)
            logger.error(f"Error during {method} {path}: {str(e)} " f"Time: {process_time}ms")
            raise
