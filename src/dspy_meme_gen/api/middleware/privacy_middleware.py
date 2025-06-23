"""Privacy middleware for request handling and consent enforcement."""

import json
import time
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import hashlib

from ...services.privacy_service import PrivacyService
from ...models.db_models.privacy_metadata import ConsentType
from ...config.config import settings

logger = logging.getLogger(__name__)


class PrivacyMiddleware(BaseHTTPMiddleware):
    """
    Middleware for privacy-first request handling.
    
    Responsibilities:
    - Consent verification
    - Request anonymization
    - Privacy headers
    - Data minimization
    - Audit logging
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.privacy_service = PrivacyService()
        self.exempt_paths = {
            "/api/health",
            "/api/privacy/policy",
            "/api/privacy/consent/grant",
            "/docs",
            "/openapi.json"
        }
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with privacy controls."""
        start_time = time.time()
        
        # Add privacy headers to response
        response = await self._process_request(request, call_next)
        
        # Add privacy-related headers
        response.headers["X-Privacy-Policy"] = settings.privacy_policy_version
        response.headers["X-Data-Minimization"] = "enabled"
        response.headers["X-Consent-Required"] = "essential"
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        
        # Log request (anonymized)
        process_time = time.time() - start_time
        await self._log_request(request, response, process_time)
        
        return response
    
    async def _process_request(self, request: Request, call_next: Callable) -> Response:
        """Process request with privacy checks."""
        # Check if path is exempt
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)
        
        # Extract user identifier (from header, cookie, or token)
        user_id = await self._get_user_identifier(request)
        
        # Store anonymized user info in request state
        if user_id:
            from ...models.db_models.privacy_metadata import generate_user_pseudonym
            request.state.user_pseudonym = generate_user_pseudonym(user_id, self.privacy_service.pseudonym_salt)
            request.state.has_user = True
        else:
            request.state.user_pseudonym = None
            request.state.has_user = False
        
        # Check consent for non-exempt endpoints
        if request.url.path.startswith("/api/v1/memes"):
            if not user_id:
                # Anonymous usage allowed for basic meme generation
                request.state.anonymous_user = True
            else:
                # Check essential consent for identified users
                # In real implementation, would check against database
                request.state.anonymous_user = False
        
        # Sanitize request data
        await self._sanitize_request(request)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Sanitize response data
            response = await self._sanitize_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Request processing error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "reference": self._generate_error_reference()}
            )
    
    async def _get_user_identifier(self, request: Request) -> Optional[str]:
        """Extract user identifier from request."""
        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In production, validate JWT and extract user ID
            return None
        
        # Check session cookie
        session_id = request.cookies.get("session_id")
        if session_id:
            # In production, validate session and get user ID
            return None
        
        # Check API key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            # In production, validate API key and get associated user
            return None
        
        return None
    
    async def _sanitize_request(self, request: Request):
        """Remove or anonymize sensitive data from request."""
        # Create a sanitized copy of headers
        safe_headers = {}
        sensitive_headers = {
            "authorization", "x-api-key", "cookie", "x-forwarded-for",
            "x-real-ip", "user-agent"
        }
        
        for key, value in request.headers.items():
            if key.lower() in sensitive_headers:
                if key.lower() == "user-agent":
                    # Generalize user agent
                    safe_headers[key] = self._generalize_user_agent(value)
                else:
                    safe_headers[key] = "[REDACTED]"
            else:
                safe_headers[key] = value
        
        # Store sanitized headers
        request.state.sanitized_headers = safe_headers
        
        # Anonymize IP address to country level
        client_ip = request.client.host if request.client else None
        if client_ip:
            # In production, use GeoIP to get country
            request.state.client_country = "US"  # Placeholder
            request.state.client_ip = None  # Don't store actual IP
    
    async def _sanitize_response(self, response: Response) -> Response:
        """Ensure response doesn't contain sensitive data."""
        # For JSON responses, check for sensitive fields
        if hasattr(response, "body"):
            try:
                # Parse JSON response
                content = json.loads(response.body)
                
                # Remove any accidentally included sensitive fields
                sensitive_fields = {
                    "ip_address", "user_email", "full_name", "phone_number",
                    "exact_location", "device_id", "session_token"
                }
                
                cleaned_content = self._remove_sensitive_fields(content, sensitive_fields)
                
                # Update response body if changes were made
                if cleaned_content != content:
                    response.body = json.dumps(cleaned_content).encode()
                    
            except (json.JSONDecodeError, AttributeError):
                # Not JSON or no body, skip sanitization
                pass
        
        return response
    
    def _remove_sensitive_fields(self, data: Any, sensitive_fields: set) -> Any:
        """Recursively remove sensitive fields from data."""
        if isinstance(data, dict):
            return {
                k: self._remove_sensitive_fields(v, sensitive_fields)
                for k, v in data.items()
                if k not in sensitive_fields
            }
        elif isinstance(data, list):
            return [
                self._remove_sensitive_fields(item, sensitive_fields)
                for item in data
            ]
        else:
            return data
    
    async def _log_request(self, request: Request, response: Response, process_time: float):
        """Log request with privacy-safe information."""
        # Only log anonymized data
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": self._generalize_path(request.url.path),
            "status_code": response.status_code,
            "process_time_ms": round(process_time * 1000),
            "has_user": getattr(request.state, "has_user", False),
            "country": getattr(request.state, "client_country", "unknown"),
            "hour_of_day": datetime.utcnow().hour
        }
        
        # Log at appropriate level
        if response.status_code >= 500:
            logger.error(f"Request failed: {log_data}")
        elif response.status_code >= 400:
            logger.warning(f"Request error: {log_data}")
        else:
            logger.info(f"Request processed: {log_data}")
    
    def _generalize_path(self, path: str) -> str:
        """Generalize URL path to remove identifiers."""
        # Replace UUIDs and IDs with placeholders
        import re
        
        # Replace UUIDs
        path = re.sub(
            r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',
            '{uuid}',
            path
        )
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        return path
    
    def _generalize_user_agent(self, user_agent: str) -> str:
        """Generalize user agent to browser and OS only."""
        # Simplified version - in production use a proper parser
        if "Chrome" in user_agent:
            browser = "Chrome"
        elif "Firefox" in user_agent:
            browser = "Firefox"
        elif "Safari" in user_agent:
            browser = "Safari"
        else:
            browser = "Other"
        
        if "Windows" in user_agent:
            os = "Windows"
        elif "Mac" in user_agent:
            os = "macOS"
        elif "Linux" in user_agent:
            os = "Linux"
        elif "Android" in user_agent:
            os = "Android"
        elif "iOS" in user_agent or "iPhone" in user_agent:
            os = "iOS"
        else:
            os = "Other"
        
        return f"{browser}/{os}"
    
    def _generate_error_reference(self) -> str:
        """Generate anonymous error reference."""
        timestamp = str(int(time.time()))
        return hashlib.sha256(timestamp.encode()).hexdigest()[:12]


class ConsentEnforcementMiddleware(BaseHTTPMiddleware):
    """
    Middleware specifically for consent enforcement.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.consent_requirements = {
            "/api/v1/analytics": [ConsentType.ANALYTICS],
            "/api/v1/personalization": [ConsentType.PERSONALIZATION],
            "/api/v1/memes/trending": [ConsentType.ANALYTICS],
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Enforce consent requirements for endpoints."""
        # Check if endpoint requires specific consent
        for path_prefix, required_consents in self.consent_requirements.items():
            if request.url.path.startswith(path_prefix):
                # Check if user has required consents
                user_pseudonym = getattr(request.state, "user_pseudonym", None)
                
                if not user_pseudonym:
                    return JSONResponse(
                        status_code=401,
                        content={
                            "error": "Authentication required",
                            "reason": "This endpoint requires user consent"
                        }
                    )
                
                # In production, check actual consent status from database
                # For now, we'll pass through
                request.state.required_consents = required_consents
        
        return await call_next(request)