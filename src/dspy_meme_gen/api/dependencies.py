"""API dependencies."""

import logging
from typing import Optional, Any
import redis.asyncio as redis
from fastapi import Depends, Header, Request, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..config.config import settings
from ..database.connection import get_db

# Configure logger
logger = logging.getLogger(__name__)

# Redis connection
redis_client: Optional[redis.Redis] = None

async def get_cache():
    """
    Get Redis cache connection.
    
    Returns:
        Redis: Redis client connection
    """
    global redis_client
    
    # Return existing connection if available
    if redis_client is not None:
        return redis_client
    
    # Create connection if Redis URL is configured
    if settings.redis_url:
        try:
            logger.info(f"Connecting to Redis at {settings.redis_url}")
            redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            return redis_client
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
    
    # Return dummy Redis client if no connection
    return DummyRedis()


async def close_connections():
    """Close Redis connections."""
    global redis_client
    if redis_client is not None:
        await redis_client.close()
        redis_client = None
        logger.info("Redis connection closed")


class DummyRedis:
    """
    Dummy Redis implementation for when Redis is not available.
    
    Implements simple cache interface with local memory storage.
    """
    
    def __init__(self):
        """Initialize the dummy cache."""
        self.cache = {}
        logger.warning("Using dummy Redis implementation (no persistence)")
    
    async def get(self, key: str) -> Optional[str]:
        """Get a value from cache."""
        return self.cache.get(key)
    
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set a value in cache."""
        self.cache[key] = value
        return True
    
    async def delete(self, key: str) -> int:
        """Delete a value from cache."""
        if key in self.cache:
            del self.cache[key]
            return 1
        return 0
    
    async def close(self) -> None:
        """Close connection (dummy operation)."""
        self.cache = {}


async def get_current_user_id(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    request: Request = None
) -> Optional[str]:
    """
    Extract user identifier from request.
    
    This is a simplified version. In production, you would:
    1. Validate JWT tokens
    2. Check API keys against database
    3. Handle session cookies
    """
    # Check Authorization header (JWT)
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        try:
            # In production, validate JWT properly
            # For now, just extract a user_id claim
            # payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
            # return payload.get("user_id")
            pass
        except Exception as e:
            logger.error(f"JWT validation error: {e}")
    
    # Check API key
    if x_api_key:
        # In production, look up API key in database
        # For now, use the key itself as identifier
        return f"api_key:{x_api_key[:8]}"  # Use first 8 chars as ID
    
    # Check session
    if request and hasattr(request, 'session'):
        session_user = request.session.get('user_id')
        if session_user:
            return session_user
    
    # Anonymous user
    return None


async def get_user_country(
    request: Request,
    x_forwarded_for: Optional[str] = Header(None),
    cf_ipcountry: Optional[str] = Header(None),  # Cloudflare
    x_country_code: Optional[str] = Header(None)  # Custom header
) -> Optional[str]:
    """
    Get user's country code from request headers.
    
    Returns two-letter country code or None.
    """
    # Check Cloudflare country header
    if cf_ipcountry and cf_ipcountry != "XX":
        return cf_ipcountry
    
    # Check custom country header
    if x_country_code:
        return x_country_code
    
    # In production, use GeoIP lookup on IP address
    # For now, return default
    return "US"


async def require_consent(
    consent_type: str,
    db: AsyncSession = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    Dependency to require specific consent type.
    
    Usage:
        @router.get("/endpoint")
        async def endpoint(
            _: None = Depends(require_consent("analytics"))
        ):
            ...
    """
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication required for this endpoint"
        )
    
    # In production, check consent in database
    # For now, pass through
    return None 