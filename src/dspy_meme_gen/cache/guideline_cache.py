"""Caching layer for content guidelines."""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aioredis
from aioredis.client import Redis

from ..models.models import ContentGuideline, GuidelineCategory, SeverityLevel
from ..utils.logging import get_logger

logger = get_logger(__name__)


class GuidelineCache:
    """Cache for content guidelines using Redis."""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl: int = 3600,  # 1 hour default TTL
        prefix: str = "guideline"
    ) -> None:
        """
        Initialize guideline cache.
        
        Args:
            redis_url: Redis connection URL
            ttl: Cache TTL in seconds
            prefix: Key prefix for cache entries
        """
        self.redis_url = redis_url or "redis://localhost:6379/0"
        self.ttl = ttl
        self.prefix = prefix
        self._redis: Optional[Redis] = None
        
    async def connect(self) -> None:
        """Connect to Redis."""
        if not self._redis:
            self._redis = await aioredis.from_url(self.redis_url)
            
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            
    def _get_category_key(self, category_id: int) -> str:
        """Get cache key for a category."""
        return f"{self.prefix}:category:{category_id}"
        
    def _get_guideline_key(self, guideline_id: int) -> str:
        """Get cache key for a guideline."""
        return f"{self.prefix}:guideline:{guideline_id}"
        
    def _get_all_categories_key(self) -> str:
        """Get cache key for all categories list."""
        return f"{self.prefix}:all_categories"
        
    async def _ensure_connected(self) -> None:
        """Ensure Redis connection is established."""
        if not self._redis:
            await self.connect()
            
    def _serialize_category(self, category: GuidelineCategory) -> Dict[str, Any]:
        """Serialize a category for caching."""
        return {
            "id": category.id,
            "name": category.name,
            "description": category.description
        }
        
    def _deserialize_category(self, data: Dict[str, Any]) -> GuidelineCategory:
        """Deserialize a category from cache."""
        return GuidelineCategory(
            id=data["id"],
            name=data["name"],
            description=data["description"]
        )
        
    def _serialize_guideline(self, guideline: ContentGuideline) -> Dict[str, Any]:
        """Serialize a guideline for caching."""
        return {
            "id": guideline.id,
            "category_id": guideline.category_id,
            "rule": guideline.rule,
            "description": guideline.description,
            "severity": guideline.severity.value,
            "keywords": guideline.keywords
        }
        
    def _deserialize_guideline(self, data: Dict[str, Any]) -> ContentGuideline:
        """Deserialize a guideline from cache."""
        return ContentGuideline(
            id=data["id"],
            category_id=data["category_id"],
            rule=data["rule"],
            description=data["description"],
            severity=SeverityLevel(data["severity"]),
            keywords=data["keywords"]
        )
        
    async def get_category(self, category_id: int) -> Optional[GuidelineCategory]:
        """
        Get a category from cache.
        
        Args:
            category_id: Category ID
            
        Returns:
            Cached category or None if not found
        """
        await self._ensure_connected()
        assert self._redis is not None
        
        key = self._get_category_key(category_id)
        data = await self._redis.get(key)
        
        if data:
            return self._deserialize_category(json.loads(data))
        return None
        
    async def set_category(self, category: GuidelineCategory) -> None:
        """
        Cache a category.
        
        Args:
            category: Category to cache
        """
        await self._ensure_connected()
        assert self._redis is not None
        
        key = self._get_category_key(category.id)
        data = json.dumps(self._serialize_category(category))
        await self._redis.set(key, data, ex=self.ttl)
        
    async def get_guideline(self, guideline_id: int) -> Optional[ContentGuideline]:
        """
        Get a guideline from cache.
        
        Args:
            guideline_id: Guideline ID
            
        Returns:
            Cached guideline or None if not found
        """
        await self._ensure_connected()
        assert self._redis is not None
        
        key = self._get_guideline_key(guideline_id)
        data = await self._redis.get(key)
        
        if data:
            return self._deserialize_guideline(json.loads(data))
        return None
        
    async def set_guideline(self, guideline: ContentGuideline) -> None:
        """
        Cache a guideline.
        
        Args:
            guideline: Guideline to cache
        """
        await self._ensure_connected()
        assert self._redis is not None
        
        key = self._get_guideline_key(guideline.id)
        data = json.dumps(self._serialize_guideline(guideline))
        await self._redis.set(key, data, ex=self.ttl)
        
    async def get_all_categories(self) -> List[GuidelineCategory]:
        """
        Get all categories from cache.
        
        Returns:
            List of cached categories
        """
        await self._ensure_connected()
        assert self._redis is not None
        
        key = self._get_all_categories_key()
        data = await self._redis.get(key)
        
        if data:
            categories_data = json.loads(data)
            return [self._deserialize_category(cat_data) for cat_data in categories_data]
        return []
        
    async def set_all_categories(self, categories: List[GuidelineCategory]) -> None:
        """
        Cache all categories.
        
        Args:
            categories: List of categories to cache
        """
        await self._ensure_connected()
        assert self._redis is not None
        
        key = self._get_all_categories_key()
        data = json.dumps([self._serialize_category(cat) for cat in categories])
        await self._redis.set(key, data, ex=self.ttl)
        
    async def invalidate_category(self, category_id: int) -> None:
        """
        Invalidate a cached category.
        
        Args:
            category_id: Category ID to invalidate
        """
        await self._ensure_connected()
        assert self._redis is not None
        
        key = self._get_category_key(category_id)
        await self._redis.delete(key)
        
        # Also invalidate all categories list
        all_cats_key = self._get_all_categories_key()
        await self._redis.delete(all_cats_key)
        
    async def invalidate_guideline(self, guideline_id: int) -> None:
        """
        Invalidate a cached guideline.
        
        Args:
            guideline_id: Guideline ID to invalidate
        """
        await self._ensure_connected()
        assert self._redis is not None
        
        key = self._get_guideline_key(guideline_id)
        await self._redis.delete(key)
        
    async def invalidate_all(self) -> None:
        """Invalidate all cached data."""
        await self._ensure_connected()
        assert self._redis is not None
        
        # Get all keys with our prefix
        keys = await self._redis.keys(f"{self.prefix}:*")
        if keys:
            await self._redis.delete(*keys)


# Global cache instance
cache = GuidelineCache() 