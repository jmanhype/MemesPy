"""
Simple usage example for the Redis Cache Actor.

This example shows how to use the cache actor directly without
relying on the full actor system to avoid circular import issues.
"""

import asyncio
import json
from typing import Any, Dict, Optional

# Direct imports to avoid circular dependency issues
from .cache_actor import CacheActor, SerializationType


class SimpleCache:
    """Simplified cache interface that wraps the cache actor functionality."""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.cache_actor = CacheActor(
            name="simple_cache",
            redis_url=redis_url,
            serialization=SerializationType.JSON,
            enable_fallback=True
        )
        self._started = False
    
    async def start(self):
        """Start the cache."""
        if not self._started:
            await self.cache_actor.on_start()
            self._started = True
    
    async def stop(self):
        """Stop the cache."""
        if self._started:
            await self.cache_actor.on_stop()
            self._started = False
    
    async def get(self, key: str) -> tuple[bool, Any]:
        """Get a value from cache. Returns (found, value)."""
        if not self._started:
            await self.start()
        
        # Create a mock request
        class MockRequest:
            def __init__(self, key: str):
                self.key = key
                self.id = f"req_{key}"
        
        request = MockRequest(key)
        response = await self.cache_actor.handle_cachegetrequest(request)
        
        return response.found, response.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache. Returns success status."""
        if not self._started:
            await self.start()
        
        # Create a mock request
        class MockRequest:
            def __init__(self, key: str, value: Any, ttl: Optional[int]):
                self.key = key
                self.value = value
                self.ttl = ttl
                self.id = f"req_{key}"
        
        request = MockRequest(key, value, ttl)
        response = await self.cache_actor.handle_cachesetrequest(request)
        
        return response.success
    
    async def health_check(self) -> Dict[str, Any]:
        """Get cache health status."""
        if not self._started:
            await self.start()
        
        # Create a mock ping message
        class MockMessage:
            pass
        
        return await self.cache_actor.handle_ping(MockMessage())


async def demo_simple_cache():
    """Demonstrate simple cache usage."""
    print("Redis Cache Actor Demo")
    print("=" * 40)
    
    # Create cache instance
    cache = SimpleCache("redis://localhost:6379")
    
    try:
        # Test basic operations
        print("1. Setting values...")
        await cache.set("user:123", {"name": "Alice", "age": 30}, ttl=60)
        await cache.set("config:theme", "dark", ttl=300)
        print("✓ Values set")
        
        print("\n2. Getting values...")
        found, user = await cache.get("user:123")
        print(f"User 123: found={found}, data={user}")
        
        found, theme = await cache.get("config:theme")
        print(f"Theme: found={found}, data={theme}")
        
        found, missing = await cache.get("nonexistent")
        print(f"Missing key: found={found}")
        
        print("\n3. Health check...")
        health = await cache.health_check()
        print(f"Health: {json.dumps(health, indent=2)}")
        
        print("\n4. Testing different data types...")
        test_data = [
            ("string", "Hello, World!"),
            ("number", 42),
            ("list", [1, 2, 3]),
            ("dict", {"nested": {"key": "value"}}),
            ("boolean", True),
        ]
        
        for key, value in test_data:
            await cache.set(f"test:{key}", value)
            found, retrieved = await cache.get(f"test:{key}")
            status = "✓" if found and retrieved == value else "✗"
            print(f"{status} {key}: {type(value).__name__}")
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await cache.stop()


if __name__ == "__main__":
    asyncio.run(demo_simple_cache())