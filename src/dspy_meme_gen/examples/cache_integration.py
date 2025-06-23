"""Example showing how to integrate the Redis cache actor with the meme generation service."""

import asyncio
import hashlib
from typing import Optional, Dict, Any

from ..actors import CacheActor, SerializationType
from ..actors.core import ActorSystem
from ..actors.messages import CacheGetRequest, CacheSetRequest
from ..config.config import settings


class CachedMemeService:
    """Example service that demonstrates cache-aside pattern with the cache actor."""
    
    def __init__(self, cache_actor_ref):
        self.cache_ref = cache_actor_ref
        
    async def generate_meme_with_cache(
        self, 
        prompt: str, 
        style: Optional[str] = None,
        user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """
        Generate a meme with caching support.
        
        This demonstrates the cache-aside pattern:
        1. Check cache first
        2. If miss, generate and cache the result
        3. Return the result
        """
        # Create cache key from prompt and style
        cache_key = self._create_cache_key(prompt, style)
        
        # Try to get from cache first
        try:
            get_request = CacheGetRequest(key=cache_key)
            cache_response = await self.cache_ref.ask(get_request, timeout=1000)  # 1 second timeout
            
            if cache_response.found:
                print(f"Cache hit for key: {cache_key}")
                return cache_response.value
                
        except Exception as e:
            print(f"Cache get failed, proceeding without cache: {e}")
        
        # Cache miss, generate meme
        print(f"Cache miss for key: {cache_key}, generating...")
        result = await self._generate_meme(prompt, style, user_id)
        
        # Cache the result
        try:
            set_request = CacheSetRequest(
                key=cache_key,
                value=result,
                ttl=settings.cache_ttl  # Use configured TTL
            )
            await self.cache_ref.ask(set_request, timeout=1000)
            print(f"Cached result for key: {cache_key}")
            
        except Exception as e:
            print(f"Cache set failed, but returning result: {e}")
        
        return result
    
    def _create_cache_key(self, prompt: str, style: Optional[str] = None) -> str:
        """Create a deterministic cache key from prompt and style."""
        key_data = f"meme:{prompt}:{style or 'default'}"
        return f"meme:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    async def _generate_meme(self, prompt: str, style: Optional[str], user_id: str) -> Dict[str, Any]:
        """Simulate meme generation (replace with actual implementation)."""
        # Simulate some processing time
        await asyncio.sleep(0.5)
        
        return {
            "id": f"meme_{hash(prompt)}",
            "prompt": prompt,
            "style": style,
            "caption": f"Generated meme for: {prompt}",
            "image_url": f"https://example.com/meme_{hash(prompt)}.jpg",
            "user_id": user_id,
            "generated_at": asyncio.get_event_loop().time(),
            "score": 0.8
        }
    
    async def invalidate_user_cache(self, user_id: str) -> int:
        """Invalidate all cached memes for a specific user."""
        # This would need to be implemented at the cache actor level
        # For now, we'll just clear all cache entries (not recommended for production)
        pattern = f"meme:*user:{user_id}*"
        # Note: This requires extending the cache actor to support pattern-based invalidation
        return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            ping_response = await self.cache_ref.ask({"type": "ping"}, timeout=1000)
            return ping_response.get("stats", {})
        except Exception as e:
            return {"error": str(e)}


async def demo_cache_integration():
    """Demonstrate cache integration with meme service."""
    print("Starting cache integration demo...")
    
    # Create actor system
    system = ActorSystem("cache_demo")
    await system.start()
    
    try:
        # Create and register cache actor
        cache_actor = CacheActor(
            name="meme_cache",
            redis_url=settings.redis_url,
            serialization=SerializationType.JSON,
            enable_fallback=True,
            default_ttl=settings.cache_ttl
        )
        
        cache_ref = await system.register_actor(cache_actor)
        
        # Create cached service
        service = CachedMemeService(cache_ref)
        
        # Test cache miss -> cache hit scenario
        prompt = "A cat wearing sunglasses"
        style = "vintage"
        
        print("\n1. First request (should be cache miss):")
        result1 = await service.generate_meme_with_cache(prompt, style)
        print(f"Result: {result1['id']}")
        
        print("\n2. Second request (should be cache hit):")
        result2 = await service.generate_meme_with_cache(prompt, style)
        print(f"Result: {result2['id']}")
        
        # Verify results are identical
        print(f"\n3. Results identical: {result1 == result2}")
        
        # Test with different prompts
        print("\n4. Testing different prompts:")
        prompts = [
            "A dog in a business suit",
            "A robot making coffee", 
            "A penguin at the beach"
        ]
        
        for prompt in prompts:
            result = await service.generate_meme_with_cache(prompt)
            print(f"Generated: {result['id']}")
        
        # Get cache statistics
        print("\n5. Cache statistics:")
        stats = await service.get_cache_stats()
        print(f"Stats: {stats}")
        
        # Test cache with concurrent requests
        print("\n6. Testing concurrent requests:")
        same_prompt = "A concurrent test meme"
        
        # Fire multiple requests simultaneously
        tasks = [
            service.generate_meme_with_cache(same_prompt)
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All results should be identical (first generates, rest hit cache)
        all_same = all(r == results[0] for r in results)
        print(f"All concurrent results identical: {all_same}")
        
        print("\nCache integration demo completed!")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await system.stop()


if __name__ == "__main__":
    asyncio.run(demo_cache_integration())