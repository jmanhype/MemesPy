"""Chaos engineering tests for MemesPy system.

This module implements chaos engineering scenarios to test system resilience
under various failure conditions and unexpected behaviors.
"""

import pytest
import asyncio
import random
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import signal
import os
import psutil
from contextlib import contextmanager

from src.dspy_meme_gen.pipeline import MemeGenerationPipeline
from src.dspy_meme_gen.agents.router import RouterAgent
from src.dspy_meme_gen.agents.image_renderer import ImageRenderingAgent
from src.dspy_meme_gen.cache.redis import RedisCache
from src.dspy_meme_gen.database.connection import DatabaseConnection


class ChaosMonkey:
    """Chaos engineering utilities for injecting failures."""
    
    def __init__(self, failure_rate: float = 0.1):
        self.failure_rate = failure_rate
        self.active = True
        self.failure_counts: Dict[str, int] = {}
    
    def maybe_fail(self, component: str, exception_type: type = Exception):
        """Randomly fail based on failure rate."""
        if self.active and random.random() < self.failure_rate:
            self.failure_counts[component] = self.failure_counts.get(component, 0) + 1
            raise exception_type(f"Chaos monkey killed {component}")
    
    def inject_latency(self, min_ms: int = 100, max_ms: int = 5000):
        """Inject random latency."""
        if self.active:
            latency = random.randint(min_ms, max_ms) / 1000
            time.sleep(latency)
    
    def corrupt_data(self, data: Any) -> Any:
        """Randomly corrupt data."""
        if not self.active or random.random() > self.failure_rate:
            return data
        
        if isinstance(data, dict):
            # Remove random keys
            if data and random.random() < 0.5:
                key_to_remove = random.choice(list(data.keys()))
                data = {k: v for k, v in data.items() if k != key_to_remove}
            # Add random keys
            if random.random() < 0.5:
                data[f"chaos_{random.randint(1000, 9999)}"] = "corrupted"
        elif isinstance(data, str):
            # Truncate string
            if len(data) > 10:
                data = data[:len(data) // 2]
        elif isinstance(data, list):
            # Remove random elements
            if data and random.random() < 0.5:
                data = data[:-1]
        
        return data


@pytest.fixture
def chaos_monkey():
    """Provide a chaos monkey instance."""
    return ChaosMonkey(failure_rate=0.3)


class TestNetworkChaos:
    """Test system behavior under network failures."""
    
    @pytest.mark.asyncio
    async def test_openai_api_timeout(self, chaos_monkey):
        """Test behavior when OpenAI API times out."""
        
        async def mock_openai_call(*args, **kwargs):
            chaos_monkey.inject_latency(5000, 10000)  # 5-10 second delay
            raise TimeoutError("OpenAI API timeout")
        
        with patch('openai.ChatCompletion.create', side_effect=mock_openai_call):
            pipeline = MemeGenerationPipeline()
            
            start = time.time()
            try:
                result = await pipeline.forward("Create a meme about timeouts")
                assert False, "Should have failed with timeout"
            except Exception as e:
                duration = time.time() - start
                assert duration < 15, "Should have failed fast"
                assert "timeout" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_intermittent_network_failures(self, chaos_monkey):
        """Test system behavior with intermittent network failures."""
        
        failure_sequence = [True, True, False, True, False, False, True]
        call_count = 0
        
        async def mock_network_call(*args, **kwargs):
            nonlocal call_count
            should_fail = failure_sequence[call_count % len(failure_sequence)]
            call_count += 1
            
            if should_fail:
                chaos_monkey.maybe_fail("network", ConnectionError)
            
            return {"status": "success"}
        
        with patch('aiohttp.ClientSession.post', side_effect=mock_network_call):
            pipeline = MemeGenerationPipeline()
            
            results = []
            for i in range(10):
                try:
                    result = await pipeline.forward(f"Create meme {i}")
                    results.append(("success", result))
                except Exception as e:
                    results.append(("failure", str(e)))
            
            # Should have mix of successes and failures
            successes = [r for r in results if r[0] == "success"]
            failures = [r for r in results if r[0] == "failure"]
            
            assert len(successes) > 0, "Should have some successes"
            assert len(failures) > 0, "Should have some failures"
            assert len(successes) + len(failures) == 10
    
    def test_dns_resolution_failure(self):
        """Test behavior when DNS resolution fails."""
        
        def mock_getaddrinfo(*args):
            raise OSError("DNS resolution failed")
        
        with patch('socket.getaddrinfo', side_effect=mock_getaddrinfo):
            pipeline = MemeGenerationPipeline()
            
            try:
                result = pipeline.forward("Create a meme")
                assert False, "Should have failed with DNS error"
            except Exception as e:
                assert "DNS" in str(e) or "resolution" in str(e).lower()


class TestDatabaseChaos:
    """Test system behavior under database failures."""
    
    @pytest.mark.asyncio
    async def test_database_connection_pool_exhaustion(self, chaos_monkey):
        """Test behavior when database connection pool is exhausted."""
        
        # Mock connection pool that gets exhausted
        class ExhaustedPool:
            def __init__(self):
                self.connections = 0
                self.max_connections = 5
            
            async def acquire(self):
                if self.connections >= self.max_connections:
                    chaos_monkey.maybe_fail("db_pool", TimeoutError)
                    raise TimeoutError("Connection pool exhausted")
                self.connections += 1
                return Mock()
            
            async def release(self, conn):
                self.connections -= 1
        
        pool = ExhaustedPool()
        
        async def stress_test():
            tasks = []
            for i in range(20):  # Try to acquire more connections than available
                async def acquire_and_hold():
                    try:
                        conn = await pool.acquire()
                        await asyncio.sleep(0.1)  # Hold connection
                        await pool.release(conn)
                    except TimeoutError:
                        pass
                
                tasks.append(asyncio.create_task(acquire_and_hold()))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should have some failures due to pool exhaustion
            failures = [r for r in results if isinstance(r, Exception)]
            assert len(failures) > 0, "Should have connection pool failures"
    
    def test_database_corruption(self, chaos_monkey):
        """Test behavior when database returns corrupted data."""
        
        def mock_query_result(*args, **kwargs):
            normal_result = {
                "id": "123",
                "topic": "programming",
                "format": "drake",
                "caption": "Test caption"
            }
            
            # Corrupt the result
            return chaos_monkey.corrupt_data(normal_result)
        
        with patch('sqlalchemy.engine.Result.fetchone', side_effect=mock_query_result):
            pipeline = MemeGenerationPipeline()
            
            # Run multiple times to test different corruption scenarios
            failures = 0
            for i in range(10):
                try:
                    result = pipeline.forward("Create a meme")
                except Exception:
                    failures += 1
            
            # Should handle some corrupted data
            assert failures < 10, "Should handle some corrupted data"
    
    def test_database_deadlock(self):
        """Test behavior during database deadlocks."""
        
        deadlock_count = 0
        
        def mock_execute(*args, **kwargs):
            nonlocal deadlock_count
            deadlock_count += 1
            
            if deadlock_count < 3:
                # Simulate deadlock for first few attempts
                raise Exception("Deadlock detected")
            
            # Succeed after retries
            return Mock(fetchone=lambda: {"id": "123"})
        
        with patch('sqlalchemy.engine.Connection.execute', side_effect=mock_execute):
            # Pipeline should handle deadlocks with retries
            pipeline = MemeGenerationPipeline()
            result = pipeline.forward("Create a meme")
            
            assert deadlock_count >= 3, "Should have retried on deadlock"


class TestMemoryChaos:
    """Test system behavior under memory pressure."""
    
    def test_memory_leak_simulation(self):
        """Simulate memory leak and test system behavior."""
        
        memory_hog = []
        
        def leak_memory():
            # Allocate 10MB of memory
            memory_hog.append(bytearray(10 * 1024 * 1024))
        
        pipeline = MemeGenerationPipeline()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Generate memes while leaking memory
        for i in range(10):
            try:
                leak_memory()
                result = pipeline.forward(f"Create meme {i}")
            except MemoryError:
                # System should handle memory errors gracefully
                break
        
        # Check memory usage
        final_memory = process.memory_info().rss
        memory_increase_mb = (final_memory - initial_memory) / (1024 * 1024)
        
        # Clear memory hog
        memory_hog.clear()
        
        assert memory_increase_mb < 200, f"Memory leak too severe: {memory_increase_mb}MB"
    
    def test_out_of_memory_handling(self):
        """Test behavior when system runs out of memory."""
        
        def mock_image_generation(*args, **kwargs):
            # Try to allocate huge amount of memory
            try:
                huge_array = bytearray(10 * 1024 * 1024 * 1024)  # 10GB
            except MemoryError:
                raise MemoryError("Out of memory during image generation")
        
        with patch.object(ImageRenderingAgent, 'forward', side_effect=mock_image_generation):
            pipeline = MemeGenerationPipeline()
            
            try:
                result = pipeline.forward("Create a meme")
                assert False, "Should have failed with memory error"
            except Exception as e:
                assert "memory" in str(e).lower()


class TestConcurrencyChaos:
    """Test system behavior under concurrency issues."""
    
    @pytest.mark.asyncio
    async def test_race_conditions(self):
        """Test for race conditions in concurrent meme generation."""
        
        shared_state = {"counter": 0, "results": []}
        
        async def generate_meme_with_shared_state(idx: int):
            pipeline = MemeGenerationPipeline()
            
            # Simulate race condition on shared state
            current = shared_state["counter"]
            await asyncio.sleep(random.random() * 0.01)  # Random delay
            shared_state["counter"] = current + 1
            
            try:
                result = await pipeline.forward(f"Create meme {idx}")
                shared_state["results"].append((idx, "success"))
            except Exception as e:
                shared_state["results"].append((idx, "failure"))
        
        # Run concurrent generations
        tasks = [generate_meme_with_shared_state(i) for i in range(20)]
        await asyncio.gather(*tasks)
        
        # Check for race condition effects
        assert shared_state["counter"] <= 20, "Race condition detected in counter"
        assert len(shared_state["results"]) == 20, "Some results were lost"
    
    def test_deadlock_detection(self):
        """Test system behavior under potential deadlock conditions."""
        
        lock1 = threading.Lock()
        lock2 = threading.Lock()
        deadlock_detected = False
        
        def task1():
            nonlocal deadlock_detected
            with lock1:
                time.sleep(0.1)
                try:
                    # Try to acquire lock2 with timeout
                    acquired = lock2.acquire(timeout=1)
                    if not acquired:
                        deadlock_detected = True
                    else:
                        lock2.release()
                except Exception:
                    deadlock_detected = True
        
        def task2():
            with lock2:
                time.sleep(0.1)
                try:
                    # Try to acquire lock1 with timeout
                    acquired = lock1.acquire(timeout=1)
                    if not acquired:
                        deadlock_detected = True
                    else:
                        lock1.release()
                except Exception:
                    deadlock_detected = True
        
        # Run tasks that could deadlock
        t1 = threading.Thread(target=task1)
        t2 = threading.Thread(target=task2)
        
        t1.start()
        t2.start()
        
        t1.join(timeout=3)
        t2.join(timeout=3)
        
        assert deadlock_detected, "System should detect potential deadlocks"


class TestCacheChaos:
    """Test system behavior under cache failures."""
    
    @pytest.mark.asyncio
    async def test_cache_inconsistency(self, chaos_monkey):
        """Test behavior when cache becomes inconsistent with database."""
        
        # Mock cache that sometimes returns stale data
        class InconsistentCache:
            def __init__(self):
                self.data = {}
                self.version = 0
            
            async def get(self, key: str):
                if chaos_monkey.active and random.random() < 0.3:
                    # Return stale data
                    return {"version": self.version - 1, "data": "stale"}
                return self.data.get(key)
            
            async def set(self, key: str, value: Any):
                self.version += 1
                self.data[key] = {"version": self.version, "data": value}
        
        cache = InconsistentCache()
        
        with patch('src.dspy_meme_gen.cache.redis.RedisCache', return_value=cache):
            pipeline = MemeGenerationPipeline()
            
            # Generate same meme multiple times
            results = []
            for i in range(10):
                result = await pipeline.forward("Create consistent meme")
                results.append(result)
            
            # Despite cache inconsistency, results should be valid
            for result in results:
                assert result["status"] in ["success", "error"]
    
    def test_cache_avalanche(self):
        """Test behavior during cache avalanche (all cache entries expire at once)."""
        
        expire_time = time.time() + 1
        
        class AvalancheCache:
            def __init__(self):
                self.data = {}
            
            async def get(self, key: str):
                if time.time() > expire_time:
                    # All cache entries expired
                    return None
                return self.data.get(key)
            
            async def set(self, key: str, value: Any):
                self.data[key] = value
        
        cache = AvalancheCache()
        
        with patch('src.dspy_meme_gen.cache.redis.RedisCache', return_value=cache):
            pipeline = MemeGenerationPipeline()
            
            # Warm up cache
            for i in range(5):
                pipeline.forward(f"Meme {i}")
            
            # Wait for cache to expire
            time.sleep(1.1)
            
            # All requests hit database now (cache avalanche)
            start = time.time()
            for i in range(5):
                pipeline.forward(f"Meme {i}")
            duration = time.time() - start
            
            # System should handle avalanche without significant degradation
            assert duration < 10, "Cache avalanche caused severe performance degradation"


class TestTimeBasedChaos:
    """Test system behavior under time-based anomalies."""
    
    def test_clock_skew(self):
        """Test behavior when system clock is skewed."""
        
        original_time = time.time
        
        def skewed_time():
            # Simulate clock jumping forward/backward randomly
            skew = random.choice([-3600, -60, 0, 60, 3600])  # ±1 hour, ±1 min
            return original_time() + skew
        
        with patch('time.time', side_effect=skewed_time):
            pipeline = MemeGenerationPipeline()
            
            # System should handle clock skew
            results = []
            for i in range(5):
                try:
                    result = pipeline.forward(f"Meme at time {i}")
                    results.append("success")
                except Exception as e:
                    if "time" in str(e).lower():
                        results.append("time_error")
                    else:
                        results.append("other_error")
            
            # Should mostly succeed despite clock skew
            success_rate = results.count("success") / len(results)
            assert success_rate > 0.6, f"Clock skew caused too many failures: {results}"
    
    def test_processing_timeout_cascade(self):
        """Test cascading timeouts when one component slows down."""
        
        def slow_component(*args, **kwargs):
            time.sleep(5)  # Simulate slow component
            return {"result": "slow"}
        
        with patch.object(RouterAgent, 'forward', side_effect=slow_component):
            pipeline = MemeGenerationPipeline()
            
            start = time.time()
            try:
                # Should timeout before 10 seconds
                result = pipeline.forward("Create meme quickly")
                duration = time.time() - start
                assert duration < 10, "Timeout cascade prevention failed"
            except Exception as e:
                duration = time.time() - start
                assert duration < 10, "Timeout cascade prevention failed"
                assert "timeout" in str(e).lower()


class TestResourceExhaustionChaos:
    """Test system behavior under resource exhaustion."""
    
    def test_file_descriptor_exhaustion(self):
        """Test behavior when file descriptors are exhausted."""
        
        open_files = []
        
        try:
            # Try to exhaust file descriptors
            for i in range(1000):
                f = open(f"/tmp/chaos_test_{i}.txt", "w")
                open_files.append(f)
        except OSError:
            # File descriptors exhausted
            pass
        
        try:
            pipeline = MemeGenerationPipeline()
            result = pipeline.forward("Create meme with no file descriptors")
            
            # Should handle file descriptor exhaustion
            assert result["status"] in ["success", "error"]
        finally:
            # Clean up
            for f in open_files:
                try:
                    f.close()
                    os.unlink(f.name)
                except:
                    pass
    
    @pytest.mark.asyncio
    async def test_thread_pool_exhaustion(self):
        """Test behavior when thread pool is exhausted."""
        
        async def exhaust_thread_pool():
            tasks = []
            
            # Create many blocking tasks
            def blocking_task():
                time.sleep(10)
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit more tasks than threads
                for i in range(50):
                    future = executor.submit(blocking_task)
                    tasks.append(future)
                
                # Try to use pipeline while threads are exhausted
                pipeline = MemeGenerationPipeline()
                
                try:
                    # Should handle thread exhaustion
                    result = await asyncio.wait_for(
                        pipeline.forward("Create meme"), 
                        timeout=5
                    )
                except asyncio.TimeoutError:
                    # Expected when threads are exhausted
                    pass
                except Exception as e:
                    # Should be a resource-related error
                    assert "thread" in str(e).lower() or "resource" in str(e).lower()
        
        await exhaust_thread_pool()


# Chaos testing utilities
@contextmanager
def chaos_environment(failure_rate: float = 0.3, components: List[str] = None):
    """Context manager for chaos testing environment."""
    
    chaos = ChaosMonkey(failure_rate)
    
    if components is None:
        components = ["network", "database", "cache", "compute"]
    
    patches = []
    
    try:
        # Inject failures into specified components
        for component in components:
            if component == "network":
                mock = Mock(side_effect=lambda *a, **k: chaos.maybe_fail("network", ConnectionError))
                patch_obj = patch('aiohttp.ClientSession.post', mock)
                patches.append(patch_obj)
                patch_obj.start()
            
            elif component == "database":
                mock = Mock(side_effect=lambda *a, **k: chaos.maybe_fail("database", TimeoutError))
                patch_obj = patch('sqlalchemy.engine.Connection.execute', mock)
                patches.append(patch_obj)
                patch_obj.start()
            
            elif component == "cache":
                mock = Mock(side_effect=lambda *a, **k: chaos.maybe_fail("cache", ConnectionError))
                patch_obj = patch('redis.asyncio.Redis.get', mock)
                patches.append(patch_obj)
                patch_obj.start()
        
        yield chaos
        
    finally:
        # Stop all patches
        for patch_obj in patches:
            patch_obj.stop()


def run_chaos_scenario(name: str, duration_seconds: int = 60, failure_rate: float = 0.3):
    """Run a chaos scenario for a specified duration."""
    
    print(f"Starting chaos scenario: {name}")
    print(f"Duration: {duration_seconds}s, Failure rate: {failure_rate}")
    
    with chaos_environment(failure_rate=failure_rate) as chaos:
        pipeline = MemeGenerationPipeline()
        
        start_time = time.time()
        results = {"success": 0, "failure": 0, "errors": {}}
        
        while time.time() - start_time < duration_seconds:
            try:
                result = pipeline.forward(f"Create meme during {name}")
                results["success"] += 1
            except Exception as e:
                results["failure"] += 1
                error_type = type(e).__name__
                results["errors"][error_type] = results["errors"].get(error_type, 0) + 1
            
            time.sleep(0.1)  # Small delay between requests
        
        # Report results
        total = results["success"] + results["failure"]
        success_rate = results["success"] / total if total > 0 else 0
        
        print(f"\nChaos scenario '{name}' completed:")
        print(f"  Total requests: {total}")
        print(f"  Successful: {results['success']} ({success_rate:.1%})")
        print(f"  Failed: {results['failure']}")
        print(f"  Error breakdown: {results['errors']}")
        print(f"  Chaos failures: {chaos.failure_counts}")
        
        return results


if __name__ == "__main__":
    # Run various chaos scenarios
    scenarios = [
        ("Network Chaos", 30, 0.2),
        ("Database Chaos", 30, 0.3),
        ("Full System Chaos", 60, 0.4),
    ]
    
    for name, duration, rate in scenarios:
        run_chaos_scenario(name, duration, rate)
        print("\n" + "="*50 + "\n")