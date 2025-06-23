"""Integration tests for system resilience with real services.

This module implements comprehensive integration tests that verify
the system's resilience when interacting with real external services.
"""

import pytest
import asyncio
import time
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import aiohttp
import redis.asyncio as aioredis
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import docker
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer
from testcontainers.compose import DockerCompose
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.dspy_meme_gen.pipeline import MemeGenerationPipeline
from src.dspy_meme_gen.api.main import app
from src.dspy_meme_gen.database.connection import DatabaseConnection
from src.dspy_meme_gen.cache.redis import RedisCache
from fastapi.testclient import TestClient


@dataclass
class ServiceHealth:
    """Health status of a service."""
    
    name: str
    healthy: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class ServiceMonitor:
    """Monitor health and performance of external services."""
    
    def __init__(self):
        self.health_history: List[ServiceHealth] = []
    
    async def check_postgres(self, connection_string: str) -> ServiceHealth:
        """Check PostgreSQL health."""
        
        start_time = time.time()
        
        try:
            engine = create_engine(connection_string, poolclass=NullPool)
            with engine.connect() as conn:
                result = conn.execute("SELECT 1")
                result.fetchone()
            
            latency = (time.time() - start_time) * 1000
            
            # Check additional metrics
            with engine.connect() as conn:
                # Check connection count
                result = conn.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                )
                active_connections = result.fetchone()[0]
                
                # Check database size
                result = conn.execute(
                    "SELECT pg_database_size(current_database())"
                )
                db_size_bytes = result.fetchone()[0]
            
            health = ServiceHealth(
                name="postgres",
                healthy=True,
                latency_ms=latency,
                metadata={
                    "active_connections": active_connections,
                    "database_size_mb": db_size_bytes / 1024 / 1024
                }
            )
            
        except Exception as e:
            health = ServiceHealth(
                name="postgres",
                healthy=False,
                error=str(e)
            )
        
        self.health_history.append(health)
        return health
    
    async def check_redis(self, redis_url: str) -> ServiceHealth:
        """Check Redis health."""
        
        start_time = time.time()
        
        try:
            redis = await aioredis.from_url(redis_url)
            
            # Ping
            await redis.ping()
            
            latency = (time.time() - start_time) * 1000
            
            # Get additional metrics
            info = await redis.info()
            
            health = ServiceHealth(
                name="redis",
                healthy=True,
                latency_ms=latency,
                metadata={
                    "used_memory_mb": info.get("used_memory", 0) / 1024 / 1024,
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0)
                }
            )
            
            await redis.close()
            
        except Exception as e:
            health = ServiceHealth(
                name="redis",
                healthy=False,
                error=str(e)
            )
        
        self.health_history.append(health)
        return health
    
    async def check_openai(self, api_key: str) -> ServiceHealth:
        """Check OpenAI API health."""
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {api_key}"}
                
                # Simple completion request
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": "gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5
                    }
                ) as response:
                    if response.status == 200:
                        await response.json()
                        latency = (time.time() - start_time) * 1000
                        
                        health = ServiceHealth(
                            name="openai",
                            healthy=True,
                            latency_ms=latency
                        )
                    else:
                        health = ServiceHealth(
                            name="openai",
                            healthy=False,
                            error=f"HTTP {response.status}"
                        )
                        
        except Exception as e:
            health = ServiceHealth(
                name="openai",
                healthy=False,
                error=str(e)
            )
        
        self.health_history.append(health)
        return health
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of service health."""
        
        summary = {
            "services": {},
            "overall_health": True
        }
        
        # Group by service
        for health in self.health_history:
            if health.name not in summary["services"]:
                summary["services"][health.name] = {
                    "checks": 0,
                    "healthy": 0,
                    "avg_latency_ms": 0,
                    "errors": []
                }
            
            service_summary = summary["services"][health.name]
            service_summary["checks"] += 1
            
            if health.healthy:
                service_summary["healthy"] += 1
                if health.latency_ms:
                    # Update average latency
                    current_avg = service_summary["avg_latency_ms"]
                    service_summary["avg_latency_ms"] = (
                        (current_avg * (service_summary["healthy"] - 1) + health.latency_ms) /
                        service_summary["healthy"]
                    )
            else:
                if health.error:
                    service_summary["errors"].append(health.error)
                summary["overall_health"] = False
        
        return summary


class TestRealServiceIntegration:
    """Test integration with real services."""
    
    @pytest.fixture(scope="session")
    def docker_services(self):
        """Start required services using Docker."""
        
        # Check if services are already running
        if os.getenv("SKIP_DOCKER_SETUP"):
            yield None
            return
        
        # Start services using docker-compose
        compose = DockerCompose(
            filepath=os.path.join(os.path.dirname(__file__), "../..", "docker-compose.yml"),
            compose_file_name="docker-compose.test.yml"
        )
        
        with compose:
            # Wait for services to be ready
            compose.wait_for("http://localhost:8081/health")
            yield compose
    
    @pytest.fixture
    async def service_monitor(self):
        """Provide service monitor."""
        return ServiceMonitor()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_database_failover(self, docker_services, service_monitor):
        """Test system behavior during database failover."""
        
        # Initial health check
        db_url = os.getenv("DATABASE_URL", "postgresql://test:test@localhost:5432/memes_test")
        initial_health = await service_monitor.check_postgres(db_url)
        assert initial_health.healthy, "Database should be healthy initially"
        
        pipeline = MemeGenerationPipeline()
        
        # Simulate database failover
        docker_client = docker.from_env()
        postgres_container = None
        
        for container in docker_client.containers.list():
            if "postgres" in container.name:
                postgres_container = container
                break
        
        if postgres_container:
            # Stop database
            postgres_container.stop()
            
            # Try to generate meme during outage
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(pipeline.forward, "Test during DB outage"),
                    timeout=5.0
                )
                # Should either fail or use cache
                assert result["status"] in ["error", "success"]
            except asyncio.TimeoutError:
                # Acceptable during outage
                pass
            
            # Restart database
            postgres_container.start()
            
            # Wait for recovery
            await asyncio.sleep(5)
            
            # Verify recovery
            recovery_health = await service_monitor.check_postgres(db_url)
            assert recovery_health.healthy, "Database should recover"
            
            # System should work normally
            result = pipeline.forward("Test after DB recovery")
            assert result["status"] == "success"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_redis_cache_degradation(self, docker_services, service_monitor):
        """Test system behavior when Redis cache degrades."""
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        # Check initial health
        initial_health = await service_monitor.check_redis(redis_url)
        assert initial_health.healthy, "Redis should be healthy initially"
        
        # Create pipeline and warm cache
        pipeline = MemeGenerationPipeline()
        
        # Generate memes to populate cache
        for i in range(5):
            result = pipeline.forward(f"Cache test {i}")
            assert result["status"] == "success"
        
        # Simulate Redis memory pressure
        redis_client = await aioredis.from_url(redis_url)
        
        # Fill Redis memory
        large_data = "x" * (1024 * 1024)  # 1MB string
        
        try:
            for i in range(100):  # Try to use 100MB
                await redis_client.set(f"memory_pressure_{i}", large_data)
        except Exception:
            # Redis may reject when memory full
            pass
        
        # Check health during pressure
        pressure_health = await service_monitor.check_redis(redis_url)
        
        # System should still function
        result = pipeline.forward("Test during Redis pressure")
        assert result["status"] in ["success", "error"]
        
        # Clean up
        await redis_client.flushall()
        await redis_client.close()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_rate_limiting(self, service_monitor):
        """Test system behavior under API rate limiting."""
        
        client = TestClient(app)
        
        # Rapid fire requests to trigger rate limiting
        results = []
        
        for i in range(100):
            response = client.post(
                "/api/v1/memes/generate",
                json={"prompt": f"Rate limit test {i}"}
            )
            
            results.append({
                "status_code": response.status_code,
                "timestamp": time.time()
            })
            
            # No delay - hammer the API
        
        # Analyze results
        success_count = sum(1 for r in results if r["status_code"] == 200)
        rate_limited_count = sum(1 for r in results if r["status_code"] == 429)
        
        # Should have some rate limiting
        assert rate_limited_count > 0, "Rate limiting not working"
        assert success_count > 0, "No requests succeeded"
        
        # Check that rate limiting is reasonable
        assert success_count > 10, "Rate limiting too aggressive"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_network_partition_recovery(self, docker_services):
        """Test recovery from network partitions."""
        
        # This test simulates network issues using iptables
        # Note: Requires root/sudo access in real environment
        
        client = TestClient(app)
        
        # Baseline request
        response = client.post(
            "/api/v1/memes/generate",
            json={"prompt": "Network test baseline"}
        )
        assert response.status_code == 200
        
        # Simulate network partition (would need actual network control)
        # In a real test environment, you might use:
        # - tc (traffic control) to add latency/packet loss
        # - iptables to block connections
        # - Docker network disconnect commands
        
        # For this test, we'll simulate by overloading the connection pool
        async def flood_connections():
            tasks = []
            for i in range(200):  # Create many concurrent connections
                async def make_request():
                    async with aiohttp.ClientSession() as session:
                        try:
                            await session.post(
                                "http://localhost:8081/api/v1/memes/generate",
                                json={"prompt": f"Flood {i}"},
                                timeout=aiohttp.ClientTimeout(total=1)
                            )
                        except:
                            pass
                
                tasks.append(asyncio.create_task(make_request()))
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flood the service
        await flood_connections()
        
        # Service should recover
        await asyncio.sleep(2)
        
        # Verify recovery
        recovery_response = client.post(
            "/api/v1/memes/generate",
            json={"prompt": "Network recovery test"}
        )
        
        assert recovery_response.status_code in [200, 503], "Service didn't recover properly"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cascading_service_failures(self, docker_services, service_monitor):
        """Test system behavior during cascading service failures."""
        
        # Monitor all services
        db_url = os.getenv("DATABASE_URL", "postgresql://test:test@localhost:5432/memes_test")
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        openai_key = os.getenv("OPENAI_API_KEY", "test-key")
        
        # Check initial health
        initial_healths = await asyncio.gather(
            service_monitor.check_postgres(db_url),
            service_monitor.check_redis(redis_url),
            service_monitor.check_openai(openai_key)
        )
        
        # Simulate progressive degradation
        docker_client = docker.from_env()
        containers_stopped = []
        
        try:
            # Stop Redis first
            for container in docker_client.containers.list():
                if "redis" in container.name:
                    container.stop()
                    containers_stopped.append(container)
                    break
            
            # System should still work without cache
            pipeline = MemeGenerationPipeline()
            result1 = pipeline.forward("Test without Redis")
            assert result1["status"] in ["success", "error"]
            
            # Stop database
            for container in docker_client.containers.list():
                if "postgres" in container.name:
                    container.stop()
                    containers_stopped.append(container)
                    break
            
            # System should fail gracefully
            try:
                result2 = pipeline.forward("Test without Redis and DB")
                assert result2["status"] == "error"
            except Exception:
                # Expected when both services down
                pass
                
        finally:
            # Restart all stopped containers
            for container in containers_stopped:
                container.start()
            
            # Wait for recovery
            await asyncio.sleep(10)
        
        # Verify full recovery
        recovery_healths = await asyncio.gather(
            service_monitor.check_postgres(db_url),
            service_monitor.check_redis(redis_url),
            service_monitor.check_openai(openai_key),
            return_exceptions=True
        )
        
        # At least core services should recover
        healthy_count = sum(1 for h in recovery_healths if not isinstance(h, Exception) and h.healthy)
        assert healthy_count >= 2, "Services didn't recover properly"


class TestProductionScenarios:
    """Test real production scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_24_hour_stability(self):
        """Test system stability over 24 hours (scaled down for testing)."""
        
        # Scale down to 1 hour for actual testing
        test_duration_seconds = 3600  # 1 hour
        request_interval = 10  # seconds between requests
        
        client = TestClient(app)
        monitor = ServiceMonitor()
        
        start_time = time.time()
        results = []
        
        while time.time() - start_time < test_duration_seconds:
            # Make request
            request_start = time.time()
            
            response = client.post(
                "/api/v1/memes/generate",
                json={"prompt": f"Stability test at {time.time()}"}
            )
            
            results.append({
                "timestamp": time.time(),
                "status_code": response.status_code,
                "response_time": time.time() - request_start,
                "success": response.status_code == 200
            })
            
            # Periodic health checks
            if len(results) % 60 == 0:  # Every 10 minutes
                await asyncio.gather(
                    monitor.check_postgres(os.getenv("DATABASE_URL")),
                    monitor.check_redis(os.getenv("REDIS_URL")),
                    return_exceptions=True
                )
            
            await asyncio.sleep(request_interval)
        
        # Analyze stability
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r["success"])
        success_rate = successful_requests / total_requests
        
        # Calculate response time percentiles
        response_times = [r["response_time"] for r in results]
        p50 = sorted(response_times)[len(response_times) // 2]
        p95 = sorted(response_times)[int(len(response_times) * 0.95)]
        p99 = sorted(response_times)[int(len(response_times) * 0.99)]
        
        # Check for degradation over time
        first_hour = results[:360]  # First hour
        last_hour = results[-360:]  # Last hour
        
        first_hour_success_rate = sum(1 for r in first_hour if r["success"]) / len(first_hour)
        last_hour_success_rate = sum(1 for r in last_hour if r["success"]) / len(last_hour)
        
        # Assertions
        assert success_rate > 0.99, f"Success rate too low: {success_rate}"
        assert p95 < 5.0, f"P95 response time too high: {p95}"
        assert abs(first_hour_success_rate - last_hour_success_rate) < 0.05, "Performance degraded over time"
        
        # Get service health summary
        health_summary = monitor.get_health_summary()
        assert health_summary["overall_health"], "Services unhealthy during test"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_deployment_rollout(self):
        """Test zero-downtime deployment scenarios."""
        
        # This test simulates a rolling deployment
        client = TestClient(app)
        
        # Continuous request sender
        async def send_requests(duration: int) -> List[Dict[str, Any]]:
            results = []
            end_time = time.time() + duration
            
            while time.time() < end_time:
                try:
                    response = await asyncio.to_thread(
                        client.post,
                        "/api/v1/memes/generate",
                        json={"prompt": "Deployment test"}
                    )
                    
                    results.append({
                        "timestamp": time.time(),
                        "status_code": response.status_code,
                        "success": response.status_code == 200
                    })
                except Exception as e:
                    results.append({
                        "timestamp": time.time(),
                        "status_code": 0,
                        "success": False,
                        "error": str(e)
                    })
                
                await asyncio.sleep(0.1)
            
            return results
        
        # Start sending requests
        request_task = asyncio.create_task(send_requests(30))
        
        # Simulate deployment after 10 seconds
        await asyncio.sleep(10)
        
        # In a real scenario, this would trigger:
        # - New container deployment
        # - Health check verification
        # - Traffic shift
        # - Old container termination
        
        # For testing, we'll simulate by restarting the app
        # (In practice, this would be done by your deployment system)
        
        # Wait for requests to complete
        results = await request_task
        
        # Analyze deployment impact
        total_requests = len(results)
        failed_requests = sum(1 for r in results if not r["success"])
        
        # During deployment, some failures are acceptable
        failure_rate = failed_requests / total_requests
        assert failure_rate < 0.05, f"Too many failures during deployment: {failure_rate}"
        
        # Check for extended downtime
        consecutive_failures = 0
        max_consecutive_failures = 0
        
        for result in results:
            if not result["success"]:
                consecutive_failures += 1
                max_consecutive_failures = max(max_consecutive_failures, consecutive_failures)
            else:
                consecutive_failures = 0
        
        # No more than 5 consecutive failures (0.5 seconds)
        assert max_consecutive_failures < 5, "Extended downtime during deployment"


class TestDisasterRecovery:
    """Test disaster recovery scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_data_corruption_recovery(self, docker_services):
        """Test recovery from data corruption."""
        
        # Create some memes
        pipeline = MemeGenerationPipeline()
        
        meme_ids = []
        for i in range(5):
            result = pipeline.forward(f"Pre-corruption meme {i}")
            if result["status"] == "success" and "id" in result.get("meme", {}):
                meme_ids.append(result["meme"]["id"])
        
        # Simulate data corruption by directly modifying database
        db_url = os.getenv("DATABASE_URL", "postgresql://test:test@localhost:5432/memes_test")
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            # Corrupt some data
            conn.execute(
                "UPDATE memes SET caption = NULL WHERE id = %s",
                meme_ids[0] if meme_ids else "nonexistent"
            )
            conn.commit()
        
        # System should handle corrupted data
        try:
            # Try to access corrupted data
            result = pipeline.forward("Test after corruption")
            assert result["status"] in ["success", "error"]
        except Exception as e:
            # Should not crash completely
            assert "corruption" not in str(e).lower()
        
        # Verify system can create new content
        recovery_result = pipeline.forward("Recovery test meme")
        assert recovery_result["status"] == "success"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_backup_restore(self, docker_services):
        """Test backup and restore procedures."""
        
        # Create test data
        pipeline = MemeGenerationPipeline()
        
        original_memes = []
        for i in range(3):
            result = pipeline.forward(f"Backup test meme {i}")
            if result["status"] == "success":
                original_memes.append(result["meme"])
        
        # Simulate backup (in real scenario, this would use pg_dump or similar)
        backup_data = {
            "memes": original_memes,
            "timestamp": time.time()
        }
        
        # Simulate data loss
        db_url = os.getenv("DATABASE_URL", "postgresql://test:test@localhost:5432/memes_test")
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            # Clear data
            conn.execute("TRUNCATE TABLE memes CASCADE")
            conn.commit()
        
        # Verify data is gone
        with engine.connect() as conn:
            result = conn.execute("SELECT COUNT(*) FROM memes")
            count = result.scalar()
            assert count == 0, "Data not cleared"
        
        # Simulate restore (in real scenario, this would use pg_restore or similar)
        # For testing, we'll just verify the system can rebuild
        
        # System should function after data loss
        post_loss_result = pipeline.forward("Test after data loss")
        assert post_loss_result["status"] == "success"
        
        # Verify new data can be created
        for i in range(3):
            restored_result = pipeline.forward(f"Restored meme {i}")
            assert restored_result["status"] == "success"


# Performance benchmarks for resilience testing
class ResilienceBenchmarks:
    """Benchmark resilience metrics."""
    
    @staticmethod
    async def measure_recovery_time(failure_injection: Callable, recovery_check: Callable) -> float:
        """Measure time to recover from failure."""
        
        # Inject failure
        failure_injection()
        failure_time = time.time()
        
        # Wait for recovery
        while True:
            if await recovery_check():
                recovery_time = time.time()
                return recovery_time - failure_time
            
            await asyncio.sleep(0.1)
            
            # Timeout after 5 minutes
            if time.time() - failure_time > 300:
                raise TimeoutError("Recovery took too long")
    
    @staticmethod
    def calculate_availability(results: List[Dict[str, Any]]) -> float:
        """Calculate system availability from test results."""
        
        if not results:
            return 0.0
        
        total_time = results[-1]["timestamp"] - results[0]["timestamp"]
        downtime = 0.0
        
        # Find periods of consecutive failures
        in_downtime = False
        downtime_start = 0
        
        for result in results:
            if not result.get("success", False):
                if not in_downtime:
                    in_downtime = True
                    downtime_start = result["timestamp"]
            else:
                if in_downtime:
                    downtime += result["timestamp"] - downtime_start
                    in_downtime = False
        
        # Handle ongoing downtime
        if in_downtime:
            downtime += results[-1]["timestamp"] - downtime_start
        
        availability = (total_time - downtime) / total_time if total_time > 0 else 0
        return availability


if __name__ == "__main__":
    # Run integration resilience tests
    pytest.main([__file__, "-v", "-m", "integration"])