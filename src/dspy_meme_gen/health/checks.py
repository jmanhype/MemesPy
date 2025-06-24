"""Health check implementation."""

from typing import Dict, Tuple

import psutil
from sqlalchemy.ext.asyncio import AsyncEngine
from redis.asyncio import Redis


class HealthCheck:
    """Health check implementation."""

    def __init__(self, db: AsyncEngine, redis: Redis) -> None:
        """Initialize health check.

        Args:
            db: Database engine.
            redis: Redis client.
        """
        self.db = db
        self.redis = redis

    async def check_health(self) -> Tuple[bool, Dict[str, str]]:
        """Check overall health of the service.

        Returns:
            Tuple[bool, Dict[str, str]]: Tuple of (is_healthy, details).
        """
        # Check database
        db_healthy, db_details = await self.check_database()

        # Check Redis
        redis_healthy, redis_details = await self.check_redis()

        # Check system resources
        system_healthy, system_details = self.check_system()

        # Check metrics
        metrics_healthy, metrics_details = self.check_metrics()

        # Aggregate results
        is_healthy = all([db_healthy, redis_healthy, system_healthy, metrics_healthy])

        details = {
            "status": "healthy" if is_healthy else "unhealthy",
            "database": db_details,
            "redis": redis_details,
            "system": system_details,
            "metrics": metrics_details,
        }

        return is_healthy, details

    async def check_database(self) -> Tuple[bool, Dict[str, str]]:
        """Check database health.

        Returns:
            Tuple[bool, Dict[str, str]]: Tuple of (is_healthy, details).
        """
        try:
            # Try to get connection from pool
            async with self.db.connect() as conn:
                await conn.execute("SELECT 1")

            return True, {"status": "connected"}
        except Exception as e:
            return False, {"status": "error", "message": str(e)}

    async def check_redis(self) -> Tuple[bool, Dict[str, str]]:
        """Check Redis health.

        Returns:
            Tuple[bool, Dict[str, str]]: Tuple of (is_healthy, details).
        """
        try:
            info = await self.redis.info()
            return True, {"status": "connected", "version": info.get("redis_version", "unknown")}
        except Exception as e:
            return False, {"status": "error", "message": str(e)}

    def check_system(self) -> Tuple[bool, Dict[str, str]]:
        """Check system resources.

        Returns:
            Tuple[bool, Dict[str, str]]: Tuple of (is_healthy, details).
        """
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Define thresholds
            cpu_threshold = 90
            memory_threshold = 90
            disk_threshold = 90

            # Check if any resource is above threshold
            is_healthy = all(
                [
                    cpu_percent < cpu_threshold,
                    memory.percent < memory_threshold,
                    disk.percent < disk_threshold,
                ]
            )

            details = {
                "status": "healthy" if is_healthy else "warning",
                "cpu_usage": f"{cpu_percent}%",
                "memory_usage": f"{memory.percent}%",
                "disk_usage": f"{disk.percent}%",
            }

            return is_healthy, details
        except Exception as e:
            return False, {"status": "error", "message": str(e)}

    def check_metrics(self) -> Tuple[bool, Dict[str, str]]:
        """Check metrics collection.

        Returns:
            Tuple[bool, Dict[str, str]]: Tuple of (is_healthy, details).
        """
        try:
            from prometheus_client import REGISTRY

            # Check if metrics are being collected
            metrics_count = len(list(REGISTRY.collect()))

            is_healthy = metrics_count > 0
            details = {
                "status": "collecting" if is_healthy else "not collecting",
                "metrics_count": str(metrics_count),
            }

            return is_healthy, details
        except Exception as e:
            return False, {"status": "error", "message": str(e)}

    async def check_readiness(self) -> Tuple[bool, Dict[str, str]]:
        """Check if service is ready to handle requests.

        Returns:
            Tuple[bool, Dict[str, str]]: Tuple of (is_ready, details).
        """
        # Check database
        db_healthy, db_details = await self.check_database()

        # Check Redis
        redis_healthy, redis_details = await self.check_redis()

        is_ready = all([db_healthy, redis_healthy])
        details = {
            "status": "ready" if is_ready else "not_ready",
            "database": db_details,
            "redis": redis_details,
        }

        return is_ready, details
