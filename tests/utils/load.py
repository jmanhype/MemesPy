"""Load test utilities and fixtures."""

import asyncio
from typing import Any, Callable, Dict, List, Optional, TypeVar
from dataclasses import dataclass
import time
import statistics

import pytest
from locust import HttpUser, task, between
from sqlalchemy.ext.asyncio import AsyncSession

from tests.config import test_config
from tests.utils.performance import PerformanceMetrics, measure_performance

T = TypeVar("T")


@dataclass
class LoadTestMetrics:
    """Load test metrics."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    response_times: List[float]
    errors: Dict[str, int]

    @property
    def success_rate(self) -> float:
        """Get success rate.

        Returns:
            float: Success rate as percentage
        """
        return (
            self.successful_requests / self.total_requests * 100 if self.total_requests > 0 else 0
        )

    @property
    def avg_response_time(self) -> float:
        """Get average response time.

        Returns:
            float: Average time in milliseconds
        """
        return statistics.mean(self.response_times) if self.response_times else 0

    @property
    def p95_response_time(self) -> float:
        """Get 95th percentile response time.

        Returns:
            float: 95th percentile time in milliseconds
        """
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx]


class MemeGeneratorUser(HttpUser):
    """Locust user class for load testing meme generation."""

    wait_time = between(1, 3)

    @task(1)
    def generate_meme(self) -> None:
        """Generate a meme."""
        self.client.post(
            "/api/generate", json={"topic": "python programming", "style": "minimalist"}
        )

    @task(2)
    def get_trending_memes(self) -> None:
        """Get trending memes."""
        self.client.get("/api/trending")

    @task(3)
    def search_memes(self) -> None:
        """Search memes."""
        self.client.get("/api/search?q=python")


async def run_load_test(
    operation: Callable[..., T],
    concurrent_users: int,
    duration_seconds: int,
    spawn_rate: int,
    *args: Any,
    **kwargs: Any,
) -> LoadTestMetrics:
    """Run a load test.

    Args:
        operation: Operation to test
        concurrent_users: Number of concurrent users
        duration_seconds: Test duration in seconds
        spawn_rate: User spawn rate per second
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation

    Returns:
        LoadTestMetrics: Load test metrics
    """
    metrics = LoadTestMetrics(
        total_requests=0, successful_requests=0, failed_requests=0, response_times=[], errors={}
    )

    async def user_task() -> None:
        while True:
            start_time = time.perf_counter()
            try:
                await operation(*args, **kwargs)
                execution_time = (time.perf_counter() - start_time) * 1000
                metrics.response_times.append(execution_time)
                metrics.successful_requests += 1
            except Exception as e:
                error_type = type(e).__name__
                metrics.errors[error_type] = metrics.errors.get(error_type, 0) + 1
                metrics.failed_requests += 1
            metrics.total_requests += 1
            await asyncio.sleep(1)

    # Create and run user tasks
    tasks = []
    for i in range(concurrent_users):
        if i > 0 and i % spawn_rate == 0:
            await asyncio.sleep(1)
        task = asyncio.create_task(user_task())
        tasks.append(task)

    # Run for specified duration
    await asyncio.sleep(duration_seconds)

    # Cancel tasks
    for task in tasks:
        task.cancel()

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass

    return metrics


@pytest.fixture
async def load_test_metrics(
    load_test_db: AsyncSession, performance_metrics: Dict[str, PerformanceMetrics]
) -> LoadTestMetrics:
    """Fixture for load test metrics.

    Args:
        load_test_db: Database session with performance monitoring
        performance_metrics: Performance metrics dictionary

    Returns:
        LoadTestMetrics: Load test metrics
    """
    metrics = LoadTestMetrics(
        total_requests=0, successful_requests=0, failed_requests=0, response_times=[], errors={}
    )
    yield metrics

    # Print load test report
    print("\nLoad Test Report:")
    print("-" * 80)
    print(f"Total Requests: {metrics.total_requests}")
    print(f"Successful Requests: {metrics.successful_requests}")
    print(f"Failed Requests: {metrics.failed_requests}")
    print(f"Success Rate: {metrics.success_rate:.1f}%")
    print(f"Average Response Time: {metrics.avg_response_time:.2f}ms")
    print(f"95th Percentile Response Time: {metrics.p95_response_time:.2f}ms")
    if metrics.errors:
        print("\nErrors:")
        for error_type, count in metrics.errors.items():
            print(f"  {error_type}: {count}")
    print("-" * 80)
