"""Performance test utilities and fixtures."""
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar
from functools import wraps
import asyncio
import statistics
from dataclasses import dataclass

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from tests.config import test_config

T = TypeVar("T")

@dataclass
class PerformanceMetrics:
    """Performance test metrics."""
    
    operation_name: str
    execution_times: List[float]
    success_count: int
    error_count: int
    
    @property
    def avg_time(self) -> float:
        """Get average execution time.
        
        Returns:
            float: Average time in milliseconds
        """
        return statistics.mean(self.execution_times) if self.execution_times else 0
    
    @property
    def p95_time(self) -> float:
        """Get 95th percentile execution time.
        
        Returns:
            float: 95th percentile time in milliseconds
        """
        if not self.execution_times:
            return 0
        sorted_times = sorted(self.execution_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx]
    
    @property
    def max_time(self) -> float:
        """Get maximum execution time.
        
        Returns:
            float: Maximum time in milliseconds
        """
        return max(self.execution_times) if self.execution_times else 0
    
    @property
    def success_rate(self) -> float:
        """Get success rate.
        
        Returns:
            float: Success rate as percentage
        """
        total = self.success_count + self.error_count
        return (self.success_count / total * 100) if total > 0 else 0

def measure_performance(threshold_ms: Optional[int] = None) -> Callable:
    """Decorator to measure function performance.
    
    Args:
        threshold_ms: Optional maximum execution time in milliseconds
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        metrics = PerformanceMetrics(
            operation_name=func.__name__,
            execution_times=[],
            success_count=0,
            error_count=0
        )
        
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                execution_time = (time.perf_counter() - start_time) * 1000
                metrics.execution_times.append(execution_time)
                metrics.success_count += 1
                
                if threshold_ms and execution_time > threshold_ms:
                    pytest.fail(
                        f"{func.__name__} exceeded threshold of {threshold_ms}ms "
                        f"(took {execution_time:.2f}ms)"
                    )
                
                return result
            except Exception as e:
                metrics.error_count += 1
                raise
            
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.perf_counter() - start_time) * 1000
                metrics.execution_times.append(execution_time)
                metrics.success_count += 1
                
                if threshold_ms and execution_time > threshold_ms:
                    pytest.fail(
                        f"{func.__name__} exceeded threshold of {threshold_ms}ms "
                        f"(took {execution_time:.2f}ms)"
                    )
                
                return result
            except Exception as e:
                metrics.error_count += 1
                raise
        
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper.metrics = metrics  # type: ignore
        return wrapper
    
    return decorator

async def run_concurrent_operations(
    operation: Callable[..., T],
    num_concurrent: int,
    *args: Any,
    **kwargs: Any
) -> List[T]:
    """Run operations concurrently.
    
    Args:
        operation: Operation to run
        num_concurrent: Number of concurrent operations
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation
        
    Returns:
        List[T]: Operation results
    """
    tasks = [
        operation(*args, **kwargs)
        for _ in range(num_concurrent)
    ]
    return await asyncio.gather(*tasks)

@pytest.fixture
def performance_metrics() -> Dict[str, PerformanceMetrics]:
    """Fixture to collect performance metrics.
    
    Returns:
        Dict[str, PerformanceMetrics]: Performance metrics by operation
    """
    metrics: Dict[str, PerformanceMetrics] = {}
    yield metrics
    
    # Print performance report
    print("\nPerformance Report:")
    print("-" * 80)
    for operation_name, operation_metrics in metrics.items():
        print(f"\nOperation: {operation_name}")
        print(f"Average Time: {operation_metrics.avg_time:.2f}ms")
        print(f"95th Percentile: {operation_metrics.p95_time:.2f}ms")
        print(f"Max Time: {operation_metrics.max_time:.2f}ms")
        print(f"Success Rate: {operation_metrics.success_rate:.1f}%")
    print("-" * 80)

@pytest.fixture
async def load_test_db(async_db_session: AsyncSession) -> AsyncSession:
    """Fixture for load testing database.
    
    Args:
        async_db_session: Async database session
        
    Returns:
        AsyncSession: Database session with performance monitoring
    """
    # Add performance monitoring to session methods
    original_execute = async_db_session.execute
    
    @measure_performance(threshold_ms=test_config.MAX_QUERY_TIME_MS)
    async def monitored_execute(*args: Any, **kwargs: Any) -> Any:
        return await original_execute(*args, **kwargs)
    
    async_db_session.execute = monitored_execute  # type: ignore
    return async_db_session 