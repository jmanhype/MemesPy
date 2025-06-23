"""Load testing and performance benchmarks for MemesPy system.

This module implements comprehensive load testing scenarios to measure
system performance under various load conditions.
"""

import pytest
import asyncio
import time
import statistics
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiohttp
import psutil
import numpy as np
from locust import HttpUser, task, between, events
import matplotlib.pyplot as plt
import pandas as pd

from src.dspy_meme_gen.pipeline import MemeGenerationPipeline
from src.dspy_meme_gen.api.main import app
from fastapi.testclient import TestClient


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    request_count: int
    success_count: int
    error_count: int
    response_times: List[float]
    error_types: Dict[str, int]
    throughput: float
    concurrent_users: int
    cpu_usage: List[float]
    memory_usage: List[float]
    timestamp: datetime
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.request_count == 0:
            return 0.0
        return self.success_count / self.request_count
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def p50_response_time(self) -> float:
        """Calculate 50th percentile response time."""
        if not self.response_times:
            return 0.0
        return np.percentile(self.response_times, 50)
    
    @property
    def p95_response_time(self) -> float:
        """Calculate 95th percentile response time."""
        if not self.response_times:
            return 0.0
        return np.percentile(self.response_times, 95)
    
    @property
    def p99_response_time(self) -> float:
        """Calculate 99th percentile response time."""
        if not self.response_times:
            return 0.0
        return np.percentile(self.response_times, 99)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
            "avg_response_time": self.avg_response_time,
            "p50_response_time": self.p50_response_time,
            "p95_response_time": self.p95_response_time,
            "p99_response_time": self.p99_response_time,
            "throughput": self.throughput,
            "concurrent_users": self.concurrent_users,
            "avg_cpu_usage": statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
            "avg_memory_usage": statistics.mean(self.memory_usage) if self.memory_usage else 0,
            "error_types": self.error_types,
            "timestamp": self.timestamp.isoformat()
        }


class LoadTester:
    """Base class for load testing."""
    
    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url
        self.metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process()
    
    async def make_request(self, session: aiohttp.ClientSession, request_data: Dict[str, Any]) -> Tuple[float, bool, Optional[str]]:
        """Make a single request and return response time and success status."""
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.base_url}/api/v1/memes/generate",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                await response.json()
                response_time = time.time() - start_time
                success = response.status == 200
                error = None if success else f"HTTP {response.status}"
                return response_time, success, error
                
        except asyncio.TimeoutError:
            return time.time() - start_time, False, "Timeout"
        except Exception as e:
            return time.time() - start_time, False, str(type(e).__name__)
    
    async def run_concurrent_requests(
        self,
        num_requests: int,
        concurrent_users: int,
        request_generator: callable
    ) -> PerformanceMetrics:
        """Run concurrent requests and collect metrics."""
        
        semaphore = asyncio.Semaphore(concurrent_users)
        response_times = []
        success_count = 0
        error_count = 0
        error_types: Dict[str, int] = {}
        cpu_samples = []
        memory_samples = []
        
        start_time = time.time()
        
        async def bounded_request(session: aiohttp.ClientSession, request_data: Dict[str, Any]):
            async with semaphore:
                response_time, success, error = await self.make_request(session, request_data)
                response_times.append(response_time)
                
                if success:
                    nonlocal success_count
                    success_count += 1
                else:
                    nonlocal error_count
                    error_count += 1
                    if error:
                        error_types[error] = error_types.get(error, 0) + 1
        
        # Start resource monitoring
        monitoring_task = asyncio.create_task(self._monitor_resources(cpu_samples, memory_samples))
        
        # Execute requests
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_requests):
                request_data = request_generator(i)
                task = asyncio.create_task(bounded_request(session, request_data))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
        
        # Stop monitoring
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
        
        duration = time.time() - start_time
        throughput = num_requests / duration if duration > 0 else 0
        
        return PerformanceMetrics(
            request_count=num_requests,
            success_count=success_count,
            error_count=error_count,
            response_times=response_times,
            error_types=error_types,
            throughput=throughput,
            concurrent_users=concurrent_users,
            cpu_usage=cpu_samples,
            memory_usage=memory_samples,
            timestamp=datetime.now()
        )
    
    async def _monitor_resources(self, cpu_samples: List[float], memory_samples: List[float]):
        """Monitor CPU and memory usage."""
        while True:
            try:
                cpu_samples.append(self.process.cpu_percent())
                memory_samples.append(self.process.memory_info().rss / 1024 / 1024)  # MB
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception:
                pass


class TestLoadScenarios:
    """Test various load scenarios."""
    
    @pytest.fixture
    def load_tester(self):
        """Provide a load tester instance."""
        return LoadTester()
    
    @pytest.fixture
    def request_generators(self):
        """Provide various request generators."""
        
        def simple_request(index: int) -> Dict[str, Any]:
            return {
                "prompt": f"Create a meme about programming #{index}",
                "format": "drake"
            }
        
        def complex_request(index: int) -> Dict[str, Any]:
            formats = ["drake", "distracted", "expanding_brain", "custom"]
            styles = ["minimalist", "retro", "modern", "cartoon"]
            
            return {
                "prompt": f"Create a detailed meme about {['AI', 'coding', 'debugging', 'deployment'][index % 4]} "
                         f"with specific requirements and constraints #{index}",
                "format": formats[index % len(formats)],
                "style": styles[index % len(styles)],
                "constraints": {
                    "appropriate": True,
                    "factual": index % 2 == 0,
                    "trending": index % 3 == 0
                }
            }
        
        def varied_complexity(index: int) -> Dict[str, Any]:
            # Mix of simple and complex requests
            if index % 5 == 0:
                return complex_request(index)
            else:
                return simple_request(index)
        
        return {
            "simple": simple_request,
            "complex": complex_request,
            "varied": varied_complexity
        }
    
    @pytest.mark.asyncio
    async def test_steady_load(self, load_tester, request_generators):
        """Test system under steady load."""
        
        configurations = [
            (100, 10),   # 100 requests, 10 concurrent users
            (500, 20),   # 500 requests, 20 concurrent users
            (1000, 50),  # 1000 requests, 50 concurrent users
        ]
        
        results = []
        
        for num_requests, concurrent_users in configurations:
            print(f"\nTesting steady load: {num_requests} requests, {concurrent_users} concurrent users")
            
            metrics = await load_tester.run_concurrent_requests(
                num_requests=num_requests,
                concurrent_users=concurrent_users,
                request_generator=request_generators["simple"]
            )
            
            results.append(metrics)
            
            # Print summary
            print(f"  Success rate: {metrics.success_rate:.2%}")
            print(f"  Avg response time: {metrics.avg_response_time:.2f}s")
            print(f"  P95 response time: {metrics.p95_response_time:.2f}s")
            print(f"  Throughput: {metrics.throughput:.2f} req/s")
            
            # Performance assertions
            assert metrics.success_rate > 0.95, f"Success rate too low: {metrics.success_rate}"
            assert metrics.avg_response_time < 5.0, f"Avg response time too high: {metrics.avg_response_time}"
            assert metrics.p95_response_time < 10.0, f"P95 response time too high: {metrics.p95_response_time}"
        
        return results
    
    @pytest.mark.asyncio
    async def test_spike_load(self, load_tester, request_generators):
        """Test system behavior during traffic spikes."""
        
        # Normal load followed by spike
        phases = [
            (50, 5, 10),    # Warmup: 50 requests, 5 users, 10s
            (200, 10, 20),  # Normal: 200 requests, 10 users, 20s
            (500, 100, 10), # Spike: 500 requests, 100 users, 10s
            (100, 10, 20),  # Recovery: 100 requests, 10 users, 20s
        ]
        
        all_metrics = []
        
        for phase_idx, (num_requests, concurrent_users, duration) in enumerate(phases):
            print(f"\nPhase {phase_idx + 1}: {num_requests} requests, {concurrent_users} users")
            
            start_time = time.time()
            metrics = await load_tester.run_concurrent_requests(
                num_requests=num_requests,
                concurrent_users=concurrent_users,
                request_generator=request_generators["varied"]
            )
            
            all_metrics.append(metrics)
            
            # Wait for remaining duration
            elapsed = time.time() - start_time
            if elapsed < duration:
                await asyncio.sleep(duration - elapsed)
            
            print(f"  Success rate: {metrics.success_rate:.2%}")
            print(f"  Avg response time: {metrics.avg_response_time:.2f}s")
            print(f"  Throughput: {metrics.throughput:.2f} req/s")
        
        # Analyze spike impact
        normal_metrics = all_metrics[1]
        spike_metrics = all_metrics[2]
        recovery_metrics = all_metrics[3]
        
        # System should handle spike with graceful degradation
        assert spike_metrics.success_rate > 0.8, "System failed during spike"
        assert spike_metrics.avg_response_time < normal_metrics.avg_response_time * 3, "Response time degraded too much"
        assert recovery_metrics.success_rate > 0.95, "System didn't recover after spike"
        
        return all_metrics
    
    @pytest.mark.asyncio
    async def test_sustained_high_load(self, load_tester, request_generators):
        """Test system under sustained high load."""
        
        # Run high load for extended period
        duration_seconds = 300  # 5 minutes
        target_rps = 50  # Target requests per second
        concurrent_users = 100
        
        print(f"\nTesting sustained high load: {target_rps} RPS for {duration_seconds}s")
        
        metrics_over_time = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            # Calculate requests for next batch
            batch_duration = 10  # seconds
            batch_requests = int(target_rps * batch_duration)
            
            metrics = await load_tester.run_concurrent_requests(
                num_requests=batch_requests,
                concurrent_users=concurrent_users,
                request_generator=request_generators["complex"]
            )
            
            metrics_over_time.append(metrics)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"  [{elapsed:.0f}s] Success: {metrics.success_rate:.2%}, "
                  f"Avg RT: {metrics.avg_response_time:.2f}s, "
                  f"Throughput: {metrics.throughput:.2f} req/s")
        
        # Analyze sustained performance
        avg_success_rate = statistics.mean(m.success_rate for m in metrics_over_time)
        avg_response_time = statistics.mean(m.avg_response_time for m in metrics_over_time)
        response_time_variance = statistics.variance(m.avg_response_time for m in metrics_over_time)
        
        print(f"\nSustained load summary:")
        print(f"  Average success rate: {avg_success_rate:.2%}")
        print(f"  Average response time: {avg_response_time:.2f}s")
        print(f"  Response time variance: {response_time_variance:.2f}")
        
        # Performance should remain stable
        assert avg_success_rate > 0.95, f"Success rate degraded: {avg_success_rate}"
        assert avg_response_time < 5.0, f"Response time too high: {avg_response_time}"
        assert response_time_variance < 2.0, f"Response time too variable: {response_time_variance}"
        
        return metrics_over_time
    
    @pytest.mark.asyncio
    async def test_concurrent_user_scaling(self, load_tester, request_generators):
        """Test how system scales with increasing concurrent users."""
        
        user_counts = [1, 5, 10, 20, 50, 100, 200]
        requests_per_user = 10
        
        scaling_results = []
        
        for user_count in user_counts:
            total_requests = user_count * requests_per_user
            
            print(f"\nTesting with {user_count} concurrent users ({total_requests} total requests)")
            
            metrics = await load_tester.run_concurrent_requests(
                num_requests=total_requests,
                concurrent_users=user_count,
                request_generator=request_generators["simple"]
            )
            
            scaling_results.append({
                "concurrent_users": user_count,
                "metrics": metrics
            })
            
            print(f"  Success rate: {metrics.success_rate:.2%}")
            print(f"  Avg response time: {metrics.avg_response_time:.2f}s")
            print(f"  Throughput: {metrics.throughput:.2f} req/s")
        
        # Analyze scaling behavior
        throughputs = [r["metrics"].throughput for r in scaling_results]
        response_times = [r["metrics"].avg_response_time for r in scaling_results]
        
        # Throughput should increase with users (up to a point)
        max_throughput_idx = throughputs.index(max(throughputs))
        optimal_users = user_counts[max_throughput_idx]
        
        print(f"\nOptimal concurrent users: {optimal_users}")
        print(f"Max throughput: {max(throughputs):.2f} req/s")
        
        # Response time should not degrade too much
        for i, rt in enumerate(response_times):
            if user_counts[i] <= optimal_users:
                assert rt < 5.0, f"Response time too high at {user_counts[i]} users: {rt}"
        
        return scaling_results


class PerformanceBenchmarks:
    """Performance benchmarking utilities."""
    
    @staticmethod
    def benchmark_component(component: callable, iterations: int = 100) -> Dict[str, float]:
        """Benchmark a single component."""
        
        times = []
        
        for i in range(iterations):
            start = time.perf_counter()
            try:
                component(f"Test input {i}")
                duration = time.perf_counter() - start
                times.append(duration)
            except Exception:
                pass
        
        if not times:
            return {}
        
        return {
            "min": min(times),
            "max": max(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "p95": np.percentile(times, 95),
            "p99": np.percentile(times, 99),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0
        }
    
    @staticmethod
    async def benchmark_pipeline_stages():
        """Benchmark individual pipeline stages."""
        
        from src.dspy_meme_gen.agents.router import RouterAgent
        from src.dspy_meme_gen.agents.prompt_generator import PromptGenerationAgent
        from src.dspy_meme_gen.agents.scorer import ScoringAgent
        
        stages = {
            "router": RouterAgent(),
            "prompt_generator": PromptGenerationAgent(),
            "scorer": ScoringAgent()
        }
        
        results = {}
        
        for stage_name, stage in stages.items():
            print(f"\nBenchmarking {stage_name}...")
            
            # Create appropriate input for each stage
            if stage_name == "router":
                test_input = "Create a meme about Python"
            elif stage_name == "prompt_generator":
                test_input = {"topic": "Python", "format": "drake"}
            else:  # scorer
                test_input = [{
                    "caption": "Test meme",
                    "format": "drake",
                    "verification_results": {"appropriateness": True}
                }]
            
            # Run benchmark
            times = []
            for i in range(50):
                start = time.perf_counter()
                try:
                    await stage.forward(test_input)
                    times.append(time.perf_counter() - start)
                except Exception:
                    pass
            
            if times:
                results[stage_name] = {
                    "mean": statistics.mean(times),
                    "p95": np.percentile(times, 95),
                    "p99": np.percentile(times, 99)
                }
                
                print(f"  Mean: {results[stage_name]['mean']:.3f}s")
                print(f"  P95: {results[stage_name]['p95']:.3f}s")
                print(f"  P99: {results[stage_name]['p99']:.3f}s")
        
        return results


class LoadTestVisualizer:
    """Visualize load test results."""
    
    @staticmethod
    def plot_response_times(metrics_list: List[PerformanceMetrics], output_file: str = "response_times.png"):
        """Plot response time distribution."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Response Time Analysis")
        
        # Collect all response times
        all_times = []
        for metrics in metrics_list:
            all_times.extend(metrics.response_times)
        
        # Histogram
        axes[0, 0].hist(all_times, bins=50, alpha=0.7)
        axes[0, 0].set_title("Response Time Distribution")
        axes[0, 0].set_xlabel("Response Time (s)")
        axes[0, 0].set_ylabel("Frequency")
        
        # Box plot by concurrent users
        data_by_users = {}
        for metrics in metrics_list:
            users = metrics.concurrent_users
            if users not in data_by_users:
                data_by_users[users] = []
            data_by_users[users].extend(metrics.response_times)
        
        axes[0, 1].boxplot(data_by_users.values(), labels=data_by_users.keys())
        axes[0, 1].set_title("Response Times by Concurrent Users")
        axes[0, 1].set_xlabel("Concurrent Users")
        axes[0, 1].set_ylabel("Response Time (s)")
        
        # Percentiles over time
        percentiles = []
        timestamps = []
        for metrics in metrics_list:
            percentiles.append({
                "p50": metrics.p50_response_time,
                "p95": metrics.p95_response_time,
                "p99": metrics.p99_response_time
            })
            timestamps.append(metrics.timestamp)
        
        if percentiles:
            df = pd.DataFrame(percentiles, index=timestamps)
            df.plot(ax=axes[1, 0])
            axes[1, 0].set_title("Response Time Percentiles Over Time")
            axes[1, 0].set_ylabel("Response Time (s)")
            axes[1, 0].legend(["P50", "P95", "P99"])
        
        # Success rate vs throughput
        success_rates = [m.success_rate for m in metrics_list]
        throughputs = [m.throughput for m in metrics_list]
        
        axes[1, 1].scatter(throughputs, success_rates, alpha=0.7)
        axes[1, 1].set_title("Success Rate vs Throughput")
        axes[1, 1].set_xlabel("Throughput (req/s)")
        axes[1, 1].set_ylabel("Success Rate")
        axes[1, 1].set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    
    @staticmethod
    def plot_resource_usage(metrics_list: List[PerformanceMetrics], output_file: str = "resource_usage.png"):
        """Plot resource usage over time."""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle("Resource Usage During Load Test")
        
        for i, metrics in enumerate(metrics_list):
            if metrics.cpu_usage:
                ax1.plot(metrics.cpu_usage, label=f"Test {i+1} ({metrics.concurrent_users} users)")
            
            if metrics.memory_usage:
                ax2.plot(metrics.memory_usage, label=f"Test {i+1} ({metrics.concurrent_users} users)")
        
        ax1.set_title("CPU Usage")
        ax1.set_ylabel("CPU %")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title("Memory Usage")
        ax2.set_ylabel("Memory (MB)")
        ax2.set_xlabel("Time (samples)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    
    @staticmethod
    def generate_report(metrics_list: List[PerformanceMetrics], output_file: str = "load_test_report.json"):
        """Generate a comprehensive load test report."""
        
        report = {
            "summary": {
                "total_requests": sum(m.request_count for m in metrics_list),
                "total_success": sum(m.success_count for m in metrics_list),
                "total_errors": sum(m.error_count for m in metrics_list),
                "overall_success_rate": sum(m.success_count for m in metrics_list) / sum(m.request_count for m in metrics_list),
                "avg_throughput": statistics.mean(m.throughput for m in metrics_list),
                "max_throughput": max(m.throughput for m in metrics_list),
                "avg_response_time": statistics.mean(m.avg_response_time for m in metrics_list),
                "test_duration": (metrics_list[-1].timestamp - metrics_list[0].timestamp).total_seconds() if len(metrics_list) > 1 else 0
            },
            "tests": [m.to_dict() for m in metrics_list],
            "recommendations": []
        }
        
        # Add recommendations based on results
        if report["summary"]["overall_success_rate"] < 0.95:
            report["recommendations"].append("Consider improving error handling - success rate is below 95%")
        
        if report["summary"]["avg_response_time"] > 5.0:
            report["recommendations"].append("Response times are high - consider optimizing slow operations")
        
        max_users = max(m.concurrent_users for m in metrics_list)
        if max_users > 100 and any(m.success_rate < 0.9 for m in metrics_list if m.concurrent_users == max_users):
            report["recommendations"].append("System struggles with high concurrent users - consider scaling strategies")
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


# Locust-based load testing for more realistic scenarios
class MemeUser(HttpUser):
    """Locust user for load testing meme generation."""
    
    wait_time = between(1, 5)
    
    @task(3)
    def generate_simple_meme(self):
        """Generate a simple meme."""
        
        response = self.client.post("/api/v1/memes/generate", json={
            "prompt": f"Create a meme about {random.choice(['Python', 'JavaScript', 'debugging', 'deployment'])}",
            "format": random.choice(["drake", "distracted", "expanding_brain"])
        })
        
        if response.status_code != 200:
            print(f"Failed request: {response.status_code} - {response.text}")
    
    @task(1)
    def generate_complex_meme(self):
        """Generate a complex meme with constraints."""
        
        response = self.client.post("/api/v1/memes/generate", json={
            "prompt": "Create a detailed technical meme with multiple requirements",
            "format": "custom",
            "style": "minimalist",
            "constraints": {
                "appropriate": True,
                "factual": True,
                "trending": True
            }
        })
    
    @task(2)
    def browse_formats(self):
        """Browse available meme formats."""
        
        self.client.get("/api/v1/formats")
    
    @task(1)
    def check_trends(self):
        """Check trending topics."""
        
        self.client.get("/api/v1/trends")
    
    def on_start(self):
        """Called when a user starts."""
        # Could add authentication here if needed
        pass


def run_locust_test(host: str = "http://localhost:8081", users: int = 100, spawn_rate: int = 10, run_time: str = "5m"):
    """Run Locust load test programmatically."""
    
    from locust import run_single_user
    from locust.env import Environment
    from locust.stats import stats_printer, stats_history
    from locust.log import setup_logging
    
    setup_logging("INFO", None)
    
    # Create environment
    env = Environment(user_classes=[MemeUser], host=host)
    env.create_local_runner()
    
    # Start test
    env.runner.start(users, spawn_rate=spawn_rate)
    
    # Run for specified time
    import gevent
    gevent.spawn(stats_printer(env.stats))
    gevent.spawn(stats_history, env.runner)
    
    # Wait for test to complete
    env.runner.greenlet.join(timeout=300)  # 5 minutes max
    
    # Stop and return stats
    env.runner.quit()
    
    return env.stats


if __name__ == "__main__":
    # Run comprehensive load tests
    import asyncio
    
    async def main():
        tester = LoadTester()
        visualizer = LoadTestVisualizer()
        
        # Run different scenarios
        print("Running load test scenarios...")
        
        # Basic load test
        simple_gen = lambda i: {"prompt": f"Create meme {i}", "format": "drake"}
        metrics1 = await tester.run_concurrent_requests(100, 10, simple_gen)
        
        # Complex load test
        complex_gen = lambda i: {
            "prompt": f"Complex meme {i}",
            "format": "custom",
            "constraints": {"appropriate": True}
        }
        metrics2 = await tester.run_concurrent_requests(100, 20, complex_gen)
        
        # Generate visualizations
        all_metrics = [metrics1, metrics2]
        visualizer.plot_response_times(all_metrics)
        visualizer.plot_resource_usage(all_metrics)
        report = visualizer.generate_report(all_metrics)
        
        print(f"\nLoad test complete. Report saved to load_test_report.json")
        print(f"Overall success rate: {report['summary']['overall_success_rate']:.2%}")
        print(f"Average throughput: {report['summary']['avg_throughput']:.2f} req/s")
    
    asyncio.run(main())