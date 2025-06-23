"""Fault injection testing for MemesPy system.

This module implements comprehensive fault injection scenarios to test
system resilience against various failure modes.
"""

import pytest
import asyncio
import random
import time
import os
import signal
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
import subprocess
import psutil

from src.dspy_meme_gen.pipeline import MemeGenerationPipeline
from src.dspy_meme_gen.agents.router import RouterAgent
from src.dspy_meme_gen.agents.image_renderer import ImageRenderingAgent
from src.dspy_meme_gen.database.connection import DatabaseConnection
from src.dspy_meme_gen.cache.redis import RedisCache


@dataclass
class FaultConfig:
    """Configuration for a fault injection."""
    
    name: str
    fault_type: str  # "exception", "delay", "corruption", "resource", "signal"
    target: str  # Component to target
    probability: float = 1.0  # Probability of fault occurring
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: Optional[float] = None
    
    def should_inject(self) -> bool:
        """Determine if fault should be injected based on probability."""
        return random.random() < self.probability


class FaultInjector:
    """Main fault injection framework."""
    
    def __init__(self):
        self.active_faults: List[FaultConfig] = []
        self.fault_history: List[Dict[str, Any]] = []
        self.original_functions: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def inject_fault(self, fault: FaultConfig):
        """Inject a fault into the system."""
        
        with self._lock:
            self.active_faults.append(fault)
            self.fault_history.append({
                "fault": fault,
                "timestamp": datetime.now(),
                "status": "injected"
            })
        
        if fault.fault_type == "exception":
            self._inject_exception_fault(fault)
        elif fault.fault_type == "delay":
            self._inject_delay_fault(fault)
        elif fault.fault_type == "corruption":
            self._inject_corruption_fault(fault)
        elif fault.fault_type == "resource":
            self._inject_resource_fault(fault)
        elif fault.fault_type == "signal":
            self._inject_signal_fault(fault)
    
    def _inject_exception_fault(self, fault: FaultConfig):
        """Inject exception-based faults."""
        
        exception_class = fault.parameters.get("exception_class", Exception)
        message = fault.parameters.get("message", f"Injected fault: {fault.name}")
        
        def faulty_function(*args, **kwargs):
            if fault.should_inject():
                raise exception_class(message)
            # Call original function if fault not injected
            return self.original_functions[fault.target](*args, **kwargs)
        
        # Patch the target
        self._patch_function(fault.target, faulty_function)
    
    def _inject_delay_fault(self, fault: FaultConfig):
        """Inject delay-based faults."""
        
        min_delay = fault.parameters.get("min_delay", 1.0)
        max_delay = fault.parameters.get("max_delay", 5.0)
        
        def delayed_function(*args, **kwargs):
            if fault.should_inject():
                delay = random.uniform(min_delay, max_delay)
                time.sleep(delay)
            return self.original_functions[fault.target](*args, **kwargs)
        
        self._patch_function(fault.target, delayed_function)
    
    def _inject_corruption_fault(self, fault: FaultConfig):
        """Inject data corruption faults."""
        
        corruption_type = fault.parameters.get("corruption_type", "truncate")
        
        def corrupt_data(data: Any) -> Any:
            if corruption_type == "truncate" and isinstance(data, (str, list, dict)):
                if isinstance(data, str):
                    return data[:len(data)//2]
                elif isinstance(data, list):
                    return data[:len(data)//2]
                elif isinstance(data, dict):
                    keys = list(data.keys())
                    return {k: data[k] for k in keys[:len(keys)//2]}
            
            elif corruption_type == "scramble" and isinstance(data, dict):
                # Randomly swap values
                keys = list(data.keys())
                values = list(data.values())
                random.shuffle(values)
                return dict(zip(keys, values))
            
            elif corruption_type == "nullify":
                return None
            
            return data
        
        def corrupting_function(*args, **kwargs):
            result = self.original_functions[fault.target](*args, **kwargs)
            if fault.should_inject():
                return corrupt_data(result)
            return result
        
        self._patch_function(fault.target, corrupting_function)
    
    def _inject_resource_fault(self, fault: FaultConfig):
        """Inject resource exhaustion faults."""
        
        resource_type = fault.parameters.get("resource_type", "memory")
        
        if resource_type == "memory":
            # Allocate memory
            size_mb = fault.parameters.get("size_mb", 100)
            self._allocate_memory(size_mb)
            
        elif resource_type == "cpu":
            # Consume CPU
            threads = fault.parameters.get("threads", 2)
            self._consume_cpu(threads, fault.duration_seconds)
            
        elif resource_type == "file_descriptors":
            # Exhaust file descriptors
            count = fault.parameters.get("count", 100)
            self._exhaust_file_descriptors(count)
    
    def _inject_signal_fault(self, fault: FaultConfig):
        """Inject signal-based faults."""
        
        signal_type = fault.parameters.get("signal_type", signal.SIGUSR1)
        delay = fault.parameters.get("delay", 1.0)
        
        def send_signal():
            time.sleep(delay)
            os.kill(os.getpid(), signal_type)
        
        threading.Thread(target=send_signal, daemon=True).start()
    
    def _patch_function(self, target: str, replacement: Callable):
        """Patch a function with fault injection."""
        
        # Parse target (e.g., "module.Class.method")
        parts = target.split(".")
        module_path = ".".join(parts[:-1])
        function_name = parts[-1]
        
        # Import and patch
        module = __import__(module_path, fromlist=[function_name])
        original = getattr(module, function_name)
        
        # Store original
        self.original_functions[target] = original
        
        # Apply patch
        setattr(module, function_name, replacement)
    
    def _allocate_memory(self, size_mb: int):
        """Allocate memory to simulate memory pressure."""
        
        # Keep reference to prevent garbage collection
        if not hasattr(self, '_memory_blocks'):
            self._memory_blocks = []
        
        block = bytearray(size_mb * 1024 * 1024)
        self._memory_blocks.append(block)
    
    def _consume_cpu(self, threads: int, duration: Optional[float]):
        """Consume CPU cycles."""
        
        def cpu_burn():
            end_time = time.time() + (duration or float('inf'))
            while time.time() < end_time:
                # Busy loop
                sum(i * i for i in range(1000))
        
        for _ in range(threads):
            threading.Thread(target=cpu_burn, daemon=True).start()
    
    def _exhaust_file_descriptors(self, count: int):
        """Exhaust file descriptors."""
        
        if not hasattr(self, '_open_files'):
            self._open_files = []
        
        for i in range(count):
            try:
                f = open(f"/tmp/fault_injection_{i}.tmp", "w")
                self._open_files.append(f)
            except OSError:
                break
    
    def clear_faults(self):
        """Clear all active faults."""
        
        with self._lock:
            # Restore original functions
            for target, original in self.original_functions.items():
                parts = target.split(".")
                module_path = ".".join(parts[:-1])
                function_name = parts[-1]
                
                module = __import__(module_path, fromlist=[function_name])
                setattr(module, function_name, original)
            
            # Clean up resources
            if hasattr(self, '_memory_blocks'):
                self._memory_blocks.clear()
            
            if hasattr(self, '_open_files'):
                for f in self._open_files:
                    try:
                        f.close()
                        os.unlink(f.name)
                    except:
                        pass
                self._open_files.clear()
            
            self.active_faults.clear()
            self.original_functions.clear()


class TestFaultInjectionScenarios:
    """Test various fault injection scenarios."""
    
    @pytest.fixture
    def fault_injector(self):
        """Provide a fault injector instance."""
        injector = FaultInjector()
        yield injector
        injector.clear_faults()
    
    def test_database_connection_failures(self, fault_injector):
        """Test system behavior with database connection failures."""
        
        # Inject connection timeout
        fault = FaultConfig(
            name="db_connection_timeout",
            fault_type="exception",
            target="sqlalchemy.engine.Connection.execute",
            probability=0.5,
            parameters={
                "exception_class": TimeoutError,
                "message": "Database connection timeout"
            }
        )
        
        fault_injector.inject_fault(fault)
        
        pipeline = MemeGenerationPipeline()
        
        # Run multiple requests
        results = {"success": 0, "failure": 0, "errors": []}
        
        for i in range(20):
            try:
                result = pipeline.forward(f"Create meme {i}")
                results["success"] += 1
            except Exception as e:
                results["failure"] += 1
                results["errors"].append(str(e))
        
        # System should handle some failures gracefully
        assert results["success"] > 0, "All requests failed"
        assert results["failure"] > 0, "No faults were injected"
        
        # Check error handling
        timeout_errors = [e for e in results["errors"] if "timeout" in e.lower()]
        assert len(timeout_errors) > 0, "Timeout errors not properly propagated"
    
    def test_intermittent_network_failures(self, fault_injector):
        """Test system behavior with intermittent network failures."""
        
        # Inject network errors with 30% probability
        fault = FaultConfig(
            name="network_failure",
            fault_type="exception",
            target="aiohttp.ClientSession.post",
            probability=0.3,
            parameters={
                "exception_class": ConnectionError,
                "message": "Network connection failed"
            }
        )
        
        fault_injector.inject_fault(fault)
        
        # Also inject random delays
        delay_fault = FaultConfig(
            name="network_delay",
            fault_type="delay",
            target="aiohttp.ClientSession.post",
            probability=0.5,
            parameters={
                "min_delay": 0.5,
                "max_delay": 3.0
            }
        )
        
        fault_injector.inject_fault(delay_fault)
        
        pipeline = MemeGenerationPipeline()
        
        # Measure impact
        start_time = time.time()
        results = []
        
        for i in range(10):
            request_start = time.time()
            try:
                result = pipeline.forward(f"Network test {i}")
                results.append({
                    "success": True,
                    "duration": time.time() - request_start
                })
            except Exception as e:
                results.append({
                    "success": False,
                    "duration": time.time() - request_start,
                    "error": str(e)
                })
        
        total_duration = time.time() - start_time
        
        # Analyze results
        successes = [r for r in results if r["success"]]
        failures = [r for r in results if not r["success"]]
        
        assert len(successes) > len(failures), "Too many failures"
        assert total_duration < 60, "System too slow under network issues"
        
        # Check that delays were injected
        slow_requests = [r for r in results if r["duration"] > 2.0]
        assert len(slow_requests) > 0, "No delays were injected"
    
    def test_memory_pressure(self, fault_injector):
        """Test system behavior under memory pressure."""
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Inject memory pressure
        fault = FaultConfig(
            name="memory_pressure",
            fault_type="resource",
            target="system",
            parameters={
                "resource_type": "memory",
                "size_mb": 500  # Allocate 500MB
            }
        )
        
        fault_injector.inject_fault(fault)
        
        # Give time for allocation
        time.sleep(1)
        
        # Check memory increased
        current_memory = process.memory_info().rss / 1024 / 1024
        assert current_memory > initial_memory + 400, "Memory not allocated"
        
        pipeline = MemeGenerationPipeline()
        
        # System should still function under memory pressure
        try:
            result = pipeline.forward("Create meme under memory pressure")
            assert result["status"] in ["success", "error"]
        except MemoryError:
            # Acceptable if system detects memory issue
            pass
    
    def test_cpu_saturation(self, fault_injector):
        """Test system behavior under CPU saturation."""
        
        # Inject CPU consumption
        fault = FaultConfig(
            name="cpu_saturation",
            fault_type="resource",
            target="system",
            parameters={
                "resource_type": "cpu",
                "threads": os.cpu_count() or 4
            },
            duration_seconds=10
        )
        
        fault_injector.inject_fault(fault)
        
        pipeline = MemeGenerationPipeline()
        
        # Measure performance under CPU pressure
        start_time = time.time()
        
        try:
            result = pipeline.forward("Create meme under CPU pressure")
            duration = time.time() - start_time
            
            # Should complete but may be slower
            assert result["status"] in ["success", "error"]
            assert duration < 30, "Request took too long under CPU pressure"
            
        except Exception as e:
            # System may fail under extreme CPU pressure
            assert "timeout" in str(e).lower() or "resource" in str(e).lower()
    
    def test_data_corruption_propagation(self, fault_injector):
        """Test how data corruption propagates through the system."""
        
        # Inject corruption in router output
        fault = FaultConfig(
            name="router_corruption",
            fault_type="corruption",
            target="src.dspy_meme_gen.agents.router.RouterAgent.forward",
            probability=0.5,
            parameters={
                "corruption_type": "scramble"
            }
        )
        
        fault_injector.inject_fault(fault)
        
        pipeline = MemeGenerationPipeline()
        
        # Track corruption effects
        corruption_detected = []
        
        for i in range(10):
            try:
                result = pipeline.forward(f"Corruption test {i}")
                # Check if result seems corrupted
                if result["status"] == "success":
                    meme = result.get("meme", {})
                    if not meme.get("topic") or not meme.get("format"):
                        corruption_detected.append("missing_fields")
            except Exception as e:
                if "key" in str(e).lower() or "attribute" in str(e).lower():
                    corruption_detected.append("key_error")
                else:
                    corruption_detected.append("other_error")
        
        # Some corruption should be detected
        assert len(corruption_detected) > 0, "No corruption detected"
        assert len(corruption_detected) < 10, "All requests corrupted"
    
    def test_cascading_failures(self, fault_injector):
        """Test cascading failure scenarios."""
        
        # Inject multiple related faults
        faults = [
            FaultConfig(
                name="cache_failure",
                fault_type="exception",
                target="src.dspy_meme_gen.cache.redis.RedisCache.get",
                probability=0.8,
                parameters={
                    "exception_class": ConnectionError,
                    "message": "Redis connection lost"
                }
            ),
            FaultConfig(
                name="db_slowdown",
                fault_type="delay",
                target="sqlalchemy.engine.Connection.execute",
                probability=1.0,
                parameters={
                    "min_delay": 2.0,
                    "max_delay": 5.0
                }
            ),
            FaultConfig(
                name="api_errors",
                fault_type="exception",
                target="openai.ChatCompletion.create",
                probability=0.3,
                parameters={
                    "exception_class": Exception,
                    "message": "OpenAI API error"
                }
            )
        ]
        
        for fault in faults:
            fault_injector.inject_fault(fault)
        
        pipeline = MemeGenerationPipeline()
        
        # Monitor cascade effects
        start_time = time.time()
        results = []
        
        for i in range(5):
            request_start = time.time()
            try:
                result = pipeline.forward(f"Cascade test {i}")
                results.append({
                    "status": "success",
                    "duration": time.time() - request_start
                })
            except Exception as e:
                results.append({
                    "status": "failure",
                    "duration": time.time() - request_start,
                    "error": str(e)
                })
        
        # Analyze cascade impact
        total_duration = time.time() - start_time
        avg_duration = sum(r["duration"] for r in results) / len(results)
        
        # System should degrade but not completely fail
        success_count = sum(1 for r in results if r["status"] == "success")
        assert success_count > 0, "Complete system failure"
        assert avg_duration < 15, "System too slow during cascade"
    
    @pytest.mark.asyncio
    async def test_async_fault_injection(self, fault_injector):
        """Test fault injection in async code paths."""
        
        # Create async fault
        async def async_faulty_function(*args, **kwargs):
            await asyncio.sleep(random.uniform(0.1, 0.5))
            if random.random() < 0.5:
                raise asyncio.TimeoutError("Async operation timeout")
            return {"result": "success"}
        
        # Patch async function
        with patch('src.dspy_meme_gen.cache.redis.RedisCache.get', side_effect=async_faulty_function):
            pipeline = MemeGenerationPipeline()
            
            # Run concurrent async requests
            tasks = []
            for i in range(10):
                task = asyncio.create_task(
                    asyncio.to_thread(pipeline.forward, f"Async test {i}")
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check mix of successes and failures
            successes = [r for r in results if not isinstance(r, Exception)]
            failures = [r for r in results if isinstance(r, Exception)]
            
            assert len(successes) > 0, "No successful async operations"
            assert len(failures) > 0, "No failed async operations"


class FaultInjectionOrchestrator:
    """Orchestrate complex fault injection scenarios."""
    
    def __init__(self, fault_injector: FaultInjector):
        self.fault_injector = fault_injector
        self.scenarios: Dict[str, List[FaultConfig]] = self._define_scenarios()
    
    def _define_scenarios(self) -> Dict[str, List[FaultConfig]]:
        """Define fault injection scenarios."""
        
        return {
            "black_friday": [
                # High load + intermittent failures
                FaultConfig(
                    name="high_latency",
                    fault_type="delay",
                    target="src.dspy_meme_gen.pipeline.MemeGenerationPipeline.forward",
                    probability=0.7,
                    parameters={"min_delay": 1.0, "max_delay": 3.0}
                ),
                FaultConfig(
                    name="cache_misses",
                    fault_type="exception",
                    target="src.dspy_meme_gen.cache.redis.RedisCache.get",
                    probability=0.9,
                    parameters={"exception_class": KeyError}
                ),
                FaultConfig(
                    name="db_contention",
                    fault_type="delay",
                    target="sqlalchemy.engine.Connection.execute",
                    probability=0.5,
                    parameters={"min_delay": 2.0, "max_delay": 5.0}
                )
            ],
            
            "datacenter_failure": [
                # Complete service failures
                FaultConfig(
                    name="redis_down",
                    fault_type="exception",
                    target="src.dspy_meme_gen.cache.redis.RedisCache.get",
                    probability=1.0,
                    parameters={"exception_class": ConnectionError}
                ),
                FaultConfig(
                    name="db_unreachable",
                    fault_type="exception",
                    target="sqlalchemy.create_engine",
                    probability=1.0,
                    parameters={"exception_class": ConnectionError}
                )
            ],
            
            "degraded_mode": [
                # Partial failures requiring fallbacks
                FaultConfig(
                    name="slow_llm",
                    fault_type="delay",
                    target="openai.ChatCompletion.create",
                    probability=0.8,
                    parameters={"min_delay": 5.0, "max_delay": 10.0}
                ),
                FaultConfig(
                    name="image_gen_errors",
                    fault_type="exception",
                    target="openai.Image.create",
                    probability=0.4,
                    parameters={"exception_class": Exception}
                )
            ],
            
            "security_incident": [
                # Simulated attack scenarios
                FaultConfig(
                    name="dos_attempt",
                    fault_type="resource",
                    target="system",
                    parameters={"resource_type": "cpu", "threads": 50}
                ),
                FaultConfig(
                    name="malformed_input",
                    fault_type="corruption",
                    target="src.dspy_meme_gen.agents.router.RouterAgent.forward",
                    probability=0.3,
                    parameters={"corruption_type": "nullify"}
                )
            ]
        }
    
    def run_scenario(self, scenario_name: str, duration_seconds: int = 60) -> Dict[str, Any]:
        """Run a named fault injection scenario."""
        
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        print(f"\nRunning fault injection scenario: {scenario_name}")
        print(f"Duration: {duration_seconds} seconds")
        
        # Inject all faults for scenario
        for fault in self.scenarios[scenario_name]:
            self.fault_injector.inject_fault(fault)
        
        # Run test workload
        pipeline = MemeGenerationPipeline()
        results = {
            "scenario": scenario_name,
            "duration": duration_seconds,
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "errors": {},
            "response_times": []
        }
        
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            request_start = time.time()
            results["requests"] += 1
            
            try:
                result = pipeline.forward(f"Test during {scenario_name}")
                results["successes"] += 1
                results["response_times"].append(time.time() - request_start)
            except Exception as e:
                results["failures"] += 1
                error_type = type(e).__name__
                results["errors"][error_type] = results["errors"].get(error_type, 0) + 1
            
            # Small delay between requests
            time.sleep(0.1)
        
        # Clear faults
        self.fault_injector.clear_faults()
        
        # Calculate metrics
        if results["response_times"]:
            results["avg_response_time"] = sum(results["response_times"]) / len(results["response_times"])
            results["max_response_time"] = max(results["response_times"])
        else:
            results["avg_response_time"] = 0
            results["max_response_time"] = 0
        
        results["success_rate"] = results["successes"] / results["requests"] if results["requests"] > 0 else 0
        
        return results
    
    def run_all_scenarios(self) -> List[Dict[str, Any]]:
        """Run all defined scenarios."""
        
        all_results = []
        
        for scenario_name in self.scenarios:
            results = self.run_scenario(scenario_name, duration_seconds=30)
            all_results.append(results)
            
            # Print summary
            print(f"\nScenario '{scenario_name}' results:")
            print(f"  Requests: {results['requests']}")
            print(f"  Success rate: {results['success_rate']:.1%}")
            print(f"  Avg response time: {results['avg_response_time']:.2f}s")
            print(f"  Errors: {results['errors']}")
            
            # Brief pause between scenarios
            time.sleep(5)
        
        return all_results


# Fault injection decorators for production use
def inject_fault(probability: float = 0.1, fault_type: str = "exception", **kwargs):
    """Decorator to inject faults into functions."""
    
    def decorator(func):
        def wrapper(*args, **kwargs_inner):
            if random.random() < probability:
                if fault_type == "exception":
                    raise kwargs.get("exception_class", Exception)(
                        kwargs.get("message", "Injected fault")
                    )
                elif fault_type == "delay":
                    time.sleep(random.uniform(
                        kwargs.get("min_delay", 0.1),
                        kwargs.get("max_delay", 1.0)
                    ))
                elif fault_type == "none":
                    return None
            
            return func(*args, **kwargs_inner)
        
        return wrapper
    
    return decorator


# Context manager for temporary fault injection
@contextmanager
def fault_injection_context(faults: List[FaultConfig]):
    """Context manager for temporary fault injection."""
    
    injector = FaultInjector()
    
    try:
        # Inject faults
        for fault in faults:
            injector.inject_fault(fault)
        
        yield injector
        
    finally:
        # Always clear faults
        injector.clear_faults()


if __name__ == "__main__":
    # Run fault injection test suite
    injector = FaultInjector()
    orchestrator = FaultInjectionOrchestrator(injector)
    
    print("Starting comprehensive fault injection tests...")
    
    # Run all scenarios
    results = orchestrator.run_all_scenarios()
    
    # Generate summary report
    print("\n" + "="*50)
    print("FAULT INJECTION TEST SUMMARY")
    print("="*50)
    
    for result in results:
        print(f"\nScenario: {result['scenario']}")
        print(f"  Total requests: {result['requests']}")
        print(f"  Success rate: {result['success_rate']:.1%}")
        print(f"  Average response time: {result['avg_response_time']:.2f}s")
        print(f"  Max response time: {result['max_response_time']:.2f}s")
        
        if result['errors']:
            print("  Error breakdown:")
            for error_type, count in result['errors'].items():
                print(f"    {error_type}: {count}")
    
    # Overall assessment
    avg_success_rate = sum(r['success_rate'] for r in results) / len(results)
    print(f"\nOverall average success rate: {avg_success_rate:.1%}")
    
    if avg_success_rate > 0.8:
        print("✓ System shows good resilience to fault injection")
    elif avg_success_rate > 0.6:
        print("⚠ System shows moderate resilience, consider improvements")
    else:
        print("✗ System shows poor resilience, significant improvements needed")