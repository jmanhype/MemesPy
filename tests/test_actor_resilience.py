"""
Resilience and fault injection tests for the actor system.

Tests chaos engineering, fault tolerance, and system recovery capabilities.
"""

import pytest
import asyncio
import random
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
from dataclasses import dataclass

from dspy_meme_gen.actors.core import Actor, Message
from dspy_meme_gen.actors.supervisor import Supervisor, RestartPolicy
from dspy_meme_gen.actors.adaptive_concurrency import ConcurrencyLimitedActor
from dspy_meme_gen.actors.work_stealing_pool import WorkStealingPool


@pytest.fixture
def mock_system():
    """Fixture to provide a mocked actor system for testing."""
    mock_system = Mock()
    mock_system.register_actor = AsyncMock(return_value="mock_ref")
    mock_system.unregister_actor = AsyncMock()
    return mock_system


def setup_supervisor_with_system(supervisor, mock_system):
    """Helper to set up supervisor with mocked system."""
    supervisor._system = mock_system
    return supervisor


@dataclass
class ChaosTestMessage(Message):
    """Message for chaos testing."""

    payload: str = ""
    inject_failure: bool = False
    delay_ms: int = 0


class ChaosActor(Actor):
    """Actor for chaos engineering tests."""

    def __init__(self, name: str, failure_rate: float = 0.1):
        super().__init__(name)
        self.failure_rate = failure_rate
        self.messages_processed = 0
        self.failures_encountered = 0

    async def _handle_message(self, message: Message) -> None:
        """Handle messages with potential chaos injection."""
        self.messages_processed += 1

        # Inject random delays
        if hasattr(message, "delay_ms") and message.delay_ms > 0:
            await asyncio.sleep(message.delay_ms / 1000.0)

        # Inject random failures
        if (hasattr(message, "inject_failure") and message.inject_failure) or (
            random.random() < self.failure_rate
        ):
            self.failures_encountered += 1
            raise RuntimeError(f"Chaos failure in {self.name}")

    async def on_start(self) -> None:
        """Called when actor starts."""
        pass

    async def on_stop(self) -> None:
        """Called when actor stops."""
        pass

    async def on_error(self, error: Exception) -> None:
        """Called on unhandled errors."""
        pass


class ResourceConstrainedActor(Actor):
    """Actor that simulates resource constraints."""

    def __init__(self, name: str, max_memory_mb: int = 100, max_cpu_percent: int = 80):
        super().__init__(name)
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.current_memory_mb = 0
        self.current_cpu_percent = 0
        self.resource_violations = 0

    async def _handle_message(self, message: Message) -> None:
        """Simulate resource usage and constraints."""
        # Simulate memory allocation
        self.current_memory_mb += random.randint(1, 10)

        # Simulate CPU usage
        self.current_cpu_percent = random.randint(10, 90)

        # Check resource constraints
        if self.current_memory_mb > self.max_memory_mb:
            self.resource_violations += 1
            raise MemoryError(
                f"Memory limit exceeded: {self.current_memory_mb}MB > {self.max_memory_mb}MB"
            )

        if self.current_cpu_percent > self.max_cpu_percent:
            self.resource_violations += 1
            raise RuntimeError(
                f"CPU limit exceeded: {self.current_cpu_percent}% > {self.max_cpu_percent}%"
            )

        # Simulate work
        await asyncio.sleep(0.01)

        # Simulate memory cleanup
        self.current_memory_mb = max(0, self.current_memory_mb - random.randint(0, 5))

    async def on_start(self) -> None:
        """Called when actor starts."""
        pass

    async def on_stop(self) -> None:
        """Called when actor stops."""
        pass

    async def on_error(self, error: Exception) -> None:
        """Called on unhandled errors."""
        pass


class TestChaosEngineering:
    """Chaos engineering tests for actor system."""

    @pytest.mark.asyncio
    async def test_random_actor_failures(self, mock_system):
        """Test system resilience with random actor failures."""
        restart_policy = RestartPolicy(max_restarts=10)
        supervisor = setup_supervisor_with_system(
            Supervisor("chaos_supervisor", restart_policy=restart_policy), mock_system
        )

        # Create multiple chaos actors with different failure rates
        chaos_actors = []
        for i in range(5):
            failure_rate = random.uniform(0.05, 0.2)  # 5-20% failure rate
            actor_ref = await supervisor.spawn_child(
                ChaosActor, f"chaos_actor_{i}", f"chaos_{i}", failure_rate
            )
            chaos_actors.append(supervisor.children[f"chaos_actor_{i}"])

        # Send messages with chaos injection
        message_count = 50
        tasks = []

        for i in range(message_count):
            actor = random.choice(chaos_actors)
            inject_failure = random.random() < 0.1  # 10% chance of forced failure
            delay = random.randint(0, 50)  # Random delay 0-50ms

            message = ChaosTestMessage(
                payload=f"chaos_message_{i}", inject_failure=inject_failure, delay_ms=delay
            )

            task = asyncio.create_task(actor.send(message))
            tasks.append(task)

        # Execute all tasks and collect results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes and failures
        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = sum(1 for r in results if isinstance(r, Exception))

        # System should handle some failures gracefully
        assert successes > 0, "No messages were processed successfully"
        assert len(supervisor.children) == 5, "All actors should still be supervised"

        await supervisor.stop()

    @pytest.mark.asyncio
    async def test_network_partition_simulation(self, mock_system):
        """Simulate network partitions between actors."""
        supervisor = setup_supervisor_with_system(Supervisor("network_supervisor"), mock_system)

        # Create actors that simulate network dependencies
        actor1_ref = await supervisor.spawn_child(ChaosActor, "actor1", "network1", 0.0)
        actor2_ref = await supervisor.spawn_child(ChaosActor, "actor2", "network2", 0.0)

        actor1 = supervisor.children["actor1"]
        actor2 = supervisor.children["actor2"]

        # Simulate network partition by introducing delays
        partition_duration = 0.5  # 500ms partition

        async def send_with_partition(actor, message_id):
            message = ChaosTestMessage(
                payload=f"partition_test_{message_id}", delay_ms=int(partition_duration * 1000)
            )
            return await actor.send(message)

        # Send messages during "partition"
        start_time = asyncio.get_event_loop().time()
        tasks = [
            send_with_partition(actor1, 1),
            send_with_partition(actor2, 2),
            send_with_partition(actor1, 3),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = asyncio.get_event_loop().time()

        # Should take at least the partition duration
        actual_duration = end_time - start_time
        assert actual_duration >= partition_duration

        # All messages should eventually be processed
        successes = sum(1 for r in results if not isinstance(r, Exception))
        assert successes == len(tasks)

        await supervisor.stop()


class TestResourceConstraints:
    """Test actor behavior under resource constraints."""

    @pytest.mark.asyncio
    async def test_memory_constraint_handling(self, mock_system):
        """Test actor behavior when hitting memory limits."""
        restart_policy = RestartPolicy(max_restarts=5)
        supervisor = setup_supervisor_with_system(
            Supervisor("resource_supervisor", restart_policy=restart_policy), mock_system
        )

        # Create resource-constrained actor
        actor_ref = await supervisor.spawn_child(
            ResourceConstrainedActor,
            "resource_actor",
            "constrained",
            max_memory_mb=50,  # Low memory limit
            max_cpu_percent=70,
        )

        resource_actor = supervisor.children["resource_actor"]

        # Send many messages to trigger resource violations
        violation_count = 0
        for i in range(100):
            try:
                message = ChaosTestMessage(payload=f"resource_test_{i}")
                await resource_actor.send(message)
                await asyncio.sleep(0.01)  # Small delay between messages
            except Exception as e:
                if "Memory limit exceeded" in str(e) or "CPU limit exceeded" in str(e):
                    violation_count += 1

        # Should have encountered some resource violations
        assert resource_actor.resource_violations > 0

        # Actor should still be supervised and potentially restarted
        assert "resource_actor" in supervisor.children

        await supervisor.stop()

    @pytest.mark.asyncio
    async def test_concurrent_resource_pressure(self, mock_system):
        """Test multiple actors under concurrent resource pressure."""
        restart_policy = RestartPolicy(max_restarts=10)
        supervisor = setup_supervisor_with_system(
            Supervisor("concurrent_supervisor", restart_policy=restart_policy), mock_system
        )

        # Create multiple resource-constrained actors
        actors = []
        for i in range(3):
            actor_ref = await supervisor.spawn_child(
                ResourceConstrainedActor,
                f"resource_actor_{i}",
                f"constrained_{i}",
                max_memory_mb=30,  # Very low limits
                max_cpu_percent=60,
            )
            actors.append(supervisor.children[f"resource_actor_{i}"])

        # Apply concurrent load
        async def stress_actor(actor, message_count):
            failures = 0
            for i in range(message_count):
                try:
                    message = ChaosTestMessage(payload=f"stress_{i}")
                    await actor.send(message)
                    await asyncio.sleep(0.001)  # High frequency
                except Exception:
                    failures += 1
            return failures

        # Run concurrent stress tests
        tasks = [stress_actor(actor, 50) for actor in actors]
        failure_counts = await asyncio.gather(*tasks)

        # Some failures are expected due to resource constraints
        total_failures = sum(failure_counts)
        assert total_failures > 0, "Expected some resource-related failures"

        # All actors should still be supervised
        assert len(supervisor.children) == 3

        await supervisor.stop()


class TestFaultInjection:
    """Systematic fault injection tests."""

    @pytest.mark.asyncio
    async def test_supervisor_failure_cascade(self, mock_system):
        """Test how failures cascade through supervision hierarchy."""
        from src.dspy_meme_gen.actors.supervisor import RestartStrategy

        restart_policy = RestartPolicy(max_restarts=3)
        root_supervisor = setup_supervisor_with_system(
            Supervisor(
                "root", restart_strategy=RestartStrategy.ONE_FOR_ALL, restart_policy=restart_policy
            ),
            mock_system,
        )

        # Create hierarchy: root -> child_supervisor -> actors
        child_sup_ref = await root_supervisor.spawn_child(Supervisor, "child_sup", "child")
        child_supervisor = root_supervisor.children["child_sup"]

        # Add actors to child supervisor
        actor_refs = []
        for i in range(3):
            actor_ref = await child_supervisor.spawn_child(
                ChaosActor, f"actor_{i}", f"chaos_{i}", 0.3  # 30% failure rate
            )
            actor_refs.append(child_supervisor.children[f"actor_{i}"])

        # Trigger failures in child actors
        failure_tasks = []
        for actor in actor_refs:
            message = ChaosTestMessage(payload="cascade_test", inject_failure=True)
            task = asyncio.create_task(actor.send(message))
            failure_tasks.append(task)

        # Execute and expect some failures
        results = await asyncio.gather(*failure_tasks, return_exceptions=True)

        # Wait for supervision to handle failures
        await asyncio.sleep(0.2)

        # Check that supervision hierarchy is maintained
        assert "child_sup" in root_supervisor.children
        assert len(child_supervisor.children) == 3  # All actors should be restarted

        await root_supervisor.stop()

    @pytest.mark.asyncio
    async def test_message_queue_overflow(self, mock_system):
        """Test behavior when message queues overflow."""
        supervisor = setup_supervisor_with_system(Supervisor("overflow_supervisor"), mock_system)

        # Create actor with limited processing capacity
        slow_actor_ref = await supervisor.spawn_child(ChaosActor, "slow_actor", "slow", 0.0)
        slow_actor = supervisor.children["slow_actor"]

        # Flood actor with messages
        overflow_count = 1000
        send_tasks = []

        for i in range(overflow_count):
            message = ChaosTestMessage(
                payload=f"overflow_{i}", delay_ms=10  # Each message takes 10ms to process
            )
            task = asyncio.create_task(slow_actor.send(message))
            send_tasks.append(task)

        # Send all messages rapidly
        start_time = asyncio.get_event_loop().time()

        # Don't wait for all to complete - test queue handling
        try:
            await asyncio.wait_for(
                asyncio.gather(*send_tasks[:100]), timeout=2.0  # Only wait for first 100
            )
        except asyncio.TimeoutError:
            pass  # Expected - we're testing overflow

        # Actor should still be responsive
        test_message = ChaosTestMessage(payload="responsiveness_test")
        response = await slow_actor.send(test_message)

        # Should be able to process new messages
        assert slow_actor.messages_processed > 0

        await supervisor.stop()


class TestRecoveryMechanisms:
    """Test system recovery after failures."""

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, mock_system):
        """Test graceful degradation when actors fail."""
        restart_policy = RestartPolicy(max_restarts=2)
        supervisor = setup_supervisor_with_system(
            Supervisor("degradation_supervisor", restart_policy=restart_policy), mock_system
        )

        # Create a pool of actors where some will fail
        reliable_actors = []
        unreliable_actors = []

        # Create mix of reliable and unreliable actors
        for i in range(3):
            # Reliable actors (low failure rate)
            reliable_ref = await supervisor.spawn_child(
                ChaosActor, f"reliable_{i}", f"reliable_{i}", 0.01  # 1% failure rate
            )
            reliable_actors.append(supervisor.children[f"reliable_{i}"])

            # Unreliable actors (high failure rate)
            unreliable_ref = await supervisor.spawn_child(
                ChaosActor, f"unreliable_{i}", f"unreliable_{i}", 0.5  # 50% failure rate
            )
            unreliable_actors.append(supervisor.children[f"unreliable_{i}"])

        # Send work to both pools
        all_actors = reliable_actors + unreliable_actors
        tasks = []

        for i in range(100):
            actor = random.choice(all_actors)
            message = ChaosTestMessage(payload=f"degradation_test_{i}")
            task = asyncio.create_task(actor.send(message))
            tasks.append(task)

        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes from reliable vs unreliable actors
        successes = sum(1 for r in results if not isinstance(r, Exception))

        # Should have some successes despite failures
        assert successes > 20, "System should gracefully degrade, not fail completely"

        # Reliable actors should have higher success rate
        reliable_messages = sum(actor.messages_processed for actor in reliable_actors)
        unreliable_messages = sum(actor.messages_processed for actor in unreliable_actors)

        # This is probabilistic but should generally hold
        assert reliable_messages >= unreliable_messages * 0.5

        await supervisor.stop()

    @pytest.mark.asyncio
    async def test_system_recovery_time(self, mock_system):
        """Test how quickly system recovers from failures."""
        restart_policy = RestartPolicy(max_restarts=5)
        supervisor = setup_supervisor_with_system(
            Supervisor("recovery_supervisor", restart_policy=restart_policy), mock_system
        )

        # Create actor that will fail then recover
        actor_ref = await supervisor.spawn_child(ChaosActor, "recovery_actor", "recovery", 0.8)
        recovery_actor = supervisor.children["recovery_actor"]

        # Measure recovery time
        recovery_times = []

        for attempt in range(3):
            # Trigger failure
            start_time = asyncio.get_event_loop().time()

            failure_message = ChaosTestMessage(payload="failure", inject_failure=True)
            try:
                await recovery_actor.send(failure_message)
            except Exception:
                pass  # Expected failure

            # Wait for restart and test recovery
            await asyncio.sleep(0.1)  # Allow time for restart

            # Test if actor is responsive again
            success = False
            max_attempts = 10
            for i in range(max_attempts):
                try:
                    test_message = ChaosTestMessage(payload=f"recovery_test_{attempt}_{i}")
                    await recovery_actor.send(test_message)
                    recovery_time = asyncio.get_event_loop().time() - start_time
                    recovery_times.append(recovery_time)
                    success = True
                    break
                except Exception:
                    await asyncio.sleep(0.05)  # Wait and retry

            assert success, f"Actor did not recover after attempt {attempt}"

        # Recovery should be reasonably fast
        avg_recovery_time = sum(recovery_times) / len(recovery_times)
        assert avg_recovery_time < 1.0, f"Average recovery time too slow: {avg_recovery_time}s"

        await supervisor.stop()


if __name__ == "__main__":
    pytest.main([__file__])
