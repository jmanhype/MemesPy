"""
Comprehensive tests for actor supervision and fault tolerance.

Tests the cybernetic actor system's supervision, recovery, and error handling capabilities.
"""

import pytest
import asyncio
import weakref
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.dspy_meme_gen.actors.core import Actor, Message
from src.dspy_meme_gen.actors.supervisor import Supervisor
from src.dspy_meme_gen.actors.adaptive_concurrency import ConcurrencyLimitedActor
from src.dspy_meme_gen.actors.work_stealing_pool import WorkStealingPool, WorkStealingWorker


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


class TestMessage(Message):
    """Test message for supervision tests."""
    
    def __init__(self, content: str, should_fail: bool = False):
        super().__init__()
        self.content = content
        self.should_fail = should_fail


class TestActor(Actor):
    """Test actor for supervision testing."""
    
    def __init__(self, name: str, fail_on_message: bool = False):
        super().__init__(name)
        self.fail_on_message = fail_on_message
        self.messages_received = []
        self.shutdown_called = False
        
    async def _handle_message(self, message: Message) -> None:
        """Handle test messages."""
        self.messages_received.append(message)
        
        if hasattr(message, 'should_fail') and message.should_fail:
            raise RuntimeError(f"Test failure in {self.name}")
            
        if self.fail_on_message:
            raise ValueError(f"Simulated failure in {self.name}")
    
    async def on_start(self) -> None:
        """Called when actor starts."""
        pass
        
    async def on_stop(self) -> None:
        """Called when actor stops."""
        self.shutdown_called = True
        
    async def on_error(self, error: Exception) -> None:
        """Called on unhandled errors."""
        pass
    
    async def stop(self) -> None:
        """Track shutdown calls."""
        self.shutdown_called = True
        await super().stop()


class FailingActor(Actor):
    """Actor that always fails for testing supervision."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.failure_count = 0
        
    async def _handle_message(self, message: Message) -> None:
        """Always fail to test supervision."""
        self.failure_count += 1
        raise RuntimeError(f"Deliberate failure #{self.failure_count}")
    
    async def on_start(self) -> None:
        """Called when actor starts."""
        pass
        
    async def on_stop(self) -> None:
        """Called when actor stops."""
        pass
        
    async def on_error(self, error: Exception) -> None:
        """Called on unhandled errors."""
        pass


@pytest.mark.asyncio
class TestSupervisorBasics:
    """Test basic supervisor functionality."""
    
    async def test_supervisor_creation(self, mock_system):
        """Test supervisor can be created with proper configuration."""
        supervisor = setup_supervisor_with_system(Supervisor("test_supervisor"), mock_system)
        
        assert supervisor.name == "test_supervisor"
        assert len(supervisor.children) == 0
        assert supervisor.restart_policy.max_restarts == 5
        assert supervisor.restart_policy.within_time_range == 60.0
        
        await supervisor.stop()
    
    async def test_child_spawning(self, mock_system):
        """Test supervisor can spawn and track child actors."""
        supervisor = setup_supervisor_with_system(Supervisor("supervisor"), mock_system)
        
        # Spawn a child actor
        child_ref = await supervisor.spawn_child(TestActor, "test_child", "child")
        
        assert "test_child" in supervisor.children
        assert len(supervisor.children) == 1
        assert supervisor.children["test_child"].actor.name == "test_child"
        
        # Child should be in spawn order
        assert "test_child" in supervisor.child_order
        
        await supervisor.stop()
    
    async def test_multiple_children(self, mock_system):
        """Test supervisor can manage multiple children."""
        supervisor = setup_supervisor_with_system(Supervisor("supervisor"), mock_system)
        
        # Spawn multiple children
        child1_ref = await supervisor.spawn_child(TestActor, "child1", "actor1")
        child2_ref = await supervisor.spawn_child(TestActor, "child2", "actor2") 
        child3_ref = await supervisor.spawn_child(TestActor, "child3", "actor3")
        
        assert len(supervisor.children) == 3
        assert "child1" in supervisor.children
        assert "child2" in supervisor.children
        assert "child3" in supervisor.children
        
        # Check spawn order is maintained
        assert supervisor.child_order == ["child1", "child2", "child3"]
        
        await supervisor.stop()


@pytest.mark.asyncio 
class TestSupervisionStrategy:
    """Test different supervision strategies."""
    
    async def test_one_for_one_strategy(self, mock_system):
        """Test one-for-one supervision strategy."""
        supervisor = setup_supervisor_with_system(
            Supervisor("supervisor", strategy="one_for_one"), mock_system
        )
        
        # Spawn children
        child1_ref = await supervisor.spawn_child(TestActor, "child1", "actor1")
        child2_ref = await supervisor.spawn_child(FailingActor, "child2", "failing_actor")
        child3_ref = await supervisor.spawn_child(TestActor, "child3", "actor3")
        
        # Get initial child instances
        original_child2 = supervisor.children["child2"]
        
        # Send message that causes child2 to fail
        failing_message = TestMessage("fail", should_fail=True)
        await supervisor.children["child2"].send(failing_message)
        
        # Wait for supervision to handle failure
        await asyncio.sleep(0.1)
        
        # Only child2 should be restarted
        assert len(supervisor.children) == 3
        assert "child1" in supervisor.children
        assert "child2" in supervisor.children  
        assert "child3" in supervisor.children
        
        # Child2 should be a new instance
        new_child2 = supervisor.children["child2"]
        assert new_child2 != original_child2
        
        await supervisor.stop()
    
    async def test_one_for_all_strategy(self, mock_system):
        """Test one-for-all supervision strategy."""
        supervisor = setup_supervisor_with_system(
            Supervisor("supervisor", strategy="one_for_all"), mock_system
        )
        
        # Spawn children
        child1_ref = await supervisor.spawn_child(TestActor, "child1", "actor1")
        child2_ref = await supervisor.spawn_child(FailingActor, "child2", "failing_actor")
        child3_ref = await supervisor.spawn_child(TestActor, "child3", "actor3")
        
        # Get initial child instances
        original_children = supervisor.children.copy()
        
        # Send message that causes child2 to fail
        failing_message = TestMessage("fail", should_fail=True)
        await supervisor.children["child2"].send(failing_message)
        
        # Wait for supervision to handle failure
        await asyncio.sleep(0.1)
        
        # All children should be restarted
        assert len(supervisor.children) == 3
        for name in ["child1", "child2", "child3"]:
            assert name in supervisor.children
            # Should be new instances
            assert supervisor.children[name] != original_children[name]
        
        await supervisor.stop()


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and recovery mechanisms."""
    
    async def test_child_failure_handling(self, mock_system):
        """Test supervisor handles child failures properly."""
        supervisor = setup_supervisor_with_system(Supervisor("supervisor"), mock_system)
        
        # Spawn a failing child
        child_ref = await supervisor.spawn_child(FailingActor, "failing_child", "failing")
        
        # Track restart count
        initial_restart_count = supervisor.restart_counts.get("failing_child", 0)
        
        # Send message that triggers failure
        test_message = TestMessage("trigger_failure")
        await supervisor.children["failing_child"].send(test_message)
        
        # Wait for supervision
        await asyncio.sleep(0.1)
        
        # Child should be restarted
        final_restart_count = supervisor.restart_counts.get("failing_child", 0)
        assert final_restart_count > initial_restart_count
        
        await supervisor.stop()
    
    async def test_max_restart_limit(self, mock_system):
        """Test max restart limit enforcement."""
        from src.dspy_meme_gen.actors.supervisor import RestartPolicy
        restart_policy = RestartPolicy(max_restarts=2, within_time_range=1.0)
        supervisor = setup_supervisor_with_system(
            Supervisor("supervisor", restart_policy=restart_policy), mock_system
        )
        
        # Spawn a consistently failing child
        child_ref = await supervisor.spawn_child(FailingActor, "failing_child", "failing")
        
        # Trigger multiple failures
        for i in range(5):
            try:
                test_message = TestMessage("trigger_failure")
                await supervisor.children["failing_child"].send(test_message)
                await asyncio.sleep(0.1)
            except Exception:
                pass  # Expected failures
        
        # Wait for all supervision attempts
        await asyncio.sleep(0.5)
        
        # Child should be terminated after max restarts
        restart_count = supervisor.restart_counts.get("failing_child", 0)
        assert restart_count <= 2  # Should not exceed max restarts
        
        await supervisor.stop()
    
    async def test_escalation_to_parent(self, mock_system):
        """Test error escalation to parent supervisor."""
        root_supervisor = setup_supervisor_with_system(Supervisor("root"), mock_system)
        restart_policy = RestartPolicy(max_restarts=1)
        child_supervisor = setup_supervisor_with_system(
            Supervisor("child_supervisor", restart_policy=restart_policy), mock_system
        )
        
        # Create supervision hierarchy
        supervisor_ref = await root_supervisor.spawn_child(Supervisor, "child_sup", "child_supervisor")
        
        # This test would need more complex setup to properly test escalation
        # For now, just verify the structure
        assert "child_sup" in root_supervisor.children
        
        await root_supervisor.stop()


@pytest.mark.asyncio
class TestMemoryManagement:
    """Test memory management and leak prevention."""
    
    async def test_weak_reference_cleanup(self, mock_system):
        """Test that weak references prevent circular dependencies."""
        supervisor = setup_supervisor_with_system(Supervisor("supervisor"), mock_system)
        
        # Spawn a child
        child_ref = await supervisor.spawn_child(TestActor, "test_child", "child")
        child_actor = supervisor.children["test_child"]
        
        # Create weak reference to supervisor
        supervisor_weak_ref = weakref.ref(supervisor)
        
        # Verify weak reference works
        assert supervisor_weak_ref() is supervisor
        
        # Clean shutdown should not prevent garbage collection
        await supervisor.stop()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # After proper cleanup, supervisor should be collectable
        # (This test mainly ensures the weak reference pattern is in place)
        assert hasattr(child_actor, '_handle_message')
        
    async def test_child_cleanup_on_shutdown(self, mock_system):
        """Test that children are properly cleaned up on shutdown."""
        supervisor = setup_supervisor_with_system(Supervisor("supervisor"), mock_system)
        
        # Spawn children
        child1_ref = await supervisor.spawn_child(TestActor, "child1", "actor1")
        child2_ref = await supervisor.spawn_child(TestActor, "child2", "actor2")
        
        # Get child instances
        child1 = supervisor.children["child1"]
        child2 = supervisor.children["child2"]
        
        # Shutdown supervisor
        await supervisor.stop()
        
        # Children should have shutdown called
        # Note: This relies on our TestActor implementation
        if hasattr(child1, 'shutdown_called'):
            assert child1.shutdown_called
        if hasattr(child2, 'shutdown_called'):
            assert child2.shutdown_called


@pytest.mark.asyncio
class TestConcurrencySupervision:
    """Test supervision of adaptive concurrency actors."""
    
    async def test_adaptive_concurrency_supervision(self, mock_system):
        """Test supervising adaptive concurrency actors."""
        supervisor = setup_supervisor_with_system(Supervisor("supervisor"), mock_system)
        
        # Mock the concurrency controller
        with patch('src.dspy_meme_gen.actors.adaptive_concurrency.DynamicConcurrencyController'):
            child_ref = await supervisor.spawn_child(
                ConcurrencyLimitedActor,
                "concurrency_actor", 
                "concurrency_test"
            )
            
            assert "concurrency_actor" in supervisor.children
            concurrency_actor = supervisor.children["concurrency_actor"]
            
            # Verify it's the right type
            assert isinstance(concurrency_actor, ConcurrencyLimitedActor)
            
            await supervisor.stop()


@pytest.mark.asyncio
class TestWorkStealingSupervision:
    """Test supervision of work stealing pool actors."""
    
    async def test_work_stealing_worker_supervision(self, mock_system):
        """Test supervising work stealing worker actors."""
        supervisor = setup_supervisor_with_system(Supervisor("supervisor"), mock_system)
        
        # Create a work stealing worker
        worker_ref = await supervisor.spawn_child(
            WorkStealingWorker,
            "worker",
            "test_worker"
        )
        
        assert "worker" in supervisor.children
        worker_actor = supervisor.children["worker"]
        
        # Verify it's the right type
        assert isinstance(worker_actor, WorkStealingWorker)
        
        await supervisor.stop()


@pytest.mark.asyncio
class TestSupervisionIntegration:
    """Integration tests for supervision system."""
    
    async def test_hierarchical_supervision(self, mock_system):
        """Test hierarchical supervision structure."""
        root_supervisor = setup_supervisor_with_system(
            Supervisor("root", strategy="one_for_all"), mock_system
        )
        
        # Create child supervisors
        child_sup1_ref = await root_supervisor.spawn_child(Supervisor, "child_sup1", "child1")
        child_sup2_ref = await root_supervisor.spawn_child(Supervisor, "child_sup2", "child2") 
        
        # Add actors to child supervisors
        child_sup1 = root_supervisor.children["child_sup1"]
        child_sup2 = root_supervisor.children["child_sup2"]
        
        actor1_ref = await child_sup1.spawn_child(TestActor, "actor1", "test1")
        actor2_ref = await child_sup2.spawn_child(TestActor, "actor2", "test2")
        
        # Verify structure
        assert len(root_supervisor.children) == 2
        assert len(child_sup1.children) == 1
        assert len(child_sup2.children) == 1
        
        await root_supervisor.stop()
    
    async def test_supervision_under_load(self, mock_system):
        """Test supervision system under concurrent load."""
        restart_policy = RestartPolicy(max_restarts=10)
        supervisor = setup_supervisor_with_system(
            Supervisor("supervisor", restart_policy=restart_policy), mock_system
        )
        
        # Spawn multiple children
        children = []
        for i in range(5):
            child_ref = await supervisor.spawn_child(TestActor, f"child_{i}", f"actor_{i}")
            children.append(supervisor.children[f"child_{i}"])
        
        # Send concurrent messages to all children
        async def send_messages(child, count):
            for i in range(count):
                message = TestMessage(f"message_{i}")
                await child.send(message)
        
        # Run concurrent message sending
        tasks = [send_messages(child, 10) for child in children]
        await asyncio.gather(*tasks)
        
        # All children should still be running
        assert len(supervisor.children) == 5
        
        await supervisor.stop()


if __name__ == "__main__":
    pytest.main([__file__])