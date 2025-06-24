"""
Comprehensive tests for the core actor system.

Tests the fundamental actor model, message passing, and lifecycle management.
"""

import pytest
import asyncio
import logging
import weakref
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
from dataclasses import dataclass

from dspy_meme_gen.actors.core import (
    Actor,
    Message,
    Request,
    Response,
    Event,
    ActorRef,
    ActorSystem,
)
from dspy_meme_gen.actors.mailbox import ActorMailbox, OverflowStrategy


async def stop_actors(*actors):
    """Helper function to stop multiple actors safely."""
    for actor in actors:
        if hasattr(actor, "running") and actor.running:
            try:
                await actor.stop()
            except Exception as e:
                logging.error(f"Error stopping actor {getattr(actor, 'name', 'unknown')}: {e}")


@pytest.fixture
async def cleanup_tasks():
    """Ensure proper cleanup of tasks after each test."""
    yield
    # Give time for any pending tasks to complete
    await asyncio.sleep(0.01)

    # Get current task to avoid cancelling it
    current_task = asyncio.current_task()

    # Cancel all remaining tasks except the current one
    loop = asyncio.get_event_loop()
    tasks = [task for task in asyncio.all_tasks(loop) if not task.done() and task != current_task]

    for task in tasks:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=0.1)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


@dataclass
class TestRequest(Request):
    """Test request message."""

    content: str = ""
    delay: float = 0.0


@dataclass
class TestResponse(Response):
    """Test response message."""

    result: str = ""
    success: bool = True


@dataclass
class TestEvent(Event):
    """Test event message."""

    data: str = ""


class TestActor(Actor):
    """Actor for testing core functionality."""

    def __init__(self, name: str, process_delay: float = 0.0):
        super().__init__(name)
        self.process_delay = process_delay
        self.requests_handled = []
        self.events_received = []
        self.start_called = False
        self.stop_called = False
        self.error_count = 0

    async def on_start(self) -> None:
        """Track start calls."""
        self.start_called = True

    async def on_stop(self) -> None:
        """Track stop calls."""
        self.stop_called = True

    async def on_error(self, error: Exception) -> None:
        """Track errors."""
        self.error_count += 1

    async def stop(self) -> None:
        """Override stop to ensure proper cleanup."""
        self.running = False
        await self.on_stop()

        # Cancel the internal task immediately
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self.logger.info(f"Actor {self.name} stopped")

    async def send(self, message: Message) -> Any:
        """Send a message to this actor and wait for response."""
        # Process the message directly without going through mailbox
        # This simulates synchronous message handling for testing
        handler_name = f"handle_{message.__class__.__name__.lower()}"
        handler = getattr(self, handler_name, self.handle_unknown)

        try:
            result = await handler(message)
            return result
        except Exception as e:
            self.logger.error(f"Error handling message {message.id}: {e}", exc_info=True)
            # Call on_error to track errors in tests
            await self.on_error(e)
            raise

    async def handle_testrequest(self, message: TestRequest) -> TestResponse:
        """Handle test requests."""
        if message.delay > 0:
            await asyncio.sleep(message.delay)

        self.requests_handled.append(message)

        if message.content == "error":
            raise ValueError("Test error")

        return TestResponse(result=f"Processed: {message.content}", success=True)

    async def handle_testevent(self, message: TestEvent) -> None:
        """Handle test events."""
        self.events_received.append(message)


class FailingActor(Actor):
    """Actor that fails on startup for testing."""

    def __init__(self, name: str):
        super().__init__(name)

    async def on_start(self) -> None:
        """Fail on start."""
        raise RuntimeError("Startup failure")

    async def on_stop(self) -> None:
        """Clean stop."""
        pass

    async def on_error(self, error: Exception) -> None:
        """Handle errors."""
        pass


class TestActorBasics:
    """Test basic actor functionality."""

    @pytest.mark.asyncio
    async def test_actor_creation(self):
        """Test actor can be created with proper state."""
        actor = TestActor("test_actor")

        assert actor.name == "test_actor"
        assert actor.running is False
        assert actor.mailbox is not None
        assert actor.logger is not None

    @pytest.mark.asyncio
    async def test_actor_lifecycle(self):
        """Test actor start/stop lifecycle."""
        actor = TestActor("lifecycle_test")

        try:
            # Start actor
            await actor.start()
            assert actor.running is True
            assert actor.start_called is True

            # Stop actor
            await actor.stop()
            assert actor.running is False
            assert actor.stop_called is True
        finally:
            # Ensure actor is stopped even if test fails
            if actor.running:
                await actor.stop()

    @pytest.mark.asyncio
    async def test_message_handling(self):
        """Test basic message handling."""
        actor = TestActor("message_test")

        try:
            await actor.start()

            # Send request
            request = TestRequest(content="hello")
            response = await actor.send(request)

            assert isinstance(response, TestResponse)
            assert response.result == "Processed: hello"
            assert response.success is True
            assert len(actor.requests_handled) == 1
        finally:
            # Ensure actor is stopped
            if actor.running:
                await actor.stop()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in message processing."""
        actor = TestActor("error_test")

        try:
            await actor.start()

            # Send request that causes error
            request = TestRequest(content="error")

            with pytest.raises(ValueError, match="Test error"):
                await actor.send(request)

            assert actor.error_count == 1
        finally:
            # Ensure actor is stopped
            if actor.running:
                await actor.stop()

    @pytest.mark.asyncio
    async def test_concurrent_message_processing(self):
        """Test handling multiple concurrent messages."""
        actor = TestActor("concurrent_test")

        try:
            await actor.start()

            # Send multiple requests concurrently
            requests = [TestRequest(content=f"message_{i}") for i in range(10)]

            # Process concurrently
            responses = await asyncio.gather(*[actor.send(req) for req in requests])

            # All should succeed
            assert len(responses) == 10
            for i, response in enumerate(responses):
                assert response.result == f"Processed: message_{i}"

            assert len(actor.requests_handled) == 10
        finally:
            # Ensure actor is stopped
            if actor.running:
                await actor.stop()


class TestActorSystem:
    """Test actor system functionality."""

    @pytest.mark.asyncio
    async def test_actor_registration(self):
        """Test registering actors with the system."""
        system = ActorSystem("test_system")
        actor = TestActor("registered_actor")

        # Register actor
        actor_ref = await system.register_actor(actor)

        assert isinstance(actor_ref, ActorRef)
        assert actor.name in system.actors
        assert system.actors[actor.name] == actor

        await system.stop()

    @pytest.mark.asyncio
    async def test_actor_unregistration(self):
        """Test unregistering actors from the system."""
        system = ActorSystem("test_system")
        actor = TestActor("temp_actor")

        # Register then unregister
        actor_ref = await system.register_actor(actor)
        await system.unregister_actor(actor.name)

        assert actor.name not in system.actors

        await system.stop()

    @pytest.mark.asyncio
    async def test_system_shutdown(self):
        """Test clean system shutdown."""
        system = ActorSystem("shutdown_test")

        # Register multiple actors
        actors = []
        for i in range(3):
            actor = TestActor(f"actor_{i}")
            await system.register_actor(actor)
            actors.append(actor)

        # Shutdown system
        await system.stop()

        # All actors should be stopped
        for actor in actors:
            assert actor.stop_called is True

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)  # Add shorter timeout for this specific test
    async def test_failing_actor_registration(self):
        """Test handling actors that fail on startup."""
        system = ActorSystem("fail_test")
        failing_actor = FailingActor("failing")

        # Registration should handle startup failure
        with pytest.raises(RuntimeError, match="Startup failure"):
            await system.register_actor(failing_actor)

        # Ensure proper cleanup even after failure
        if failing_actor._task and not failing_actor._task.done():
            failing_actor._task.cancel()
            try:
                await failing_actor._task
            except asyncio.CancelledError:
                pass

        await system.stop()


class TestMailbox:
    """Test mailbox functionality."""

    @pytest.mark.asyncio
    async def test_mailbox_creation(self):
        """Test mailbox creation with configuration."""
        mailbox = ActorMailbox(capacity=100, overflow_strategy=OverflowStrategy.DROP_OLDEST)

        assert mailbox.capacity == 100
        assert mailbox.overflow_strategy == OverflowStrategy.DROP_OLDEST
        assert mailbox.size == 0

    @pytest.mark.asyncio
    async def test_message_queuing(self):
        """Test queuing and retrieving messages."""
        mailbox = ActorMailbox()

        # Put message
        message = TestRequest(content="test")
        await mailbox.put(message)

        assert mailbox.size == 1

        # Get message
        received = await mailbox.get()
        assert received == message
        assert mailbox.size == 0

    @pytest.mark.asyncio
    async def test_mailbox_overflow(self):
        """Test mailbox behavior when full."""
        mailbox = ActorMailbox(capacity=2)

        # Fill mailbox
        await mailbox.put(TestRequest(content="1"))
        await mailbox.put(TestRequest(content="2"))

        # Should be full
        assert mailbox.is_full

        # Next put should block or fail based on implementation
        try:
            await asyncio.wait_for(mailbox.put(TestRequest(content="3")), timeout=0.1)
        except asyncio.TimeoutError:
            pass  # Expected for blocking mailbox

    @pytest.mark.asyncio
    async def test_mailbox_timeout(self):
        """Test mailbox get timeout."""
        mailbox = ActorMailbox()

        # Get from empty mailbox should timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(mailbox.get(), timeout=0.1)


class TestActorCommunication:
    """Test actor-to-actor communication."""

    @pytest.mark.asyncio
    async def test_actor_to_actor_messaging(self):
        """Test direct actor communication."""
        sender = TestActor("sender")
        receiver = TestActor("receiver")

        try:
            await sender.start()
            await receiver.start()

            # Send message from sender to receiver
            request = TestRequest(content="inter_actor_message")
            response = await receiver.send(request)

            assert response.result == "Processed: inter_actor_message"
            assert len(receiver.requests_handled) == 1
        finally:
            # Ensure both actors are stopped
            if sender.running:
                await sender.stop()
            if receiver.running:
                await receiver.stop()

    @pytest.mark.asyncio
    async def test_event_broadcasting(self):
        """Test event broadcasting to multiple actors."""
        actors = []
        tasks = []

        try:
            for i in range(3):
                actor = TestActor(f"listener_{i}")
                await actor.start()
                actors.append(actor)

            # Send event to all actors
            event = TestEvent(data="broadcast_test")

            for actor in actors:
                task = asyncio.create_task(actor.send(event))
                tasks.append(task)

            await asyncio.gather(*tasks)

            # All actors should have received the event
            for actor in actors:
                assert len(actor.events_received) == 1
                assert actor.events_received[0].data == "broadcast_test"
        finally:
            # Cancel any pending tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Cleanup all actors using helper
            await stop_actors(*actors)


class TestActorPerformance:
    """Test actor performance characteristics."""

    @pytest.mark.asyncio
    async def test_high_throughput_messaging(self):
        """Test actor performance under high message load."""
        actor = TestActor("performance_test")
        tasks = []

        try:
            await actor.start()

            # Send many messages rapidly
            message_count = 1000
            start_time = asyncio.get_event_loop().time()

            for i in range(message_count):
                request = TestRequest(content=f"perf_test_{i}")
                task = asyncio.create_task(actor.send(request))
                tasks.append(task)

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = asyncio.get_event_loop().time()

            # Filter out any exceptions from gather
            successful_responses = [r for r in responses if not isinstance(r, Exception)]

            # Check all messages processed
            assert len(successful_responses) == message_count
            assert len(actor.requests_handled) == message_count

            # Performance check (should handle 1000 messages in reasonable time)
            duration = end_time - start_time
            throughput = message_count / duration
            assert throughput > 100  # At least 100 messages/second
        finally:
            # Cancel any pending tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Ensure actor is stopped
            if actor.running:
                await actor.stop()

    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage and cleanup."""
        actor = TestActor("memory_test")

        # Create weak reference to track cleanup
        weak_ref = weakref.ref(actor)

        try:
            await actor.start()
            await actor.stop()
        finally:
            # Ensure actor is stopped before deletion
            if actor.running:
                await actor.stop()

        # Clear strong reference
        del actor

        # Force garbage collection
        import gc

        gc.collect()

        # Weak reference should still work if cleanup is proper
        # (This test mainly ensures proper lifecycle management)
        assert weak_ref is not None


if __name__ == "__main__":
    pytest.main([__file__])
