"""Basic actor system tests."""

import pytest
import asyncio
from dataclasses import dataclass

from dspy_meme_gen.actors.core import Actor
from dspy_meme_gen.actors.base_messages import Message


@dataclass 
class TestMessage(Message):
    """Simple test message."""
    content: str = ""


class EchoActor(Actor):
    """Simple echo actor for testing."""
    
    async def receive(self, message: Message) -> None:
        """Echo the message content."""
        if isinstance(message, TestMessage):
            self.processed_messages.append(message.content)
    
    async def on_start(self) -> None:
        """Called when actor starts."""
        pass
    
    async def on_stop(self) -> None:
        """Called when actor stops."""
        pass
    
    async def on_error(self, error: Exception, message: Message) -> None:
        """Called when an error occurs."""
        pass
    
    def __init__(self, name: str):
        super().__init__(name)
        self.processed_messages = []


@pytest.mark.asyncio
async def test_actor_creation():
    """Test basic actor creation."""
    actor = EchoActor("test_actor")
    assert actor.name == "test_actor"
    assert isinstance(actor.state, dict)


@pytest.mark.asyncio
async def test_actor_lifecycle():
    """Test actor start and stop."""
    actor = EchoActor("test_actor")
    
    # Start actor
    await actor.start()
    # Just verify no exceptions thrown
    
    # Stop actor
    await actor.stop()
    # Verify actor stopped without errors