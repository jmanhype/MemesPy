"""Core actor system implementation."""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional, TypeVar, Generic
from weakref import WeakValueDictionary

from .base_messages import Message, Request, Response, Event
from .mailbox import ActorMailbox, OverflowStrategy
from .circuit_breaker import CircuitBreaker

T = TypeVar('T')


class ActorRef:
    """Reference to an actor for message passing."""
    
    def __init__(self, actor: 'Actor', system: 'ActorSystem'):
        self._actor = actor
        self._system = system
        self.path = actor.path
        
    async def tell(self, message: Message) -> None:
        """Send a message without expecting a response."""
        await self._actor.receive(message)
        
    async def ask(self, message: Message, timeout: int = 5000) -> Any:
        """Send a message and wait for a response."""
        future = asyncio.Future()
        request_id = message.id
        
        # Store the future for response correlation
        self._system._pending_requests[request_id] = future
        
        # Set reply_to so the actor knows where to send the response
        message.reply_to = self._system._self_ref
        
        try:
            await self.tell(message)
            result = await asyncio.wait_for(
                future,
                timeout=timeout / 1000  # Convert to seconds
            )
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request {request_id} timed out after {timeout}ms")
        finally:
            self._system._pending_requests.pop(request_id, None)


class Actor(ABC):
    """Base actor class."""
    
    def __init__(
        self,
        name: str,
        mailbox_size: int = 1000,
        overflow_strategy: OverflowStrategy = OverflowStrategy.DROP_OLDEST
    ):
        self.name = name
        self.path = f"/user/{name}"
        self.mailbox = ActorMailbox(mailbox_size, overflow_strategy)
        self.state: Dict[str, Any] = {}
        self.running = False
        self.logger = logging.getLogger(f"actor.{name}")
        self._task: Optional[asyncio.Task] = None
        self._system: Optional['ActorSystem'] = None
        self._ref: Optional[ActorRef] = None
        
    async def start(self) -> None:
        """Start the actor's message processing loop."""
        self.running = True
        self._task = asyncio.create_task(self._run())
        await self.on_start()
        self.logger.info(f"Actor {self.name} started")
        
    async def stop(self) -> None:
        """Gracefully stop the actor."""
        self.logger.info(f"Stopping actor {self.name}")
        self.running = False
        await self.on_stop()
        if self._task:
            await self._task
        self.logger.info(f"Actor {self.name} stopped")
            
    async def receive(self, message: Message) -> None:
        """Receive a message into the mailbox."""
        await self.mailbox.put(message)
        
    async def _run(self) -> None:
        """Main message processing loop."""
        while self.running:
            try:
                message = await self.mailbox.get()
                if message:
                    await self._handle_message(message)
            except Exception as e:
                self.logger.error(f"Error processing message: {e}", exc_info=True)
                await self.on_error(e)
                
    async def _handle_message(self, message: Message) -> None:
        """Route message to appropriate handler."""
        handler_name = f"handle_{message.__class__.__name__.lower()}"
        handler = getattr(self, handler_name, self.handle_unknown)
        
        try:
            result = await handler(message)
            
            # If this is a request with reply_to, send the response
            if isinstance(message, Request) and message.reply_to:
                response = Response(
                    request_id=message.id,
                    status='success'
                )
                if isinstance(result, dict):
                    for k, v in result.items():
                        setattr(response, k, v)
                await message.reply_to.tell(response)
                
        except Exception as e:
            self.logger.error(f"Error handling message {message.id}: {e}", exc_info=True)
            if isinstance(message, Request) and message.reply_to:
                await message.reply_to.tell(Response(
                    request_id=message.id,
                    status='error',
                    error=str(e)
                ))
            raise
            
    @abstractmethod
    async def on_start(self) -> None:
        """Called when actor starts."""
        pass
        
    @abstractmethod
    async def on_stop(self) -> None:
        """Called when actor stops."""
        pass
        
    @abstractmethod
    async def on_error(self, error: Exception) -> None:
        """Called on unhandled errors."""
        pass
        
    async def handle_unknown(self, message: Message) -> None:
        """Handle unknown message types."""
        self.logger.warning(f"Unknown message type: {type(message).__name__}")
        
    def get_ref(self) -> ActorRef:
        """Get actor reference."""
        if not self._ref:
            raise RuntimeError(f"Actor {self.name} not registered with system")
        return self._ref


class ActorSystem:
    """Manages actor lifecycle and message routing."""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.actors: Dict[str, Actor] = {}
        self.actor_refs: WeakValueDictionary[str, ActorRef] = WeakValueDictionary()
        self.logger = logging.getLogger(f"system.{name}")
        self._running = False
        self._pending_requests: Dict[str, asyncio.Future] = {}
        
        # Create a system actor for handling responses
        self._system_actor = SystemActor()
        self._self_ref = None
        
    async def start(self) -> None:
        """Start the actor system."""
        self._running = True
        
        # Register system actor
        await self.register_actor(self._system_actor)
        self._self_ref = self._system_actor.get_ref()
        
        self.logger.info(f"Actor system {self.name} started")
        
    async def stop(self) -> None:
        """Stop all actors and shutdown the system."""
        self.logger.info(f"Stopping actor system {self.name}")
        
        # Stop all actors
        tasks = []
        for actor in self.actors.values():
            tasks.append(actor.stop())
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self._running = False
        self.logger.info(f"Actor system {self.name} stopped")
        
    async def register_actor(self, actor: Actor) -> ActorRef:
        """Register an actor with the system."""
        if actor.name in self.actors:
            raise ValueError(f"Actor {actor.name} already registered")
            
        self.actors[actor.name] = actor
        actor._system = self
        
        # Create actor reference
        ref = ActorRef(actor, self)
        actor._ref = ref
        self.actor_refs[actor.name] = ref
        
        # Start the actor
        await actor.start()
        
        return ref
        
    async def unregister_actor(self, name: str) -> None:
        """Unregister an actor from the system."""
        if name not in self.actors:
            return
            
        actor = self.actors[name]
        await actor.stop()
        
        del self.actors[name]
        # WeakValueDictionary will automatically remove the ref
        
    def get_actor(self, name: str) -> Optional[ActorRef]:
        """Get an actor reference by name."""
        return self.actor_refs.get(name)
        
    async def spawn(self, actor_class: type[Actor], *args, **kwargs) -> ActorRef:
        """Spawn a new actor."""
        actor = actor_class(*args, **kwargs)
        return await self.register_actor(actor)


class SystemActor(Actor):
    """Internal system actor for handling responses."""
    
    def __init__(self):
        super().__init__("system")
        
    async def on_start(self) -> None:
        pass
        
    async def on_stop(self) -> None:
        pass
        
    async def on_error(self, error: Exception) -> None:
        self.logger.error(f"System actor error: {error}")
        
    async def handle_response(self, message: Response) -> None:
        """Handle response messages."""
        if message.request_id in self._system._pending_requests:
            future = self._system._pending_requests[message.request_id]
            if message.status == 'error':
                future.set_exception(Exception(message.error))
            else:
                future.set_result(message)