"""Supervisor implementation with restart strategies for actor fault tolerance."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any, Type
from collections import deque

from .core import Actor, ActorRef, ActorSystem, Message


class RestartStrategy(Enum):
    """Supervisor restart strategies."""
    ONE_FOR_ONE = "one_for_one"        # Restart only the failed actor
    ONE_FOR_ALL = "one_for_all"        # Restart all supervised actors
    REST_FOR_ONE = "rest_for_one"      # Restart the failed actor and all actors started after it
    ESCALATE = "escalate"              # Escalate to parent supervisor


class SupervisorDirective(Enum):
    """Supervisor directives for handling failures."""
    RESTART = "restart"                # Restart the actor
    RESUME = "resume"                  # Resume the actor without restart
    STOP = "stop"                      # Stop the actor
    ESCALATE = "escalate"              # Escalate to parent supervisor


@dataclass
class RestartPolicy:
    """Policy for actor restarts."""
    max_restarts: int = 5
    within_time_range: float = 60.0  # seconds
    backoff_strategy: str = "exponential"  # linear, exponential, fixed
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    jitter: bool = True


@dataclass
class FailureRecord:
    """Record of actor failures."""
    actor_name: str
    timestamp: float
    exception: Exception
    restart_count: int = 0


@dataclass
class SupervisedActor:
    """Container for supervised actor information."""
    actor: Actor
    actor_ref: ActorRef
    actor_class: Type[Actor]
    init_args: tuple
    init_kwargs: dict
    restart_count: int = 0
    failure_history: deque = field(default_factory=lambda: deque(maxlen=100))
    last_restart: Optional[float] = None
    state: str = "running"  # running, stopped, restarting, failed


class Supervisor(Actor):
    """Base supervisor class for managing child actors."""
    
    def __init__(
        self,
        name: str,
        restart_strategy: RestartStrategy = RestartStrategy.ONE_FOR_ONE,
        restart_policy: Optional[RestartPolicy] = None,
        max_children: int = 100
    ):
        super().__init__(name)
        self.restart_strategy = restart_strategy
        self.restart_policy = restart_policy or RestartPolicy()
        self.max_children = max_children
        
        # Child management
        self.children: Dict[str, SupervisedActor] = {}
        self.child_order: List[str] = []  # Maintains start order for REST_FOR_ONE
        self.failure_history: deque = deque(maxlen=1000)
        
        # Parent supervisor reference
        self.parent_supervisor: Optional['Supervisor'] = None
        
        # Metrics
        self.total_restarts = 0
        self.total_failures = 0
        
    async def on_start(self) -> None:
        """Initialize supervisor."""
        self.logger.info(f"Supervisor {self.name} started with strategy {self.restart_strategy}")
        
    async def on_stop(self) -> None:
        """Stop all children when supervisor stops."""
        await self.stop_all_children()
        
    async def on_error(self, error: Exception) -> None:
        """Handle supervisor errors."""
        self.logger.error(f"Supervisor {self.name} error: {error}", exc_info=True)
        
        # If we have a parent supervisor, escalate
        if self.parent_supervisor:
            await self.parent_supervisor.handle_child_failure(self.name, error)
        
    async def spawn_child(
        self,
        actor_class: Type[Actor],
        name: str,
        *args,
        **kwargs
    ) -> ActorRef:
        """Spawn a new child actor under supervision."""
        if len(self.children) >= self.max_children:
            raise RuntimeError(f"Supervisor {self.name} at maximum child capacity")
            
        if name in self.children:
            raise ValueError(f"Child actor {name} already exists")
            
        # Create and register the actor
        actor = actor_class(name, *args, **kwargs)
        actor_ref = await self._system.register_actor(actor)
        
        # Track as supervised child
        supervised = SupervisedActor(
            actor=actor,
            actor_ref=actor_ref,
            actor_class=actor_class,
            init_args=args,
            init_kwargs=kwargs
        )
        
        self.children[name] = supervised
        self.child_order.append(name)
        
        # Set up error monitoring with weak reference to avoid circular dependency
        import weakref
        supervisor_ref = weakref.ref(self)
        original_handle_message = actor._handle_message
        
        async def monitored_handle_message(message: Message) -> None:
            try:
                await original_handle_message(message)
            except Exception as e:
                supervisor = supervisor_ref()
                if supervisor:
                    await supervisor.handle_child_failure(name, e)
                raise
                
        actor._handle_message = monitored_handle_message
        
        self.logger.info(f"Spawned child actor {name} under supervision")
        return actor_ref
        
    async def handle_child_failure(self, child_name: str, exception: Exception) -> None:
        """Handle failure of a child actor."""
        if child_name not in self.children:
            self.logger.warning(f"Received failure notification for unknown child {child_name}")
            return
            
        supervised = self.children[child_name]
        failure_record = FailureRecord(
            actor_name=child_name,
            timestamp=time.time(),
            exception=exception
        )
        
        supervised.failure_history.append(failure_record)
        self.failure_history.append(failure_record)
        self.total_failures += 1
        
        self.logger.error(
            f"Child actor {child_name} failed: {exception}",
            exc_info=True
        )
        
        # Determine supervisor directive
        directive = await self.decide_directive(supervised, exception)
        
        # Execute the directive
        await self.execute_directive(child_name, directive, exception)
        
    async def decide_directive(
        self,
        supervised: SupervisedActor,
        exception: Exception
    ) -> SupervisorDirective:
        """Decide what directive to apply based on failure."""
        # Check if we've exceeded restart limits
        now = time.time()
        recent_failures = [
            f for f in supervised.failure_history
            if now - f.timestamp <= self.restart_policy.within_time_range
        ]
        
        if len(recent_failures) > self.restart_policy.max_restarts:
            self.logger.warning(
                f"Child {supervised.actor.name} exceeded restart limit "
                f"({len(recent_failures)} failures in {self.restart_policy.within_time_range}s)"
            )
            return SupervisorDirective.ESCALATE
            
        # Default to restart for most exceptions
        return SupervisorDirective.RESTART
        
    async def execute_directive(
        self,
        child_name: str,
        directive: SupervisorDirective,
        exception: Exception
    ) -> None:
        """Execute the supervisor directive."""
        if directive == SupervisorDirective.RESTART:
            await self.restart_child(child_name)
            
        elif directive == SupervisorDirective.RESUME:
            # For resume, we just log and continue
            self.logger.info(f"Resuming child {child_name} after failure")
            
        elif directive == SupervisorDirective.STOP:
            await self.stop_child(child_name)
            
        elif directive == SupervisorDirective.ESCALATE:
            if self.parent_supervisor:
                await self.parent_supervisor.handle_child_failure(self.name, exception)
            else:
                self.logger.critical(
                    f"No parent supervisor to escalate failure of {child_name}: {exception}"
                )
                
    async def restart_child(self, child_name: str) -> None:
        """Restart a specific child based on restart strategy."""
        if self.restart_strategy == RestartStrategy.ONE_FOR_ONE:
            await self._restart_one_for_one(child_name)
            
        elif self.restart_strategy == RestartStrategy.ONE_FOR_ALL:
            await self._restart_one_for_all()
            
        elif self.restart_strategy == RestartStrategy.REST_FOR_ONE:
            await self._restart_rest_for_one(child_name)
            
        elif self.restart_strategy == RestartStrategy.ESCALATE:
            if self.parent_supervisor:
                exception = RuntimeError(f"Child {child_name} requires restart escalation")
                await self.parent_supervisor.handle_child_failure(self.name, exception)
                
    async def _restart_one_for_one(self, child_name: str) -> None:
        """Restart only the failed child."""
        await self._restart_single_child(child_name)
        
    async def _restart_one_for_all(self) -> None:
        """Restart all children."""
        self.logger.info(f"Restarting all {len(self.children)} children (ONE_FOR_ALL)")
        
        # Stop all children first
        stop_tasks = []
        for name in list(self.children.keys()):
            stop_tasks.append(self._stop_single_child(name))
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Restart all children
        restart_tasks = []
        for name in self.child_order:
            if name in self.children:  # May have been removed during stop
                restart_tasks.append(self._start_single_child(name))
        await asyncio.gather(*restart_tasks, return_exceptions=True)
        
    async def _restart_rest_for_one(self, failed_child: str) -> None:
        """Restart the failed child and all children started after it."""
        if failed_child not in self.child_order:
            await self._restart_one_for_one(failed_child)
            return
            
        failed_index = self.child_order.index(failed_child)
        to_restart = self.child_order[failed_index:]
        
        self.logger.info(
            f"Restarting {len(to_restart)} children from {failed_child} onwards (REST_FOR_ONE)"
        )
        
        # Stop affected children
        stop_tasks = []
        for name in reversed(to_restart):  # Stop in reverse order
            if name in self.children:
                stop_tasks.append(self._stop_single_child(name))
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Restart in original order
        restart_tasks = []
        for name in to_restart:
            if name in self.children:
                restart_tasks.append(self._start_single_child(name))
        await asyncio.gather(*restart_tasks, return_exceptions=True)
        
    async def _restart_single_child(self, child_name: str) -> None:
        """Restart a single child actor."""
        if child_name not in self.children:
            return
            
        supervised = self.children[child_name]
        
        # Calculate restart delay with backoff
        delay = self._calculate_restart_delay(supervised)
        if delay > 0:
            self.logger.info(f"Waiting {delay:.2f}s before restarting {child_name}")
            await asyncio.sleep(delay)
            
        supervised.state = "restarting"
        
        try:
            # Stop the old actor
            await self._stop_single_child(child_name, remove=False)
            
            # Start new instance
            await self._start_single_child(child_name)
            
            supervised.restart_count += 1
            supervised.last_restart = time.time()
            self.total_restarts += 1
            
            self.logger.info(
                f"Successfully restarted child {child_name} "
                f"(restart #{supervised.restart_count})"
            )
            
        except Exception as e:
            supervised.state = "failed"
            self.logger.error(f"Failed to restart child {child_name}: {e}", exc_info=True)
            raise
            
    async def _start_single_child(self, child_name: str) -> None:
        """Start a single child actor."""
        if child_name not in self.children:
            return
            
        supervised = self.children[child_name]
        
        # Create new actor instance
        actor = supervised.actor_class(
            child_name,
            *supervised.init_args,
            **supervised.init_kwargs
        )
        
        # Register with system
        actor_ref = await self._system.register_actor(actor)
        
        # Update supervised actor
        supervised.actor = actor
        supervised.actor_ref = actor_ref
        supervised.state = "running"
        
        # Set up monitoring again
        original_handle_message = actor._handle_message
        
        async def monitored_handle_message(message: Message) -> None:
            try:
                await original_handle_message(message)
            except Exception as e:
                await self.handle_child_failure(child_name, e)
                raise
                
        actor._handle_message = monitored_handle_message
        
    async def _stop_single_child(self, child_name: str, remove: bool = True) -> None:
        """Stop a single child actor."""
        if child_name not in self.children:
            return
            
        supervised = self.children[child_name]
        supervised.state = "stopped"
        
        try:
            # Stop the actor
            await supervised.actor.stop()
            
            # Unregister from system
            await self._system.unregister_actor(child_name)
            
        except Exception as e:
            self.logger.error(f"Error stopping child {child_name}: {e}", exc_info=True)
            
        if remove:
            del self.children[child_name]
            if child_name in self.child_order:
                self.child_order.remove(child_name)
                
    def _calculate_restart_delay(self, supervised: SupervisedActor) -> float:
        """Calculate restart delay based on backoff strategy."""
        if self.restart_policy.backoff_strategy == "fixed":
            delay = self.restart_policy.initial_delay
            
        elif self.restart_policy.backoff_strategy == "linear":
            delay = self.restart_policy.initial_delay * (supervised.restart_count + 1)
            
        elif self.restart_policy.backoff_strategy == "exponential":
            delay = self.restart_policy.initial_delay * (2 ** supervised.restart_count)
            
        else:
            delay = self.restart_policy.initial_delay
            
        # Apply maximum delay limit
        delay = min(delay, self.restart_policy.max_delay)
        
        # Add jitter if enabled
        if self.restart_policy.jitter:
            import random
            jitter = random.uniform(0.8, 1.2)
            delay *= jitter
            
        return delay
        
    async def stop_child(self, child_name: str) -> None:
        """Stop a specific child actor."""
        await self._stop_single_child(child_name, remove=True)
        self.logger.info(f"Stopped child actor {child_name}")
        
    async def stop_all_children(self) -> None:
        """Stop all child actors."""
        if not self.children:
            return
            
        self.logger.info(f"Stopping all {len(self.children)} children")
        
        # Stop children in reverse order
        stop_tasks = []
        for name in reversed(self.child_order):
            if name in self.children:
                stop_tasks.append(self._stop_single_child(name))
                
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        self.children.clear()
        self.child_order.clear()
        
    def get_child_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all child actors."""
        status = {}
        for name, supervised in self.children.items():
            status[name] = {
                "state": supervised.state,
                "restart_count": supervised.restart_count,
                "last_restart": supervised.last_restart,
                "failure_count": len(supervised.failure_history),
                "mailbox_size": supervised.actor.mailbox.size() if supervised.actor else 0
            }
        return status
        
    def get_supervisor_metrics(self) -> Dict[str, Any]:
        """Get supervisor metrics."""
        return {
            "name": self.name,
            "restart_strategy": self.restart_strategy.value,
            "child_count": len(self.children),
            "total_restarts": self.total_restarts,
            "total_failures": self.total_failures,
            "children": self.get_child_status()
        }


class SupervisorTree:
    """Manages a tree of supervisors for hierarchical fault tolerance."""
    
    def __init__(self, root_supervisor: Supervisor):
        self.root = root_supervisor
        self.supervisors: Dict[str, Supervisor] = {root_supervisor.name: root_supervisor}
        self.parent_map: Dict[str, str] = {}  # child -> parent mapping
        self.logger = logging.getLogger("supervisor_tree")
        
    def add_supervisor(self, supervisor: Supervisor, parent_name: str) -> None:
        """Add a supervisor under a parent supervisor."""
        if parent_name not in self.supervisors:
            raise ValueError(f"Parent supervisor {parent_name} not found")
            
        if supervisor.name in self.supervisors:
            raise ValueError(f"Supervisor {supervisor.name} already exists")
            
        parent = self.supervisors[parent_name]
        supervisor.parent_supervisor = parent
        
        self.supervisors[supervisor.name] = supervisor
        self.parent_map[supervisor.name] = parent_name
        
        self.logger.info(f"Added supervisor {supervisor.name} under {parent_name}")
        
    def remove_supervisor(self, supervisor_name: str) -> None:
        """Remove a supervisor and all its children."""
        if supervisor_name not in self.supervisors:
            return
            
        # Remove all child supervisors recursively
        children = [
            child for child, parent in self.parent_map.items()
            if parent == supervisor_name
        ]
        
        for child in children:
            self.remove_supervisor(child)
            
        # Remove the supervisor itself
        del self.supervisors[supervisor_name]
        self.parent_map.pop(supervisor_name, None)
        
        self.logger.info(f"Removed supervisor {supervisor_name}")
        
    def get_tree_metrics(self) -> Dict[str, Any]:
        """Get metrics for the entire supervisor tree."""
        return {
            "supervisors": {
                name: supervisor.get_supervisor_metrics()
                for name, supervisor in self.supervisors.items()
            },
            "tree_structure": self._build_tree_structure()
        }
        
    def _build_tree_structure(self) -> Dict[str, Any]:
        """Build a tree structure representation."""
        def build_node(supervisor_name: str) -> Dict[str, Any]:
            children = [
                child for child, parent in self.parent_map.items()
                if parent == supervisor_name
            ]
            
            return {
                "name": supervisor_name,
                "children": [build_node(child) for child in children]
            }
            
        return build_node(self.root.name)
        
    async def start_tree(self, system: ActorSystem) -> None:
        """Start all supervisors in the tree."""
        for supervisor in self.supervisors.values():
            await system.register_actor(supervisor)
            
    async def stop_tree(self) -> None:
        """Stop all supervisors in the tree."""
        # Stop in reverse topological order (leaves first)
        ordered_supervisors = []
        
        def visit(supervisor_name: str, visited: Set[str]) -> None:
            if supervisor_name in visited:
                return
                
            visited.add(supervisor_name)
            
            # Visit children first
            children = [
                child for child, parent in self.parent_map.items()
                if parent == supervisor_name
            ]
            
            for child in children:
                visit(child, visited)
                
            ordered_supervisors.append(supervisor_name)
            
        visit(self.root.name, set())
        
        # Stop in reverse order
        for supervisor_name in reversed(ordered_supervisors):
            if supervisor_name in self.supervisors:
                await self.supervisors[supervisor_name].stop()