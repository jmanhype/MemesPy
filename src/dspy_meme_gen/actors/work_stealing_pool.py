"""Work stealing pool for load balancing between actors."""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from weakref import WeakSet

from .core import Actor, ActorRef, ActorSystem, Message, Request, Response
from .flow_control import FlowController, PressureLevel


class StealingStrategy(Enum):
    """Work stealing strategies."""
    RANDOM = "random"              # Steal from random victim
    LEAST_LOADED = "least_loaded"  # Steal from least loaded victim
    ROUND_ROBIN = "round_robin"    # Steal in round-robin fashion
    ADAPTIVE = "adaptive"          # Adapt strategy based on load patterns


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class WorkItem:
    """Work item for the stealing pool."""
    id: str
    message: Message
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    attempts: int = 0
    max_attempts: int = 3
    timeout: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if work item has expired."""
        if self.timeout is None:
            return False
        return time.time() - self.created_at > self.timeout
        
    def should_retry(self) -> bool:
        """Check if work item should be retried."""
        return self.attempts < self.max_attempts


class WorkerMetrics:
    """Metrics for a worker in the pool."""
    
    def __init__(self):
        self.tasks_processed = 0
        self.tasks_stolen = 0
        self.tasks_given = 0
        self.total_processing_time = 0.0
        self.queue_size = 0
        self.last_activity = time.time()
        self.failures = 0
        
    def get_load_factor(self) -> float:
        """Calculate load factor (0.0 = idle, 1.0+ = overloaded)."""
        # Simple load calculation based on queue size and recent activity
        time_since_activity = time.time() - self.last_activity
        activity_factor = max(0.0, 1.0 - (time_since_activity / 60.0))  # Decay over 1 minute
        queue_factor = min(1.0, self.queue_size / 100.0)  # Normalize queue size
        
        return (activity_factor + queue_factor) / 2.0
        
    def get_average_processing_time(self) -> float:
        """Get average processing time per task."""
        if self.tasks_processed == 0:
            return 0.0
        return self.total_processing_time / self.tasks_processed


class WorkStealingWorker(Actor):
    """Worker actor that participates in work stealing."""
    
    def __init__(
        self,
        name: str,
        pool: 'WorkStealingPool',
        max_queue_size: int = 1000,
        steal_threshold: int = 10,
        steal_batch_size: int = 5
    ):
        super().__init__(name)
        self.pool = pool
        self.max_queue_size = max_queue_size
        self.steal_threshold = steal_threshold
        self.steal_batch_size = steal_batch_size
        
        # Work queues (deque supports efficient operations on both ends)
        self.local_queue: deque[WorkItem] = deque()
        self.high_priority_queue: deque[WorkItem] = deque()
        
        # Metrics and state
        self.metrics = WorkerMetrics()
        self.is_stealing = False
        self.steal_cooldown = 1.0  # seconds
        self.last_steal_attempt = 0.0
        
        # Processing state
        self.current_work: Optional[WorkItem] = None
        self.processing_start: Optional[float] = None
        
    async def on_start(self) -> None:
        """Initialize worker."""
        await self.pool.register_worker(self)
        self.logger.info(f"Work stealing worker {self.name} started")
        
    async def on_stop(self) -> None:
        """Clean up worker."""
        await self.pool.unregister_worker(self)
        
        # Return unprocessed work to pool
        unprocessed_items = list(self.high_priority_queue) + list(self.local_queue)
        if self.current_work:
            unprocessed_items.append(self.current_work)
            
        for item in unprocessed_items:
            await self.pool.redistribute_work(item)
            
        self.logger.info(f"Work stealing worker {self.name} stopped")
        
    async def on_error(self, error: Exception) -> None:
        """Handle worker errors."""
        self.metrics.failures += 1
        self.logger.error(f"Worker {self.name} error: {error}", exc_info=True)
        
        # If processing work when error occurred, mark it for retry
        if self.current_work:
            self.current_work.attempts += 1
            if self.current_work.should_retry():
                await self.pool.redistribute_work(self.current_work)
            else:
                self.logger.error(f"Work item {self.current_work.id} failed after max attempts")
            self.current_work = None
            
    async def submit_work(self, work_item: WorkItem) -> bool:
        """Submit work to this worker."""
        if work_item.is_expired():
            self.logger.warning(f"Rejecting expired work item {work_item.id}")
            return False
            
        # Choose queue based on priority
        if work_item.priority in (TaskPriority.HIGH, TaskPriority.CRITICAL):
            if len(self.high_priority_queue) >= self.max_queue_size:
                return False
            self.high_priority_queue.append(work_item)
        else:
            if len(self.local_queue) >= self.max_queue_size:
                return False
            self.local_queue.append(work_item)
            
        self.metrics.queue_size += 1
        self.metrics.last_activity = time.time()
        
        # Wake up the worker if it's idle
        await self.mailbox.put(Message())
        
        return True
        
    async def _run(self) -> None:
        """Main worker loop with work stealing."""
        while self.running:
            try:
                # Process messages first
                message = await self.mailbox.get()
                if message and hasattr(message, '__class__') and message.__class__ != Message:
                    await self._handle_message(message)
                    continue
                    
                # Try to get work
                work_item = await self._get_next_work()
                
                if work_item:
                    await self._process_work(work_item)
                else:
                    # No work available, try to steal
                    if await self._should_attempt_steal():
                        await self._attempt_steal()
                    else:
                        # Wait a bit before checking again
                        await asyncio.sleep(0.1)
                        
            except Exception as e:
                await self.on_error(e)
                
    async def _get_next_work(self) -> Optional[WorkItem]:
        """Get next work item, prioritizing high-priority queue."""
        # Check high priority queue first
        while self.high_priority_queue:
            item = self.high_priority_queue.popleft()
            if not item.is_expired():
                self.metrics.queue_size -= 1
                return item
            else:
                self.logger.debug(f"Discarding expired high-priority item {item.id}")
                
        # Check local queue
        while self.local_queue:
            item = self.local_queue.popleft()
            if not item.is_expired():
                self.metrics.queue_size -= 1
                return item
            else:
                self.logger.debug(f"Discarding expired item {item.id}")
                
        return None
        
    async def _process_work(self, work_item: WorkItem) -> None:
        """Process a work item."""
        self.current_work = work_item
        self.processing_start = time.time()
        
        try:
            work_item.attempts += 1
            
            # Route to appropriate handler based on message type
            await self._handle_message(work_item.message)
            
            # Update metrics
            processing_time = time.time() - self.processing_start
            self.metrics.tasks_processed += 1
            self.metrics.total_processing_time += processing_time
            self.metrics.last_activity = time.time()
            
            self.logger.debug(
                f"Processed work item {work_item.id} in {processing_time:.3f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing work item {work_item.id}: {e}")
            
            # Retry if possible
            if work_item.should_retry():
                await self.pool.redistribute_work(work_item)
            else:
                self.logger.error(f"Work item {work_item.id} failed after max attempts")
                
            raise
        finally:
            self.current_work = None
            self.processing_start = None
            
    async def _should_attempt_steal(self) -> bool:
        """Determine if worker should attempt to steal work."""
        now = time.time()
        
        # Check cooldown
        if now - self.last_steal_attempt < self.steal_cooldown:
            return False
            
        # Check if we're already stealing
        if self.is_stealing:
            return False
            
        # Check if we have enough work
        total_queue_size = len(self.local_queue) + len(self.high_priority_queue)
        if total_queue_size >= self.steal_threshold:
            return False
            
        return True
        
    async def _attempt_steal(self) -> None:
        """Attempt to steal work from other workers."""
        self.is_stealing = True
        self.last_steal_attempt = time.time()
        
        try:
            victims = await self.pool.find_steal_victims(self, self.steal_batch_size)
            
            if not victims:
                self.logger.debug("No steal victims found")
                return
                
            stolen_items = []
            for victim in victims:
                items = await victim.steal_work(self.steal_batch_size // len(victims))
                stolen_items.extend(items)
                
            if stolen_items:
                self.metrics.tasks_stolen += len(stolen_items)
                self.logger.debug(f"Stole {len(stolen_items)} work items")
                
                # Add stolen items to our queues
                for item in stolen_items:
                    if item.priority in (TaskPriority.HIGH, TaskPriority.CRITICAL):
                        self.high_priority_queue.append(item)
                    else:
                        self.local_queue.append(item)
                        
                self.metrics.queue_size += len(stolen_items)
                
        except Exception as e:
            self.logger.error(f"Error during work stealing: {e}")
        finally:
            self.is_stealing = False
            
    async def steal_work(self, max_items: int) -> List[WorkItem]:
        """Allow other workers to steal work from this worker."""
        stolen_items = []
        
        # Only steal from local queue, keep high priority items
        while len(stolen_items) < max_items and self.local_queue:
            # Steal from the end (LIFO) to minimize cache misses
            item = self.local_queue.pop()
            if not item.is_expired():
                stolen_items.append(item)
            else:
                self.logger.debug(f"Discarding expired item during steal: {item.id}")
                
        if stolen_items:
            self.metrics.tasks_given += len(stolen_items)
            self.metrics.queue_size -= len(stolen_items)
            self.logger.debug(f"Gave away {len(stolen_items)} work items")
            
        return stolen_items
        
    def get_load_info(self) -> Dict[str, Any]:
        """Get current load information."""
        total_queue_size = len(self.local_queue) + len(self.high_priority_queue)
        
        return {
            "name": self.name,
            "queue_size": total_queue_size,
            "high_priority_size": len(self.high_priority_queue),
            "load_factor": self.metrics.get_load_factor(),
            "is_processing": self.current_work is not None,
            "is_stealing": self.is_stealing,
            "metrics": {
                "tasks_processed": self.metrics.tasks_processed,
                "tasks_stolen": self.metrics.tasks_stolen,
                "tasks_given": self.metrics.tasks_given,
                "average_processing_time": self.metrics.get_average_processing_time(),
                "failures": self.metrics.failures
            }
        }


class WorkStealingPool:
    """Pool of work-stealing workers for load balancing."""
    
    def __init__(
        self,
        name: str,
        stealing_strategy: StealingStrategy = StealingStrategy.ADAPTIVE,
        rebalance_interval: float = 5.0
    ):
        self.name = name
        self.stealing_strategy = stealing_strategy
        self.rebalance_interval = rebalance_interval
        
        self.workers: Dict[str, WorkStealingWorker] = {}
        self.worker_refs: Dict[str, ActorRef] = {}
        self.round_robin_index = 0
        
        # Pool metrics
        self.total_work_submitted = 0
        self.total_work_completed = 0
        self.total_work_failed = 0
        
        self.logger = logging.getLogger(f"work_stealing_pool.{name}")
        
        # Start rebalancing task
        self._rebalance_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self, system: ActorSystem) -> None:
        """Start the work stealing pool."""
        self._running = True
        self._system = system
        
        # Start rebalancing task
        self._rebalance_task = asyncio.create_task(self._rebalance_loop())
        
        self.logger.info(f"Work stealing pool {self.name} started")
        
    async def stop(self) -> None:
        """Stop the work stealing pool."""
        self._running = False
        
        if self._rebalance_task:
            self._rebalance_task.cancel()
            try:
                await self._rebalance_task
            except asyncio.CancelledError:
                pass
                
        # Stop all workers
        stop_tasks = []
        for worker in self.workers.values():
            stop_tasks.append(worker.stop())
            
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
            
        self.logger.info(f"Work stealing pool {self.name} stopped")
        
    async def add_worker(
        self,
        worker_name: str,
        max_queue_size: int = 1000,
        steal_threshold: int = 10
    ) -> ActorRef:
        """Add a new worker to the pool."""
        if worker_name in self.workers:
            raise ValueError(f"Worker {worker_name} already exists")
            
        worker = WorkStealingWorker(
            worker_name,
            self,
            max_queue_size=max_queue_size,
            steal_threshold=steal_threshold
        )
        
        worker_ref = await self._system.register_actor(worker)
        
        self.workers[worker_name] = worker
        self.worker_refs[worker_name] = worker_ref
        
        self.logger.info(f"Added worker {worker_name} to pool")
        return worker_ref
        
    async def remove_worker(self, worker_name: str) -> None:
        """Remove a worker from the pool."""
        if worker_name not in self.workers:
            return
            
        worker = self.workers[worker_name]
        await worker.stop()
        
        del self.workers[worker_name]
        del self.worker_refs[worker_name]
        
        self.logger.info(f"Removed worker {worker_name} from pool")
        
    async def register_worker(self, worker: WorkStealingWorker) -> None:
        """Register a worker with the pool (called by worker on start)."""
        # This is called by the worker itself, no need to add to our tracking
        pass
        
    async def unregister_worker(self, worker: WorkStealingWorker) -> None:
        """Unregister a worker from the pool (called by worker on stop)."""
        # This is called by the worker itself, cleanup is handled in remove_worker
        pass
        
    async def submit_work(
        self,
        message: Message,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None
    ) -> bool:
        """Submit work to the pool."""
        if not self.workers:
            self.logger.warning("No workers available to handle work")
            return False
            
        work_item = WorkItem(
            id=message.id,
            message=message,
            priority=priority,
            timeout=timeout
        )
        
        # Find best worker for the work
        worker = await self._select_worker_for_work(work_item)
        
        if worker and await worker.submit_work(work_item):
            self.total_work_submitted += 1
            return True
        else:
            self.logger.warning(f"Failed to submit work item {work_item.id}")
            return False
            
    async def redistribute_work(self, work_item: WorkItem) -> None:
        """Redistribute failed work to another worker."""
        worker = await self._select_worker_for_work(work_item)
        
        if worker:
            if await worker.submit_work(work_item):
                self.logger.debug(f"Redistributed work item {work_item.id}")
            else:
                self.logger.error(f"Failed to redistribute work item {work_item.id}")
                self.total_work_failed += 1
        else:
            self.logger.error(f"No worker available for redistribution of {work_item.id}")
            self.total_work_failed += 1
            
    async def _select_worker_for_work(self, work_item: WorkItem) -> Optional[WorkStealingWorker]:
        """Select the best worker for a work item."""
        if not self.workers:
            return None
            
        workers_list = list(self.workers.values())
        
        if self.stealing_strategy == StealingStrategy.ROUND_ROBIN:
            worker = workers_list[self.round_robin_index % len(workers_list)]
            self.round_robin_index += 1
            return worker
            
        elif self.stealing_strategy == StealingStrategy.LEAST_LOADED:
            # Find worker with lowest load
            best_worker = None
            lowest_load = float('inf')
            
            for worker in workers_list:
                load = worker.metrics.get_load_factor()
                total_queue = len(worker.local_queue) + len(worker.high_priority_queue)
                
                if total_queue < worker.max_queue_size and load < lowest_load:
                    lowest_load = load
                    best_worker = worker
                    
            return best_worker
            
        elif self.stealing_strategy == StealingStrategy.RANDOM:
            return random.choice(workers_list)
            
        elif self.stealing_strategy == StealingStrategy.ADAPTIVE:
            # Use least loaded for high priority, round robin for normal
            if work_item.priority in (TaskPriority.HIGH, TaskPriority.CRITICAL):
                return await self._select_least_loaded_worker()
            else:
                worker = workers_list[self.round_robin_index % len(workers_list)]
                self.round_robin_index += 1
                return worker
                
        return workers_list[0]  # Fallback
        
    async def _select_least_loaded_worker(self) -> Optional[WorkStealingWorker]:
        """Select the least loaded worker."""
        if not self.workers:
            return None
            
        best_worker = None
        lowest_load = float('inf')
        
        for worker in self.workers.values():
            load = worker.metrics.get_load_factor()
            total_queue = len(worker.local_queue) + len(worker.high_priority_queue)
            
            if total_queue < worker.max_queue_size and load < lowest_load:
                lowest_load = load
                best_worker = worker
                
        return best_worker
        
    async def find_steal_victims(
        self,
        thief: WorkStealingWorker,
        max_victims: int
    ) -> List[WorkStealingWorker]:
        """Find workers that the thief can steal from."""
        candidates = []
        
        for worker in self.workers.values():
            if worker == thief:
                continue
                
            # Only steal from workers with enough work
            total_queue = len(worker.local_queue) + len(worker.high_priority_queue)
            if total_queue > worker.steal_threshold:
                load = worker.metrics.get_load_factor()
                candidates.append((worker, load, total_queue))
                
        if not candidates:
            return []
            
        # Sort by load factor (highest first) and queue size
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Return top candidates
        return [worker for worker, _, _ in candidates[:max_victims]]
        
    async def _rebalance_loop(self) -> None:
        """Periodically rebalance work across workers."""
        while self._running:
            try:
                await asyncio.sleep(self.rebalance_interval)
                await self._rebalance_work()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error during rebalancing: {e}")
                
    async def _rebalance_work(self) -> None:
        """Rebalance work across workers."""
        if len(self.workers) < 2:
            return
            
        # Calculate load distribution using safe load info
        worker_loads = []
        total_work = 0
        
        for worker in self.workers.values():
            load_info = worker.get_load_info()
            queue_size = load_info["queue_size"]
            load_factor = load_info["load_factor"]
            worker_loads.append((worker, queue_size, load_factor))
            total_work += queue_size
            
        if total_work == 0:
            return
            
        # Sort by load (highest first)
        worker_loads.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        # Find overloaded and underloaded workers
        avg_work = total_work / len(self.workers)
        threshold = max(10, avg_work * 0.3)  # 30% deviation threshold
        
        overloaded = [(w, q, l) for w, q, l in worker_loads if q > avg_work + threshold]
        underloaded = [(w, q, l) for w, q, l in worker_loads if q < avg_work - threshold]
        
        # Redistribute work
        redistributions = 0
        for overloaded_worker, overload_queue, _ in overloaded:
            if not underloaded:
                break
                
            # Calculate how much to redistribute
            excess = overload_queue - avg_work
            to_redistribute = min(excess // 2, overloaded_worker.steal_batch_size)
            
            if to_redistribute > 0:
                # Find best underloaded worker
                underloaded_worker, _, _ = underloaded[redistributions % len(underloaded)]
                
                # Steal work
                stolen_items = await overloaded_worker.steal_work(int(to_redistribute))
                
                # Give to underloaded worker
                for item in stolen_items:
                    await underloaded_worker.submit_work(item)
                    
                if stolen_items:
                    self.logger.debug(
                        f"Rebalanced {len(stolen_items)} items from "
                        f"{overloaded_worker.name} to {underloaded_worker.name}"
                    )
                    redistributions += 1
                    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status."""
        worker_info = {}
        total_queue_size = 0
        total_processed = 0
        total_stolen = 0
        
        for worker_name, worker in self.workers.items():
            info = worker.get_load_info()
            worker_info[worker_name] = info
            total_queue_size += info["queue_size"]
            total_processed += info["metrics"]["tasks_processed"]
            total_stolen += info["metrics"]["tasks_stolen"]
            
        return {
            "name": self.name,
            "strategy": self.stealing_strategy.value,
            "worker_count": len(self.workers),
            "total_queue_size": total_queue_size,
            "total_work_submitted": self.total_work_submitted,
            "total_work_completed": total_processed,
            "total_work_failed": self.total_work_failed,
            "total_work_stolen": total_stolen,
            "workers": worker_info
        }