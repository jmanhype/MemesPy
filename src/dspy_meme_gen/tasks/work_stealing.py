"""Work stealing scheduler for load balancing."""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Set
from collections import deque
from datetime import datetime

from .task_types import Task, TaskPriority
from .task_pool import TaskPool

logger = logging.getLogger(__name__)


class WorkQueue:
    """Thread-safe work queue for a worker."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self._queue: deque[Task] = deque()
        self._lock = asyncio.Lock()

    async def push(self, task: Task):
        """Push task to queue."""
        async with self._lock:
            self._queue.append(task)

    async def pop(self) -> Optional[Task]:
        """Pop task from own end of queue."""
        async with self._lock:
            if self._queue:
                return self._queue.pop()
        return None

    async def steal(self) -> Optional[Task]:
        """Steal task from other end of queue."""
        async with self._lock:
            if self._queue:
                return self._queue.popleft()
        return None

    async def size(self) -> int:
        """Get queue size."""
        async with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        """Check if queue is empty without locking."""
        return len(self._queue) == 0


class WorkStealingScheduler:
    """Work stealing scheduler for better load distribution."""

    def __init__(self, num_workers: int = None):
        """Initialize scheduler.

        Args:
            num_workers: Number of workers (defaults to CPU count)
        """
        self.num_workers = (
            num_workers or asyncio.get_event_loop()._thread_pool_executor._max_workers
        )

        # Create work queues for each worker
        self._work_queues: List[WorkQueue] = [WorkQueue(i) for i in range(self.num_workers)]

        # Task assignment strategy
        self._next_worker = 0
        self._worker_loads: Dict[int, int] = {i: 0 for i in range(self.num_workers)}

        # Stealing statistics
        self._steal_attempts = 0
        self._successful_steals = 0

    async def schedule_task(self, task: Task) -> int:
        """Schedule a task to a worker.

        Args:
            task: Task to schedule

        Returns:
            Worker ID that received the task
        """
        # Find least loaded worker
        worker_id = await self._find_least_loaded_worker()

        # Add to worker's queue
        await self._work_queues[worker_id].push(task)
        self._worker_loads[worker_id] += 1

        logger.debug(f"Scheduled task {task.task_id} to worker {worker_id}")
        return worker_id

    async def get_task(self, worker_id: int) -> Optional[Task]:
        """Get next task for a worker (with work stealing).

        Args:
            worker_id: ID of the requesting worker

        Returns:
            Next task to execute, or None
        """
        # Try to get from own queue first
        task = await self._work_queues[worker_id].pop()
        if task:
            self._worker_loads[worker_id] -= 1
            return task

        # Try to steal from other workers
        task = await self._steal_work(worker_id)
        if task:
            self._worker_loads[worker_id] += 1
            return task

        return None

    async def _find_least_loaded_worker(self) -> int:
        """Find the worker with least load."""
        # Simple round-robin with load consideration
        min_load = float("inf")
        best_worker = 0

        for i in range(self.num_workers):
            queue_size = await self._work_queues[i].size()
            if queue_size < min_load:
                min_load = queue_size
                best_worker = i

        return best_worker

    async def _steal_work(self, thief_id: int) -> Optional[Task]:
        """Attempt to steal work from other workers.

        Args:
            thief_id: ID of the worker attempting to steal

        Returns:
            Stolen task, or None
        """
        self._steal_attempts += 1

        # Create list of victim candidates (exclude self)
        victims = [i for i in range(self.num_workers) if i != thief_id]

        # Shuffle to randomize stealing pattern
        random.shuffle(victims)

        # Try to steal from each victim
        for victim_id in victims:
            victim_queue = self._work_queues[victim_id]

            # Only steal if victim has more than 1 task
            if await victim_queue.size() > 1:
                task = await victim_queue.steal()
                if task:
                    self._successful_steals += 1
                    self._worker_loads[victim_id] -= 1
                    logger.debug(f"Worker {thief_id} stole task from worker {victim_id}")
                    return task

        return None

    def get_stats(self) -> Dict[str, any]:
        """Get work stealing statistics."""
        steal_rate = 0.0
        if self._steal_attempts > 0:
            steal_rate = self._successful_steals / self._steal_attempts

        return {
            "num_workers": self.num_workers,
            "worker_loads": self._worker_loads.copy(),
            "steal_attempts": self._steal_attempts,
            "successful_steals": self._successful_steals,
            "steal_success_rate": steal_rate,
            "total_tasks": sum(self._worker_loads.values()),
        }


class WorkStealingPool(TaskPool):
    """Task pool with work stealing capabilities."""

    def __init__(self, max_workers: int = 10, max_queue_size: int = 1000):
        """Initialize work stealing pool."""
        super().__init__(max_workers, max_queue_size)
        self._scheduler = WorkStealingScheduler(max_workers)

    async def submit(self, task: Task) -> str:
        """Submit task with work stealing scheduling."""
        task_id = await super().submit(task)

        # Schedule to a worker if no dependencies
        if not task.dependencies:
            await self._scheduler.schedule_task(task)

        return task_id

    async def _worker_loop(self, worker_id: int):
        """Worker loop with work stealing."""
        logger.debug(f"Work-stealing worker {worker_id} started")

        while not self._shutdown:
            try:
                # Try to get task from scheduler
                task = await self._scheduler.get_task(worker_id)
                if not task:
                    # No work available, sleep briefly
                    await asyncio.sleep(0.01)
                    continue

                # Execute task
                async with self._worker_semaphore:
                    await self._execute_task(task)

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.debug(f"Work-stealing worker {worker_id} stopped")

    def get_stats(self) -> Dict[str, any]:
        """Get pool statistics including work stealing."""
        stats = super().get_stats()
        stats["work_stealing"] = self._scheduler.get_stats()
        return stats
