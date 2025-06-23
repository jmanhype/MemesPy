"""Task pool for managing concurrent task execution."""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict
from datetime import datetime
import heapq

from .task_types import Task, TaskResult, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)


class TaskPool:
    """Pool for managing and executing tasks concurrently."""
    
    def __init__(self, max_workers: int = 10, max_queue_size: int = 1000):
        """Initialize task pool.
        
        Args:
            max_workers: Maximum number of concurrent workers
            max_queue_size: Maximum number of tasks in queue
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        # Task queues by priority
        self._priority_queues: Dict[TaskPriority, List[Task]] = {
            priority: [] for priority in TaskPriority
        }
        
        # Active tasks
        self._active_tasks: Dict[str, Task] = {}
        self._running_tasks: Set[str] = set()
        
        # Task results
        self._results: Dict[str, TaskResult] = {}
        
        # Dependency tracking
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._dependents: Dict[str, Set[str]] = defaultdict(set)
        
        # Worker semaphore
        self._worker_semaphore = asyncio.Semaphore(max_workers)
        
        # Pool state
        self._shutdown = False
        self._workers: List[asyncio.Task] = []
        
    async def submit(self, task: Task) -> str:
        """Submit a task to the pool.
        
        Args:
            task: Task to submit
            
        Returns:
            Task ID
            
        Raises:
            ValueError: If pool is full or shutdown
        """
        if self._shutdown:
            raise ValueError("Cannot submit task to shutdown pool")
            
        if self._get_queue_size() >= self.max_queue_size:
            raise ValueError("Task queue is full")
            
        # Track task
        self._active_tasks[task.task_id] = task
        
        # Track dependencies
        if task.dependencies:
            self._dependencies[task.task_id] = set(task.dependencies)
            for dep_id in task.dependencies:
                self._dependents[dep_id].add(task.task_id)
        
        # Add to priority queue if no dependencies
        if not task.dependencies:
            self._enqueue_task(task)
            
        logger.debug(f"Submitted task {task.task_id} (type: {task.task_type.value})")
        return task.task_id
    
    async def submit_batch(self, tasks: List[Task]) -> List[str]:
        """Submit multiple tasks at once.
        
        Args:
            tasks: List of tasks to submit
            
        Returns:
            List of task IDs
        """
        task_ids = []
        for task in tasks:
            task_id = await self.submit(task)
            task_ids.append(task_id)
        return task_ids
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Wait for a specific task to complete.
        
        Args:
            task_id: ID of task to wait for
            timeout: Optional timeout in seconds
            
        Returns:
            Task result
            
        Raises:
            asyncio.TimeoutError: If timeout exceeded
            KeyError: If task not found
        """
        if task_id not in self._active_tasks:
            if task_id in self._results:
                return self._results[task_id]
            raise KeyError(f"Task {task_id} not found")
        
        # Wait for task to complete
        start_time = datetime.now()
        while task_id not in self._results:
            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= timeout:
                    raise asyncio.TimeoutError(f"Timeout waiting for task {task_id}")
            
            await asyncio.sleep(0.1)
        
        return self._results[task_id]
    
    async def wait_for_all(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Wait for multiple tasks to complete.
        
        Args:
            task_ids: List of task IDs to wait for
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary mapping task ID to result
        """
        if timeout:
            tasks = [self.wait_for_task(task_id, timeout) for task_id in task_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                task_id: result if not isinstance(result, Exception) else TaskResult(
                    task_id=task_id,
                    status=TaskStatus.TIMEOUT,
                    error=str(result)
                )
                for task_id, result in zip(task_ids, results)
            }
        else:
            results = {}
            for task_id in task_ids:
                results[task_id] = await self.wait_for_task(task_id)
            return results
    
    async def start(self):
        """Start the task pool workers."""
        if self._workers:
            return
            
        logger.info(f"Starting task pool with {self.max_workers} workers")
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)
    
    async def shutdown(self, wait: bool = True):
        """Shutdown the task pool.
        
        Args:
            wait: Whether to wait for running tasks to complete
        """
        logger.info("Shutting down task pool")
        self._shutdown = True
        
        if wait and self._workers:
            # Wait for workers to complete
            await asyncio.gather(*self._workers, return_exceptions=True)
        else:
            # Cancel workers
            for worker in self._workers:
                worker.cancel()
        
        self._workers.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics.
        
        Returns:
            Dictionary with pool statistics
        """
        return {
            "total_tasks": len(self._active_tasks),
            "running_tasks": len(self._running_tasks),
            "pending_tasks": self._get_queue_size(),
            "completed_tasks": len(self._results),
            "queue_by_priority": {
                priority.name: len(queue)
                for priority, queue in self._priority_queues.items()
            }
        }
    
    def _enqueue_task(self, task: Task):
        """Add task to appropriate priority queue."""
        queue = self._priority_queues[task.priority]
        # Use negative priority for max heap behavior
        heapq.heappush(queue, (-task.priority.value, task.created_at, task))
    
    def _dequeue_task(self) -> Optional[Task]:
        """Get next task from highest priority queue."""
        for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
            queue = self._priority_queues[priority]
            if queue:
                _, _, task = heapq.heappop(queue)
                return task
        return None
    
    def _get_queue_size(self) -> int:
        """Get total number of queued tasks."""
        return sum(len(queue) for queue in self._priority_queues.values())
    
    async def _worker_loop(self, worker_id: int):
        """Main worker loop for executing tasks."""
        logger.debug(f"Worker {worker_id} started")
        
        while not self._shutdown:
            try:
                # Get next task
                task = self._dequeue_task()
                if not task:
                    await asyncio.sleep(0.1)
                    continue
                
                # Acquire semaphore
                async with self._worker_semaphore:
                    await self._execute_task(task)
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.debug(f"Worker {worker_id} stopped")
    
    async def _execute_task(self, task: Task):
        """Execute a single task."""
        logger.debug(f"Executing task {task.task_id}")
        
        # Mark as running
        self._running_tasks.add(task.task_id)
        
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Execute with timeout
            if task.func:
                task_result = await asyncio.wait_for(
                    task.func(*task.args, **task.kwargs),
                    timeout=task.timeout_seconds
                )
                
                result.status = TaskStatus.COMPLETED
                result.result = task_result
            else:
                raise ValueError("Task has no function to execute")
                
        except asyncio.TimeoutError:
            result.status = TaskStatus.TIMEOUT
            result.error = f"Task timed out after {task.timeout_seconds}s"
            logger.warning(f"Task {task.task_id} timed out")
            
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            logger.error(f"Task {task.task_id} failed: {e}")
            
        finally:
            # Update result
            result.end_time = datetime.now()
            result.duration_ms = (result.end_time - result.start_time).total_seconds() * 1000
            
            # Store result
            self._results[task.task_id] = result
            self._running_tasks.remove(task.task_id)
            
            # Handle dependent tasks
            await self._handle_task_completion(task.task_id)
    
    async def _handle_task_completion(self, task_id: str):
        """Handle completion of a task and check dependents."""
        # Check dependent tasks
        for dependent_id in self._dependents.get(task_id, set()):
            # Remove this dependency
            self._dependencies[dependent_id].discard(task_id)
            
            # If no more dependencies, enqueue the task
            if not self._dependencies[dependent_id]:
                if dependent_id in self._active_tasks:
                    dependent_task = self._active_tasks[dependent_id]
                    self._enqueue_task(dependent_task)
                    logger.debug(f"Enqueued dependent task {dependent_id}")
        
        # Clean up tracking
        self._dependents.pop(task_id, None)
        self._dependencies.pop(task_id, None)
        self._active_tasks.pop(task_id, None)