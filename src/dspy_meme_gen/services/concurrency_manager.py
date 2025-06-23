"""Concurrency manager for handling multiple meme generation requests efficiently."""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    """Status of a generation request."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class GenerationRequest:
    """A meme generation request."""
    id: str
    topic: str
    format: str
    created_at: float
    status: RequestStatus = RequestStatus.QUEUED
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_start: Optional[float] = None
    processing_end: Optional[float] = None


class ConcurrencyManager:
    """
    Manages concurrent meme generation requests with proper backpressure and flow control.
    
    Based on cybernetic principles:
    - Bounded queues prevent memory overflow
    - Backpressure signals when system is overloaded
    - Circuit breaker pattern for fault tolerance
    - Fair scheduling with FIFO queue
    """
    
    def __init__(
        self,
        max_concurrent: int = 5,
        max_queue_size: int = 50,
        request_timeout: float = 120.0,
        circuit_breaker_threshold: int = 5
    ):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.request_timeout = request_timeout
        self.circuit_breaker_threshold = circuit_breaker_threshold
        
        # Request tracking
        self.requests: Dict[str, GenerationRequest] = {}
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing_semaphore = asyncio.Semaphore(max_concurrent)
        
        # Circuit breaker state
        self.failure_count = 0
        self.circuit_open = False
        self.circuit_open_time: Optional[float] = None
        self.circuit_recovery_timeout = 60.0  # 1 minute
        
        # Metrics
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.timeout_requests = 0
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_requests())
    
    async def submit_request(
        self,
        topic: str,
        format: str,
        generation_func: Callable[[str, str], Awaitable[Dict[str, Any]]]
    ) -> str:
        """
        Submit a meme generation request.
        
        Args:
            topic: The meme topic
            format: The meme format
            generation_func: Async function that generates the meme
            
        Returns:
            Request ID for tracking
            
        Raises:
            asyncio.QueueFull: If queue is full (backpressure)
            RuntimeError: If circuit breaker is open
        """
        # Check circuit breaker
        if self.circuit_open:
            if time.time() - self.circuit_open_time < self.circuit_recovery_timeout:
                raise RuntimeError("Circuit breaker is open - system overloaded")
            else:
                # Try to close circuit breaker
                self.circuit_open = False
                self.failure_count = 0
                logger.info("Circuit breaker closed - attempting recovery")
        
        # Create request
        request_id = str(uuid.uuid4())
        request = GenerationRequest(
            id=request_id,
            topic=topic,
            format=format,
            created_at=time.time()
        )
        
        self.requests[request_id] = request
        self.total_requests += 1
        
        # Try to add to queue (this will raise QueueFull if full)
        try:
            await self.queue.put((request, generation_func))
            logger.info(f"Request queued (queue size: {self.queue.qsize()})")
        except asyncio.QueueFull:
            # Remove from tracking and re-raise
            del self.requests[request_id]
            raise
        
        # Start processing if not already running
        asyncio.create_task(self._process_queue())
        
        return request_id
    
    async def get_request_status(self, request_id: str) -> Optional[GenerationRequest]:
        """Get the status of a request."""
        return self.requests.get(request_id)
    
    async def wait_for_completion(self, request_id: str, timeout: Optional[float] = None) -> GenerationRequest:
        """
        Wait for a request to complete.
        
        Args:
            request_id: The request ID
            timeout: Optional timeout in seconds
            
        Returns:
            The completed request
            
        Raises:
            asyncio.TimeoutError: If timeout is reached
            KeyError: If request ID is not found
        """
        if request_id not in self.requests:
            raise KeyError("Request not found")
        
        request = self.requests[request_id]
        timeout = timeout or self.request_timeout
        
        start_time = time.time()
        while request.status in (RequestStatus.QUEUED, RequestStatus.PROCESSING):
            if time.time() - start_time > timeout:
                request.status = RequestStatus.TIMEOUT
                request.error = f"Request timed out after {timeout} seconds"
                self.timeout_requests += 1
                raise asyncio.TimeoutError("Request timed out")
            
            await asyncio.sleep(0.1)  # Check every 100ms
        
        return request
    
    async def _process_queue(self):
        """Process the request queue with concurrency control."""
        try:
            while not self.queue.empty():
                async with self.processing_semaphore:  # Limit concurrent processing
                    try:
                        request, generation_func = await asyncio.wait_for(
                            self.queue.get(), timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        break  # No more items in queue
                    
                    # Start processing
                    request.status = RequestStatus.PROCESSING
                    request.processing_start = time.time()
                    
                    logger.info(f"Processing request (concurrent: {self.max_concurrent - self.processing_semaphore._value})")
                    
                    try:
                        # Execute generation with timeout
                        result = await asyncio.wait_for(
                            generation_func(request.topic, request.format),
                            timeout=self.request_timeout
                        )
                        
                        # Success
                        request.status = RequestStatus.COMPLETED
                        request.result = result
                        request.processing_end = time.time()
                        self.completed_requests += 1
                        
                        # Reset circuit breaker on success
                        if self.failure_count > 0:
                            self.failure_count = max(0, self.failure_count - 1)
                        
                        logger.info(f"Request completed successfully in {request.processing_end - request.processing_start:.1f}s")
                        
                    except asyncio.TimeoutError:
                        # Timeout
                        request.status = RequestStatus.TIMEOUT
                        request.error = f"Generation timed out after {self.request_timeout} seconds"
                        request.processing_end = time.time()
                        self.timeout_requests += 1
                        self._handle_failure()
                        
                        logger.warning("Request timed out during processing")
                        
                    except Exception as e:
                        # Failure
                        request.status = RequestStatus.FAILED
                        request.error = str(e)
                        request.processing_end = time.time()
                        self.failed_requests += 1
                        self._handle_failure()
                        
                        logger.error(f"Request failed: {e}")
                    
                    finally:
                        self.queue.task_done()
        
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
    
    def _handle_failure(self):
        """Handle failure for circuit breaker."""
        self.failure_count += 1
        if self.failure_count >= self.circuit_breaker_threshold:
            self.circuit_open = True
            self.circuit_open_time = time.time()
            logger.error(f"Circuit breaker opened after {self.failure_count} failures")
    
    async def _cleanup_expired_requests(self):
        """Background task to clean up old requests."""
        while True:
            try:
                current_time = time.time()
                expired_ids = []
                
                for request_id, request in self.requests.items():
                    # Remove requests older than 10 minutes
                    if current_time - request.created_at > 600:
                        expired_ids.append(request_id)
                
                for request_id in expired_ids:
                    del self.requests[request_id]
                    logger.debug("Cleaned up expired request")
                
                await asyncio.sleep(60)  # Run cleanup every minute
                
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(60)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get concurrency metrics."""
        active_processing = self.max_concurrent - self.processing_semaphore._value
        
        return {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "timeout_requests": self.timeout_requests,
            "queue_size": self.queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "active_processing": active_processing,
            "max_concurrent": self.max_concurrent,
            "circuit_open": self.circuit_open,
            "failure_count": self.failure_count,
            "success_rate": self.completed_requests / max(1, self.total_requests) * 100,
            "queue_utilization": self.queue.qsize() / self.max_queue_size * 100,
            "processing_utilization": active_processing / self.max_concurrent * 100
        }
    
    async def shutdown(self):
        """Shutdown the concurrency manager."""
        self._cleanup_task.cancel()
        
        # Wait for queue to empty
        await self.queue.join()
        
        logger.info("Concurrency manager shutdown complete")


# Global instance
_concurrency_manager: Optional[ConcurrencyManager] = None


def get_concurrency_manager() -> ConcurrencyManager:
    """Get the global concurrency manager instance."""
    global _concurrency_manager
    if _concurrency_manager is None:
        _concurrency_manager = ConcurrencyManager(
            max_concurrent=5,  # Process up to 5 requests concurrently
            max_queue_size=50,  # Queue up to 50 requests
            request_timeout=120.0,  # 2 minute timeout per request
            circuit_breaker_threshold=5  # Open circuit after 5 failures
        )
    return _concurrency_manager