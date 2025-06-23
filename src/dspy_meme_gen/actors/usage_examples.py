"""Individual usage examples for each flow control component."""

import asyncio
import logging
from typing import Dict, Any

from .core import Actor, ActorSystem, Message, Request
from .flow_control import (
    TokenBucketFlowController,
    SlidingWindowFlowController,
    AdaptiveWindowFlowController,
    BackpressureManager,
    FlowControlledActor
)
from .supervisor import (
    Supervisor,
    RestartStrategy,
    RestartPolicy
)
from .work_stealing_pool import (
    WorkStealingPool,
    StealingStrategy,
    TaskPriority
)
from .adaptive_concurrency import (
    AdaptiveConcurrencyController,
    ConcurrencyStrategy,
    ConcurrencyLimitedActor
)


# Configure logging
logging.basicConfig(level=logging.INFO)


class SimpleWorkRequest(Request):
    """Simple work request for examples."""
    def __init__(self, work_id: str, data: str):
        super().__init__()
        self.work_id = work_id
        self.data = data


class SimpleWorker(Actor):
    """Simple worker for examples."""
    
    def __init__(self, name: str, processing_time: float = 0.5):
        super().__init__(name)
        self.processing_time = processing_time
        self.processed_count = 0
        
    async def on_start(self) -> None:
        self.logger.info(f"Simple worker {self.name} started")
        
    async def on_stop(self) -> None:
        self.logger.info(f"Simple worker {self.name} stopped (processed {self.processed_count} items)")
        
    async def on_error(self, error: Exception) -> None:
        self.logger.error(f"Worker {self.name} error: {error}")
        
    async def handle_simpleworkrequest(self, message: SimpleWorkRequest) -> Dict[str, Any]:
        """Process work request."""
        await asyncio.sleep(self.processing_time)
        self.processed_count += 1
        
        return {
            "result": f"Processed {message.work_id}: {message.data}",
            "worker": self.name,
            "processed_count": self.processed_count
        }


async def example_flow_control():
    """Example of using flow control components."""
    print("\n=== Flow Control Example ===")
    
    # Create actor system
    system = ActorSystem("flow-control-example")
    await system.start()
    
    try:
        # Create different flow controllers
        token_bucket = TokenBucketFlowController(
            name="token_bucket",
            rate=10.0,  # 10 requests per second
            burst_capacity=20
        )
        
        sliding_window = SlidingWindowFlowController(
            name="sliding_window",
            window_size=10.0,  # 10 second window
            max_requests=50
        )
        
        adaptive_window = AdaptiveWindowFlowController(
            name="adaptive_window",
            initial_limit=15,
            target_response_time=0.5
        )
        
        # Create backpressure manager
        bp_manager = BackpressureManager()
        bp_manager.register_controller(token_bucket)
        bp_manager.register_controller(sliding_window)
        bp_manager.register_controller(adaptive_window)
        
        # Create flow-controlled actors
        worker1 = FlowControlledActor("worker1", token_bucket)
        worker2 = FlowControlledActor("worker2", sliding_window)
        worker3 = FlowControlledActor("worker3", adaptive_window)
        
        # Register actors
        ref1 = await system.register_actor(worker1)
        ref2 = await system.register_actor(worker2) 
        ref3 = await system.register_actor(worker3)
        
        # Send requests to test flow control
        for i in range(100):
            request = SimpleWorkRequest(f"req-{i}", f"data-{i}")
            
            # Try to send to each worker
            await ref1.tell(request)
            await ref2.tell(request)
            await ref3.tell(request)
            
            if i % 10 == 0:
                pressure = bp_manager.calculate_global_pressure()
                print(f"Sent {i} requests, global pressure: {pressure}")
                
            await asyncio.sleep(0.05)  # Small delay
            
        # Wait a bit and check final metrics
        await asyncio.sleep(2)
        
        metrics = bp_manager.get_controller_metrics()
        for name, metric in metrics.items():
            print(f"{name}: {metric.requests_total} total, {metric.requests_allowed} allowed, {metric.requests_rejected} rejected")
            
        # Stop controllers
        await token_bucket.stop()
        await sliding_window.stop()
        await adaptive_window.stop()
        
    finally:
        await system.stop()


async def example_supervisor():
    """Example of using supervisor with restart strategies."""
    print("\n=== Supervisor Example ===")
    
    system = ActorSystem("supervisor-example")
    await system.start()
    
    try:
        # Create supervisor with different restart policies
        supervisor = Supervisor(
            name="example_supervisor",
            restart_strategy=RestartStrategy.ONE_FOR_ONE,
            restart_policy=RestartPolicy(
                max_restarts=3,
                within_time_range=60.0,
                backoff_strategy="exponential",
                initial_delay=1.0
            )
        )
        
        # Register supervisor
        supervisor_ref = await system.register_actor(supervisor)
        
        # Spawn children under supervision
        worker_refs = []
        for i in range(3):
            ref = await supervisor.spawn_child(SimpleWorker, f"supervised_worker_{i}", 0.1)
            worker_refs.append(ref)
            
        print(f"Spawned {len(worker_refs)} workers under supervision")
        
        # Send some work
        for i in range(50):
            worker_ref = worker_refs[i % len(worker_refs)]
            request = SimpleWorkRequest(f"work-{i}", f"test-data-{i}")
            await worker_ref.tell(request)
            
        await asyncio.sleep(2)
        
        # Check supervisor status
        status = supervisor.get_child_status()
        print("Child status:", status)
        
        # Get supervisor metrics
        metrics = supervisor.get_supervisor_metrics()
        print(f"Supervisor metrics: {metrics['total_restarts']} restarts, {metrics['total_failures']} failures")
        
    finally:
        await system.stop()


async def example_work_stealing():
    """Example of using work stealing pool."""
    print("\n=== Work Stealing Example ===")
    
    system = ActorSystem("work-stealing-example")
    await system.start()
    
    try:
        # Create work stealing pool
        pool = WorkStealingPool(
            name="example_pool",
            stealing_strategy=StealingStrategy.LEAST_LOADED
        )
        await pool.start(system)
        
        # Add workers to the pool
        worker_names = []
        for i in range(4):
            worker_name = f"stealing_worker_{i}"
            await pool.add_worker(worker_name, max_queue_size=20, steal_threshold=5)
            worker_names.append(worker_name)
            
        print(f"Added {len(worker_names)} workers to stealing pool")
        
        # Submit work with different priorities
        for i in range(200):
            request = SimpleWorkRequest(f"steal-work-{i}", f"data-{i}")
            
            # Vary priority
            if i % 10 == 0:
                priority = TaskPriority.HIGH
            elif i % 20 == 0:
                priority = TaskPriority.CRITICAL
            else:
                priority = TaskPriority.NORMAL
                
            success = await pool.submit_work(request, priority)
            if not success and i % 50 == 0:
                print(f"Failed to submit work item {i}")
                
        # Let work stealing happen
        await asyncio.sleep(5)
        
        # Check pool status
        status = pool.get_pool_status()
        print(f"Pool status: {status['total_work_submitted']} submitted, {status['total_work_completed']} completed")
        print(f"Work stolen: {status['total_work_stolen']}")
        
        # Show individual worker stats
        for worker_name, worker_info in status['workers'].items():
            print(f"Worker {worker_name}: queue={worker_info['queue_size']}, "
                  f"processed={worker_info['metrics']['tasks_processed']}, "
                  f"stolen={worker_info['metrics']['tasks_stolen']}")
        
        await pool.stop()
        
    finally:
        await system.stop()


async def example_adaptive_concurrency():
    """Example of using adaptive concurrency control."""
    print("\n=== Adaptive Concurrency Example ===")
    
    system = ActorSystem("concurrency-example")
    await system.start()
    
    try:
        # Create different concurrency controllers
        littles_controller = AdaptiveConcurrencyController(
            name="littles_law",
            strategy=ConcurrencyStrategy.LITTLES_LAW,
            initial_concurrency=5,
            target_latency=0.5
        )
        
        vegas_controller = AdaptiveConcurrencyController(
            name="vegas",
            strategy=ConcurrencyStrategy.VEGAS,
            initial_concurrency=3,
            target_latency=0.3
        )
        
        # Create concurrency-limited actors
        worker1 = ConcurrencyLimitedActor("concurrent_worker1", littles_controller)
        worker2 = ConcurrencyLimitedActor("concurrent_worker2", vegas_controller)
        
        # Register actors
        ref1 = await system.register_actor(worker1)
        ref2 = await system.register_actor(worker2)
        
        # Generate load to test adaptation
        async def generate_load(worker_ref, worker_name, count):
            for i in range(count):
                request = SimpleWorkRequest(f"{worker_name}-req-{i}", f"concurrent-data-{i}")
                await worker_ref.tell(request)
                await asyncio.sleep(0.02)  # 50 requests/second
                
        # Generate concurrent load
        await asyncio.gather(
            generate_load(ref1, "littles", 100),
            generate_load(ref2, "vegas", 100)
        )
        
        # Wait for adaptation
        await asyncio.sleep(10)
        
        # Check concurrency metrics
        littles_metrics = littles_controller.get_concurrency_metrics()
        vegas_metrics = vegas_controller.get_concurrency_metrics()
        
        print(f"Little's Law Controller: concurrency={littles_metrics['concurrency']['current']}, "
              f"latency={littles_metrics['performance']['average_latency']:.3f}s")
        print(f"Vegas Controller: concurrency={vegas_metrics['concurrency']['current']}, "
              f"latency={vegas_metrics['performance']['average_latency']:.3f}s")
        
        # Stop controllers
        await littles_controller.stop()
        await vegas_controller.stop()
        
    finally:
        await system.stop()


async def run_all_examples():
    """Run all individual examples."""
    print("Running Flow Control and Backpressure Examples")
    print("=" * 50)
    
    await example_flow_control()
    await example_supervisor()
    await example_work_stealing()
    await example_adaptive_concurrency()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    # Run all examples
    asyncio.run(run_all_examples())