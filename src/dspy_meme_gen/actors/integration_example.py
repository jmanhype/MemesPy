"""Integration example showing how all flow control components work together."""

import asyncio
import logging
from typing import Dict, Any

from .core import Actor, ActorSystem, Message, Request
from .flow_control import (
    TokenBucketFlowController,
    BackpressureManager,
    FlowControlledActor
)
from .supervisor import (
    Supervisor,
    RestartStrategy,
    RestartPolicy,
    SupervisorTree
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


class WorkerRequest(Request):
    """Example work request."""
    def __init__(self, work_data: str, priority: int = 1):
        super().__init__()
        self.work_data = work_data
        self.priority = priority


class ExampleWorkerActor(FlowControlledActor, ConcurrencyLimitedActor):
    """Example worker that combines flow control and concurrency limiting."""
    
    def __init__(self, name: str, flow_controller, concurrency_controller):
        # Initialize both parent classes
        FlowControlledActor.__init__(self, name, flow_controller)
        ConcurrencyLimitedActor.__init__(self, name, concurrency_controller)
        self.processed_count = 0
        
    async def on_start(self) -> None:
        """Start both components."""
        await FlowControlledActor.on_start(self)
        await ConcurrencyLimitedActor.on_start(self)
        
    async def on_stop(self) -> None:
        """Stop both components."""
        await FlowControlledActor.on_stop(self)
        await ConcurrencyLimitedActor.on_stop(self)
        
    async def on_error(self, error: Exception) -> None:
        """Handle errors from both components."""
        await FlowControlledActor.on_error(self, error)
        await ConcurrencyLimitedActor.on_error(self, error)
        
    async def handle_workerrequest(self, message: WorkerRequest) -> Dict[str, Any]:
        """Process work request with simulated work."""
        # Simulate work with variable duration
        import random
        work_duration = random.uniform(0.1, 2.0)
        await asyncio.sleep(work_duration)
        
        self.processed_count += 1
        
        # Simulate occasional failures
        if random.random() < 0.05:  # 5% failure rate
            raise RuntimeError(f"Simulated failure processing {message.work_data}")
        
        return {
            "result": f"Processed: {message.work_data}",
            "processed_count": self.processed_count,
            "worker": self.name
        }


class LoadGeneratorActor(Actor):
    """Generates load to test the system."""
    
    def __init__(self, name: str, work_pool: WorkStealingPool):
        super().__init__(name)
        self.work_pool = work_pool
        self.requests_sent = 0
        self._load_task = None
        
    async def on_start(self) -> None:
        """Start generating load."""
        self._load_task = asyncio.create_task(self._generate_load())
        
    async def on_stop(self) -> None:
        """Stop generating load."""
        if self._load_task:
            self._load_task.cancel()
            try:
                await self._load_task
            except asyncio.CancelledError:
                pass
                
    async def on_error(self, error: Exception) -> None:
        """Handle load generator errors."""
        self.logger.error(f"Load generator error: {error}")
        
    async def _generate_load(self) -> None:
        """Generate continuous load."""
        import random
        
        while True:
            try:
                # Variable load generation
                await asyncio.sleep(random.uniform(0.05, 0.5))
                
                # Create work request
                request = WorkerRequest(
                    work_data=f"work-{self.requests_sent}",
                    priority=random.randint(1, 4)
                )
                
                # Convert priority to TaskPriority
                priority_map = {1: TaskPriority.LOW, 2: TaskPriority.NORMAL, 
                              3: TaskPriority.HIGH, 4: TaskPriority.CRITICAL}
                task_priority = priority_map[request.priority]
                
                # Submit to work pool
                success = await self.work_pool.submit_work(request, task_priority)
                
                if success:
                    self.requests_sent += 1
                    if self.requests_sent % 100 == 0:
                        self.logger.info(f"Generated {self.requests_sent} requests")
                else:
                    self.logger.warning("Failed to submit work to pool")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error generating load: {e}")


async def create_integrated_system() -> Dict[str, Any]:
    """Create a complete integrated system with all components."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create actor system
    system = ActorSystem("integrated-system")
    await system.start()
    
    # Create backpressure manager
    backpressure_manager = BackpressureManager()
    
    # Create flow controllers for different worker types
    flow_controllers = {}
    concurrency_controllers = {}
    
    for worker_type in ["fast", "medium", "slow"]:
        # Token bucket for rate limiting
        if worker_type == "fast":
            rate, burst = 50.0, 100
            concurrency_strategy = ConcurrencyStrategy.LITTLES_LAW
            target_latency = 0.5
        elif worker_type == "medium":
            rate, burst = 20.0, 40
            concurrency_strategy = ConcurrencyStrategy.VEGAS
            target_latency = 1.0
        else:  # slow
            rate, burst = 5.0, 10
            concurrency_strategy = ConcurrencyStrategy.CUBIC
            target_latency = 2.0
            
        flow_controller = TokenBucketFlowController(
            name=f"{worker_type}_flow_control",
            rate=rate,
            burst_capacity=burst
        )
        
        concurrency_controller = AdaptiveConcurrencyController(
            name=f"{worker_type}_concurrency_control",
            strategy=concurrency_strategy,
            target_latency=target_latency,
            initial_concurrency=10,
            min_concurrency=1,
            max_concurrency=50
        )
        
        flow_controllers[worker_type] = flow_controller
        concurrency_controllers[worker_type] = concurrency_controller
        
        # Register with backpressure manager
        backpressure_manager.register_controller(flow_controller)
        
    # Create supervisor tree
    root_supervisor = Supervisor(
        name="root_supervisor",
        restart_strategy=RestartStrategy.ONE_FOR_ALL,
        restart_policy=RestartPolicy(
            max_restarts=3,
            within_time_range=60.0,
            backoff_strategy="exponential"
        )
    )
    
    worker_supervisors = {}
    for worker_type in ["fast", "medium", "slow"]:
        supervisor = Supervisor(
            name=f"{worker_type}_supervisor",
            restart_strategy=RestartStrategy.ONE_FOR_ONE,
            restart_policy=RestartPolicy(
                max_restarts=5,
                within_time_range=30.0,
                backoff_strategy="linear"
            )
        )
        worker_supervisors[worker_type] = supervisor
        
    # Create supervisor tree
    supervisor_tree = SupervisorTree(root_supervisor)
    for worker_type, supervisor in worker_supervisors.items():
        supervisor_tree.add_supervisor(supervisor, "root_supervisor")
        
    # Start supervisor tree
    await supervisor_tree.start_tree(system)
    
    # Create work stealing pools
    work_pools = {}
    for worker_type in ["fast", "medium", "slow"]:
        pool = WorkStealingPool(
            name=f"{worker_type}_pool",
            stealing_strategy=StealingStrategy.ADAPTIVE
        )
        await pool.start(system)
        work_pools[worker_type] = pool
        
    # Create and spawn workers
    workers = {}
    for worker_type in ["fast", "medium", "slow"]:
        workers[worker_type] = []
        supervisor = worker_supervisors[worker_type]
        pool = work_pools[worker_type]
        
        # Create 3 workers of each type
        for i in range(3):
            worker_name = f"{worker_type}_worker_{i}"
            
            # Create combined worker
            worker = ExampleWorkerActor(
                worker_name,
                flow_controllers[worker_type],
                concurrency_controllers[worker_type]
            )
            
            # Spawn under supervision
            worker_ref = await supervisor.spawn_child(
                ExampleWorkerActor,
                worker_name,
                flow_controllers[worker_type],
                concurrency_controllers[worker_type]
            )
            
            # Add to work pool
            await pool.add_worker(worker_name)
            
            workers[worker_type].append(worker_ref)
            
    # Create load generators
    load_generators = []
    for worker_type, pool in work_pools.items():
        load_gen = LoadGeneratorActor(f"{worker_type}_load_gen", pool)
        load_gen_ref = await system.register_actor(load_gen)
        load_generators.append(load_gen_ref)
        
    return {
        "system": system,
        "backpressure_manager": backpressure_manager,
        "supervisor_tree": supervisor_tree,
        "work_pools": work_pools,
        "flow_controllers": flow_controllers,
        "concurrency_controllers": concurrency_controllers,
        "workers": workers,
        "load_generators": load_generators
    }


async def monitor_system(components: Dict[str, Any], duration: int = 300) -> None:
    """Monitor the integrated system and log metrics."""
    
    logger = logging.getLogger("system_monitor")
    
    for _ in range(duration // 10):  # Monitor every 10 seconds
        await asyncio.sleep(10)
        
        logger.info("=== System Metrics ===")
        
        # Backpressure status
        global_pressure = components["backpressure_manager"].calculate_global_pressure()
        logger.info(f"Global Pressure: {global_pressure}")
        
        # Flow controller metrics
        flow_metrics = components["backpressure_manager"].get_controller_metrics()
        for name, metrics in flow_metrics.items():
            logger.info(
                f"Flow Control {name}: "
                f"requests={metrics.requests_total}, "
                f"allowed={metrics.requests_allowed}, "
                f"rejected={metrics.requests_rejected}, "
                f"pressure={metrics.current_pressure}"
            )
            
        # Concurrency controller metrics
        for worker_type, controller in components["concurrency_controllers"].items():
            metrics = controller.get_concurrency_metrics()
            logger.info(
                f"Concurrency {worker_type}: "
                f"current={metrics['concurrency']['current']}, "
                f"active={metrics['concurrency']['active_requests']}, "
                f"latency={metrics['performance']['average_latency']:.3f}s, "
                f"throughput={metrics['performance']['throughput']:.2f}/s"
            )
            
        # Work pool metrics
        for worker_type, pool in components["work_pools"].items():
            status = pool.get_pool_status()
            logger.info(
                f"Work Pool {worker_type}: "
                f"workers={status['worker_count']}, "
                f"queue_size={status['total_queue_size']}, "
                f"submitted={status['total_work_submitted']}, "
                f"completed={status['total_work_completed']}, "
                f"stolen={status['total_work_stolen']}"
            )
            
        # Supervisor metrics
        supervisor_metrics = components["supervisor_tree"].get_tree_metrics()
        for name, metrics in supervisor_metrics["supervisors"].items():
            if metrics["child_count"] > 0:  # Only show supervisors with children
                logger.info(
                    f"Supervisor {name}: "
                    f"children={metrics['child_count']}, "
                    f"restarts={metrics['total_restarts']}, "
                    f"failures={metrics['total_failures']}"
                )


async def run_integration_example():
    """Run the complete integration example."""
    
    logger = logging.getLogger("integration_example")
    logger.info("Starting integrated actor system example...")
    
    try:
        # Create the integrated system
        components = await create_integrated_system()
        logger.info("Integrated system created successfully")
        
        # Monitor the system
        monitor_task = asyncio.create_task(monitor_system(components, duration=120))
        
        # Let the system run
        await asyncio.sleep(120)  # Run for 2 minutes
        
        # Stop monitoring
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
            
        logger.info("Shutting down integrated system...")
        
        # Graceful shutdown
        await components["backpressure_manager"].stop_all_controllers()
        
        for pool in components["work_pools"].values():
            await pool.stop()
            
        await components["supervisor_tree"].stop_tree()
        await components["system"].stop()
        
        logger.info("Integrated system shutdown complete")
        
    except Exception as e:
        logger.error(f"Error in integration example: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Run the integration example
    asyncio.run(run_integration_example())