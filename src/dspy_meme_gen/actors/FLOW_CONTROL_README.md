# Actor System Flow Control and Backpressure

This document describes the comprehensive flow control and backpressure system implemented for the actor framework.

## Overview

The flow control system provides four main components:

1. **Flow Control** (`flow_control.py`) - Rate limiting and backpressure mechanisms
2. **Supervisor** (`supervisor.py`) - Fault tolerance with restart strategies
3. **Work Stealing Pool** (`work_stealing_pool.py`) - Load balancing between actors
4. **Adaptive Concurrency** (`adaptive_concurrency.py`) - Dynamic concurrency control

## Components

### 1. Flow Control (`flow_control.py`)

Implements various flow control strategies to prevent system overload:

#### Flow Controllers

- **TokenBucketFlowController**: Token bucket rate limiting with burst capacity
- **SlidingWindowFlowController**: Sliding window rate limiting
- **AdaptiveWindowFlowController**: Adaptive limits based on system performance

#### Usage Example

```python
from dspy_meme_gen.actors import (
    TokenBucketFlowController, 
    FlowControlledActor,
    BackpressureManager
)

# Create token bucket controller (10 req/sec, burst of 20)
controller = TokenBucketFlowController(
    name="api_rate_limit",
    rate=10.0,
    burst_capacity=20
)

# Create flow-controlled actor
actor = FlowControlledActor("worker", controller)

# Manage multiple controllers
bp_manager = BackpressureManager()
bp_manager.register_controller(controller)

# Check global pressure
pressure = bp_manager.calculate_global_pressure()
```

#### Key Features

- **Token Bucket**: Classic rate limiting with configurable burst capacity
- **Sliding Window**: Time-based request limiting
- **Adaptive Window**: Self-adjusting limits based on response times
- **Backpressure Signaling**: Propagates pressure signals across the system
- **Metrics Collection**: Detailed metrics for monitoring and alerting

### 2. Supervisor (`supervisor.py`)

Provides fault tolerance through supervision trees with configurable restart strategies:

#### Restart Strategies

- **ONE_FOR_ONE**: Restart only the failed actor
- **ONE_FOR_ALL**: Restart all supervised actors  
- **REST_FOR_ONE**: Restart failed actor and all actors started after it
- **ESCALATE**: Escalate failure to parent supervisor

#### Usage Example

```python
from dspy_meme_gen.actors import (
    Supervisor, 
    RestartStrategy, 
    RestartPolicy,
    SupervisorTree
)

# Create supervisor with restart policy
supervisor = Supervisor(
    name="worker_supervisor",
    restart_strategy=RestartStrategy.ONE_FOR_ONE,
    restart_policy=RestartPolicy(
        max_restarts=5,
        within_time_range=60.0,
        backoff_strategy="exponential",
        initial_delay=1.0,
        max_delay=30.0
    )
)

# Spawn child actors under supervision
worker_ref = await supervisor.spawn_child(
    WorkerActor, 
    "worker_1", 
    *args, 
    **kwargs
)

# Create supervision tree
root = Supervisor("root", RestartStrategy.ONE_FOR_ALL)
tree = SupervisorTree(root)
tree.add_supervisor(supervisor, "root")
```

#### Key Features

- **Hierarchical Supervision**: Build supervision trees for complex fault tolerance
- **Configurable Restart Policies**: Control when and how actors restart
- **Backoff Strategies**: Linear, exponential, or fixed delay between restarts
- **Failure Tracking**: Track failure patterns and restart counts
- **Metrics and Monitoring**: Detailed supervision metrics

### 3. Work Stealing Pool (`work_stealing_pool.py`)

Implements work stealing for dynamic load balancing between actors:

#### Stealing Strategies

- **RANDOM**: Steal from random workers
- **LEAST_LOADED**: Steal from least loaded workers
- **ROUND_ROBIN**: Steal in round-robin fashion
- **ADAPTIVE**: Adapt strategy based on load patterns

#### Usage Example

```python
from dspy_meme_gen.actors import (
    WorkStealingPool,
    StealingStrategy,
    TaskPriority,
    WorkItem
)

# Create work stealing pool
pool = WorkStealingPool(
    name="worker_pool",
    stealing_strategy=StealingStrategy.LEAST_LOADED
)
await pool.start(system)

# Add workers to pool
for i in range(4):
    await pool.add_worker(
        f"worker_{i}",
        max_queue_size=100,
        steal_threshold=10
    )

# Submit work with priority
success = await pool.submit_work(
    message=request,
    priority=TaskPriority.HIGH,
    timeout=30.0
)
```

#### Key Features

- **Dynamic Load Balancing**: Automatically redistribute work between actors
- **Priority Queues**: Support for different task priorities
- **Theft Prevention**: Configurable thresholds to prevent excessive stealing
- **Work Redistribution**: Handle failed work gracefully
- **Comprehensive Metrics**: Track stealing patterns and performance

### 4. Adaptive Concurrency (`adaptive_concurrency.py`)

Implements adaptive concurrency control using various algorithms:

#### Concurrency Strategies

- **LITTLES_LAW**: Use Little's Law (L = λW) for optimal concurrency
- **GRADIENT_DESCENT**: PID controller based on latency
- **ADDITIVE_INCREASE**: AIMD (Additive Increase, Multiplicative Decrease)
- **VEGAS**: TCP Vegas-like congestion control
- **CUBIC**: CUBIC congestion control algorithm

#### Usage Example

```python
from dspy_meme_gen.actors import (
    AdaptiveConcurrencyController,
    ConcurrencyStrategy,
    ConcurrencyLimitedActor
)

# Create concurrency controller
controller = AdaptiveConcurrencyController(
    name="api_concurrency",
    strategy=ConcurrencyStrategy.LITTLES_LAW,
    initial_concurrency=10,
    min_concurrency=1,
    max_concurrency=100,
    target_latency=1.0  # 1 second target
)

# Create concurrency-limited actor
actor = ConcurrencyLimitedActor("worker", controller)

# Monitor concurrency metrics
metrics = controller.get_concurrency_metrics()
print(f"Current concurrency: {metrics['concurrency']['current']}")
print(f"Average latency: {metrics['performance']['average_latency']}")
```

#### Key Features

- **Multiple Algorithms**: Choose the best algorithm for your workload
- **Little's Law Implementation**: Mathematically optimal concurrency
- **PID Control**: Precise control using proportional-integral-derivative
- **Congestion Detection**: TCP-inspired congestion control
- **Real-time Adaptation**: Continuously adjust based on performance

## Integration

### Combined Usage

All components can be used together for maximum effectiveness:

```python
from dspy_meme_gen.actors import *

# Create integrated worker
class IntegratedWorker(FlowControlledActor, ConcurrencyLimitedActor):
    def __init__(self, name, flow_controller, concurrency_controller):
        FlowControlledActor.__init__(self, name, flow_controller)
        ConcurrencyLimitedActor.__init__(self, name, concurrency_controller)
        
    async def on_start(self):
        await FlowControlledActor.on_start(self)
        await ConcurrencyLimitedActor.on_start(self)

# Create system with all components
system = ActorSystem("integrated")
await system.start()

# Flow control
flow_controller = TokenBucketFlowController("flow", 50.0, 100)

# Concurrency control  
concurrency_controller = AdaptiveConcurrencyController(
    "concurrency", ConcurrencyStrategy.LITTLES_LAW
)

# Supervision
supervisor = Supervisor("supervisor", RestartStrategy.ONE_FOR_ONE)
await system.register_actor(supervisor)

# Work stealing
pool = WorkStealingPool("pool", StealingStrategy.ADAPTIVE)
await pool.start(system)

# Create integrated workers
for i in range(3):
    worker = IntegratedWorker(f"worker_{i}", flow_controller, concurrency_controller)
    await supervisor.spawn_child(IntegratedWorker, f"worker_{i}", 
                                flow_controller, concurrency_controller)
    await pool.add_worker(f"worker_{i}")
```

### Monitoring

The system provides comprehensive metrics for monitoring:

```python
# Flow control metrics
bp_manager = BackpressureManager()
bp_manager.register_controller(flow_controller)
pressure = bp_manager.calculate_global_pressure()
flow_metrics = bp_manager.get_controller_metrics()

# Supervisor metrics
supervisor_metrics = supervisor.get_supervisor_metrics()

# Work stealing metrics  
pool_status = pool.get_pool_status()

# Concurrency metrics
concurrency_metrics = controller.get_concurrency_metrics()
```

## Configuration Guidelines

### Flow Control Configuration

- **High-throughput APIs**: Use TokenBucket with high rate, moderate burst
- **Batch processing**: Use AdaptiveWindow for self-tuning
- **Real-time systems**: Use SlidingWindow for predictable limiting

### Supervisor Configuration

- **Stateless workers**: Use ONE_FOR_ONE with aggressive restart
- **Coordinated workers**: Use ONE_FOR_ALL for consistency
- **Pipeline workers**: Use REST_FOR_ONE for ordered processing

### Work Stealing Configuration

- **CPU-bound tasks**: Use LEAST_LOADED strategy
- **I/O-bound tasks**: Use ADAPTIVE strategy  
- **Mixed workloads**: Use ADAPTIVE with priority queues

### Concurrency Configuration

- **Known workload patterns**: Use LITTLES_LAW
- **Variable workloads**: Use VEGAS or CUBIC
- **Strict SLA requirements**: Use GRADIENT_DESCENT with PID tuning

## Performance Considerations

1. **Flow Control Overhead**: Token bucket has lowest overhead, adaptive window has highest
2. **Supervision Overhead**: ONE_FOR_ONE has lowest impact, ONE_FOR_ALL highest
3. **Work Stealing Overhead**: More workers = more stealing overhead
4. **Concurrency Adaptation**: Frequent adaptation can cause instability

## Best Practices

1. **Start Conservative**: Begin with low limits and gradually increase
2. **Monitor Continuously**: Use metrics to understand system behavior  
3. **Test Thoroughly**: Validate under realistic load conditions
4. **Plan for Failures**: Design supervision strategies for your failure modes
5. **Tune Gradually**: Make small adjustments and measure impact

## Examples

See the provided example files:
- `usage_examples.py` - Individual component examples
- `integration_example.py` - Complete integrated system example

Run examples:
```bash
python -m src.dspy_meme_gen.actors.usage_examples
python -m src.dspy_meme_gen.actors.integration_example
```