# MemesPy Actor-Based Architecture Design

## Table of Contents
1. [Overview](#overview)
2. [Core Principles](#core-principles)
3. [Supervision Tree Structure](#supervision-tree-structure)
4. [Actor Hierarchy and Responsibilities](#actor-hierarchy-and-responsibilities)
5. [Message Protocols](#message-protocols)
6. [State Machine Definitions](#state-machine-definitions)
7. [Error Handling and Fault Isolation](#error-handling-and-fault-isolation)
8. [Backpressure and Flow Control](#backpressure-and-flow-control)
9. [Implementation Guidelines](#implementation-guidelines)

## Overview

This document defines a comprehensive actor-based architecture for MemesPy following Erlang/OTP principles. The design emphasizes fault tolerance, isolation, and the "let it crash" philosophy to create a robust, production-ready system.

### Key Technologies
- **Actor Framework**: Python's `ray` or `pykka` for actor implementation
- **Supervision**: Custom supervision tree implementation
- **Message Passing**: Async message queues with mailbox pattern
- **State Management**: Immutable state with event sourcing

## Core Principles

### 1. Let It Crash Philosophy
- Actors fail fast on errors
- Supervisors handle recovery
- No defensive programming within actors
- Clear failure boundaries

### 2. Actor Isolation
- No shared mutable state
- Communication only through messages
- Resource isolation per actor
- Independent failure domains

### 3. Supervision and Recovery
- Hierarchical supervision trees
- Configurable restart strategies
- Escalating failure handling
- Health monitoring

## Supervision Tree Structure

```
RootSupervisor
├── SystemSupervisor (one_for_all)
│   ├── MetricsCollector
│   ├── HealthMonitor
│   └── ConfigManager
│
├── APISupervisor (rest_for_one)
│   ├── HTTPServerActor
│   ├── WebSocketManager
│   └── RequestRouterPool
│       ├── RequestRouter-1
│       ├── RequestRouter-2
│       └── RequestRouter-N
│
├── ProcessingSupervisor (one_for_one)
│   ├── MemeGenerationSupervisor (simple_one_for_one)
│   │   └── MemeGeneratorPool
│   │       ├── MemeGenerator-1
│   │       ├── MemeGenerator-2
│   │       └── MemeGenerator-N
│   │
│   ├── VerificationSupervisor (one_for_one)
│   │   ├── ContentVerifierPool
│   │   ├── FactCheckerPool
│   │   └── QualityScorerPool
│   │
│   └── RefinementSupervisor (one_for_one)
│       └── MemeRefinerPool
│
├── DataSupervisor (rest_for_one)
│   ├── DatabaseConnectionPool
│   ├── CacheManager
│   │   └── RedisActorPool
│   └── StorageManager
│       └── S3ActorPool
│
└── ExternalServicesSupervisor (one_for_one)
    ├── OpenAIServicePool
    ├── ImageServicePool
    └── WebhookManager
```

### Restart Strategies

#### one_for_one
- When a child fails, only that child is restarted
- Used for independent services

#### one_for_all
- When any child fails, all children are restarted
- Used for tightly coupled services that must maintain consistency

#### rest_for_one
- When a child fails, that child and all children started after it are restarted
- Used for services with startup dependencies

#### simple_one_for_one
- Specialized for dynamically created children of the same type
- Used for worker pools

### Supervisor Configuration

```python
class SupervisorConfig:
    max_restarts: int = 3
    max_restart_interval: int = 60  # seconds
    restart_strategy: RestartStrategy
    shutdown_timeout: int = 5000  # milliseconds
    
class ChildSpec:
    id: str
    start_func: Callable
    restart: RestartType  # permanent, temporary, transient
    shutdown: ShutdownType  # brutal_kill, infinity, timeout
    type: ActorType  # worker, supervisor
```

## Actor Hierarchy and Responsibilities

### System Actors

#### MetricsCollector
- **Responsibility**: Aggregate system metrics
- **State**: Current metrics, aggregation windows
- **Messages**: MetricUpdate, GetMetrics, FlushMetrics

#### HealthMonitor
- **Responsibility**: Monitor actor health and system resources
- **State**: Health status map, thresholds
- **Messages**: HealthCheck, HealthReport, AlertTrigger

#### ConfigManager
- **Responsibility**: Manage configuration and hot reloading
- **State**: Current configuration, watchers
- **Messages**: ConfigUpdate, GetConfig, ReloadConfig

### API Layer Actors

#### HTTPServerActor
- **Responsibility**: Handle HTTP lifecycle and connection management
- **State**: Server instance, active connections
- **Messages**: StartServer, StopServer, ConnectionUpdate

#### RequestRouter
- **Responsibility**: Route requests to appropriate handlers
- **State**: Routing table, request queue
- **Messages**: RouteRequest, UpdateRoutes, GetQueueStatus

### Processing Layer Actors

#### MemeGenerator
- **Responsibility**: Generate memes using DSPy
- **State**: Generation context, template cache
- **Messages**: GenerateMeme, CancelGeneration, GetProgress

#### ContentVerifier
- **Responsibility**: Verify content appropriateness
- **State**: Verification rules, blacklists
- **Messages**: VerifyContent, UpdateRules, GetVerificationResult

#### FactChecker
- **Responsibility**: Check factual accuracy
- **State**: Fact database, verification sources
- **Messages**: CheckFacts, UpdateSources, GetFactResult

#### QualityScorer
- **Responsibility**: Score meme quality
- **State**: Scoring models, thresholds
- **Messages**: ScoreMeme, UpdateModel, GetScore

#### MemeRefiner
- **Responsibility**: Refine low-quality memes
- **State**: Refinement strategies, history
- **Messages**: RefineMeme, GetRefinementStatus

### Data Layer Actors

#### DatabaseConnectionPool
- **Responsibility**: Manage database connections
- **State**: Connection pool, transaction state
- **Messages**: GetConnection, ReleaseConnection, ExecuteQuery

#### CacheManager
- **Responsibility**: Manage distributed cache
- **State**: Cache instances, eviction policies
- **Messages**: Get, Set, Delete, Invalidate

#### StorageManager
- **Responsibility**: Handle file storage operations
- **State**: Storage backends, upload queue
- **Messages**: StoreFile, RetrieveFile, DeleteFile

### External Service Actors

#### OpenAIServiceActor
- **Responsibility**: Manage OpenAI API interactions
- **State**: API clients, rate limits, retry state
- **Messages**: CallAPI, GetRateLimit, HandleRetry

## Message Protocols

### Message Base Structure

```python
@dataclass
class Message:
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    reply_to: Optional[ActorRef] = None
    timeout: Optional[int] = None  # milliseconds

@dataclass
class Request(Message):
    pass

@dataclass
class Response(Message):
    request_id: str
    status: ResponseStatus
    
@dataclass
class Event(Message):
    event_type: str
    source: ActorRef
```

### Core Message Types

#### System Messages

```python
@dataclass
class Ping(Message):
    pass

@dataclass
class Pong(Message):
    pass

@dataclass
class Terminate(Message):
    reason: str
    graceful: bool = True

@dataclass
class Restart(Message):
    reason: str
    clear_state: bool = False
```

#### Meme Generation Messages

```python
@dataclass
class GenerateMemeRequest(Request):
    prompt: str
    style: Optional[str] = None
    user_id: str
    priority: Priority = Priority.NORMAL

@dataclass
class GenerateMemeResponse(Response):
    meme_id: Optional[str] = None
    image_url: Optional[str] = None
    error: Optional[str] = None

@dataclass
class MemeGenerationProgress(Event):
    meme_id: str
    stage: str
    progress: float  # 0.0 to 1.0
    message: str
```

#### Verification Messages

```python
@dataclass
class VerifyContentRequest(Request):
    content: str
    meme_id: str
    verification_type: VerificationType

@dataclass
class VerifyContentResponse(Response):
    passed: bool
    issues: List[str]
    confidence: float
    
@dataclass
class VerificationComplete(Event):
    meme_id: str
    all_passed: bool
    results: Dict[str, VerificationResult]
```

### Message Flow Patterns

#### Request-Response Pattern
```
Client -> Router -> Worker -> Client
   |        |         |         ^
   |        |         |         |
   +--------+---------+---------+
            correlation_id
```

#### Event Broadcasting Pattern
```
Producer -> EventBus -> Subscriber1
                    |-> Subscriber2
                    |-> Subscriber3
```

#### Pipeline Pattern
```
Generator -> Verifier -> Scorer -> Refiner -> Storage
    |           |          |         |          |
    +-------------------------------------------+
                    supervision flow
```

## State Machine Definitions

### MemeGenerator State Machine

```python
class MemeGeneratorState(Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    GENERATING = "generating"
    VERIFYING = "verifying"
    REFINING = "refining"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"

class MemeGeneratorFSM:
    transitions = [
        # trigger, source, dest, conditions, actions
        ("start", IDLE, INITIALIZING, None, "init_generation"),
        ("generate", INITIALIZING, GENERATING, "has_valid_prompt", "call_dspy"),
        ("verify", GENERATING, VERIFYING, "generation_complete", "send_to_verifier"),
        ("refine", VERIFYING, REFINING, "needs_refinement", "send_to_refiner"),
        ("finalize", VERIFYING, FINALIZING, "verification_passed", "prepare_result"),
        ("finalize", REFINING, FINALIZING, "refinement_complete", "prepare_result"),
        ("complete", FINALIZING, COMPLETED, None, "return_result"),
        ("fail", "*", FAILED, None, "handle_failure"),
        ("reset", [COMPLETED, FAILED], IDLE, None, "cleanup"),
    ]
```

### RequestRouter State Machine

```python
class RouterState(Enum):
    READY = "ready"
    ROUTING = "routing"
    QUEUED = "queued"
    PROCESSING = "processing"
    RESPONDING = "responding"
    ERROR = "error"

class RouterFSM:
    transitions = [
        ("receive_request", READY, ROUTING, None, "analyze_request"),
        ("queue", ROUTING, QUEUED, "at_capacity", "add_to_queue"),
        ("route", ROUTING, PROCESSING, "worker_available", "dispatch_to_worker"),
        ("process_queue", QUEUED, PROCESSING, "worker_available", "dispatch_queued"),
        ("receive_response", PROCESSING, RESPONDING, None, "prepare_response"),
        ("send_response", RESPONDING, READY, None, "cleanup_request"),
        ("handle_error", "*", ERROR, None, "log_error"),
        ("recover", ERROR, READY, None, "reset_state"),
    ]
```

### Supervisor State Machine

```python
class SupervisorState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    RESTARTING_CHILD = "restarting_child"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"

class SupervisorFSM:
    transitions = [
        ("start", STARTING, RUNNING, "all_children_started", None),
        ("child_failed", RUNNING, RESTARTING_CHILD, None, "apply_restart_strategy"),
        ("child_restarted", RESTARTING_CHILD, RUNNING, "within_restart_limit", None),
        ("escalate", RESTARTING_CHILD, SHUTTING_DOWN, "exceeded_restart_limit", "notify_parent"),
        ("shutdown", RUNNING, SHUTTING_DOWN, None, "stop_children"),
        ("terminate", SHUTTING_DOWN, TERMINATED, "all_children_stopped", None),
    ]
```

## Error Handling and Fault Isolation

### Error Categories

#### Transient Errors
- Network timeouts
- Rate limits
- Temporary resource unavailability
- **Strategy**: Exponential backoff retry

#### Permanent Errors
- Invalid input
- Business rule violations
- Authentication failures
- **Strategy**: Fail fast, return error

#### System Errors
- Out of memory
- Corrupted state
- Hardware failures
- **Strategy**: Let it crash, supervisor restart

### Fault Isolation Boundaries

```python
class FaultBoundary:
    def __init__(self, name: str, isolation_level: IsolationLevel):
        self.name = name
        self.isolation_level = isolation_level
        self.error_budget = ErrorBudget()
        
    def execute(self, func: Callable, timeout: int):
        try:
            return asyncio.wait_for(func(), timeout=timeout)
        except Exception as e:
            self.error_budget.record_error(e)
            if self.error_budget.exceeded():
                raise CircuitBreakerOpen()
            raise

# Boundary Definitions
BOUNDARIES = {
    "external_api": FaultBoundary("external_api", IsolationLevel.STRICT),
    "database": FaultBoundary("database", IsolationLevel.MODERATE),
    "cache": FaultBoundary("cache", IsolationLevel.RELAXED),
    "internal_processing": FaultBoundary("internal", IsolationLevel.MODERATE),
}
```

### Error Escalation Path

```
Actor Error -> Local Handler -> Supervisor -> Parent Supervisor -> Root Supervisor
     |              |                |                |                    |
     +-- Log -------+-- Retry -------+-- Restart -----+-- Shutdown -------+
```

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    def call(self, func: Callable):
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpen()
                
        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

## Backpressure and Flow Control

### Mailbox Management

```python
class ActorMailbox:
    def __init__(self, capacity: int = 1000, overflow_strategy: OverflowStrategy = OverflowStrategy.DROP_OLDEST):
        self.capacity = capacity
        self.overflow_strategy = overflow_strategy
        self.messages: Deque[Message] = deque(maxlen=capacity)
        self.high_water_mark = int(capacity * 0.8)
        self.low_water_mark = int(capacity * 0.2)
        
    def put(self, message: Message) -> bool:
        if len(self.messages) >= self.capacity:
            return self._handle_overflow(message)
        
        self.messages.append(message)
        
        if len(self.messages) >= self.high_water_mark:
            self._signal_backpressure()
            
        return True
```

### Rate Limiting

```python
class RateLimiter:
    def __init__(self, rate: int, burst: int):
        self.rate = rate  # messages per second
        self.burst = burst  # burst capacity
        self.tokens = burst
        self.last_update = time.time()
        
    def acquire(self, count: int = 1) -> bool:
        self._refill()
        
        if self.tokens >= count:
            self.tokens -= count
            return True
            
        return False
        
    def _refill(self):
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now
```

### Work Stealing

```python
class WorkStealingPool:
    def __init__(self, workers: List[ActorRef]):
        self.workers = workers
        self.work_queues = {w: deque() for w in workers}
        
    async def submit(self, work: Message):
        # Find least loaded worker
        worker = min(self.workers, key=lambda w: len(self.work_queues[w]))
        self.work_queues[worker].append(work)
        await worker.tell(work)
        
    async def steal_work(self, thief: ActorRef):
        # Find most loaded worker
        victim = max(self.workers, key=lambda w: len(self.work_queues[w]))
        
        if len(self.work_queues[victim]) > 1:
            work = self.work_queues[victim].pop()
            self.work_queues[thief].append(work)
            await thief.tell(work)
```

### Flow Control Strategies

#### 1. Adaptive Concurrency
```python
class AdaptiveConcurrency:
    def __init__(self, min_concurrency: int = 1, max_concurrency: int = 100):
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.current_concurrency = min_concurrency
        self.latency_tracker = LatencyTracker()
        
    def adjust(self):
        gradient = self.latency_tracker.get_gradient()
        
        if gradient < 0:  # Latency improving
            self.current_concurrency = min(
                self.max_concurrency,
                int(self.current_concurrency * 1.1)
            )
        else:  # Latency degrading
            self.current_concurrency = max(
                self.min_concurrency,
                int(self.current_concurrency * 0.9)
            )
```

#### 2. Bulkhead Pattern
```python
class Bulkhead:
    def __init__(self, name: str, size: int):
        self.name = name
        self.semaphore = asyncio.Semaphore(size)
        self.active_requests = 0
        self.rejected_requests = 0
        
    async def execute(self, func: Callable):
        if not self.semaphore.locked():
            async with self.semaphore:
                self.active_requests += 1
                try:
                    return await func()
                finally:
                    self.active_requests -= 1
        else:
            self.rejected_requests += 1
            raise BulkheadRejection(f"Bulkhead {self.name} is full")
```

## Implementation Guidelines

### Actor Base Class

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import asyncio
import logging

class Actor(ABC):
    def __init__(self, name: str, mailbox_size: int = 1000):
        self.name = name
        self.mailbox = ActorMailbox(mailbox_size)
        self.state = {}
        self.running = False
        self.logger = logging.getLogger(f"actor.{name}")
        self._task = None
        
    async def start(self):
        """Start the actor's message processing loop"""
        self.running = True
        self._task = asyncio.create_task(self._run())
        await self.on_start()
        
    async def stop(self):
        """Gracefully stop the actor"""
        self.running = False
        await self.on_stop()
        if self._task:
            await self._task
            
    async def _run(self):
        """Main message processing loop"""
        while self.running:
            try:
                message = await self.mailbox.get()
                await self._handle_message(message)
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                await self.on_error(e)
                
    async def _handle_message(self, message: Message):
        """Route message to appropriate handler"""
        handler_name = f"handle_{message.__class__.__name__.lower()}"
        handler = getattr(self, handler_name, self.handle_unknown)
        
        try:
            result = await handler(message)
            if message.reply_to:
                await message.reply_to.tell(result)
        except Exception as e:
            if message.reply_to:
                await message.reply_to.tell(ErrorResponse(
                    request_id=message.id,
                    error=str(e)
                ))
            raise
            
    @abstractmethod
    async def on_start(self):
        """Called when actor starts"""
        pass
        
    @abstractmethod
    async def on_stop(self):
        """Called when actor stops"""
        pass
        
    @abstractmethod
    async def on_error(self, error: Exception):
        """Called on unhandled errors"""
        pass
        
    async def handle_unknown(self, message: Message):
        """Handle unknown message types"""
        self.logger.warning(f"Unknown message type: {type(message)}")
```

### Supervisor Implementation

```python
class Supervisor(Actor):
    def __init__(self, name: str, restart_strategy: RestartStrategy, config: SupervisorConfig):
        super().__init__(name)
        self.restart_strategy = restart_strategy
        self.config = config
        self.children: Dict[str, ChildSpec] = {}
        self.restart_counts: Dict[str, List[datetime]] = {}
        
    async def start_child(self, spec: ChildSpec) -> ActorRef:
        """Start a child actor"""
        actor = await spec.start_func()
        self.children[spec.id] = spec
        self.restart_counts[spec.id] = []
        
        # Monitor child
        asyncio.create_task(self._monitor_child(spec.id, actor))
        
        return actor
        
    async def _monitor_child(self, child_id: str, actor: ActorRef):
        """Monitor child actor health"""
        while child_id in self.children:
            try:
                await actor.ask(Ping(), timeout=5000)
                await asyncio.sleep(10)  # Health check interval
            except Exception as e:
                await self._handle_child_failure(child_id, e)
                break
                
    async def _handle_child_failure(self, child_id: str, error: Exception):
        """Handle child actor failure"""
        self.logger.error(f"Child {child_id} failed: {error}")
        
        # Check restart budget
        now = datetime.utcnow()
        self.restart_counts[child_id].append(now)
        
        # Remove old restart entries
        cutoff = now - timedelta(seconds=self.config.max_restart_interval)
        self.restart_counts[child_id] = [
            t for t in self.restart_counts[child_id] if t > cutoff
        ]
        
        if len(self.restart_counts[child_id]) > self.config.max_restarts:
            # Escalate to parent
            await self._escalate_failure(child_id, error)
        else:
            # Apply restart strategy
            await self._apply_restart_strategy(child_id)
            
    async def _apply_restart_strategy(self, failed_child_id: str):
        """Apply configured restart strategy"""
        if self.restart_strategy == RestartStrategy.ONE_FOR_ONE:
            await self._restart_child(failed_child_id)
            
        elif self.restart_strategy == RestartStrategy.ONE_FOR_ALL:
            for child_id in self.children:
                await self._restart_child(child_id)
                
        elif self.restart_strategy == RestartStrategy.REST_FOR_ONE:
            # Find children started after the failed one
            child_order = list(self.children.keys())
            failed_index = child_order.index(failed_child_id)
            
            for child_id in child_order[failed_index:]:
                await self._restart_child(child_id)
```

### Message Bus Implementation

```python
class MessageBus:
    def __init__(self):
        self.topics: Dict[str, List[ActorRef]] = {}
        self.logger = logging.getLogger("messagebus")
        
    def subscribe(self, topic: str, actor: ActorRef):
        """Subscribe an actor to a topic"""
        if topic not in self.topics:
            self.topics[topic] = []
        self.topics[topic].append(actor)
        
    def unsubscribe(self, topic: str, actor: ActorRef):
        """Unsubscribe an actor from a topic"""
        if topic in self.topics:
            self.topics[topic].remove(actor)
            
    async def publish(self, topic: str, message: Message):
        """Publish a message to all subscribers"""
        if topic in self.topics:
            tasks = []
            for actor in self.topics[topic]:
                tasks.append(actor.tell(message))
            await asyncio.gather(*tasks, return_exceptions=True)
```

### Production Deployment Considerations

#### 1. Monitoring and Metrics
- Implement comprehensive metrics collection
- Track actor lifecycle events
- Monitor message queue depths
- Alert on supervision tree changes

#### 2. Distributed Deployment
- Use distributed actor frameworks (Ray, Orleans)
- Implement actor discovery and registration
- Handle network partitions gracefully
- Ensure message delivery guarantees

#### 3. Performance Optimization
- Tune mailbox sizes based on load
- Implement actor pooling for hot paths
- Use work stealing for load balancing
- Profile and optimize message serialization

#### 4. Testing Strategy
- Unit test individual actors
- Integration test supervision trees
- Chaos test failure scenarios
- Load test with production-like traffic

### Example Implementation

```python
# Example: MemeGeneratorActor
class MemeGeneratorActor(Actor):
    def __init__(self, name: str, dspy_client: DSPyClient):
        super().__init__(name)
        self.dspy_client = dspy_client
        self.fsm = MemeGeneratorFSM()
        
    async def handle_generatememe(self, message: GenerateMemeRequest):
        """Handle meme generation request"""
        # Initialize state
        self.state["request"] = message
        self.state["meme_id"] = str(uuid4())
        
        # Start state machine
        self.fsm.start()
        
        try:
            # Generate meme
            result = await self.dspy_client.generate(
                prompt=message.prompt,
                style=message.style
            )
            
            # Transition through states
            self.fsm.generate()
            
            # Send to verification
            verifier = await self.context.actor_of(ContentVerifier)
            verification_result = await verifier.ask(
                VerifyContentRequest(
                    content=result.text,
                    meme_id=self.state["meme_id"]
                )
            )
            
            self.fsm.verify()
            
            if not verification_result.passed:
                # Send to refinement
                refiner = await self.context.actor_of(MemeRefiner)
                refined_result = await refiner.ask(
                    RefineMemeRequest(
                        original=result,
                        issues=verification_result.issues
                    )
                )
                self.fsm.refine()
                result = refined_result
                
            # Finalize
            self.fsm.finalize()
            
            return GenerateMemeResponse(
                request_id=message.id,
                meme_id=self.state["meme_id"],
                image_url=result.image_url,
                status=ResponseStatus.SUCCESS
            )
            
        except Exception as e:
            self.fsm.fail()
            raise
            
    async def on_error(self, error: Exception):
        """Handle errors - let it crash"""
        self.logger.error(f"Fatal error in {self.name}: {error}")
        # Don't try to recover - supervisor will handle it
        raise error
```

## Conclusion

This actor-based architecture provides:

1. **Fault Tolerance**: Through supervision trees and isolated failure domains
2. **Scalability**: Via actor pools and work stealing
3. **Maintainability**: Clear separation of concerns and message protocols
4. **Observability**: Comprehensive monitoring and state tracking
5. **Performance**: Backpressure handling and adaptive concurrency

The design follows Erlang/OTP principles while being implementable in Python, providing a robust foundation for the MemesPy application.