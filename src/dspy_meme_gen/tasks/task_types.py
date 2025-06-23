"""Task type definitions for the Task Master system."""

from enum import Enum
from typing import Any, Dict, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
import uuid


class TaskStatus(Enum):
    """Status of a task in the system."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Types of tasks in the meme generation pipeline."""
    # Verification tasks
    FACTUALITY_CHECK = "factuality_check"
    APPROPRIATENESS_CHECK = "appropriateness_check"
    INSTRUCTION_CHECK = "instruction_check"
    
    # Generation tasks
    TREND_ANALYSIS = "trend_analysis"
    FORMAT_SELECTION = "format_selection"
    PROMPT_GENERATION = "prompt_generation"
    IMAGE_RENDERING = "image_rendering"
    
    # Scoring and refinement
    SCORING = "scoring"
    REFINEMENT = "refinement"
    
    # Composite tasks
    VERIFICATION_BUNDLE = "verification_bundle"
    GENERATION_BUNDLE = "generation_bundle"


class TaskPriority(Enum):
    """Priority levels for task execution."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        """Check if task completed successfully."""
        return self.status == TaskStatus.COMPLETED
    
    @property
    def is_failure(self) -> bool:
        """Check if task failed."""
        return self.status in [TaskStatus.FAILED, TaskStatus.TIMEOUT]


@dataclass
class Task:
    """Definition of a task to be executed."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType = TaskType.PROMPT_GENERATION
    priority: TaskPriority = TaskPriority.NORMAL
    func: Optional[Callable[..., Awaitable[Any]]] = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    max_retries: int = 2
    dependencies: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __hash__(self):
        """Make task hashable for use in sets."""
        return hash(self.task_id)
    
    def __eq__(self, other):
        """Compare tasks by ID."""
        if isinstance(other, Task):
            return self.task_id == other.task_id
        return False