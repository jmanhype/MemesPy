"""Base command definitions for CQRS."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, TypeVar, Generic
from uuid import UUID, uuid4


@dataclass
class CommandMetadata:
    """Metadata attached to every command."""

    command_id: UUID = field(default_factory=uuid4)
    command_type: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    correlation_id: Optional[UUID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Command(ABC):
    """Base class for all commands."""

    metadata: CommandMetadata = field(default_factory=CommandMetadata, init=False)

    def __post_init__(self):
        """Initialize command after dataclass creation."""
        # Default implementation - subclasses can override
        pass

    @abstractmethod
    def validate(self) -> List[str]:
        """Validate command data. Returns list of validation errors."""
        pass


T = TypeVar("T")


class CommandResult(Generic[T]):
    """Result of command execution."""

    def __init__(
        self,
        success: bool,
        value: Optional[T] = None,
        error: Optional[str] = None,
        events: Optional[List[Any]] = None,
    ):
        self.success = success
        self.value = value
        self.error = error
        self.events = events or []

    @classmethod
    def success(cls, value: T, events: Optional[List[Any]] = None) -> "CommandResult[T]":
        """Create successful result."""
        return cls(success=True, value=value, events=events)

    @classmethod
    def failure(cls, error: str) -> "CommandResult[T]":
        """Create failure result."""
        return cls(success=False, error=error)


class CommandHandler(ABC, Generic[T]):
    """Base class for command handlers."""

    @abstractmethod
    async def handle(self, command: Command) -> CommandResult[T]:
        """Handle the command and return result."""
        pass

    @abstractmethod
    def can_handle(self, command: Command) -> bool:
        """Check if this handler can handle the command."""
        pass


class CommandBus:
    """Command bus for routing commands to handlers."""

    def __init__(self):
        self._handlers: List[CommandHandler] = []

    def register_handler(self, handler: CommandHandler) -> None:
        """Register a command handler."""
        self._handlers.append(handler)

    async def send(self, command: Command) -> CommandResult:
        """Send command to appropriate handler."""
        # Validate command
        errors = command.validate()
        if errors:
            return CommandResult.failure(f"Validation errors: {', '.join(errors)}")

        # Find handler
        for handler in self._handlers:
            if handler.can_handle(command):
                return await handler.handle(command)

        return CommandResult.failure(
            f"No handler found for command type: {command.metadata.command_type}"
        )
