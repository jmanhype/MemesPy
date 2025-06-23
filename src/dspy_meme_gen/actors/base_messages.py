"""Base message classes for the actor system."""

import uuid
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Message(ABC):
    """Base message class."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    reply_to: Optional[Any] = None  # ActorRef - avoiding circular import
    timeout: Optional[int] = None  # milliseconds


@dataclass
class Request(Message):
    """Base request message."""
    pass


@dataclass
class Response(Message):
    """Base response message."""
    request_id: str = ""
    status: str = ""  # 'success' or 'error'
    error: Optional[str] = None


@dataclass
class Event(Message):
    """Base event message."""
    event_type: str = ""
    source: Optional[Any] = None  # ActorRef - avoiding circular import