"""Meme-specific commands."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from uuid import UUID

from .base import Command, CommandHandler, CommandResult
from ..events import (
    MemeGenerationRequested,
    MemeGenerated,
    MemeGenerationFailed,
    MemeScored,
    MemeRefined,
    MemeApproved,
    MemeRejected,
    MemeDeleted,
)
from ..store.event_store import EventStore
from ..aggregates.meme import MemeAggregate


@dataclass
class GenerateMemeCommand(Command):
    """Command to generate a new meme."""

    topic: str
    format: str
    style: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.metadata.command_type = "GenerateMeme"

    def validate(self) -> List[str]:
        errors = []
        if not self.topic or not self.topic.strip():
            errors.append("Topic is required")
        if not self.format or not self.format.strip():
            errors.append("Format is required")
        if len(self.topic) > 500:
            errors.append("Topic must be less than 500 characters")
        return errors


@dataclass
class ScoreMemeCommand(Command):
    """Command to score a meme."""

    meme_id: UUID

    def __post_init__(self):
        super().__post_init__()
        self.metadata.command_type = "ScoreMeme"

    def validate(self) -> List[str]:
        errors = []
        if not self.meme_id:
            errors.append("Meme ID is required")
        return errors


@dataclass
class RefineMemeCommand(Command):
    """Command to refine a low-scoring meme."""

    meme_id: UUID
    refinement_reason: str

    def __post_init__(self):
        super().__post_init__()
        self.metadata.command_type = "RefineMeme"

    def validate(self) -> List[str]:
        errors = []
        if not self.meme_id:
            errors.append("Meme ID is required")
        if not self.refinement_reason:
            errors.append("Refinement reason is required")
        return errors


@dataclass
class ApproveMemeCommand(Command):
    """Command to approve a meme after verification."""

    meme_id: UUID
    verification_scores: Dict[str, float]

    def __post_init__(self):
        super().__post_init__()
        self.metadata.command_type = "ApproveMeme"

    def validate(self) -> List[str]:
        errors = []
        if not self.meme_id:
            errors.append("Meme ID is required")
        if not self.verification_scores:
            errors.append("Verification scores are required")
        return errors


@dataclass
class RejectMemeCommand(Command):
    """Command to reject a meme."""

    meme_id: UUID
    rejection_reasons: List[str]
    violation_categories: List[str]

    def __post_init__(self):
        super().__post_init__()
        self.metadata.command_type = "RejectMeme"

    def validate(self) -> List[str]:
        errors = []
        if not self.meme_id:
            errors.append("Meme ID is required")
        if not self.rejection_reasons:
            errors.append("At least one rejection reason is required")
        return errors


@dataclass
class DeleteMemeCommand(Command):
    """Command to delete a meme."""

    meme_id: UUID
    deletion_reason: str
    soft_delete: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.metadata.command_type = "DeleteMeme"

    def validate(self) -> List[str]:
        errors = []
        if not self.meme_id:
            errors.append("Meme ID is required")
        if not self.deletion_reason:
            errors.append("Deletion reason is required")
        return errors


class GenerateMemeCommandHandler(CommandHandler[UUID]):
    """Handler for meme generation commands."""

    def __init__(self, event_store: EventStore, meme_generator):
        self.event_store = event_store
        self.meme_generator = meme_generator

    def can_handle(self, command: Command) -> bool:
        return isinstance(command, GenerateMemeCommand)

    async def handle(self, command: GenerateMemeCommand) -> CommandResult[UUID]:
        """Handle meme generation command."""
        try:
            # Create aggregate
            aggregate = MemeAggregate()

            # Generate request ID
            request_id = command.metadata.command_id

            # Emit generation requested event
            request_event = MemeGenerationRequested(
                request_id=request_id,
                topic=command.topic,
                format=command.format,
                style=command.style,
                parameters=command.parameters,
            )
            request_event.metadata.aggregate_id = aggregate.id
            request_event.metadata.aggregate_type = "Meme"
            request_event.metadata.aggregate_version = 1
            request_event.metadata.user_id = command.metadata.user_id
            request_event.metadata.correlation_id = command.metadata.correlation_id

            # Generate meme using DSPy agent
            try:
                result = await self.meme_generator.generate(
                    topic=command.topic,
                    format=command.format,
                    style=command.style,
                    parameters=command.parameters,
                )

                # Emit generation success event
                success_event = MemeGenerated(
                    meme_id=aggregate.id,
                    request_id=request_id,
                    topic=command.topic,
                    format=command.format,
                    text=result["text"],
                    image_url=result["image_url"],
                    template_id=result.get("template_id"),
                    generation_time_ms=result.get("generation_time_ms", 0),
                    model_used=result.get("model_used", "unknown"),
                )
                success_event.metadata.aggregate_id = aggregate.id
                success_event.metadata.aggregate_type = "Meme"
                success_event.metadata.aggregate_version = 2
                success_event.metadata.user_id = command.metadata.user_id
                success_event.metadata.correlation_id = command.metadata.correlation_id
                success_event.metadata.causation_id = request_event.metadata.event_id

                # Store events
                await self.event_store.append_batch([request_event, success_event])

                return CommandResult.success(
                    value=aggregate.id, events=[request_event, success_event]
                )

            except Exception as e:
                # Emit generation failure event
                failure_event = MemeGenerationFailed(
                    request_id=request_id,
                    error_code="GENERATION_ERROR",
                    error_message=str(e),
                    retry_count=0,
                )
                failure_event.metadata.aggregate_id = aggregate.id
                failure_event.metadata.aggregate_type = "Meme"
                failure_event.metadata.aggregate_version = 2
                failure_event.metadata.user_id = command.metadata.user_id
                failure_event.metadata.correlation_id = command.metadata.correlation_id
                failure_event.metadata.causation_id = request_event.metadata.event_id

                # Store events
                await self.event_store.append_batch([request_event, failure_event])

                return CommandResult.failure(f"Meme generation failed: {str(e)}")

        except Exception as e:
            return CommandResult.failure(f"Command handling failed: {str(e)}")


class ScoreMemeCommandHandler(CommandHandler[float]):
    """Handler for meme scoring commands."""

    def __init__(self, event_store: EventStore, scoring_agent):
        self.event_store = event_store
        self.scoring_agent = scoring_agent

    def can_handle(self, command: Command) -> bool:
        return isinstance(command, ScoreMemeCommand)

    async def handle(self, command: ScoreMemeCommand) -> CommandResult[float]:
        """Handle meme scoring command."""
        try:
            # Load aggregate
            events = await self.event_store.get_events(command.meme_id)
            if not events:
                return CommandResult.failure("Meme not found")

            aggregate = MemeAggregate()
            for event in events:
                aggregate.apply(event)

            # Score the meme
            scores = await self.scoring_agent.score(aggregate.state)

            # Emit scored event
            scored_event = MemeScored(
                meme_id=command.meme_id,
                score=scores["overall"],
                humor_score=scores["humor"],
                relevance_score=scores["relevance"],
                appropriateness_score=scores["appropriateness"],
                scoring_model=scores.get("model", "unknown"),
                scorer_agent_id=scores.get("agent_id", "scorer"),
            )
            scored_event.metadata.aggregate_id = command.meme_id
            scored_event.metadata.aggregate_type = "Meme"
            scored_event.metadata.aggregate_version = aggregate.version + 1
            scored_event.metadata.user_id = command.metadata.user_id
            scored_event.metadata.correlation_id = command.metadata.correlation_id

            await self.event_store.append(scored_event)

            return CommandResult.success(value=scores["overall"], events=[scored_event])

        except Exception as e:
            return CommandResult.failure(f"Scoring failed: {str(e)}")


class DeleteMemeCommandHandler(CommandHandler[bool]):
    """Handler for meme deletion commands."""

    def __init__(self, event_store: EventStore):
        self.event_store = event_store

    def can_handle(self, command: Command) -> bool:
        return isinstance(command, DeleteMemeCommand)

    async def handle(self, command: DeleteMemeCommand) -> CommandResult[bool]:
        """Handle meme deletion command."""
        try:
            # Load aggregate
            events = await self.event_store.get_events(command.meme_id)
            if not events:
                return CommandResult.failure("Meme not found")

            aggregate = MemeAggregate()
            for event in events:
                aggregate.apply(event)

            # Check if already deleted
            if aggregate.state.get("deleted", False):
                return CommandResult.failure("Meme already deleted")

            # Emit deletion event
            deletion_event = MemeDeleted(
                meme_id=command.meme_id,
                deletion_reason=command.deletion_reason,
                deleted_by=command.metadata.user_id or "system",
                soft_delete=command.soft_delete,
            )
            deletion_event.metadata.aggregate_id = command.meme_id
            deletion_event.metadata.aggregate_type = "Meme"
            deletion_event.metadata.aggregate_version = aggregate.version + 1
            deletion_event.metadata.user_id = command.metadata.user_id
            deletion_event.metadata.correlation_id = command.metadata.correlation_id

            await self.event_store.append(deletion_event)

            return CommandResult.success(value=True, events=[deletion_event])

        except Exception as e:
            return CommandResult.failure(f"Deletion failed: {str(e)}")
