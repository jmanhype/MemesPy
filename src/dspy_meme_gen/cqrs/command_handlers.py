"""Command handlers for meme operations with async event sourcing."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4

from ..actors.core import ActorSystem, ActorRef
from ..actors.messages import *
from .commands.base import Command, CommandHandler, CommandResult
from .commands.meme_commands import (
    GenerateMemeCommand, ScoreMemeCommand, RefineMemeCommand,
    ApproveMemeCommand, RejectMemeCommand, DeleteMemeCommand
)
from .events import (
    MemeGenerationStarted, MemeGenerated, MemeGenerationFailed,
    TextGenerated, ImageGenerated, QualityScored, MemeCompleted,
    VerificationRequested, VerificationCompleted, RefinementRequested,
    PipelineStageStarted, PipelineStageCompleted
)
from .event_store import get_event_store, MemeEventStore
from .aggregates.meme import MemeAggregate


logger = logging.getLogger(__name__)


class AsyncMemeGenerationCommandHandler(CommandHandler[UUID]):
    """Async command handler for meme generation using actor system."""
    
    def __init__(
        self,
        actor_system: ActorSystem,
        text_generator_ref: ActorRef,
        image_generator_ref: ActorRef,
        quality_scorer_ref: ActorRef
    ):
        self.actor_system = actor_system
        self.text_generator_ref = text_generator_ref
        self.image_generator_ref = image_generator_ref
        self.quality_scorer_ref = quality_scorer_ref
        
    def can_handle(self, command: Command) -> bool:
        return isinstance(command, GenerateMemeCommand)
    
    async def handle(self, command: GenerateMemeCommand) -> CommandResult[UUID]:
        """Handle meme generation command using async actor pipeline."""
        try:
            event_store = await get_event_store()
            
            # Create new meme aggregate
            meme_id = uuid4()
            request_id = command.metadata.command_id
            
            # Emit generation started event
            started_event = MemeGenerationStarted(
                request_id=request_id,
                topic=command.topic,
                format=command.format,
                parameters=command.parameters or {},
                pipeline_config={
                    "stages": ["text_generation", "image_generation", "quality_scoring"],
                    "async_pipeline": True,
                    "actor_based": True
                },
                expected_stages=["text_generation", "image_generation", "quality_scoring"]
            )
            started_event.metadata.aggregate_id = meme_id
            started_event.metadata.aggregate_type = "Meme"
            started_event.metadata.aggregate_version = 1
            started_event.metadata.user_id = command.metadata.user_id
            started_event.metadata.correlation_id = command.metadata.correlation_id
            
            await event_store.append(started_event)
            
            # Start async pipeline
            pipeline_result = await self._execute_generation_pipeline(
                meme_id=meme_id,
                request_id=request_id,
                topic=command.topic,
                format=command.format,
                parameters=command.parameters or {},
                correlation_id=command.metadata.correlation_id,
                user_id=command.metadata.user_id
            )
            
            return CommandResult.success(
                value=meme_id,
                events=[started_event] + pipeline_result.get('events', [])
            )
            
        except Exception as e:
            logger.error(f"Meme generation command failed: {e}", exc_info=True)
            return CommandResult.failure(f"Command handling failed: {str(e)}")
    
    async def _execute_generation_pipeline(
        self,
        meme_id: UUID,
        request_id: UUID,
        topic: str,
        format: str,
        parameters: Dict[str, Any],
        correlation_id: Optional[UUID] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the async meme generation pipeline."""
        events = []
        event_store = await get_event_store()
        
        try:
            # Stage 1: Text Generation
            stage1_event = PipelineStageStarted(
                pipeline_id=request_id,
                stage_name="text_generation",
                stage_order=1,
                input_data={"topic": topic, "format": format, "parameters": parameters},
                expected_outputs=["text", "confidence_score"]
            )
            stage1_event.metadata.aggregate_id = meme_id
            stage1_event.metadata.aggregate_type = "Meme"
            stage1_event.metadata.correlation_id = correlation_id
            events.append(stage1_event)
            
            # Send text generation request to actor
            text_request = GenerateTextRequest(
                topic=topic,
                format=format,
                style=parameters.get('style'),
                max_length=parameters.get('max_length', 100)
            )
            
            start_time = datetime.utcnow()
            text_result = await self.text_generator_ref.ask(text_request, timeout=30000)
            generation_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            if text_result.status == 'success':
                # Emit text generated event
                text_event = TextGenerated(
                    meme_id=meme_id,
                    text=text_result.text,
                    generation_method="actor_dspy",
                    model_used=text_result.model_used,
                    prompt_used=text_result.prompt_used,
                    generation_time_ms=generation_time,
                    confidence_score=text_result.confidence_score,
                    alternatives_considered=text_result.alternatives
                )
                text_event.metadata.aggregate_id = meme_id
                text_event.metadata.aggregate_type = "Meme"
                text_event.metadata.correlation_id = correlation_id
                text_event.metadata.causation_id = stage1_event.metadata.event_id
                events.append(text_event)
                
                # Complete stage 1
                stage1_complete = PipelineStageCompleted(
                    pipeline_id=request_id,
                    stage_name="text_generation",
                    stage_order=1,
                    output_data={"text": text_result.text, "confidence": text_result.confidence_score},
                    success=True,
                    duration_ms=generation_time,
                    next_stage="image_generation"
                )
                stage1_complete.metadata.aggregate_id = meme_id
                stage1_complete.metadata.correlation_id = correlation_id
                events.append(stage1_complete)
                
            else:
                # Handle text generation failure
                failure_event = MemeGenerationFailed(
                    request_id=request_id,
                    error_code="TEXT_GENERATION_FAILED",
                    error_message=text_result.error,
                    retry_count=0
                )
                failure_event.metadata.aggregate_id = meme_id
                failure_event.metadata.correlation_id = correlation_id
                events.append(failure_event)
                
                await event_store.append_batch(events)
                return {"success": False, "error": "Text generation failed", "events": events}
            
            # Stage 2: Image Generation
            stage2_event = PipelineStageStarted(
                pipeline_id=request_id,
                stage_name="image_generation",
                stage_order=2,
                input_data={"text": text_result.text, "format": format},
                expected_outputs=["image_url", "dimensions"]
            )
            stage2_event.metadata.aggregate_id = meme_id
            stage2_event.metadata.correlation_id = correlation_id
            events.append(stage2_event)
            
            # Send image generation request to actor
            image_request = GenerateImageRequest(
                text=text_result.text,
                format=format,
                template_id=parameters.get('template_id'),
                style=parameters.get('style', 'meme')
            )
            
            start_time = datetime.utcnow()
            image_result = await self.image_generator_ref.ask(image_request, timeout=60000)
            generation_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            if image_result.status == 'success':
                # Emit image generated event
                image_event = ImageGenerated(
                    meme_id=meme_id,
                    image_url=image_result.image_url,
                    image_type="generated",
                    template_used=image_result.template_id,
                    generation_time_ms=generation_time,
                    dimensions={"width": image_result.width, "height": image_result.height},
                    file_size_bytes=image_result.file_size
                )
                image_event.metadata.aggregate_id = meme_id
                image_event.metadata.correlation_id = correlation_id
                image_event.metadata.causation_id = stage2_event.metadata.event_id
                events.append(image_event)
                
                # Complete stage 2
                stage2_complete = PipelineStageCompleted(
                    pipeline_id=request_id,
                    stage_name="image_generation", 
                    stage_order=2,
                    output_data={"image_url": image_result.image_url, "dimensions": image_event.dimensions},
                    success=True,
                    duration_ms=generation_time,
                    next_stage="quality_scoring"
                )
                stage2_complete.metadata.aggregate_id = meme_id
                stage2_complete.metadata.correlation_id = correlation_id
                events.append(stage2_complete)
                
            else:
                # Handle image generation failure
                failure_event = MemeGenerationFailed(
                    request_id=request_id,
                    error_code="IMAGE_GENERATION_FAILED",
                    error_message=image_result.error,
                    retry_count=0
                )
                failure_event.metadata.aggregate_id = meme_id
                failure_event.metadata.correlation_id = correlation_id
                events.append(failure_event)
                
                await event_store.append_batch(events)
                return {"success": False, "error": "Image generation failed", "events": events}
            
            # Stage 3: Quality Scoring
            stage3_event = PipelineStageStarted(
                pipeline_id=request_id,
                stage_name="quality_scoring",
                stage_order=3,
                input_data={"text": text_result.text, "image_url": image_result.image_url},
                expected_outputs=["scores", "overall_score"]
            )
            stage3_event.metadata.aggregate_id = meme_id
            stage3_event.metadata.correlation_id = correlation_id
            events.append(stage3_event)
            
            # Send scoring request to actor
            score_request = ScoreMemeRequest(
                text=text_result.text,
                image_url=image_result.image_url,
                topic=topic,
                format=format
            )
            
            start_time = datetime.utcnow()
            score_result = await self.quality_scorer_ref.ask(score_request, timeout=30000)
            scoring_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            if score_result.status == 'success':
                # Emit quality scored event
                quality_event = QualityScored(
                    meme_id=meme_id,
                    overall_score=score_result.overall_score,
                    humor_score=score_result.humor_score,
                    relevance_score=score_result.relevance_score,
                    originality_score=score_result.originality_score,
                    visual_appeal_score=score_result.visual_appeal_score,
                    appropriateness_score=score_result.appropriateness_score,
                    scoring_agent=score_result.agent_id,
                    scoring_model=score_result.model_used,
                    criteria_details=score_result.criteria_details
                )
                quality_event.metadata.aggregate_id = meme_id  
                quality_event.metadata.correlation_id = correlation_id
                quality_event.metadata.causation_id = stage3_event.metadata.event_id
                events.append(quality_event)
                
                # Complete stage 3
                stage3_complete = PipelineStageCompleted(
                    pipeline_id=request_id,
                    stage_name="quality_scoring",
                    stage_order=3,
                    output_data={"overall_score": score_result.overall_score, "scores": score_result.scores},
                    success=True,
                    duration_ms=scoring_time,
                    next_stage=None  # Final stage
                )
                stage3_complete.metadata.aggregate_id = meme_id
                stage3_complete.metadata.correlation_id = correlation_id
                events.append(stage3_complete)
                
                # Emit meme completed event
                completed_event = MemeCompleted(
                    meme_id=meme_id,
                    request_id=request_id,
                    final_score=score_result.overall_score,
                    status="completed" if score_result.overall_score >= 0.7 else "needs_refinement",
                    completion_time_ms=int((datetime.utcnow() - started_event.metadata.timestamp).total_seconds() * 1000),
                    pipeline_stages=["text_generation", "image_generation", "quality_scoring"],
                    final_metadata={
                        "text": text_result.text,
                        "image_url": image_result.image_url,
                        "overall_score": score_result.overall_score
                    }
                )
                completed_event.metadata.aggregate_id = meme_id
                completed_event.metadata.correlation_id = correlation_id
                events.append(completed_event)
                
            else:
                # Handle scoring failure
                failure_event = MemeGenerationFailed(
                    request_id=request_id,
                    error_code="QUALITY_SCORING_FAILED",
                    error_message=score_result.error,
                    retry_count=0
                )
                failure_event.metadata.aggregate_id = meme_id
                failure_event.metadata.correlation_id = correlation_id
                events.append(failure_event)
                
                await event_store.append_batch(events)
                return {"success": False, "error": "Quality scoring failed", "events": events}
            
            # Store all events
            await event_store.append_batch(events)
            
            return {
                "success": True,
                "meme_id": meme_id,
                "text": text_result.text,
                "image_url": image_result.image_url,
                "score": score_result.overall_score,
                "events": events
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            
            # Emit failure event
            failure_event = MemeGenerationFailed(
                request_id=request_id,
                error_code="PIPELINE_EXECUTION_FAILED",
                error_message=str(e),
                retry_count=0
            )
            failure_event.metadata.aggregate_id = meme_id
            failure_event.metadata.correlation_id = correlation_id
            events.append(failure_event)
            
            await event_store.append_batch(events)
            
            return {"success": False, "error": str(e), "events": events}


class AsyncMemeVerificationCommandHandler(CommandHandler[Dict[str, Any]]):
    """Async command handler for meme verification."""
    
    def __init__(self, verification_actor_ref: ActorRef):
        self.verification_actor_ref = verification_actor_ref
    
    def can_handle(self, command: Command) -> bool:
        return isinstance(command, VerifyMemeCommand)  # This command would need to be defined
    
    async def handle(self, command) -> CommandResult[Dict[str, Any]]:
        """Handle meme verification command."""
        try:
            event_store = await get_event_store()
            
            # Emit verification requested event
            verification_requested = VerificationRequested(
                meme_id=command.meme_id,
                verification_type=command.verification_type,
                content_to_verify=command.content,
                verification_criteria=command.criteria
            )
            verification_requested.metadata.aggregate_id = command.meme_id
            verification_requested.metadata.aggregate_type = "Meme"
            verification_requested.metadata.correlation_id = command.metadata.correlation_id
            
            await event_store.append(verification_requested)
            
            # Send verification request to actor
            verify_request = VerifyContentRequest(
                content=command.content,
                verification_type=command.verification_type,
                criteria=command.criteria
            )
            
            result = await self.verification_actor_ref.ask(verify_request, timeout=30000)
            
            # Emit verification completed event
            verification_completed = VerificationCompleted(
                meme_id=command.meme_id,
                verification_type=command.verification_type,
                verification_result=result.result,
                confidence_score=result.confidence,
                details=result.details,
                verification_agent=result.agent_id,
                flags_raised=result.flags
            )
            verification_completed.metadata.aggregate_id = command.meme_id
            verification_completed.metadata.correlation_id = command.metadata.correlation_id
            verification_completed.metadata.causation_id = verification_requested.metadata.event_id
            
            await event_store.append(verification_completed)
            
            return CommandResult.success(
                value={"result": result.result, "confidence": result.confidence, "details": result.details},
                events=[verification_requested, verification_completed]
            )
            
        except Exception as e:
            logger.error(f"Verification command failed: {e}", exc_info=True)
            return CommandResult.failure(f"Verification failed: {str(e)}")


class AsyncMemeRefinementCommandHandler(CommandHandler[UUID]):
    """Async command handler for meme refinement."""
    
    def __init__(
        self,
        text_generator_ref: ActorRef,
        image_generator_ref: ActorRef,
        quality_scorer_ref: ActorRef
    ):
        self.text_generator_ref = text_generator_ref
        self.image_generator_ref = image_generator_ref
        self.quality_scorer_ref = quality_scorer_ref
    
    def can_handle(self, command: Command) -> bool:
        return isinstance(command, RefineMemeCommand)
    
    async def handle(self, command: RefineMemeCommand) -> CommandResult[UUID]:
        """Handle meme refinement command."""
        try:
            event_store = await get_event_store()
            
            # Load original meme aggregate
            events = await event_store.get_events(command.meme_id)
            if not events:
                return CommandResult.failure("Original meme not found")
            
            original_aggregate = MemeAggregate()
            for event in events:
                original_aggregate.apply(event)
            
            # Create new meme ID for refined version
            refined_meme_id = uuid4()
            
            # Emit refinement requested event
            refinement_requested = RefinementRequested(
                original_meme_id=command.meme_id,
                refinement_reason=command.refinement_reason,
                current_score=original_aggregate.state.get('score', 0.0),
                target_score=0.8,  # Target improvement
                refinement_suggestions=["improve_humor", "enhance_relevance"],
                refinement_type="both",
                iteration_count=original_aggregate.state.get('refinement_count', 0) + 1
            )
            refinement_requested.metadata.aggregate_id = refined_meme_id
            refinement_requested.metadata.aggregate_type = "Meme"
            refinement_requested.metadata.correlation_id = command.metadata.correlation_id
            
            await event_store.append(refinement_requested)
            
            # Execute refinement pipeline (similar to generation but with refinement context)
            # This would involve regenerating text/image with feedback from the refinement reason
            
            return CommandResult.success(
                value=refined_meme_id,
                events=[refinement_requested]
            )
            
        except Exception as e:
            logger.error(f"Refinement command failed: {e}", exc_info=True)
            return CommandResult.failure(f"Refinement failed: {str(e)}")


# Command handler registry
command_handlers: Dict[str, CommandHandler] = {}


def register_command_handler(command_type: str, handler: CommandHandler):
    """Register a command handler."""
    command_handlers[command_type] = handler


async def initialize_command_handlers(actor_system: ActorSystem) -> None:
    """Initialize all command handlers with actor system."""
    # Get actor references
    text_generator_ref = actor_system.get_actor("text_generator")
    image_generator_ref = actor_system.get_actor("image_generator") 
    quality_scorer_ref = actor_system.get_actor("quality_scorer")
    verification_ref = actor_system.get_actor("verifier")
    
    if not all([text_generator_ref, image_generator_ref, quality_scorer_ref]):
        raise RuntimeError("Required actors not found in actor system")
    
    # Register handlers
    register_command_handler(
        "GenerateMeme",
        AsyncMemeGenerationCommandHandler(
            actor_system, text_generator_ref, image_generator_ref, quality_scorer_ref
        )
    )
    
    if verification_ref:
        register_command_handler(
            "VerifyMeme", 
            AsyncMemeVerificationCommandHandler(verification_ref)
        )
    
    register_command_handler(
        "RefineMeme",
        AsyncMemeRefinementCommandHandler(
            text_generator_ref, image_generator_ref, quality_scorer_ref
        )
    )
    
    logger.info("Command handlers initialized with actor system")


async def handle_command(command: Command) -> CommandResult:
    """Route command to appropriate handler."""
    handler = command_handlers.get(command.metadata.command_type)
    if not handler:
        return CommandResult.failure(f"No handler for command type: {command.metadata.command_type}")
    
    return await handler.handle(command)