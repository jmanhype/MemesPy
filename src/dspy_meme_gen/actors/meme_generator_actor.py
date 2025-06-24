"""Meme Generator Actor - Main orchestrator for meme generation."""

import asyncio
import logging
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time

from .core import Actor, ActorRef, Message, Request, Response, Event
from .circuit_breaker import CircuitBreaker
from ..agents.meme_generator import MemeGenerator
from ..config.config import settings


class MemeGenerationState(Enum):
    """States for meme generation process."""

    IDLE = "idle"
    GENERATING_TEXT = "generating_text"
    SCORING = "scoring"
    REFINING = "refining"
    GENERATING_IMAGE = "generating_image"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerateMemeRequest(Request):
    """Request to generate a meme."""

    topic: str = ""
    format: str = "standard"
    style: Optional[str] = None
    max_refinements: int = 3
    target_score: float = 0.7


@dataclass
class MemeGeneratedEvent(Event):
    """Event emitted when a meme is generated."""

    meme_id: str = ""
    topic: str = ""
    text: str = ""
    image_url: str = ""
    score: float = 0.0
    refinement_count: int = 0


@dataclass
class GenerateTextRequest(Request):
    """Request to generate meme text."""

    topic: str = ""
    format: str = ""
    style: Optional[str] = None


@dataclass
class GenerateImageRequest(Request):
    """Request to generate meme image."""

    text: str = ""
    format: str = ""
    style: Optional[str] = None


@dataclass
class ScoreMemeRequest(Request):
    """Request to score a meme."""

    text: str = ""
    topic: str = ""
    format: str = ""


@dataclass
class RefineMemeRequest(Request):
    """Request to refine a meme."""

    text: str = ""
    topic: str = ""
    format: str = ""
    score: float = 0.0
    feedback: Optional[str] = None


class MemeGeneratorActor(Actor):
    """
    Main orchestrator actor for meme generation.

    Coordinates the entire meme generation pipeline:
    1. Text generation via DSPy
    2. Quality scoring
    3. Refinement if needed
    4. Image generation
    """

    def __init__(
        self,
        text_generator_ref: Optional[ActorRef] = None,
        image_generator_ref: Optional[ActorRef] = None,
        name: str = "meme_generator",
    ):
        super().__init__(name)
        self.text_generator_ref = text_generator_ref
        self.image_generator_ref = image_generator_ref

        # Initialize state
        self.state = MemeGenerationState.IDLE
        self.current_request: Optional[GenerateMemeRequest] = None
        self.current_meme_data: Dict[str, Any] = {}
        self.refinement_count = 0

        # Circuit breakers for external services
        from .circuit_breaker import CircuitBreakerConfig

        self.text_gen_breaker = CircuitBreaker(
            "text_generation", CircuitBreakerConfig(failure_threshold=3, timeout=30)
        )
        self.image_gen_breaker = CircuitBreaker(
            "image_generation", CircuitBreakerConfig(failure_threshold=2, timeout=60)
        )

        # Initialize DSPy agents
        self.meme_generator = MemeGenerator()

        # Simple scoring function for compatibility
        class SimpleScoringAgent:
            def forward(self, **kwargs):
                # Simple scoring logic - would be replaced with actual agent
                class Result:
                    def __init__(self):
                        self.score = 0.8  # Default score
                        self.feedback = "Generated successfully"

                return Result()

        self.scorer = SimpleScoringAgent()

        # Simple refinement function for compatibility
        class SimpleRefinementAgent:
            def forward(self, **kwargs):
                # Simple refinement logic - would be replaced with actual agent
                class Result:
                    def __init__(self, text):
                        self.refined_text = text + " (refined)"

                return Result(kwargs.get("original_text", "default text"))

        self.refiner = SimpleRefinementAgent()

        # Metrics
        self.metrics = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "average_score": 0.0,
            "average_refinements": 0.0,
            "total_generation_time": 0.0,
        }

    async def on_start(self) -> None:
        """Initialize the actor."""
        self.logger.info(f"MemeGeneratorActor {self.name} starting")

    async def on_stop(self) -> None:
        """Cleanup when stopping."""
        self.logger.info(f"MemeGeneratorActor {self.name} stopping")
        self.logger.info(f"Final metrics: {self.metrics}")

    async def on_error(self, error: Exception) -> None:
        """Handle errors."""
        self.logger.error(f"Error in MemeGeneratorActor: {error}")
        self.state = MemeGenerationState.FAILED

    async def handle_generatememerequest(self, message: GenerateMemeRequest) -> Dict[str, Any]:
        """Handle meme generation request."""
        start_time = time.time()
        self.metrics["total_requests"] += 1

        try:
            self.logger.info(f"Starting meme generation for topic: {message.topic}")
            self.current_request = message
            self.current_meme_data = {
                "topic": message.topic,
                "format": message.format,
                "style": message.style,
                "request_id": message.id,
            }
            self.refinement_count = 0

            # Generate text
            self.state = MemeGenerationState.GENERATING_TEXT
            text_result = await self._generate_text()
            self.current_meme_data["text"] = text_result["text"]

            # Score the meme
            self.state = MemeGenerationState.SCORING
            score_result = await self._score_meme()
            self.current_meme_data["score"] = score_result["score"]

            # Refine if needed
            while (
                self.current_meme_data["score"] < message.target_score
                and self.refinement_count < message.max_refinements
            ):
                self.state = MemeGenerationState.REFINING
                self.refinement_count += 1

                refine_result = await self._refine_meme(score_result.get("feedback"))
                self.current_meme_data["text"] = refine_result["text"]

                # Re-score
                self.state = MemeGenerationState.SCORING
                score_result = await self._score_meme()
                self.current_meme_data["score"] = score_result["score"]

            # Generate image
            self.state = MemeGenerationState.GENERATING_IMAGE
            image_result = await self._generate_image()
            self.current_meme_data["image_url"] = image_result["image_url"]

            # Complete
            self.state = MemeGenerationState.COMPLETED
            generation_time = time.time() - start_time

            # Update metrics
            self.metrics["successful_generations"] += 1
            self.metrics["total_generation_time"] += generation_time
            self._update_averages()

            # Emit event
            event = MemeGeneratedEvent(
                event_type="meme_generated",
                meme_id=self.current_meme_data.get("id", message.id),
                topic=message.topic,
                text=self.current_meme_data["text"],
                image_url=self.current_meme_data["image_url"],
                score=self.current_meme_data["score"],
                refinement_count=self.refinement_count,
            )
            await self._emit_event(event)

            self.logger.info(
                f"Meme generation completed in {generation_time:.2f}s "
                f"with score {self.current_meme_data['score']:.2f} "
                f"after {self.refinement_count} refinements"
            )

            return {
                "id": self.current_meme_data.get("id", message.id),
                "text": self.current_meme_data["text"],
                "image_url": self.current_meme_data["image_url"],
                "score": self.current_meme_data["score"],
                "refinement_count": self.refinement_count,
                "generation_time": generation_time,
            }

        except Exception as e:
            self.logger.error(f"Failed to generate meme: {e}")
            self.metrics["failed_generations"] += 1
            self.state = MemeGenerationState.FAILED
            raise

    async def _generate_text(self) -> Dict[str, Any]:
        """Generate meme text using text generator actor or fallback."""
        try:
            if self.text_generator_ref:
                # Use actor if available
                request = GenerateTextRequest(
                    topic=self.current_request.topic,
                    format=self.current_request.format,
                    style=self.current_request.style,
                )

                response = await self.text_gen_breaker.call(
                    self.text_generator_ref.ask, request, timeout=10000
                )
                return response
            else:
                # Fallback to direct DSPy call
                result = await self.text_gen_breaker.call(
                    self.meme_generator.generate_meme,
                    self.current_request.topic,
                    self.current_request.format,
                )
                return {"text": result["text"]}

        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            raise

    async def _score_meme(self) -> Dict[str, Any]:
        """Score the generated meme."""
        try:
            score_result = self.scorer.forward(
                text=self.current_meme_data["text"],
                topic=self.current_request.topic,
                format=self.current_request.format,
            )

            return {
                "score": score_result.score,
                "feedback": getattr(score_result, "feedback", None),
            }

        except Exception as e:
            self.logger.error(f"Scoring failed: {e}")
            # Default score on failure
            return {"score": 0.5, "feedback": "Scoring failed"}

    async def _refine_meme(self, feedback: Optional[str] = None) -> Dict[str, Any]:
        """Refine the meme text."""
        try:
            refine_result = self.refiner.forward(
                original_text=self.current_meme_data["text"],
                topic=self.current_request.topic,
                format=self.current_request.format,
                score=self.current_meme_data["score"],
                feedback=feedback,
            )

            return {"text": refine_result.refined_text}

        except Exception as e:
            self.logger.error(f"Refinement failed: {e}")
            # Return original text on failure
            return {"text": self.current_meme_data["text"]}

    async def _generate_image(self) -> Dict[str, Any]:
        """Generate meme image using image generator actor or fallback."""
        try:
            if self.image_generator_ref:
                # Use actor if available
                request = GenerateImageRequest(
                    text=self.current_meme_data["text"],
                    format=self.current_request.format,
                    style=self.current_request.style,
                )

                response = await self.image_gen_breaker.call(
                    self.image_generator_ref.ask, request, timeout=30000
                )
                return response
            else:
                # Fallback to placeholder
                return {
                    "image_url": f"https://placeholder.pics/svg/600x600/DEDEDE/555555/{self.current_request.topic}"
                }

        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            # Return placeholder on failure
            return {"image_url": f"https://placeholder.pics/svg/600x600/FF0000/FFFFFF/Error"}

    async def _emit_event(self, event: Event) -> None:
        """Emit an event to interested actors."""
        # In a real system, this would publish to an event bus
        self.logger.info(f"Event emitted: {event.event_type}")

    def _update_averages(self) -> None:
        """Update running averages for metrics."""
        total = self.metrics["successful_generations"]
        if total > 0:
            # Update average score
            current_avg_score = self.metrics["average_score"]
            new_score = self.current_meme_data["score"]
            self.metrics["average_score"] = (current_avg_score * (total - 1) + new_score) / total

            # Update average refinements
            current_avg_ref = self.metrics["average_refinements"]
            self.metrics["average_refinements"] = (
                current_avg_ref * (total - 1) + self.refinement_count
            ) / total

    async def handle_getmetricsrequest(self, message: Request) -> Dict[str, Any]:
        """Return current metrics."""
        avg_time = 0.0
        if self.metrics["successful_generations"] > 0:
            avg_time = (
                self.metrics["total_generation_time"] / self.metrics["successful_generations"]
            )

        return {
            **self.metrics,
            "average_generation_time": avg_time,
            "current_state": self.state.value,
            "circuit_breakers": {
                "text_gen": self.text_gen_breaker.get_state().value,
                "image_gen": self.image_gen_breaker.get_state().value,
            },
        }


@dataclass
class GetMetricsRequest(Request):
    """Request to get actor metrics."""

    pass
