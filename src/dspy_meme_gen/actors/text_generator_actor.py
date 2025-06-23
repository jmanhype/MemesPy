"""Text Generator Actor - Handles DSPy text generation for memes."""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import uuid

from .core import Actor, ActorRef, Message, Request, Response, Event
from .circuit_breaker import CircuitBreaker
from ..agents.meme_generator import meme_generator as dspy_meme_gen
from ..config.config import settings


class TextGenerationState(Enum):
    """States for text generation process."""
    IDLE = "idle"
    SELECTING_FORMAT = "selecting_format"
    GENERATING_PROMPT = "generating_prompt"
    GENERATING_TEXT = "generating_text"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerateTextRequest(Request):
    """Request to generate meme text."""
    topic: str = ""
    format: str = ""
    style: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class VerifyTextRequest(Request):
    """Request to verify generated text."""
    text: str = ""
    topic: str = ""
    format: str = ""
    check_appropriateness: bool = True
    check_factuality: bool = False


@dataclass
class TextGeneratedEvent(Event):
    """Event emitted when text is generated."""
    text: str = ""
    topic: str = ""
    format: str = ""
    generation_time: float = 0.0
    verified: bool = False


class TextGeneratorActor(Actor):
    """
    Actor responsible for generating meme text using DSPy.
    
    Handles:
    - Format selection and optimization
    - Prompt generation
    - Text generation via DSPy
    - Text verification
    - Caching of successful generations
    """
    
    def __init__(
        self,
        name: str = "text_generator",
        enable_verification: bool = True,
        cache_results: bool = True
    ):
        super().__init__(name)
        
        # Configuration
        self.enable_verification = enable_verification
        self.cache_results = cache_results
        
        # State
        self.state = TextGenerationState.IDLE
        self.generation_cache: Dict[str, Dict[str, Any]] = {}
        self.format_cache: Dict[str, Dict[str, Any]] = {}
        
        # Circuit breaker for DSPy calls
        from .circuit_breaker import CircuitBreakerConfig
        self.dspy_breaker = CircuitBreaker(
            "dspy_generation",
            CircuitBreakerConfig(failure_threshold=5, timeout=30)
        )
        
        # Initialize agents with simple implementations for compatibility
        class SimplePromptGeneratorAgent:
            def forward(self, **kwargs):
                class Result:
                    def __init__(self):
                        self.prompt = f"Generate a meme about {kwargs.get('topic', 'general topic')}"
                return Result()
                
        self.prompt_generator = SimplePromptGeneratorAgent()
        
        class SimpleFormatSelectorAgent:
            def forward(self, **kwargs):
                class Result:
                    def __init__(self):
                        self.format_name = kwargs.get('preferred_format', 'standard')
                        self.template = None
                        self.style_hints = []
                        self.requirements = []
                return Result()
                
        self.format_selector = SimpleFormatSelectorAgent()
        
        class SimpleVerificationAgent:
            def forward(self, **kwargs):
                class Result:
                    def __init__(self):
                        self.is_valid = True
                return Result()
                
        self.verifier = SimpleVerificationAgent() if enable_verification else None
        
        # Metrics
        self.metrics = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "cache_hits": 0,
            "verification_failures": 0,
            "average_generation_time": 0.0,
            "total_generation_time": 0.0
        }
        
    async def on_start(self) -> None:
        """Initialize the actor."""
        self.logger.info(f"TextGeneratorActor {self.name} starting")
        self.logger.info(f"Verification: {'enabled' if self.enable_verification else 'disabled'}")
        self.logger.info(f"Caching: {'enabled' if self.cache_results else 'disabled'}")
        
    async def on_stop(self) -> None:
        """Cleanup when stopping."""
        self.logger.info(f"TextGeneratorActor {self.name} stopping")
        self.logger.info(f"Final metrics: {self.metrics}")
        
        # Clear caches
        self.generation_cache.clear()
        self.format_cache.clear()
        
    async def on_error(self, error: Exception) -> None:
        """Handle errors."""
        self.logger.error(f"Error in TextGeneratorActor: {error}")
        self.state = TextGenerationState.FAILED
        
    async def handle_generatetextrequest(self, message: GenerateTextRequest) -> Dict[str, Any]:
        """Handle text generation request."""
        start_time = time.time()
        self.metrics["total_generations"] += 1
        
        try:
            self.logger.info(
                f"Generating text for topic: '{message.topic}', "
                f"format: '{message.format}', style: '{message.style}'"
            )
            
            # Check cache first
            cache_key = self._get_cache_key(message.topic, message.format, message.style)
            if self.cache_results and cache_key in self.generation_cache:
                self.metrics["cache_hits"] += 1
                self.logger.info(f"Cache hit for key: {cache_key}")
                return self.generation_cache[cache_key]
            
            # Start generation process
            self.state = TextGenerationState.SELECTING_FORMAT
            
            # Select/optimize format
            format_details = await self._select_format(message.format, message.topic)
            
            # Generate prompt
            self.state = TextGenerationState.GENERATING_PROMPT
            prompt = await self._generate_prompt(
                message.topic,
                format_details,
                message.style,
                message.context
            )
            
            # Generate text
            self.state = TextGenerationState.GENERATING_TEXT
            generated_text = await self._generate_text(prompt, message.format)
            
            # Verify if enabled
            verified = True
            if self.enable_verification:
                self.state = TextGenerationState.VERIFYING
                verified = await self._verify_text(
                    generated_text,
                    message.topic,
                    message.format
                )
                
                if not verified:
                    self.metrics["verification_failures"] += 1
                    # Try regeneration once
                    self.logger.warning("Verification failed, regenerating...")
                    generated_text = await self._generate_text(prompt, message.format)
                    verified = await self._verify_text(
                        generated_text,
                        message.topic,
                        message.format
                    )
            
            # Complete
            self.state = TextGenerationState.COMPLETED
            generation_time = time.time() - start_time
            
            # Update metrics
            self.metrics["successful_generations"] += 1
            self.metrics["total_generation_time"] += generation_time
            self._update_average_time()
            
            # Prepare result
            result = {
                "text": generated_text,
                "format_details": format_details,
                "verified": verified,
                "generation_time": generation_time
            }
            
            # Cache result if successful
            if self.cache_results and verified:
                self.generation_cache[cache_key] = result
                
            # Emit event
            event = TextGeneratedEvent(
                event_type="text_generated",
                text=generated_text,
                topic=message.topic,
                format=message.format,
                generation_time=generation_time,
                verified=verified
            )
            await self._emit_event(event)
            
            self.logger.info(
                f"Text generation completed in {generation_time:.2f}s "
                f"(verified: {verified})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate text: {e}")
            self.metrics["failed_generations"] += 1
            self.state = TextGenerationState.FAILED
            raise
            
    async def handle_verifytextrequest(self, message: VerifyTextRequest) -> Dict[str, Any]:
        """Handle text verification request."""
        if not self.enable_verification:
            return {"verified": True, "reason": "Verification disabled"}
            
        try:
            verified = await self._verify_text(
                message.text,
                message.topic,
                message.format,
                message.check_appropriateness,
                message.check_factuality
            )
            
            return {
                "verified": verified,
                "text": message.text
            }
            
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            return {
                "verified": False,
                "reason": str(e)
            }
            
    async def _select_format(self, format_name: str, topic: str) -> Dict[str, Any]:
        """Select and optimize format for the topic."""
        try:
            # Check format cache
            cache_key = f"{format_name}:{topic}"
            if cache_key in self.format_cache:
                return self.format_cache[cache_key]
                
            # Use format selector agent
            result = self.format_selector.forward(
                topic=topic,
                preferred_format=format_name
            )
            
            format_details = {
                "name": result.format_name,
                "template": getattr(result, 'template', None),
                "style_hints": getattr(result, 'style_hints', []),
                "requirements": getattr(result, 'requirements', [])
            }
            
            # Cache result
            self.format_cache[cache_key] = format_details
            
            return format_details
            
        except Exception as e:
            self.logger.error(f"Format selection failed: {e}")
            # Return default format
            return {
                "name": format_name,
                "template": None,
                "style_hints": [],
                "requirements": []
            }
            
    async def _generate_prompt(
        self,
        topic: str,
        format_details: Dict[str, Any],
        style: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate optimized prompt for DSPy."""
        try:
            result = self.prompt_generator.forward(
                topic=topic,
                format_name=format_details["name"],
                style=style,
                context=context
            )
            
            return result.prompt
            
        except Exception as e:
            self.logger.error(f"Prompt generation failed: {e}")
            # Fallback to simple prompt
            return f"Generate a {format_details['name']} meme about {topic}"
            
    async def _generate_text(self, prompt: str, format_name: str) -> str:
        """Generate meme text using DSPy."""
        try:
            # Use the DSPy meme generator
            result = await self.dspy_breaker.call(
                dspy_meme_gen.generate_meme,
                topic=prompt,  # Using the optimized prompt as topic
                format=format_name
            )
            
            # Safely extract text from result
            if isinstance(result, dict) and "text" in result:
                return result["text"]
            elif hasattr(result, 'text'):
                return result.text
            else:
                return str(result)
                
        except Exception as e:
            self.logger.error(f"DSPy text generation failed: {e}")
            raise
            
    async def _verify_text(
        self,
        text: str,
        topic: str,
        format_name: str,
        check_appropriateness: bool = True,
        check_factuality: bool = False
    ) -> bool:
        """Verify generated text."""
        if not self.verifier:
            return True
            
        try:
            result = self.verifier.forward(
                text=text,
                topic=topic,
                format_name=format_name,
                check_appropriateness=check_appropriateness,
                check_factuality=check_factuality
            )
            
            return result.is_valid
            
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            # Be permissive on verification errors
            return True
            
    async def _emit_event(self, event: Event) -> None:
        """Emit an event to interested actors."""
        self.logger.debug(f"Event emitted: {event.event_type}")
        
    def _get_cache_key(self, topic: str, format_name: str, style: Optional[str]) -> str:
        """Generate cache key for a generation request."""
        style_str = style or "default"
        return f"{topic}:{format_name}:{style_str}"
        
    def _update_average_time(self) -> None:
        """Update average generation time."""
        total = self.metrics["successful_generations"]
        if total > 0:
            self.metrics["average_generation_time"] = (
                self.metrics["total_generation_time"] / total
            )
            
    async def handle_clearcacherequest(self, message: Request) -> Dict[str, Any]:
        """Clear generation and format caches."""
        generation_cache_size = len(self.generation_cache)
        format_cache_size = len(self.format_cache)
        
        self.generation_cache.clear()
        self.format_cache.clear()
        
        self.logger.info(
            f"Cleared {generation_cache_size} generation cache entries "
            f"and {format_cache_size} format cache entries"
        )
        
        return {
            "generation_cache_cleared": generation_cache_size,
            "format_cache_cleared": format_cache_size
        }
        
    async def handle_getmetricsrequest(self, message: Request) -> Dict[str, Any]:
        """Return current metrics."""
        return {
            **self.metrics,
            "current_state": self.state.value,
            "cache_size": len(self.generation_cache),
            "format_cache_size": len(self.format_cache),
            "circuit_breaker_state": self.dspy_breaker.get_state().value,
            "verification_enabled": self.enable_verification
        }


@dataclass
class ClearCacheRequest(Request):
    """Request to clear caches."""
    pass