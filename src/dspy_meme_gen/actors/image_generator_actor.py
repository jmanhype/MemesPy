"""Image Generator Actor - Handles image generation for memes using DALL-E/gpt-image-1."""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import uuid
from io import BytesIO
import base64

from .core import Actor, ActorRef, Message, Request, Response, Event
from .circuit_breaker import CircuitBreaker
from ..agents.image_renderer import ImageRenderingAgent
from ..config.config import settings

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

try:
    import cloudinary
    import cloudinary.uploader
    CLOUDINARY_AVAILABLE = True
except ImportError:
    CLOUDINARY_AVAILABLE = False


class ImageGenerationState(Enum):
    """States for image generation process."""
    IDLE = "idle"
    GENERATING_PROMPT = "generating_prompt"
    GENERATING_IMAGE = "generating_image"
    POST_PROCESSING = "post_processing"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerateImageRequest(Request):
    """Request to generate meme image."""
    text: str = ""
    format: str = ""
    style: Optional[str] = None
    size: str = "1024x1024"
    quality: str = "standard"  # standard or hd
    add_text_overlay: bool = True
    upload_to_cdn: bool = True


@dataclass
class ProcessImageRequest(Request):
    """Request to process an existing image."""
    image_data: Union[str, bytes] = ""  # base64 string or bytes
    text: str = ""
    format: str = ""
    operations: List[str] = field(default_factory=list)  # ["resize", "overlay", "filter"]


@dataclass
class ImageGeneratedEvent(Event):
    """Event emitted when image is generated."""
    image_url: str = ""
    text: str = ""
    format: str = ""
    generation_time: float = 0.0
    size: str = ""
    cdn_uploaded: bool = False


class ImageGeneratorActor(Actor):
    """
    Actor responsible for generating meme images.
    
    Handles:
    - Image generation via OpenAI DALL-E/gpt-image-1
    - Post-processing (text overlay, resizing, etc.)
    - CDN upload (Cloudinary)
    - Caching of generated images
    - Retry logic with circuit breakers
    """
    
    def __init__(
        self,
        name: str = "image_generator",
        provider: str = "openai",
        enable_cdn_upload: bool = True
    ):
        super().__init__(name)
        
        # Configuration
        self.provider = provider
        self.enable_cdn_upload = enable_cdn_upload and CLOUDINARY_AVAILABLE
        
        # State
        self.state = ImageGenerationState.IDLE
        self.image_cache: Dict[str, Dict[str, Any]] = {}
        
        # Circuit breakers
        from .circuit_breaker import CircuitBreakerConfig
        self.openai_breaker = CircuitBreaker(
            "openai_images",
            CircuitBreakerConfig(failure_threshold=3, timeout=60)
        )
        self.cloudinary_breaker = CircuitBreaker(
            "cloudinary_upload",
            CircuitBreakerConfig(failure_threshold=2, timeout=30)
        )
        
        # Initialize clients
        self.openai_client = None
        if OPENAI_AVAILABLE and settings.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
            
        # Initialize image rendering agent
        self.image_renderer = ImageRenderingAgent(
            api_key=settings.openai_api_key,
            image_service="openai"
        )
        
        # Metrics
        self.metrics = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "cache_hits": 0,
            "cdn_uploads": 0,
            "failed_uploads": 0,
            "average_generation_time": 0.0,
            "total_generation_time": 0.0
        }
        
    async def on_start(self) -> None:
        """Initialize the actor."""
        self.logger.info(f"ImageGeneratorActor {self.name} starting")
        self.logger.info(f"Provider: {self.provider}")
        self.logger.info(f"OpenAI available: {OPENAI_AVAILABLE}")
        self.logger.info(f"CDN upload: {'enabled' if self.enable_cdn_upload else 'disabled'}")
        
        if not OPENAI_AVAILABLE:
            self.logger.warning("OpenAI not available - image generation will be limited")
            
        if not CLOUDINARY_AVAILABLE and self.enable_cdn_upload:
            self.logger.warning("Cloudinary not available - CDN upload disabled")
            self.enable_cdn_upload = False
            
    async def on_stop(self) -> None:
        """Cleanup when stopping."""
        self.logger.info(f"ImageGeneratorActor {self.name} stopping")
        self.logger.info(f"Final metrics: {self.metrics}")
        
        # Clear cache
        self.image_cache.clear()
        
        # Close OpenAI client
        if self.openai_client:
            await self.openai_client.close()
            
    async def on_error(self, error: Exception) -> None:
        """Handle errors."""
        self.logger.error(f"Error in ImageGeneratorActor: {error}")
        self.state = ImageGenerationState.FAILED
        
    async def handle_generateimagerequest(self, message: GenerateImageRequest) -> Dict[str, Any]:
        """Handle image generation request."""
        start_time = time.time()
        self.metrics["total_generations"] += 1
        
        try:
            self.logger.info(
                f"Generating image for text: '{message.text[:50]}...', "
                f"format: '{message.format}', size: '{message.size}'"
            )
            
            # Check cache first
            cache_key = self._get_cache_key(message.text, message.format, message.size)
            if cache_key in self.image_cache:
                self.metrics["cache_hits"] += 1
                self.logger.info(f"Cache hit for key: {cache_key}")
                return self.image_cache[cache_key]
            
            # Generate image prompt
            self.state = ImageGenerationState.GENERATING_PROMPT
            image_prompt = await self._generate_image_prompt(
                message.text,
                message.format,
                message.style
            )
            
            # Generate image
            self.state = ImageGenerationState.GENERATING_IMAGE
            image_data = await self._generate_image(
                image_prompt,
                message.size,
                message.quality
            )
            
            # Post-process if needed
            if message.add_text_overlay:
                self.state = ImageGenerationState.POST_PROCESSING
                image_data = await self._add_text_overlay(
                    image_data,
                    message.text,
                    message.format
                )
            
            # Upload to CDN
            image_url = None
            cdn_uploaded = False
            if message.upload_to_cdn and self.enable_cdn_upload:
                self.state = ImageGenerationState.UPLOADING
                upload_result = await self._upload_to_cdn(image_data)
                image_url = upload_result["url"]
                cdn_uploaded = True
                self.metrics["cdn_uploads"] += 1
            else:
                # Create a data URL for local use
                image_b64 = base64.b64encode(image_data).decode()
                image_url = f"data:image/png;base64,{image_b64}"
            
            # Complete
            self.state = ImageGenerationState.COMPLETED
            generation_time = time.time() - start_time
            
            # Update metrics
            self.metrics["successful_generations"] += 1
            self.metrics["total_generation_time"] += generation_time
            self._update_average_time()
            
            # Prepare result
            result = {
                "image_url": image_url,
                "size": message.size,
                "format": message.format,
                "generation_time": generation_time,
                "cdn_uploaded": cdn_uploaded,
                "prompt_used": image_prompt
            }
            
            # Cache result
            self.image_cache[cache_key] = result
            
            # Emit event
            event = ImageGeneratedEvent(
                event_type="image_generated",
                image_url=image_url,
                text=message.text,
                format=message.format,
                generation_time=generation_time,
                size=message.size,
                cdn_uploaded=cdn_uploaded
            )
            await self._emit_event(event)
            
            self.logger.info(
                f"Image generation completed in {generation_time:.2f}s "
                f"(CDN: {cdn_uploaded})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate image: {e}")
            self.metrics["failed_generations"] += 1
            self.state = ImageGenerationState.FAILED
            raise
            
    async def handle_processimagerequest(self, message: ProcessImageRequest) -> Dict[str, Any]:
        """Handle image processing request."""
        try:
            self.logger.info(f"Processing image with operations: {message.operations}")
            
            # Convert input to bytes if needed
            if isinstance(message.image_data, str):
                # Assume base64
                image_data = base64.b64decode(message.image_data)
            else:
                image_data = message.image_data
            
            # Apply operations
            for operation in message.operations:
                if operation == "overlay":
                    image_data = await self._add_text_overlay(
                        image_data,
                        message.text,
                        message.format
                    )
                elif operation == "resize":
                    image_data = await self._resize_image(image_data, "512x512")
                elif operation == "filter":
                    image_data = await self._apply_filter(image_data, "enhance")
            
            # Return processed image as base64
            image_b64 = base64.b64encode(image_data).decode()
            
            return {
                "image_data": image_b64,
                "operations_applied": message.operations
            }
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            raise
            
    async def _generate_image_prompt(
        self,
        meme_text: str,
        format_name: str,
        style: Optional[str]
    ) -> str:
        """Generate optimized prompt for image generation."""
        try:
            # Format-specific prompt templates
            format_prompts = {
                "drake": "Drake pointing meme format, split screen with Drake rejecting top and approving bottom",
                "distracted_boyfriend": "Distracted boyfriend meme format with three people",
                "woman_yelling_at_cat": "Woman yelling at cat meme format",
                "expanding_brain": "Expanding brain meme format showing progression",
                "change_my_mind": "Change my mind meme format with person at table",
                "two_buttons": "Two buttons meme format showing difficult choice",
                "standard": "Clean, simple meme background suitable for text overlay"
            }
            
            base_prompt = format_prompts.get(format_name, format_prompts["standard"])
            
            # Add style modifiers
            style_modifiers = {
                "retro": "in retro style with vintage colors",
                "modern": "in modern flat design style",
                "cartoon": "in cartoon/animated style",
                "realistic": "in realistic photographic style",
                "minimalist": "in clean minimalist style"
            }
            
            if style and style in style_modifiers:
                base_prompt += f", {style_modifiers[style]}"
            
            # Add general meme requirements
            base_prompt += ", high quality, clear, suitable for meme text overlay, 1024x1024 resolution"
            
            return base_prompt
            
        except Exception as e:
            self.logger.error(f"Prompt generation failed: {e}")
            return f"Meme image for: {meme_text}"
            
    async def _generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard"
    ) -> bytes:
        """Generate image using OpenAI DALL-E."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not available")
            
        try:
            # Use DALL-E 3 for better quality
            async def generate_image_call():
                response = await self.openai_client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=1,
                    response_format="b64_json"
                )
                
                # Get the base64 image data
                image_b64 = response.data[0].b64_json
                image_data = base64.b64decode(image_b64)
                
                self.logger.info(f"Generated image ({len(image_data)} bytes)")
                return image_data
                
            return await self.openai_breaker.call(generate_image_call)
                
        except Exception as e:
            self.logger.error(f"OpenAI image generation failed: {e}")
            # Try fallback with image rendering agent
            try:
                result = self.image_renderer.forward(
                    image_prompt=prompt
                )
                # This would typically return a URL, but we need bytes
                # For now, raise the original error
                raise e
            except:
                raise e
                
    async def _add_text_overlay(
        self,
        image_data: bytes,
        text: str,
        format_name: str
    ) -> bytes:
        """Add text overlay to image."""
        try:
            # Use the image rendering agent's text overlay function
            result = self.image_renderer.add_text_overlay(
                image_data=image_data,
                caption=text,
                position="bottom"  # Default position
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Text overlay failed: {e}")
            # Return original image on failure
            return image_data
            
    async def _resize_image(self, image_data: bytes, new_size: str) -> bytes:
        """Resize image to new dimensions."""
        try:
            from PIL import Image
            
            # Parse size
            width, height = map(int, new_size.split('x'))
            
            # Open and resize
            image = Image.open(BytesIO(image_data))
            resized = image.resize((width, height), Image.Resampling.LANCZOS)
            
            # Convert back to bytes
            output = BytesIO()
            resized.save(output, format="PNG")
            return output.getvalue()
            
        except Exception as e:
            self.logger.error(f"Image resize failed: {e}")
            return image_data
            
    async def _apply_filter(self, image_data: bytes, filter_type: str) -> bytes:
        """Apply filter to image."""
        try:
            from PIL import Image, ImageEnhance
            
            image = Image.open(BytesIO(image_data))
            
            if filter_type == "enhance":
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.1)
            
            # Convert back to bytes
            output = BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
            
        except Exception as e:
            self.logger.error(f"Filter application failed: {e}")
            return image_data
            
    async def _upload_to_cdn(self, image_data: bytes) -> Dict[str, Any]:
        """Upload image to Cloudinary CDN."""
        if not CLOUDINARY_AVAILABLE:
            raise RuntimeError("Cloudinary not available")
            
        try:
            # Upload to Cloudinary
            async def upload_call():
                upload_result = cloudinary.uploader.upload(
                    BytesIO(image_data),
                    folder="meme_generator",
                    resource_type="image",
                    format="png"
                )
                
                return {
                    "url": upload_result["secure_url"],
                    "public_id": upload_result["public_id"],
                    "width": upload_result["width"],
                    "height": upload_result["height"]
                }
                
            return await self.cloudinary_breaker.call(upload_call)
                
        except Exception as e:
            self.logger.error(f"CDN upload failed: {e}")
            self.metrics["failed_uploads"] += 1
            raise
            
    async def _emit_event(self, event: Event) -> None:
        """Emit an event to interested actors."""
        self.logger.debug(f"Event emitted: {event.event_type}")
        
    def _get_cache_key(self, text: str, format_name: str, size: str) -> str:
        """Generate cache key for an image generation request."""
        # Create a hash of the text to keep keys reasonable length
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{text_hash}:{format_name}:{size}"
        
    def _update_average_time(self) -> None:
        """Update average generation time."""
        total = self.metrics["successful_generations"]
        if total > 0:
            self.metrics["average_generation_time"] = (
                self.metrics["total_generation_time"] / total
            )
            
    async def handle_clearcacherequest(self, message: Request) -> Dict[str, Any]:
        """Clear image cache."""
        cache_size = len(self.image_cache)
        self.image_cache.clear()
        
        self.logger.info(f"Cleared {cache_size} image cache entries")
        
        return {
            "cache_cleared": cache_size
        }
        
    async def handle_getmetricsrequest(self, message: Request) -> Dict[str, Any]:
        """Return current metrics."""
        return {
            **self.metrics,
            "current_state": self.state.value,
            "cache_size": len(self.image_cache),
            "provider": self.provider,
            "cdn_enabled": self.enable_cdn_upload,
            "circuit_breakers": {
                "openai": self.openai_breaker.get_state().value,
                "cloudinary": self.cloudinary_breaker.get_state().value
            }
        }


@dataclass
class GetMetricsRequest(Request):
    """Request to get actor metrics."""
    pass


@dataclass
class ClearCacheRequest(Request):
    """Request to clear caches."""
    pass