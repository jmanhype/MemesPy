"""Image generation module for memes."""

import os
import random
import logging
import base64
import time
from pathlib import Path
import tempfile
import uuid  # Import uuid for generating unique filenames
from typing import Optional, Dict, Any, Literal, Union, List, Tuple

import requests
from pydantic import BaseModel

from dspy_meme_gen.config.config import settings
from ..services.metadata_collector import MetadataCollector

# Configure logger
logger = logging.getLogger(__name__)

# Define the persistent storage path
STATIC_MEME_PATH = Path("static/images/memes")
# Ensure the directory exists
STATIC_MEME_PATH.mkdir(parents=True, exist_ok=True)


# Define a local upload_to_s3 function in case the module is not available
def upload_to_s3(file_path: str, is_local_file: bool = False) -> Optional[str]:
    """
    Stub for uploading a file to S3.

    Args:
        file_path: The path to the file to upload
        is_local_file: Whether the file is a local file or a URL

    Returns:
        The URL of the uploaded file, or None if upload failed
    """
    logger.warning("S3 upload functionality not available. Using local file path.")
    return file_path


class ImageGenerator(BaseModel):
    """Generate images for memes."""

    provider: str
    sample_urls: List[str] = [
        "https://picsum.photos/600/400",
        "https://picsum.photos/600/401",
        "https://picsum.photos/600/402",
        "https://picsum.photos/600/403",
        "https://picsum.photos/600/404",
        "https://picsum.photos/600/405",
        "https://picsum.photos/600/406",
        "https://picsum.photos/600/407",
    ]
    metadata_collector: Optional[MetadataCollector] = None

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True

    def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        style: Optional[str] = None,
        meme_text: Optional[str] = None,
    ) -> str:
        """
        Generate an image based on the prompt.

        Args:
            prompt: The image description
            size: Image size (width x height)
            style: Optional style to apply
            meme_text: Optional text to overlay on the image

        Returns:
            URL or relative path to the generated image
        """
        # Get provider from instance or settings
        provider = self.provider or settings.image_provider

        # Handle different providers
        if provider == "placeholder":
            return self._generate_placeholder()
        elif provider in ["dall-e", "dalle"]:
            return self._generate_dalle(prompt, size, style, meme_text)
        elif provider == "gpt4o":
            return self._generate_gpt4o(prompt, size, style, meme_text)
        elif provider == "gpt-image-1":
            return self._generate_gpt_image_1(prompt, size, style, meme_text)
        else:
            logger.warning(f"Unknown provider: {provider}. Using placeholder.")
            return self._generate_placeholder()

    def _generate_placeholder(self) -> str:
        """Generate a placeholder image."""
        return random.choice(self.sample_urls)

    def _generate_dalle(
        self,
        prompt: str,
        size: str = "1024x1024",
        style: Optional[str] = None,
        meme_text: Optional[str] = None,
    ) -> str:
        """
        Generate an image using DALL-E.

        Args:
            prompt: The image description
            size: Image size (width x height)
            style: Optional style to apply
            meme_text: Optional text to overlay on the image

        Returns:
            URL to the generated image
        """
        start_time = time.time()

        try:
            # First check if OpenAI API key is set
            api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")

            if not api_key:
                logger.warning("OpenAI API key not found. Using placeholder image.")
                return self._generate_placeholder()

            # Enhance prompt with style and text
            enhanced_prompt = prompt

            if style:
                enhanced_prompt += f", {style}"

            if meme_text:
                # Include the text instruction in the prompt for DALL-E
                enhanced_prompt += f'. Include the text "{meme_text}" formatted as a meme caption.'

            # Import OpenAI here to avoid dependency for those not using it
            try:
                from openai import OpenAI
            except ImportError:
                logger.error("OpenAI package not installed. Install with 'pip install openai'")
                return self._generate_placeholder()

            client = OpenAI(api_key=api_key)

            # Parse size
            try:
                width, height = map(int, size.split("x"))
                size_str = f"{width}x{height}"
            except (ValueError, AttributeError):
                logger.warning(f"Invalid size format: {size}, defaulting to 1024x1024")
                size_str = "1024x1024"
                width, height = 1024, 1024

            # DALL-E 3 model
            logger.info(f"Generating DALL-E image with prompt: {enhanced_prompt}")

            generation_start = time.time()
            response = client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                size=size_str,
                quality="standard",
                n=1,
                response_format="url",
            )
            generation_time = (time.time() - generation_start) * 1000

            image_url = response.data[0].url
            logger.debug(f"Generated DALL-E image URL: {image_url}")

            # Track metadata if collector is available
            if self.metadata_collector:
                self.metadata_collector.track_image_generation(
                    provider="openai",
                    prompt=enhanced_prompt,
                    model="dall-e-3",
                    size=size_str,
                    duration_ms=generation_time,
                    success=True,
                    image_url=image_url,
                )

                # Add cost tracking (DALL-E 3 pricing)
                # Standard quality: $0.040 per image for 1024x1024
                cost = 0.040 if size_str == "1024x1024" else 0.080
                self.metadata_collector.add_cost_tracking(image_cost=cost)

                # Note: We can't analyze remote images or embed EXIF without downloading
                # But we track the URL and generation parameters
                self.metadata_collector.metadata["image_metadata"] = {
                    "provider": "dall-e-3",
                    "remote_url": image_url,
                    "width": width,
                    "height": height,
                    "format": "png",  # DALL-E returns PNG
                    "quality": "standard",
                    "storage_location": "openai_cdn",
                }

            # Upload to S3 if configured
            if (
                hasattr(settings, "use_s3_for_images")
                and settings.use_s3_for_images
                and hasattr(settings, "s3_bucket")
                and settings.s3_bucket
            ):
                try:
                    s3_url = upload_to_s3(image_url)
                    if s3_url:
                        logger.info(f"Uploaded DALL-E image to S3: {s3_url}")
                        if self.metadata_collector:
                            self.metadata_collector.metadata["image_metadata"]["cdn_url"] = s3_url
                            self.metadata_collector.metadata["image_metadata"][
                                "storage_location"
                            ] = "s3"
                        return s3_url
                except Exception as e:
                    logger.error(f"Failed to upload DALL-E image to S3: {str(e)}")

            return image_url

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            logger.error(f"Error generating DALL-E image: {str(e)}")

            # Track error
            if self.metadata_collector:
                self.metadata_collector.track_image_generation(
                    provider="openai",
                    prompt=prompt,
                    model="dall-e-3",
                    size=size,
                    duration_ms=total_time,
                    success=False,
                    error=str(e),
                )

            return self._generate_placeholder()

    def _generate_gpt_image_1(
        self,
        prompt: str,
        size: str = "1024x1024",
        style: Optional[str] = None,
        meme_text: Optional[str] = None,
    ) -> str:
        """
        Generate an image using OpenAI's gpt-image-1 model.

        Args:
            prompt: The image description
            size: Image size (width x height)
            style: Optional style to apply
            meme_text: Optional text to overlay on the image

        Returns:
            Relative web path to the generated image or fallback URL
        """
        start_time = time.time()

        try:
            # First check if OpenAI API key is set
            api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")

            if not api_key:
                logger.warning("OpenAI API key not found. Using placeholder image.")
                return self._generate_placeholder()

            # Enhance prompt with style and text
            enhanced_prompt = prompt

            if style:
                enhanced_prompt += f", {style} style"

            if meme_text:
                # Include the text instruction in the prompt for gpt-image-1
                enhanced_prompt += f'. Include the text "{meme_text}" formatted as a meme caption.'

            # Import OpenAI here to avoid dependency for those not using it
            try:
                from openai import OpenAI
            except ImportError:
                logger.error("OpenAI package not installed. Install with 'pip install openai'")
                return self._generate_placeholder()

            client = OpenAI(api_key=api_key)

            # Parse size
            try:
                width, height = map(int, size.split("x"))
                size_str = f"{width}x{height}"
            except (ValueError, AttributeError):
                logger.warning(f"Invalid size format: {size}, defaulting to 1024x1024")
                size_str = "1024x1024"
                width, height = 1024, 1024

            # gpt-image-1 model - it returns base64 encoded image by default
            logger.info(f"Generating gpt-image-1 image with prompt: {enhanced_prompt}")

            generation_start = time.time()
            try:
                response = client.images.generate(
                    model="gpt-image-1", prompt=enhanced_prompt, size=size_str, n=1
                )

                generation_time = (time.time() - generation_start) * 1000

                # Extract base64 image data
                b64_json = response.data[0].b64_json

                # Generate a unique filename
                unique_filename = f"{uuid.uuid4()}.png"
                save_path = STATIC_MEME_PATH / unique_filename

                # Decode and save to the persistent directory
                image_data = base64.b64decode(b64_json)
                with open(save_path, "wb") as f:
                    f.write(image_data)

                logger.info(f"Saved gpt-image-1 image to persistent storage: {save_path}")

                # Construct the relative web path
                web_path = f"/static/images/memes/{unique_filename}"

                # Track metadata if collector is available
                if self.metadata_collector:
                    self.metadata_collector.track_image_generation(
                        provider="openai",
                        prompt=enhanced_prompt,
                        model="gpt-image-1",
                        size=size_str,
                        duration_ms=generation_time,
                        success=True,
                        image_url=web_path,
                    )

                    # Analyze the generated image
                    image_metadata = self.metadata_collector.analyze_generated_image(str(save_path))
                    if image_metadata:
                        self.metadata_collector.metadata["image_metadata"] = image_metadata

                    # Embed metadata in EXIF
                    self.metadata_collector.embed_in_image_exif(str(save_path))

                    # Add cost tracking (estimated)
                    # gpt-image-1 pricing (example: $0.016 per image for 1024x1024)
                    cost = 0.016 if size_str == "1024x1024" else 0.020
                    self.metadata_collector.add_cost_tracking(image_cost=cost)

                return web_path

            except Exception as e:
                generation_time = (time.time() - generation_start) * 1000

                if (
                    "organization must be verified" in str(e).lower()
                    or "403" in str(e)
                    or "moderation_blocked" in str(e).lower()
                ):
                    log_message = "GPT-Image-1 not available (verification/moderation). Falling back to DALL-E."
                    fallback_reason = "organization_verification"
                    if "moderation_blocked" in str(e).lower():
                        log_message = (
                            "GPT-Image-1 prompt blocked by moderation. Falling back to DALL-E."
                        )
                        fallback_reason = "moderation_blocked"
                    logger.warning(log_message)

                    # Track failed attempt
                    if self.metadata_collector:
                        self.metadata_collector.track_image_generation(
                            provider="openai",
                            prompt=enhanced_prompt,
                            model="gpt-image-1",
                            size=size_str,
                            duration_ms=generation_time,
                            success=False,
                            error=str(e),
                            fallback_info={"reason": fallback_reason, "to_model": "dall-e-3"},
                        )

                    return self._generate_dalle(prompt, size, style, meme_text)
                else:
                    logger.error(f"Unexpected error generating gpt-image-1 image: {str(e)}")
                    raise  # Re-raise unexpected errors

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            logger.error(f"Error in gpt-image-1 generation process: {str(e)}")

            # Track error
            if self.metadata_collector:
                self.metadata_collector.track_image_generation(
                    provider="openai",
                    prompt=prompt,
                    model="gpt-image-1",
                    size=size,
                    duration_ms=total_time,
                    success=False,
                    error=str(e),
                )

            return self._generate_dalle(prompt, size, style, meme_text)

    def _generate_gpt4o(
        self,
        prompt: str,
        size: str = "1024x1024",
        style: Optional[str] = None,
        meme_text: Optional[str] = None,
    ) -> str:
        """
        Generate an image using GPT-4o.
        This is a future implementation and currently falls back to DALL-E.

        Args:
            prompt: The image description
            size: Image size (width x height)
            style: Optional style to apply
            meme_text: Optional text to overlay on the image

        Returns:
            URL to the generated image
        """
        # Check if GPT-4o image generation is available
        if not self._check_if_gpt4o_image_available():
            logger.info("GPT-4o image generation not available. Using DALL-E instead.")
            return self._generate_dalle(prompt, size, style, meme_text)

        # This is a placeholder for future implementation
        # For now, we'll just fall back to DALL-E
        logger.info("GPT-4o image generation implementation pending. Using DALL-E.")
        return self._generate_dalle(prompt, size, style, meme_text)

    def _check_if_gpt4o_image_available(self) -> bool:
        """
        Check if GPT-4o image generation is available.

        Returns:
            True if GPT-4o image generation is available, False otherwise
        """
        # This is a placeholder for future implementation
        # For now, we'll just return False
        return False
