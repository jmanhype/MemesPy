"""Image Rendering Agent for generating and processing meme images."""

from typing import Dict, Any, TypedDict, Optional
import os
from io import BytesIO
import requests
from PIL import Image, ImageDraw, ImageFont
import cloudinary
import cloudinary.uploader
from openai import OpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import dspy
import openai


class ImageResult(TypedDict):
    """Type definition for rendered image results."""
    image_url: str
    cloudinary_id: str
    width: int
    height: int


class ImageRenderingAgent(dspy.Module):
    """
    Agent responsible for generating and processing meme images.
    
    This agent:
    - Interfaces with image generation APIs (OpenAI DALL-E)
    - Handles image post-processing (text overlay, formatting)
    - Manages image upload to Cloudinary
    - Provides CDN URLs for frontend display
    """

    def __init__(self, api_key: str = None, image_service: str = "openai") -> None:
        """
        Initialize the image rendering agent.

        Args:
            api_key: API key for the image generation service
            image_service: Name of the image service to use (openai, stability, midjourney)
        """
        super().__init__()
        self.api_key = api_key
        self.image_service = image_service

    def forward(
        self,
        image_prompt: str,
        caption: Optional[str] = None,
        format_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate and process a meme image.

        Args:
            image_prompt: Prompt for image generation
            caption: Optional caption to overlay
            format_details: Optional format specifications

        Returns:
            Dict containing image details
        """
        try:
            # Generate base image
            if self.image_service == "openai":
                image_data = self._generate_openai_image(image_prompt)
            elif self.image_service == "stability":
                image_data = self._generate_stability_image(image_prompt)
            elif self.image_service == "midjourney":
                image_data = self._generate_midjourney_image(image_prompt)
            else:
                raise ValueError(f"Unsupported image service: {self.image_service}")

            # Add text overlay if needed
            if caption and format_details and "text_overlay" in format_details.get("requirements", []):
                image_data = self._add_text_overlay(image_data, caption, format_details)

            # Upload to Cloudinary
            try:
                upload_result = cloudinary.uploader.upload(image_data)
            except Exception as e:
                raise RuntimeError(f"Failed to upload image: {str(e)}")

            return {
                "image_url": upload_result["secure_url"],
                "cloudinary_id": upload_result["public_id"],
                "width": upload_result["width"],
                "height": upload_result["height"]
            }

        except Exception as e:
            raise RuntimeError(f"Failed to generate image: {str(e)}")

    def _generate_openai_image(self, prompt: str) -> bytes:
        """Generate image using OpenAI's DALL-E."""
        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024"
            }
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI API error: {response.text}")
            
        image_url = response.json()["data"][0]["url"]
        image_response = requests.get(image_url)
        
        if image_response.status_code != 200:
            raise RuntimeError("Failed to download generated image")
            
        openai.api_key = self.api_key
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response["data"][0]["url"]
        return requests.get(image_url).content

    def _generate_stability_image(self, prompt: str) -> bytes:
        """Generate image using Stability AI."""
        try:
            from stability_sdk import client
            stability_api = client.StabilityInference(
                key=self.api_key,
                engine="stable-diffusion-v1-5"
            )
            response = stability_api.generate(
                prompt=prompt,
                width=512,
                height=512
            )
            return response.images[0].tobytes()
        except ImportError:
            raise RuntimeError("stability_sdk not installed. Please install it to use Stability AI.")

    def _add_text_overlay(
        self,
        image_data: bytes,
        caption: str,
        format_details: Dict[str, Any]
    ) -> bytes:
        """Add text overlay to the image."""
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_data))
        draw = ImageDraw.Draw(image)

        # Load font
        try:
            font = ImageFont.truetype("impact.ttf", 40)
        except IOError:
            font = ImageFont.load_default()

        # Calculate text position
        width, height = image.size
        text_width, text_height = draw.textsize(caption, font=font)
        text_position = ((width - text_width) // 2, height - text_height - 20)

        # Draw text with outline
        outline_color = (0, 0, 0)
        text_color = (255, 255, 255)

        # Draw outline
        for offset_x, offset_y in [(+1, 0), (-1, 0), (0, +1), (0, -1)]:
            draw.text(
                (text_position[0] + offset_x, text_position[1] + offset_y),
                caption,
                font=font,
                fill=outline_color
            )

        # Draw main text
        draw.text(text_position, caption, font=font, fill=text_color)

        # Convert back to bytes
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    def add_text_overlay(
        self,
        image_data: bytes,
        caption: str,
        position: str = "bottom"
    ) -> bytes:
        """
        Add text overlay to an image.
        
        Args:
            image_data: Raw image data
            caption: Text to overlay
            position: Where to place the text
            
        Returns:
            Processed image data as bytes
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(image_data))
            draw = ImageDraw.Draw(image)
            
            # Load font (using default for now)
            font = ImageFont.load_default()
            
            # Calculate text position
            width, height = image.size
            text_width, text_height = draw.textsize(caption, font=font)
            
            if position == "top":
                text_position = ((width - text_width) // 2, 20)
            elif position == "bottom":
                text_position = ((width - text_width) // 2, height - text_height - 20)
            else:  # center
                text_position = ((width - text_width) // 2, (height - text_height) // 2)
            
            # Draw text with outline for visibility
            outline_color = (0, 0, 0)
            text_color = (255, 255, 255)
            
            # Draw outline
            for offset_x, offset_y in [(+1, 0), (-1, 0), (0, +1), (0, -1)]:
                draw.text(
                    (text_position[0] + offset_x, text_position[1] + offset_y),
                    caption,
                    font=font,
                    fill=outline_color
                )
            
            # Draw main text
            draw.text(text_position, caption, font=font, fill=text_color)
            
            # Convert back to bytes
            output = BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error adding text overlay: {str(e)}")
            raise RuntimeError(f"Failed to add text overlay: {str(e)}")

    def upload_to_cloudinary(self, image_data: bytes) -> Dict[str, Any]:
        """
        Upload an image to Cloudinary.
        
        Args:
            image_data: Raw image data to upload
            
        Returns:
            Cloudinary upload response
        """
        try:
            # Create a BytesIO object from the image data
            image_io = BytesIO(image_data)
            
            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(
                image_io,
                folder="meme_generator",
                resource_type="image"
            )
            
            return upload_result
            
        except Exception as e:
            logger.error(f"Error uploading to Cloudinary: {str(e)}")
            raise RuntimeError(f"Failed to upload image: {str(e)}") 