#!/usr/bin/env python
"""
Test script for trying out OpenAI's gpt-image-1 API directly.

This script demonstrates how to use the gpt-image-1 model for image generation
with various prompts and options.

Usage:
    python scripts/test_gpt_image_1.py

Environment variables:
    OPENAI_API_KEY: Your OpenAI API key
"""

import os
import sys
import argparse
from typing import Optional
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI package not installed. Run 'pip install openai'")
    sys.exit(1)

def generate_image(
    prompt: str,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
    meme_text: Optional[str] = None,
    style: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Generate an image using OpenAI's image generation API.
    
    Args:
        prompt: The image description
        model: Model to use (gpt-image-1 or dall-e-3)
        size: Image size (width x height)
        meme_text: Optional text to overlay on the image
        style: Optional style to apply
        api_key: OpenAI API key (reads from env if not provided)
        
    Returns:
        URL to the generated image
    """
    # Use provided API key or get from environment
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided and not found in environment")
    
    # Create OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Enhance prompt with style and text overlay
    enhanced_prompt = prompt
    
    if style:
        enhanced_prompt += f", {style} style"
        
    if meme_text:
        enhanced_prompt += f". Include the text \"{meme_text}\" in the image, formatted as a meme caption."
    
    print(f"Using model: {model}")
    print(f"Size: {size}")
    print(f"Enhanced prompt: {enhanced_prompt}")
    
    # Parse size
    try:
        width, height = map(int, size.split("x"))
    except (ValueError, AttributeError):
        print(f"Invalid size format: {size}, defaulting to 1024x1024")
        width, height = 1024, 1024
    
    # Generate image
    response = client.images.generate(
        model=model,
        prompt=enhanced_prompt,
        n=1,
        size=f"{width}x{height}"
    )
    
    image_url = response.data[0].url
    print(f"Generated image URL: {image_url}")
    
    return image_url

def main():
    """Main function for testing gpt-image-1 API."""
    parser = argparse.ArgumentParser(description="Test OpenAI's gpt-image-1 API")
    parser.add_argument("--prompt", type=str, default="A funny programming meme about debugging code",
                        help="Image prompt")
    parser.add_argument("--model", type=str, default="gpt-image-1", 
                        help="Model to use (gpt-image-1 or dall-e-3)")
    parser.add_argument("--size", type=str, default="1024x1024", 
                        help="Image size (width x height)")
    parser.add_argument("--text", type=str, 
                        help="Optional text to overlay on the image")
    parser.add_argument("--style", type=str, 
                        help="Optional style to apply (e.g., 'anime', 'photorealistic')")
    
    args = parser.parse_args()
    
    try:
        # Generate image
        image_url = generate_image(
            prompt=args.prompt,
            model=args.model,
            size=args.size,
            meme_text=args.text,
            style=args.style,
        )
        
        print("\nImage generation successful!")
        print(f"Image URL: {image_url}")
        
    except Exception as e:
        print(f"Error generating image: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 