#!/usr/bin/env python
"""
Test script for trying out OpenAI's image generation APIs.

This script demonstrates how to use both the gpt-image-1 and DALL-E 3 models
for image generation with various prompts and options.

Usage:
    python scripts/test_image_gen.py

Environment variables:
    OPENAI_API_KEY: Your OpenAI API key
"""

import os
import sys
import base64
import argparse
import requests
from typing import Optional, Tuple, Union
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI package not installed. Run 'pip install openai'")
    sys.exit(1)

def generate_image(
    prompt: str,
    model: str = "gpt-image-1",  # Now that organization is verified, use gpt-image-1 as default
    size: str = "1024x1024",
    meme_text: Optional[str] = None,
    style: Optional[str] = None,
    api_key: Optional[str] = None,
    output_file: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate an image using OpenAI's image generation API.
    
    Args:
        prompt: The image description
        model: Model to use (gpt-image-1 or dall-e-3)
        size: Image size (width x height)
        meme_text: Optional text to overlay on the image
        style: Optional style to apply
        api_key: OpenAI API key (reads from env if not provided)
        output_file: If provided and b64_json is returned, save image to this file
        
    Returns:
        Tuple containing:
        - URL to the generated image (if available)
        - Path to saved image (if b64_json was returned and output_file was provided)
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
    try:
        image_url = None
        b64_json = None
        
        if model == "gpt-image-1":
            # Try gpt-image-1 first - it returns base64 encoded image data by default
            try:
    response = client.images.generate(
                    model="gpt-image-1",
        prompt=enhanced_prompt,
        n=1,
        size=f"{width}x{height}"
    )
                # The response from gpt-image-1 includes b64_json by default
                b64_json = response.data[0].b64_json
                
            except Exception as e:
                if "organization must be verified" in str(e).lower() or "403" in str(e):
                    print("GPT-Image-1 not available due to organization verification requirements.")
                    print("Falling back to DALL-E 3...")
                    model = "dall-e-3"
                    # Continue to DALL-E 3 generation
                else:
                    raise  # Re-raise if it's a different error
        
        # If we're using DALL-E 3 (either by choice or fallback)
        if model == "dall-e-3":
            # DALL-E 3 specific parameters
            response = client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                n=1,
                size=f"{width}x{height}",
                quality="standard",
                response_format="url"
            )
            # DALL-E 3 returns a URL
    image_url = response.data[0].url
        
        # Handle b64_json
        saved_path = None
        if b64_json:
            print(f"Received base64 encoded image data (length: {len(b64_json)})")
            
            # Save to file if output_file is provided
            if output_file:
                # Generate a default filename if none provided
                if output_file == True:
                    output_file = f"generated_image_{model.replace('-', '_')}.png"
                
                # Decode and save
                image_data = base64.b64decode(b64_json)
                with open(output_file, "wb") as f:
                    f.write(image_data)
                saved_path = output_file
                print(f"Image saved to {saved_path}")
    
        return image_url, saved_path
        
    except Exception as e:
        print(f"Error generating image: {e}")
        raise

def main():
    """Main function for testing image generation APIs."""
    parser = argparse.ArgumentParser(description="Test OpenAI's image generation APIs")
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
    parser.add_argument("--save", type=str, nargs="?", const=True,
                        help="Save image to disk with optional filename")
    
    args = parser.parse_args()
    
    try:
        # Generate image
        image_url, saved_path = generate_image(
            prompt=args.prompt,
            model=args.model,
            size=args.size,
            meme_text=args.text,
            style=args.style,
            output_file=args.save
        )
        
        print("\nImage generation successful!")
        if image_url:
        print(f"Image URL: {image_url}")
        if saved_path:
            print(f"Image saved to: {saved_path}")
        
    except Exception as e:
        print(f"Error generating image: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 