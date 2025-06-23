#!/usr/bin/env python3
"""Test gpt-image-1 generation with simple prompts."""

import requests
import json
import time
from pathlib import Path


BASE_URL = "http://localhost:8081"


def test_simple_meme():
    """Test generating a meme with a simple prompt."""
    print("🎨 Testing GPT-Image-1 Generation")
    print("=" * 60)
    
    # Use simple, wholesome topics
    test_cases = [
        {"topic": "coding", "format": "Drake meme"},
        {"topic": "coffee", "format": "This is fine"},
        {"topic": "cats", "format": "Woman yelling at cat"},
        {"topic": "debugging", "format": "Expanding brain"},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['topic']} ({test_case['format']})")
        
        response = requests.post(f"{BASE_URL}/api/v1/memes/", json=test_case)
        
        if response.status_code == 201:
            meme_data = response.json()
            print(f"   ✅ Generated meme ID: {meme_data['id']}")
            print(f"   📝 Text: {meme_data.get('text', 'N/A')}")
            print(f"   🖼️  Image URL: {meme_data.get('image_url', 'N/A')}")
            
            # Check if it's a local file (gpt-image-1) or remote (DALL-E)
            image_url = meme_data.get('image_url', '')
            if image_url.startswith('/static/'):
                print("   ✨ SUCCESS: Used gpt-image-1 (local file)")
                
                # Check if EXIF was embedded
                image_path = image_url.replace("/static/", "static/")
                if Path(image_path).exists():
                    print(f"   📁 Local file exists: {image_path}")
            elif 'oaidalleapiprodscus.blob.core.windows.net' in image_url:
                print("   ⚠️  Fell back to DALL-E 3 (remote URL)")
            else:
                print(f"   ❓ Unknown provider: {image_url}")
                
            # Get metadata
            metadata_response = requests.get(
                f"{BASE_URL}/api/v1/analytics/memes/{meme_data['id']}/metadata"
            )
            if metadata_response.status_code == 200:
                metadata = metadata_response.json()
                if metadata.get('image_metadata'):
                    image_meta = metadata['image_metadata']
                    print(f"   🔍 Provider: {image_meta.get('provider', 'N/A')}")
                    if image_meta.get('file_size_bytes'):
                        print(f"   📊 File size: {image_meta['file_size_mb']} MB")
                    if image_meta.get('has_exif'):
                        print("   📸 EXIF data: Present")
        else:
            print(f"   ❌ Failed to generate meme: {response.status_code}")
            print(f"   Error: {response.text}")
        
        time.sleep(2)  # Rate limiting


if __name__ == "__main__":
    print("🎭 MEMESPY GPT-IMAGE-1 TEST")
    print("=" * 60)
    print("Testing various simple prompts to avoid moderation")
    
    test_simple_meme()
    
    print("\n\n✅ Test completed!")