#!/usr/bin/env python3
"""Test script to demonstrate comprehensive metadata collection."""

import asyncio
import json
import time
from datetime import datetime
import requests
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import piexif

# API base URL
BASE_URL = "http://localhost:8081"


def test_meme_generation_with_metadata():
    """Test meme generation and verify metadata collection."""
    
    print("🚀 Testing Meme Generation with Comprehensive Metadata\n")
    
    # Generate a meme
    payload = {
        "topic": "artificial intelligence taking over debugging",
        "format": "Drake meme"
    }
    
    print(f"📤 Generating meme: {payload}")
    start_time = time.time()
    
    response = requests.post(f"{BASE_URL}/api/v1/memes/", json=payload)
    generation_time = (time.time() - start_time) * 1000
    
    if response.status_code == 201:
        meme = response.json()
        print(f"✅ Meme generated successfully in {generation_time:.0f}ms")
        print(f"   ID: {meme['id']}")
        print(f"   Text: {meme['text']}")
        print(f"   Image URL: {meme['image_url']}")
        
        return meme['id'], meme['image_url']
    else:
        print(f"❌ Failed to generate meme: {response.text}")
        return None, None


def test_get_metadata(meme_id: str):
    """Test retrieving comprehensive metadata."""
    
    print(f"\n📊 Retrieving Metadata for Meme {meme_id}\n")
    
    response = requests.get(f"{BASE_URL}/api/v1/analytics/memes/{meme_id}/metadata")
    
    if response.status_code == 200:
        metadata = response.json()
        
        print("📋 Basic Information:")
        print(f"   Topic: {metadata['topic']}")
        print(f"   Format: {metadata['format']}")
        print(f"   Score: {metadata['score']}")
        print(f"   Created: {metadata['created_at']}")
        
        if metadata.get('generation_metadata'):
            print("\n⚙️  Generation Metadata:")
            gen_meta = metadata['generation_metadata']
            print(f"   Model Used: {gen_meta.get('model_used', 'N/A')}")
            print(f"   Generation Time: {gen_meta.get('total_duration_ms', 'N/A')}ms")
            print(f"   Retry Count: {gen_meta.get('retry_count', 0)}")
            print(f"   Efficiency Score: {gen_meta.get('efficiency_score', 'N/A')}")
        
        if metadata.get('image_metadata'):
            print("\n🖼️  Image Metadata:")
            img_meta = metadata['image_metadata']
            print(f"   Dimensions: {img_meta.get('width', 'N/A')}x{img_meta.get('height', 'N/A')}")
            print(f"   Format: {img_meta.get('format', 'N/A')}")
            print(f"   Size: {img_meta.get('size_bytes', 'N/A')} bytes")
            print(f"   Storage: {img_meta.get('storage_location', 'N/A')}")
        
        if metadata.get('dspy_metadata'):
            print("\n🤖 DSPy Metadata:")
            dspy_meta = metadata['dspy_metadata']
            print(f"   Model: {dspy_meta.get('model_name', 'N/A')}")
            print(f"   Temperature: {dspy_meta.get('temperature', 'N/A')}")
            print(f"   Total Tokens: {dspy_meta.get('total_tokens', 'N/A')}")
        
        if metadata.get('generation_cost'):
            print(f"\n💰 Generation Cost: ${metadata['generation_cost']:.4f}")
        
        return metadata
    else:
        print(f"❌ Failed to get metadata: {response.text}")
        return None


def test_exif_metadata(image_url: str):
    """Test EXIF metadata embedded in generated images."""
    
    if not image_url.startswith("/static/"):
        print("\n🔍 EXIF Metadata: Only available for locally stored images")
        return
    
    print("\n🔍 Checking EXIF Metadata in Generated Image\n")
    
    # Extract filename from URL
    filename = image_url.split("/")[-1]
    image_path = Path("static/images/memes") / filename
    
    if not image_path.exists():
        print(f"❌ Image file not found: {image_path}")
        return
    
    try:
        # Open image and read EXIF
        img = Image.open(image_path)
        exif_dict = piexif.load(str(image_path))
        
        print("📸 EXIF Data Found:")
        
        # Basic EXIF info
        if "0th" in exif_dict:
            for tag, value in exif_dict["0th"].items():
                tag_name = TAGS.get(tag, tag)
                if tag_name in ["Software", "DateTime", "HostComputer"]:
                    print(f"   {tag_name}: {value}")
        
        # Custom metadata in UserComment
        if "Exif" in exif_dict and piexif.ExifIFD.UserComment in exif_dict["Exif"]:
            user_comment = exif_dict["Exif"][piexif.ExifIFD.UserComment]
            if isinstance(user_comment, bytes):
                try:
                    # Remove encoding prefix if present
                    if user_comment.startswith(b'ASCII\x00\x00\x00'):
                        user_comment = user_comment[8:]
                    metadata = json.loads(user_comment.decode('utf-8'))
                    print("\n📝 Embedded Metadata:")
                    for key, value in metadata.items():
                        print(f"   {key}: {value}")
                except:
                    print(f"   Raw UserComment: {user_comment}")
        
    except Exception as e:
        print(f"❌ Error reading EXIF data: {e}")


def test_generation_stats():
    """Test generation statistics endpoint."""
    
    print("\n📈 Testing Generation Statistics\n")
    
    for period in ["hourly", "daily"]:
        response = requests.get(f"{BASE_URL}/api/v1/analytics/stats/{period}")
        
        if response.status_code == 200:
            stats = response.json()
            
            print(f"\n📊 {period.capitalize()} Statistics:")
            print(f"   Period: {stats['start_time']} to {stats['end_time']}")
            print(f"   Total Generations: {stats['total_generations']}")
            print(f"   Success Rate: {stats['success_rate']}%")
            print(f"   Avg Generation Time: {stats['avg_generation_time_ms']}ms")
            print(f"   P95 Generation Time: {stats['p95_generation_time_ms']}ms")
            
            if stats.get('model_breakdown'):
                print(f"\n   Model Usage:")
                for model, count in stats['model_breakdown'].items():
                    print(f"      {model}: {count}")
            
            print(f"\n   Quality Metrics:")
            print(f"      Avg Score: {stats['avg_score']}")
            print(f"      Viral Memes: {stats['viral_meme_count']}")
            
            print(f"\n   Cost Analysis:")
            print(f"      Total Cost: ${stats['total_cost']}")
            print(f"      Avg Cost/Meme: ${stats['avg_cost_per_meme']}")
            print(f"      Cache Hit Rate: {stats['cache_hit_rate']}%")


def test_metadata_search():
    """Test searching memes by metadata."""
    
    print("\n🔎 Testing Metadata Search\n")
    
    # Search for high-quality memes
    search_payload = {
        "min_score": 0.7,
        "sort_by": "score",
        "sort_order": "desc",
        "limit": 5
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/analytics/search", json=search_payload)
    
    if response.status_code == 200:
        results = response.json()
        print(f"Found {len(results)} high-quality memes:\n")
        
        for meme in results:
            print(f"   • {meme['topic']} ({meme['format']})")
            print(f"     Score: {meme['score']}, Cost: ${meme.get('generation_cost', 0):.3f}")
            print(f"     Generated in: {meme.get('total_duration_ms', 'N/A')}ms")
            print()


def test_trending_memes():
    """Test trending memes endpoint."""
    
    print("\n🔥 Testing Trending Memes\n")
    
    response = requests.get(f"{BASE_URL}/api/v1/analytics/trending?hours=24&limit=5")
    
    if response.status_code == 200:
        trending = response.json()
        
        if trending:
            print("Top trending memes (last 24 hours):\n")
            for idx, meme in enumerate(trending, 1):
                print(f"{idx}. {meme['topic']} - {meme['format']}")
                print(f"   Trending Score: {meme['trending_score']}")
                print(f"   Views: {meme['views']}, Likes: {meme['likes']}, Shares: {meme['shares']}")
                print()
        else:
            print("No trending memes found")


def test_batch_generation_for_analytics():
    """Generate multiple memes to populate analytics data."""
    
    print("\n🏭 Generating Multiple Memes for Analytics\n")
    
    topics = [
        ("machine learning", "Distracted boyfriend"),
        ("python vs javascript", "Drake meme"),
        ("debugging at 3am", "This is fine"),
        ("code reviews", "Expanding brain"),
        ("production deployments", "Sweating superhero")
    ]
    
    generated_ids = []
    
    for topic, format in topics:
        payload = {"topic": topic, "format": format}
        response = requests.post(f"{BASE_URL}/api/v1/memes/", json=payload)
        
        if response.status_code == 201:
            meme = response.json()
            generated_ids.append(meme['id'])
            print(f"✅ Generated: {topic} ({format}) - ID: {meme['id'][:8]}...")
        else:
            print(f"❌ Failed: {topic} ({format})")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    return generated_ids


async def main():
    """Run all metadata tests."""
    
    print("=" * 60)
    print("🎭 MEMESPY COMPREHENSIVE METADATA SYSTEM TEST")
    print("=" * 60)
    
    # Test 1: Generate a meme with full metadata tracking
    meme_id, image_url = test_meme_generation_with_metadata()
    
    if meme_id:
        # Test 2: Retrieve and display metadata
        metadata = test_get_metadata(meme_id)
        
        # Test 3: Check EXIF metadata in image
        if image_url:
            test_exif_metadata(image_url)
    
    # Test 4: Generate multiple memes for analytics
    print("\n" + "=" * 60)
    generated_ids = test_batch_generation_for_analytics()
    
    # Test 5: Test analytics endpoints
    print("\n" + "=" * 60)
    test_generation_stats()
    
    # Test 6: Test metadata search
    print("\n" + "=" * 60)
    test_metadata_search()
    
    # Test 7: Test trending memes
    print("\n" + "=" * 60)
    test_trending_memes()
    
    print("\n" + "=" * 60)
    print("✅ Metadata System Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())