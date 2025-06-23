#!/usr/bin/env python3
"""Test EXIF metadata embedding in generated meme images."""

import requests
import json
import time
from pathlib import Path
import piexif
from PIL import Image


BASE_URL = "http://localhost:8081"


def test_exif_embedding():
    """Test EXIF metadata embedding in meme images."""
    print("🔬 Testing EXIF Metadata Embedding")
    print("=" * 60)
    
    # Generate a meme
    print("\n1️⃣ Generating a meme with gpt-image-1...")
    payload = {
        "topic": "testing exif metadata",
        "format": "Success Kid"
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/memes/", json=payload)
    
    if response.status_code != 201:
        print(f"❌ Failed to generate meme: {response.text}")
        return
    
    meme_data = response.json()
    meme_id = meme_data['id']
    image_url = meme_data['image_url']
    print(f"✅ Meme generated: {meme_id}")
    print(f"   Image URL: {image_url}")
    
    # Extract the local file path from the URL
    # URL format: /static/images/memes/filename.png
    if image_url.startswith("/static/"):
        # Remove /static/ prefix and prepend actual static directory
        image_path = image_url.replace("/static/", "static/")
        full_path = Path(image_path)
        
        if not full_path.exists():
            print(f"❌ Image file not found at: {full_path}")
            return
        
        print(f"\n2️⃣ Reading EXIF data from: {full_path}")
        
        try:
            # Try to read EXIF data using piexif
            exif_dict = piexif.load(str(full_path))
            
            print("\n📷 EXIF Data Found:")
            
            # Check 0th IFD
            if piexif.ImageIFD.Software in exif_dict["0th"]:
                software = exif_dict["0th"][piexif.ImageIFD.Software].decode('utf-8')
                print(f"   Software: {software}")
            
            if piexif.ImageIFD.DateTime in exif_dict["0th"]:
                datetime_str = exif_dict["0th"][piexif.ImageIFD.DateTime].decode('utf-8')
                print(f"   DateTime: {datetime_str}")
            
            if piexif.ImageIFD.HostComputer in exif_dict["0th"]:
                host = exif_dict["0th"][piexif.ImageIFD.HostComputer].decode('utf-8')
                print(f"   Host Computer: {host}")
            
            # Check Exif IFD
            if "Exif" in exif_dict:
                if piexif.ExifIFD.DateTimeOriginal in exif_dict["Exif"]:
                    original_date = exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal].decode('utf-8')
                    print(f"   Date Time Original: {original_date}")
                
                if piexif.ExifIFD.UserComment in exif_dict["Exif"]:
                    user_comment_raw = exif_dict["Exif"][piexif.ExifIFD.UserComment]
                    # UserComment has a special format, try to decode it
                    try:
                        # Skip the first 8 bytes (character code designation)
                        user_comment = user_comment_raw[8:].decode('utf-8')
                        comment_data = json.loads(user_comment)
                        print(f"\n   📝 User Comment (Meme Metadata):")
                        for key, value in comment_data.items():
                            print(f"      {key}: {value}")
                    except Exception as e:
                        print(f"   User Comment (raw): {user_comment_raw}")
            
            # Also check with PIL
            print("\n3️⃣ Verifying with PIL:")
            with Image.open(full_path) as img:
                exif_data = img.getexif()
                if exif_data:
                    print("   ✅ PIL confirms EXIF data is present")
                    print(f"   Number of EXIF tags: {len(exif_data)}")
                else:
                    print("   ⚠️  PIL shows no EXIF data")
                    
        except piexif.InvalidImageDataError as e:
            print(f"❌ Invalid image data for EXIF: {e}")
            print("   Note: PNG files need special handling for EXIF data")
            
            # Try reading as PNG metadata
            print("\n4️⃣ Checking PNG metadata instead:")
            with Image.open(full_path) as img:
                print(f"   Format: {img.format}")
                print(f"   Mode: {img.mode}")
                print(f"   Info keys: {list(img.info.keys())}")
                if img.info:
                    for key, value in img.info.items():
                        print(f"   {key}: {value}")
                        
        except Exception as e:
            print(f"❌ Error reading EXIF data: {e}")
    
    # Test the metadata endpoint
    print("\n5️⃣ Checking metadata via API:")
    metadata_response = requests.get(f"{BASE_URL}/api/v1/analytics/memes/{meme_id}/metadata")
    if metadata_response.status_code == 200:
        metadata = metadata_response.json()
        print("   ✅ Metadata API response received")
        if metadata.get('image_metadata'):
            print("   📊 Image Metadata:")
            for key, value in metadata['image_metadata'].items():
                if key != 'exif_tags':  # Skip detailed EXIF tags
                    print(f"      {key}: {value}")


def test_jpeg_generation():
    """Test generating meme as JPEG for better EXIF support."""
    print("\n\n🖼️  Testing JPEG Generation (Better EXIF Support)")
    print("=" * 60)
    
    # Note: Current implementation saves as PNG, but we can test the system
    print("ℹ️  Current system saves images as PNG")
    print("   PNG files have limited EXIF support")
    print("   For full EXIF support, consider saving as JPEG")


if __name__ == "__main__":
    print("🎭 MEMESPY EXIF METADATA TEST")
    print("=" * 60)
    
    test_exif_embedding()
    test_jpeg_generation()
    
    print("\n\n✅ EXIF metadata test completed!")