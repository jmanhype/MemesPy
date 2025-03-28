import logging
import asyncio
from typing import Dict, Any, Optional

import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath("."))

# Import the meme generator
from src.dspy_meme_gen.agents.meme_generator import MemeGenerator

async def test_meme_generation():
    """Test the meme generator with fallback functionality."""
    # Create a new instance of the MemeGenerator
    generator = MemeGenerator()
    
    # Define test topics
    test_topics = [
        "Python Programming",
        "Machine Learning",
        "JavaScript",
        "Database Design"
    ]
    
    # Generate memes for each topic
    for i, topic in enumerate(test_topics, 1):
        try:
            meme = await generator.generate_meme(topic=topic, format="standard")
            print(f"\n--- Meme {i} ---")
            print(f"Topic: {topic}")
            print(f"Format: standard")
            print(f"Text: {meme['text']}")
            print(f"ID: {meme['id']}")
            print(f"Success: True")
        except Exception as e:
            print(f"\n--- Meme {i} ---")
            print(f"Topic: {topic}")
            print(f"Format: standard")
            print(f"Text: Error: {e}")
            print(f"Success: False")

# Entry point
if __name__ == "__main__":
    asyncio.run(test_meme_generation()) 