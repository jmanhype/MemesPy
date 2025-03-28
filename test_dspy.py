"""Simple script to test DSPy integration."""

import sys
import logging
import asyncio
import dspy
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
print("Loading .env file...")
load_dotenv(verbose=True)

# Get API key directly from environment
api_key = os.environ.get("OPENAI_API_KEY")
print(f"Environment variables loaded. API Key present: {api_key is not None}")
print(f"API Key: {api_key}")

# Hardcoded API key for testing
HARDCODED_API_KEY = "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"
print(f"Hardcoded API Key: {HARDCODED_API_KEY[:10]}...{HARDCODED_API_KEY[-4:]}")


def setup_dspy() -> Optional[dspy.LM]:
    """Set up DSPy with the OpenAI API key."""
    # Use hardcoded key for testing
    test_api_key = HARDCODED_API_KEY
    
    try:
        model_name = os.environ.get("DSPY_MODEL_NAME", "gpt-4")
        logger.info(f"Initializing DSPy with model: {model_name}")
        
        print(f"Using API key: {test_api_key[:10]}...{test_api_key[-4:]}")
        
        # Creating LM with direct API key
        lm = dspy.LM(
            model=f"openai/{model_name}",
            api_key=test_api_key
        )
        dspy.settings.configure(lm=lm)
        logger.info("DSPy successfully configured!")
        return lm
    except Exception as e:
        logger.error(f"Failed to configure DSPy: {e}")
        return None


class MemePrompt(dspy.Signature):
    """DSPy signature for meme generation."""
    
    topic: str = dspy.InputField(description="Topic for the meme")
    format: str = dspy.InputField(description="Format for the meme")
    
    text: str = dspy.OutputField(description="Generated meme text")


async def generate_meme_text(topic: str, format_id: str) -> Dict[str, Any]:
    """Generate meme text using DSPy."""
    try:
        # Create a DSPy Predict module for meme generation
        meme_generator = dspy.Predict(MemePrompt)
        
        # Generate meme text
        logger.info(f"Generating meme for topic: {topic}, format: {format_id}")
        prediction = meme_generator(topic=topic, format=format_id)
        
        logger.info(f"Generated meme text: {prediction.text}")
        
        # Create meme data dictionary
        meme_data = {
            "topic": topic,
            "format": format_id,
            "text": prediction.text,
            "success": True
        }
        
        return meme_data
    except Exception as e:
        logger.error(f"Error generating meme: {e}")
        return {
            "topic": topic,
            "format": format_id,
            "text": f"Error: {str(e)}",
            "success": False
        }


async def main():
    """Main function to test DSPy integration."""
    # Initialize DSPy
    lm = setup_dspy()
    
    if not lm:
        logger.error("Failed to initialize DSPy. Exiting.")
        return
    
    # Test topics
    test_topics = [
        "Python Programming"
    ]
    
    # Generate memes for test topics
    results = []
    for topic in test_topics:
        meme = await generate_meme_text(topic, "standard")
        results.append(meme)
    
    # Print results
    for i, result in enumerate(results):
        print(f"\n--- Meme {i+1} ---")
        print(f"Topic: {result['topic']}")
        print(f"Format: {result['format']}")
        print(f"Text: {result['text']}")
        print(f"Success: {result['success']}")


if __name__ == "__main__":
    asyncio.run(main()) 