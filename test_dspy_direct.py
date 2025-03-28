"""Simple script to test DSPy using the LM class directly."""

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
load_dotenv()

# Get API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
logger.info(f"API Key found: {api_key is not None}")


def setup_dspy() -> Optional[dspy.LM]:
    """Set up DSPy with the OpenAI API key."""
    if not api_key:
        logger.error("OpenAI API key not found!")
        return None
    
    try:
        # Use LM class directly instead of more specific OpenAI class
        logger.info("Initializing DSPy with model: gpt-3.5-turbo-0125")
        lm = dspy.LM(
            model="openai/gpt-3.5-turbo-0125",
            api_key=api_key
        )
        
        # Configure DSPy to use this language model
        dspy.settings.configure(lm=lm)
        logger.info("DSPy successfully configured!")
        return lm
    except Exception as e:
        logger.error(f"Failed to configure DSPy: {e}")
        return None


class MemePrompt(dspy.Signature):
    """DSPy signature for meme generation."""
    
    topic: str = dspy.InputField()
    format: str = dspy.InputField()
    
    text: str = dspy.OutputField()


async def test_dspy():
    """Test DSPy using the LM directly."""
    # Initialize DSPy
    lm = setup_dspy()
    
    if not lm:
        logger.error("Failed to initialize DSPy. Exiting.")
        return
    
    # Define a predictor
    meme_generator = dspy.Predict(MemePrompt)
    
    # Test topic
    topic = "Python Programming"
    format_id = "standard"
    
    try:
        # Generate meme text
        logger.info(f"Generating meme for topic: {topic}, format: {format_id}")
        prediction = meme_generator(topic=topic, format=format_id)
        
        logger.info(f"Generated meme text: {prediction.text}")
        
        # Print the result
        print(f"\n--- Generated Meme ---")
        print(f"Topic: {topic}")
        print(f"Format: {format_id}")
        print(f"Text: {prediction.text}")
        print(f"Success: True")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\n--- Generated Meme ---")
        print(f"Topic: {topic}")
        print(f"Format: {format_id}")
        print(f"Text: Error: {e}")
        print(f"Success: False")
    
    # Test the raw LM directly
    try:
        # Create a direct prompt
        prompt = f"""Create a funny meme text about {topic} in the {format_id} format.
Keep it short and witty."""
        
        logger.info("Testing direct LM call")
        response = lm(prompt)
        
        logger.info(f"Direct LM response: {response}")
        print(f"\n--- Direct LM Call ---")
        print(f"Prompt: Create a meme about {topic}")
        print(f"Response: {response}")
        print(f"Success: True")
    
    except Exception as e:
        logger.error(f"Direct LM error: {e}")
        print(f"\n--- Direct LM Call ---")
        print(f"Error: {e}")
        print(f"Success: False")


if __name__ == "__main__":
    asyncio.run(test_dspy()) 