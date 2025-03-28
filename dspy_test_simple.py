#!/usr/bin/env python3
"""
Simplified DSPy test script that focuses on making the API work
without organization ID.
"""
import os
import sys
import logging
import re
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import DSPy after setting up logging
try:
    import dspy
except ImportError:
    logger.error("DSPy is not installed. Please install it with: pip install dspy")
    sys.exit(1)

def read_api_key_from_env_file() -> str:
    """
    Read the OpenAI API key directly from the .env file
    
    Returns:
        str: The OpenAI API key
    """
    try:
        with open('.env', 'r') as f:
            content = f.read()
            
        # Use regex to find the OPENAI_API_KEY line
        match = re.search(r'OPENAI_API_KEY=([^\n]+)', content)
        if match:
            return match.group(1).strip()
        else:
            logger.error("Could not find OPENAI_API_KEY in .env file")
            return ""
    except Exception as e:
        logger.error(f"Error reading .env file: {e}")
        return ""

def setup_dspy() -> None:
    """
    Set up the DSPy library with OpenAI configuration
    """
    api_key = read_api_key_from_env_file()
    
    # For debugging, log the first few characters of the API key
    if api_key:
        logger.info(f"API Key found: {api_key[:10]}...")
    else:
        logger.error("No API key found")
        return
    
    # Hardcode these values for now
    model_name = "gpt-3.5-turbo-0125"
    temperature = 0.7
    max_tokens = 1024
    
    logger.info(f"Using model: {model_name}, temperature: {temperature}, max_tokens: {max_tokens}")
    
    # Debug print (remove in production)
    print(f"DEBUG - Using API key (first 10 chars): {api_key[:10]}...")
    
    # Initialize DSPy with the OpenAI LM
    try:
        # Configure with dspy.LM using the format "provider/model_name"
        # For OpenAI, the format is "openai/model_name"
        openai_model_string = f"openai/{model_name}"
        
        # Based on direct API testing, the key works without organization parameter
        llm = dspy.LM(
            model=openai_model_string,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        dspy.settings.configure(lm=llm)
        logger.info(f"Configured DSPy with LM using model: {openai_model_string}")
    except Exception as e:
        logger.error(f"Error initializing DSPy: {e}")
        raise

class MemePrompt(dspy.Signature):
    """Signature for meme generation."""
    topic: str = dspy.InputField(description="The topic of the meme")
    format: str = dspy.InputField(description="The format of the meme")
    text: str = dspy.OutputField(description="The text for the meme")
    caption: str = dspy.OutputField(description="A caption for the meme")

def generate_meme_text(topic: str, meme_format: str = "standard") -> Dict[str, Any]:
    """
    Generate a meme text for the given topic and format using DSPy.
    
    Args:
        topic: The topic to generate a meme for
        meme_format: The format of the meme (standard, motivational, etc.)
        
    Returns:
        Dictionary with meme text information
    """
    print(f"Starting meme generation for topic: {topic}")
    logger.info(f"Generating meme for topic: {topic}, format: {meme_format}")
    
    try:
        # Initialize DSPy
        print("Setting up DSPy...")
        setup_dspy()
        print("DSPy setup complete")
        
        # Create a predictor using dspy.Predict
        print("Creating meme generator...")
        meme_generator = dspy.Predict(MemePrompt)
        print("Meme generator created")
        
        # Generate the meme text by calling the predictor with the inputs
        print("Generating meme with predictor...")
        try:
            response = meme_generator(topic=topic, format=meme_format)
            print("Meme generation completed successfully")
            logger.info(f"Successfully generated meme for topic: {topic}")
            
            # Return the meme text
            return {
                "topic": topic,
                "format": meme_format,
                "caption": response.caption,
                "text": response.text
            }
        except Exception as inner_e:
            print(f"Error in meme generation step: {type(inner_e).__name__}: {inner_e}")
            logger.error(f"Error in meme generation step: {inner_e}")
            raise  # Re-raise to be caught by outer try-except
    except Exception as e:
        print(f"Error generating meme: {type(e).__name__}: {e}")
        logger.error(f"Error generating meme: {e}")
        # Return an error result with the topic included
        return {
            "topic": topic,
            "format": meme_format,
            "error": str(e)
        }

def main() -> None:
    """Main function to generate memes for different topics."""
    print("Starting main function")
    topics = ["Python Programming"]  # Only testing one topic for debugging
    
    for topic in topics:
        print(f"Processing topic: {topic}")
        result = generate_meme_text(topic)
        
        if "error" in result:
            print(f"\nError generating meme for topic: {result['topic']}")
            print(f"Error: {result['error']}")
        else:
            print(f"\nMeme for topic: {result['topic']}")
            print(f"Caption: {result['caption']}")
            print(f"Text: {result['text']}")
            print("---")
    
    print("Main function completed")

if __name__ == "__main__":
    main() 