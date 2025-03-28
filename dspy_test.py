#!/usr/bin/env python3
import os
import sys
import logging
import json
import re
from typing import Optional, Dict, Any, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import DSPy after setting up logging
try:
    import dspy
    from dspy.predict import Predict
except ImportError:
    logger.error("DSPy is not installed. Please install it with: pip install dspy")
    sys.exit(1)

# Function to directly read the API key from .env file
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

def read_org_id_from_env_file() -> str:
    """
    Read the OpenAI organization ID directly from the .env file
    
    Returns:
        str: The OpenAI organization ID or empty string if not found
    """
    try:
        with open('.env', 'r') as f:
            content = f.read()
            
        # Use regex to find the OPENAI_ORGANIZATION line
        match = re.search(r'OPENAI_ORGANIZATION(?:_ID)?=([^\n]+)', content)
        if match:
            return match.group(1).strip()
        else:
            logger.info("Could not find OPENAI_ORGANIZATION or OPENAI_ORGANIZATION_ID in .env file")
            return ""
    except Exception as e:
        logger.error(f"Error reading .env file: {e}")
        return ""

def extract_project_id_from_key(api_key: str) -> str:
    """
    Try to extract a project ID from an OpenAI API key.
    For project API keys (sk-proj-*), the project ID is sometimes embedded in the key.
    
    Args:
        api_key: The OpenAI API key
        
    Returns:
        str: The extracted project ID, or an empty string if not found
    """
    if not api_key.startswith("sk-proj-"):
        return ""
    
    # For project API keys, try using the first part of the key after the "sk-proj-" prefix
    # This is a guess based on the API key format
    try:
        # Split by hyphen and take project ID part (typically 3rd segment)
        parts = api_key.split("-")
        if len(parts) >= 3:
            project_id = parts[2]
            # Limit to a reasonable length
            return project_id[:24]
    except Exception as e:
        logger.error(f"Error extracting project ID from API key: {e}")
    
    # As an alternative, let's just use the first 24 characters after the prefix
    if len(api_key) > 8:  # "sk-proj-" is 8 chars
        return api_key[8:32]
    
    return ""

def setup_dspy() -> None:
    """
    Set up the DSPy library with OpenAI configuration.
    
    This function reads the API key from the .env file and sets
    environment variables appropriately before configuring DSPy.
    """
    # Read API key from .env file
    api_key = read_api_key_from_env_file()
    
    # For debugging, log the first few characters of the API key
    if api_key:
        logger.info(f"API Key found: {api_key[:10]}...")
    else:
        logger.error("No API key found")
        return
    
    # Set important environment variables
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_ORG_ID"] = ""  # Empty string works for project API keys
    
    logger.info("Environment variables set: OPENAI_API_KEY and OPENAI_ORG_ID (empty)")
    
    # Configuration parameters
    model_name = "gpt-3.5-turbo-0125"
    temperature = 0.7
    max_tokens = 1024
    
    logger.info(f"Using model: {model_name}, temperature: {temperature}, max_tokens: {max_tokens}")
    
    try:
        # Configure with dspy.LM using the format "provider/model_name"
        openai_model_string = f"openai/{model_name}"
        
        # Create config without specifying organization - it will use environment variables
        config = {
            "model": openai_model_string,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Create the LM
        llm = dspy.LM(**config)
        
        # Configure DSPy with this LM
        dspy.settings.configure(lm=llm)
        logger.info(f"Successfully configured DSPy with model: {openai_model_string}")
    except Exception as e:
        logger.error(f"Error initializing DSPy: {e}")
        raise

class MemePrompt(dspy.Signature):
    """Signature for meme generation."""
    topic: str = dspy.InputField(description="The topic of the meme")
    format: str = dspy.InputField(description="The format of the meme")
    text: str = dspy.OutputField(description="The text for the meme")

def generate_meme_text(topic: str, meme_format: str = "standard") -> Dict[str, str]:
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
        response = meme_generator(topic=topic, format=meme_format)
        
        print(f"Meme generated successfully for topic: {topic}")
        logger.info(f"Successfully generated meme for topic: {topic}")
        print(f"Generated text: {response.text}")
        
        # Return the meme text - note that we only have the 'text' field
        return {
            "topic": topic,
            "format": meme_format,
            "text": response.text
        }
    except Exception as e:
        print(f"Error in meme generation step: {e}")
        logger.error(f"Error in meme generation step: {e}")
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
            logger.error(f"Error generating meme: {result['error']}")
        else:
            print(f"\nMeme for topic: {result['topic']}")
            print(f"Text: {result['text']}")
            logger.info(f"Successfully generated meme for topic: {result['topic']}")
    
    print("Main function completed")

if __name__ == "__main__":
    main() 