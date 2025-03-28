#!/usr/bin/env python3
"""
Minimal DSPy test that only uses the empty organization parameter.
"""

import os
import re
import logging
import sys
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import DSPy
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

class SimplePrompt(dspy.Signature):
    """Simple DSPy signature for testing."""
    question: str = dspy.InputField(description="The user's question")
    answer: str = dspy.OutputField(description="The answer to the question")

def main():
    """Test DSPy with empty organization."""
    print("Starting DSPy minimal test...")
    
    # Read API key
    api_key = read_api_key_from_env_file()
    if not api_key:
        logger.error("No API key found")
        return
    
    logger.info(f"API Key found: {api_key[:10]}...")
    
    try:
        # Configure DSPy with empty organization
        model_name = "gpt-3.5-turbo-0125"
        llm = dspy.LM(
            model=f"openai/{model_name}",
            api_key=api_key,
            temperature=0.7,
            max_tokens=100,
            organization=""  # Empty organization string
        )
        
        dspy.settings.configure(lm=llm)
        logger.info("DSPy configured successfully")
        
        # Create a simple predictor
        predictor = dspy.Predict(SimplePrompt)
        
        # Test with a simple question
        print("Making prediction...")
        result = predictor(question="What is 2+2?")
        
        print(f"\nQuestion: What is 2+2?")
        print(f"Answer: {result.answer}")
        print("DSPy test completed successfully.")
        
    except Exception as e:
        logger.error(f"Error testing DSPy: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 