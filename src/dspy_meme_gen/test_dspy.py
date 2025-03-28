"""Simple test file to verify DSPy configuration."""

import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dspy_configuration():
    """Test that DSPy can be properly configured with OpenAI."""
    try:
        import dspy
        logger.info("Successfully imported DSPy")
        
        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            return False
            
        # Try to configure DSPy with OpenAI
        try:
            lm = dspy.LM("openai/gpt-3.5-turbo-0125", api_key=api_key)
            dspy.configure(lm=lm)
            logger.info("Successfully configured DSPy with OpenAI")
            
            # Test a simple completion to verify it works
            result = lm("What is DSPy?")
            logger.info(f"LM Response: {result}")
            logger.info("DSPy is working correctly!")
            return True
        except Exception as e:
            logger.error(f"Failed to configure DSPy: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    except ImportError:
        logger.error("Failed to import DSPy")
        return False

if __name__ == "__main__":
    success = test_dspy_configuration()
    sys.exit(0 if success else 1) 