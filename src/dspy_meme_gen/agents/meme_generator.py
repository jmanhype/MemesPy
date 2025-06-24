"""DSPy-powered meme text generation."""

from typing import Optional, Dict, Any, List, cast
import logging
import uuid
from datetime import datetime
import os
import random
import time

import dspy
from pydantic import BaseModel, Field

from ..config.config import settings

# Set up logging
logging.basicConfig(
    level=logging.getLevelName(settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MemePrompt(dspy.Signature):
    """DSPy signature for meme generation."""

    topic: str = dspy.InputField(description="The topic of the meme")
    format: str = dspy.InputField(description="The format of the meme")

    text: str = dspy.OutputField(description="The generated text for the meme")


class MemeGenerator:
    """Generate memes using DSPy."""

    def __init__(self):
        """Initialize the MemeGenerator."""
        logger.info("Initializing MemeGenerator with DSPy")

        # Initialize DSPy with OpenAI
        try:
            api_key = settings.openai_api_key
            if not api_key:
                # In test environments, we might use mocks
                import sys

                if "pytest" in sys.modules:
                    api_key = "test-key"  # Tests should mock this anyway
                else:
                    logger.error("OpenAI API key not found in environment variables")
                    raise ValueError("OpenAI API key not found")

            logger.info(f"Initializing DSPy with OpenAI API key (first chars: {api_key[:5]}...)")

            # Create the OpenAI model
            self.llm = dspy.LM(
                model=f"openai/{settings.dspy_model}",
                api_key=api_key,
                temperature=0.7,
                max_tokens=1000,
            )

            # Configure DSPy with this LM
            dspy.settings.configure(lm=self.llm)

            # Create the meme generator predictor
            self.meme_generator = dspy.Predict(MemePrompt)
            logger.info("DSPy initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DSPy: {e}")
            raise

    async def generate_meme(self, topic: str, format: str = "standard") -> Dict[str, Any]:
        """Generate a meme for the given topic using DSPy.

        Args:
            topic: The topic for the meme
            format: The meme format (e.g., standard, comparison, distracted)

        Returns:
            Dict containing the generated meme details
        """
        logger.info(f"Generating meme for topic: {topic}, format: {format}")

        try:
            # Generate the meme text using DSPy
            start_time = time.time()
            result = self.meme_generator(topic=topic, format=format)
            generation_time = time.time() - start_time

            logger.info(f"Generated meme text in {generation_time:.2f}s: {result.text}")

            # Create a unique ID for the meme
            meme_id = str(uuid.uuid4())

            # Build the response
            response = {
                "id": meme_id,
                "topic": topic,
                "format": format,
                "text": result.text,
                "image_url": f"https://placeholder.pics/svg/300/DEDEDE/555555/{topic}",  # Placeholder
                "created_at": time.time(),
                "score": 0.95,  # Placeholder score
            }

            return response
        except Exception as e:
            logger.error(f"Error generating meme: {e}")
            raise


# Singleton instance - create lazily to avoid import-time API key requirements
_meme_generator = None


def get_meme_generator():
    """Get or create the singleton MemeGenerator instance."""
    global _meme_generator
    if _meme_generator is None:
        _meme_generator = MemeGenerator()
    return _meme_generator
