"""DSPy-powered trend analysis for meme generation."""

from typing import Dict, List, Any, Optional
import logging
import uuid
import time
import json

import dspy
from ..config.config import settings

# Set up logging
logging.basicConfig(
    level=logging.getLevelName(settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TrendTopic(dspy.Signature):
    """DSPy signature for trend topic analysis."""

    query: str = dspy.InputField(description="The query to analyze trends for")
    topics: List[Dict[str, Any]] = dspy.OutputField(
        description="List of trending topics with details"
    )


class TrendAnalyzer:
    """DSPy-powered trend analyzer for meme generation."""

    def __init__(self):
        """Initialize the DSPy trend analyzer."""
        logger.info("Initializing DSPy trend analyzer")

        try:
            api_key = settings.openai_api_key
            if not api_key:
                logger.error("OpenAI API key not found in environment variables")
                raise ValueError("OpenAI API key not found")

            logger.info(f"Initializing DSPy with OpenAI API key (first chars: {api_key[:5]}...)")

            # Create the OpenAI model
            self.llm = dspy.LM(
                model=f"openai/{settings.dspy_model}",
                api_key=api_key,
                temperature=settings.dspy_temperature,
                max_tokens=settings.dspy_max_tokens,
            )

            # Configure DSPy with this LM
            dspy.settings.configure(lm=self.llm)

            # Create the trend analyzer predictor
            self.trend_predictor = dspy.ChainOfThought(TrendTopic)
            logger.info("DSPy trend analyzer initialized successfully")

            # Cache for trend topics
            self.topics_cache = {}

        except Exception as e:
            logger.error(f"Failed to initialize DSPy trend analyzer: {e}")
            raise

    async def get_trending_topics(
        self, query: Optional[str] = "current internet meme trends"
    ) -> List[Dict[str, Any]]:
        """
        Get trending topics for meme generation using DSPy.

        Args:
            query: Query for trends to analyze

        Returns:
            List of trending topics with details
        """
        if not query:
            query = "current internet meme trends"

        logger.info(f"Getting trending topics with query: {query}")

        try:
            # Generate trending topics using DSPy
            start_time = time.time()
            result = self.trend_predictor(query=query)
            generation_time = time.time() - start_time

            logger.info(f"Generated trending topics in {generation_time:.2f}s")

            # Process the topics
            topics = result.topics

            # Add IDs and cache the topics
            for topic in topics:
                if "id" not in topic:
                    topic_id = str(uuid.uuid4())
                    topic["id"] = topic_id
                    self.topics_cache[topic_id] = topic

            logger.info(f"Generated {len(topics)} trending topics")
            return topics
        except Exception as e:
            logger.error(f"Error generating trending topics: {e}")
            # Return empty list in case of error
            return []

    async def get_trending_topic(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific trending topic.

        Args:
            topic_id: ID of the trending topic

        Returns:
            Trending topic details if found, None otherwise
        """
        logger.info(f"Looking for trending topic with ID: {topic_id}")

        # Check if the topic is in the cache
        if topic_id in self.topics_cache:
            logger.info(f"Found trending topic in cache: {self.topics_cache[topic_id]['name']}")
            return self.topics_cache[topic_id]

        logger.warning(f"Trending topic with ID {topic_id} not found")
        return None


# Create a singleton instance
trend_analyzer = TrendAnalyzer()
