"""DSPy-powered format generation for memes."""

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemeFormat(dspy.Signature):
    """DSPy signature for meme format generation."""
    query: str = dspy.InputField(description="The query to find meme formats for")
    formats: List[Dict[str, Any]] = dspy.OutputField(description="List of meme formats with details")

class FormatGenerator:
    """DSPy-powered format generator for memes."""
    
    def __init__(self):
        """Initialize the DSPy format generator."""
        logger.info("Initializing DSPy format generator")
        
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
                max_tokens=settings.dspy_max_tokens
            )
            
            # Configure DSPy with this LM
            dspy.settings.configure(lm=self.llm)
            
            # Create the format generator predictor
            self.format_predictor = dspy.ChainOfThought(MemeFormat)
            logger.info("DSPy format generator initialized successfully")
            
            # Cache for meme formats
            self.formats_cache = {}
            
        except Exception as e:
            logger.error(f"Failed to initialize DSPy format generator: {e}")
            raise
    
    async def get_formats(self, query: Optional[str] = "popular meme formats") -> List[Dict[str, Any]]:
        """
        Get available meme formats using DSPy.
        
        Args:
            query: Query for meme formats
            
        Returns:
            List of meme formats with details
        """
        if not query:
            query = "popular meme formats"
            
        logger.info(f"Getting meme formats with query: {query}")
        
        try:
            # Generate meme formats using DSPy
            start_time = time.time()
            result = self.format_predictor(query=query)
            generation_time = time.time() - start_time
            
            logger.info(f"Generated meme formats in {generation_time:.2f}s")
            
            # Process the formats
            formats = result.formats
            
            # Add IDs and cache the formats
            for format_item in formats:
                if "id" not in format_item:
                    format_id = format_item.get("name", "").lower().replace(" ", "_")
                    if not format_id:
                        format_id = str(uuid.uuid4())
                    format_item["id"] = format_id
                    self.formats_cache[format_id] = format_item
            
            logger.info(f"Generated {len(formats)} meme formats")
            return formats
        except Exception as e:
            logger.error(f"Error generating meme formats: {e}")
            # Return empty list in case of error
            return []
    
    async def get_format(self, format_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific meme format.
        
        Args:
            format_id: ID of the meme format
            
        Returns:
            Format details if found, None otherwise
        """
        logger.info(f"Looking for format with ID: {format_id}")
        
        # Check if the format is in the cache
        if format_id in self.formats_cache:
            logger.info(f"Found format in cache: {self.formats_cache[format_id]['name']}")
            return self.formats_cache[format_id]
        
        # If not in cache, try to get all formats and check again
        try:
            await self.get_formats()
            if format_id in self.formats_cache:
                logger.info(f"Found format after refresh: {self.formats_cache[format_id]['name']}")
                return self.formats_cache[format_id]
        except Exception as e:
            logger.error(f"Error refreshing formats: {e}")
        
        logger.warning(f"Format with ID {format_id} not found")
        return None


# Create a singleton instance
format_generator = FormatGenerator() 